import os
import sys
import torch
import faiss

import numpy as np
import torch.nn.functional as F

from scipy import signal

sys.path.append(os.getcwd())

from main.app.variables import translations
from main.library.predictors.Generator import Generator
from main.inference.extracting.rms import RMSEnergyExtractor
from main.library.utils import extract_features, change_rms, clear_gpu_cache, get_onnx_argument

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

class Pipeline:
    def __init__(self, tgt_sr, config):
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.f0_min = 50
        self.f0_max = 1100
        self.device = config.device
        self.is_half = config.is_half
        self.tgt_sr = tgt_sr

    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, index_rate, version, protect, energy):
        pitch_guidance = pitch != None and pitchf != None
        energy_use = energy != None

        feats = torch.from_numpy(audio0)
        feats = feats.half() if self.is_half else feats.float()

        feats = feats.mean(-1) if feats.dim() == 2 else feats
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)

        with torch.no_grad():
            if self.embed_suffix == ".pt":
                padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
                logits = model.extract_features(**{"source": feats.to(self.device), "padding_mask": padding_mask, "output_layer": 9 if version == "v1" else 12})
                feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
            elif self.embed_suffix == ".onnx": feats = extract_features(model, feats.to(self.device), version).to(self.device)
            elif self.embed_suffix == ".safetensors":
                logits = model(feats.to(self.device))["last_hidden_state"]
                feats = model.final_proj(logits[0]).unsqueeze(0) if version == "v1" else logits
            else: raise ValueError(translations["option_not_valid"])

            feats0 = feats.clone() if protect < 0.5 and pitch_guidance else None

            if (not isinstance(index, type(None)) and not isinstance(big_npy, type(None)) and index_rate != 0):
                npy = feats[0].cpu().numpy()
                if self.is_half: npy = npy.astype(np.float32)

                score, ix = index.search(npy, k=8)
                weight = np.square(1 / score)

                npy = np.sum(big_npy[ix] * np.expand_dims(weight / weight.sum(axis=1, keepdims=True), axis=2), axis=1)
                if self.is_half: npy = npy.astype(np.float16)

                feats = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats)

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            p_len = min(audio0.shape[0] // self.window, feats.shape[1])

            if pitch_guidance: pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
            if energy_use: energy = energy[:, :p_len]

            if feats0 is not None:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)

                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
                feats = (feats * pitchff + feats0 * (1 - pitchff)).to(feats0.dtype)

            p_len = torch.tensor([p_len], device=self.device).long()
            feats = feats.half() if self.is_half else feats.float()

            if not pitch_guidance: pitch, pitchf = None, None
            else: pitchf = pitchf.half() if self.is_half else pitchf.float()
            if not energy_use: energy = None
            else: energy = energy.half() if self.is_half else energy.float()

            audio1 = (
                (
                    net_g.infer(
                        feats, 
                        p_len, 
                        pitch, 
                        pitchf,
                        sid,
                        energy
                    )[0][0, 0]
                ).data.cpu().float().numpy()
            ) if self.suffix == ".pth" else (
                net_g.run(
                    [net_g.get_outputs()[0].name], (
                        get_onnx_argument(
                            net_g, 
                            feats, 
                            p_len, 
                            sid, 
                            pitch, 
                            pitchf, 
                            energy, 
                            pitch_guidance, 
                            energy_use
                        )
                    )
                )[0][0, 0]
            )

        if self.embed_suffix == ".pt": del padding_mask
        del feats, feats0, p_len

        clear_gpu_cache()
        return audio1
    
    def pipeline(self, logger, model, net_g, sid, audio, f0_up_key, f0_method, file_index, index_rate, pitch_guidance, filter_radius, rms_mix_rate, version, protect, hop_length, f0_autotune, f0_autotune_strength, suffix, embed_suffix, f0_file=None, f0_onnx=False, pbar=None, proposal_pitch=False, proposal_pitch_threshold=0, energy_use=False, del_onnx=True):
        self.embed_suffix = embed_suffix
        self.suffix = suffix

        if file_index != "" and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                logger.error(translations["read_faiss_index_error"].format(e=e))
                index = big_npy = None
        else: index = big_npy = None

        if pbar: pbar.update(1)

        opt_ts, audio_opt = [], []
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)

            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]

            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min())[0][0])

        s = 0
        t, inp_f0 = None, None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        p_len = audio_pad.shape[0] // self.window

        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    raw_lines = f.read()

                    if len(raw_lines) > 0:
                        inp_f0 = []

                        for line in raw_lines.strip("\n").split("\n"):
                            inp_f0.append([float(i) for i in line.split(",")])

                        inp_f0 = np.array(inp_f0, dtype=np.float32)
            except:
                logger.error(translations["error_readfile"])
                inp_f0 = None

        if pbar: pbar.update(1)

        if pitch_guidance:
            if not hasattr(self, "f0_generator"): self.f0_generator = Generator(self.sample_rate, hop_length, self.f0_min, self.f0_max, self.is_half, self.device, f0_onnx, del_onnx)
            pitch, pitchf = self.f0_generator.calculator(self.x_pad, f0_method, audio_pad, f0_up_key, p_len, filter_radius, f0_autotune, f0_autotune_strength, manual_f0=inp_f0, proposal_pitch=proposal_pitch, proposal_pitch_threshold=proposal_pitch_threshold)

            if self.device == "mps": pitchf = pitchf.astype(np.float32)
            pitch, pitchf = torch.tensor(pitch[:p_len], device=self.device).unsqueeze(0).long(), torch.tensor(pitchf[:p_len], device=self.device).unsqueeze(0).float()

        if pbar: pbar.update(1)

        if energy_use:
            if not hasattr(self, "rms_extract"): self.rms_extract = RMSEnergyExtractor(frame_length=2048, hop_length=self.window, center=True, pad_mode = "reflect").to(self.device).eval()
            energy = self.rms_extract(torch.from_numpy(audio_pad).to(self.device).unsqueeze(0)).cpu().numpy()
            
            if self.device == "mps": energy = energy.astype(np.float32)
            energy = torch.tensor(energy[:p_len], device=self.device).unsqueeze(0).float()

        if pbar: pbar.update(1)

        for t in opt_ts:
            t = t // self.window * self.window
            audio_opt.append(
                self.voice_conversion(
                    model, 
                    net_g, 
                    sid, 
                    audio_pad[s : t + self.t_pad2 + self.window], 
                    pitch[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, 
                    pitchf[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, 
                    index, 
                    big_npy, 
                    index_rate, 
                    version, 
                    protect, 
                    energy[:, s // self.window : (t + self.t_pad2) // self.window] if energy_use else None
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )    
            s = t
            
        audio_opt.append(
            self.voice_conversion(
                model, 
                net_g, 
                sid, 
                audio_pad[t:], 
                (pitch[:, t // self.window :] if t is not None else pitch) if pitch_guidance else None, 
                (pitchf[:, t // self.window :] if t is not None else pitchf) if pitch_guidance else None, 
                index, 
                big_npy, 
                index_rate, 
                version, 
                protect, 
                (energy[:, t // self.window :] if t is not None else energy) if energy_use else None
            )[self.t_pad_tgt : -self.t_pad_tgt]
        )

        if pbar: pbar.update(1)

        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1: audio_opt = change_rms(audio, self.sample_rate, audio_opt, self.sample_rate, rms_mix_rate)

        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1: audio_opt /= audio_max

        if pitch_guidance: del pitch, pitchf
        del sid

        clear_gpu_cache()
        return audio_opt