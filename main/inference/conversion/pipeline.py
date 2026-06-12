import os
import sys
import torch

import numpy as np
import torch.nn.functional as F

from scipy import signal

sys.path.append(os.getcwd())

from main.library.audio.features import change_rms
from main.app.variables import translations, logger
from main.library.utils import extract_features, clear_gpu_cache

bh, ah = signal.butter(
    N=5, 
    Wn=48, 
    btype="high", 
    fs=16000
)

class Pipeline:
    def __init__(
        self, 
        tgt_sr, 
        config, 
        net_g,
        hubert_model,
        f0_generator, 
        rms_extract, 
        version,
        sid,
        dtype
    ):

        self.window = 160
        self.sample_rate = 16000

        self.x_pad = config.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad = self.sample_rate * self.x_pad
        self.t_max = self.sample_rate * config.x_max
        self.t_query = self.sample_rate * config.x_query
        self.t_center = self.sample_rate * config.x_center
        self.t_pad2 = self.t_pad * 2

        self.sid = sid
        self.dtype = dtype
        self.net_g = net_g
        self.tgt_sr = tgt_sr
        self.version = version
        self.device = config.device
        self.rms_extract = rms_extract
        self.f0_generator = f0_generator
        self.hubert_model = hubert_model
        self.energy_use = rms_extract is not None
        self.pitch_guidance = f0_generator is not None

    def voice_conversion(
        self, 
        audio0, 
        pitch, 
        pitchf, 
        index, 
        big_tsr, 
        index_rate, 
        protect, 
        energy,
        embedders_mix = False,
        embedders_mix_layers = 9,
        embedders_mix_ratio = 0.5
    ):
        feats = torch.from_numpy(audio0).to(self.device).to(self.dtype)
        feats = feats.mean(-1) if feats.dim() == 2 else feats
        assert feats.dim() == 1, feats.dim()

        with torch.inference_mode():
            feats = extract_features(
                self.hubert_model, 
                feats.view(1, -1), 
                self.version, 
                mix=embedders_mix, 
                mix_layers=embedders_mix_layers, 
                mix_ratio=embedders_mix_ratio
            )

            feats0 = feats.clone() if protect < 0.5 and self.pitch_guidance else None

            if index is not None and big_tsr is not None and index_rate != 0:
                score, ix = index.search(feats[0], k=8)
                weight = (1 / score).square()
                query = (big_tsr[ix] * (weight / weight.sum(dim=1, keepdim=True)).unsqueeze(2)).sum(dim=1)
                feats = query.unsqueeze(0) * index_rate + (1.0 - index_rate) * feats

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            p_len = min(audio0.shape[0] // self.window, feats.shape[1])

            if self.pitch_guidance: pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
            if self.energy_use: energy = energy[:p_len].unsqueeze(0).to(self.dtype)

            if feats0 is not None:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)

                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
                feats = (feats * pitchff + feats0 * (1 - pitchff)).to(feats0.dtype)

            p_len = torch.tensor([p_len], device=self.device).long()
            if self.pitch_guidance: pitchf.to(self.dtype)
            feats = feats.to(self.dtype) 

            audio1 = self.net_g.infer(
                feats, 
                p_len, 
                pitch, 
                pitchf,
                self.sid,
                energy
            )[0][0, 0].cpu().float().numpy()

        del feats, feats0, p_len
        clear_gpu_cache()
        return audio1
    
    def pipeline(
        self, 
        audio, 
        f0_up_key, 
        f0_method, 
        index,
        big_tsr, 
        index_rate, 
        filter_radius, 
        rms_mix_rate, 
        protect, 
        f0_autotune, 
        f0_autotune_strength, 
        f0_file=None, 
        pbar=None, 
        proposal_pitch=False, 
        proposal_pitch_threshold=255.0, 
        embedders_mix = False,
        embedders_mix_layers = 9,
        embedders_mix_ratio = 0.5,
    ):
        s = 0
        t, inp_f0 = None, None
        opt_ts, audio_opt = [], []
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)

            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]

            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min())[0][0]
                )

        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        pbar.update(1)

        if f0_file and os.path.exists(f0_file) and f0_file.endswith(".txt"):
            try:
                with open(f0_file, "r") as f:
                    raw_lines = f.read()

                    if len(raw_lines) > 0:
                        inp_f0 = []

                        for line in raw_lines.strip("\n").split("\n"):
                            inp_f0.append([float(i) for i in line.split(",")])

                        inp_f0 = np.array(inp_f0, dtype=np.float32)
            except:
                logger.error(translations["error_readfile"])
                inp_f0 = None

        pbar.update(1)

        if self.pitch_guidance:
            pitch, pitchf = self.f0_generator.calculator(
                self.x_pad, 
                f0_method, 
                audio_pad, 
                f0_up_key, 
                p_len, 
                filter_radius, 
                f0_autotune, 
                f0_autotune_strength, 
                manual_f0=inp_f0, 
                proposal_pitch=proposal_pitch, 
                proposal_pitch_threshold=proposal_pitch_threshold
            )

            if self.device == "mps": pitchf = pitchf.astype(np.float32)

            pitch, pitchf = (
                torch.tensor(pitch[:p_len], device=self.device).unsqueeze(0).long(), 
                torch.tensor(pitchf[:p_len], device=self.device).unsqueeze(0).float()
            )

        pbar.update(1)

        if self.energy_use:
            energy = self.rms_extract(
                torch.from_numpy(audio_pad).to(self.device).unsqueeze(0)
            )[:p_len].to(self.device).float()

        pbar.update(1)
        pbar.total = pbar.total + len(opt_ts)
        pbar.refresh()

        for t in opt_ts:
            t = t // self.window * self.window
            start = s // self.window
            end = (t + self.t_pad2) // self.window

            audio_opt.append(
                self.voice_conversion(
                    audio_pad[s : t + self.t_pad2 + self.window], 
                    pitch[:, start:end] if self.pitch_guidance else None, 
                    pitchf[:, start:end] if self.pitch_guidance else None, 
                    index, 
                    big_tsr, 
                    index_rate, 
                    protect, 
                    energy[:, start:end] if self.energy_use else None,
                    embedders_mix=embedders_mix, 
                    embedders_mix_layers=embedders_mix_layers, 
                    embedders_mix_ratio=embedders_mix_ratio
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )

            s = t
            pbar.update(1)
        
        start_opt = (t // self.window) if t is not None else 0
        audio_opt.append(
            self.voice_conversion(
                audio_pad[t:], 
                pitch[:, start_opt:] if self.pitch_guidance else None, 
                pitchf[:, start_opt:] if self.pitch_guidance else None, 
                index, 
                big_tsr, 
                index_rate, 
                protect, 
                energy[:, start_opt:] if self.energy_use else None,
                embedders_mix=embedders_mix, 
                embedders_mix_layers=embedders_mix_layers, 
                embedders_mix_ratio=embedders_mix_ratio
            )[self.t_pad_tgt : -self.t_pad_tgt]
        )

        pbar.update(1)
        audio_opt = np.concatenate(audio_opt)

        if rms_mix_rate != 1:
            audio_opt = change_rms(
                audio, 
                self.sample_rate, 
                audio_opt, 
                self.tgt_sr, 
                rms_mix_rate,
                device=self.device
            ).cpu().numpy()

        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1: audio_opt /= audio_max
        if self.pitch_guidance: del pitch, pitchf

        clear_gpu_cache()
        return audio_opt