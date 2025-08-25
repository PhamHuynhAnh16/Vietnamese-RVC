import os
import sys
import onnx
import json
import torch

import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as tat

sys.path.append(os.getcwd())

from main.app.variables import config
from main.library.predictors.Generator import Generator
from main.library.algorithm.synthesizers import Synthesizer
from main.inference.extracting.rms import RMSEnergyExtractor
from main.library.utils import get_onnx_argument, load_embedders_model, extract_features, change_rms, load_faiss_index, load_model

class Inference:
    def get_synthesizer(self, model_path):
        model = load_model(model_path)

        if model_path.endswith(".pth"):
            self.tgt_sr = model["config"][-1]
            model["config"][-3] = model["weight"]["emb_g.weight"].shape[0]

            self.use_f0 = model.get("f0", 1)
            self.version = model.get("version", "v1")
            self.vocoder = model.get("vocoder", "Default")
            self.energy = model.get("energy", False)

            if self.vocoder != "Default": config.is_half = False

            net_g = Synthesizer(
                *model["config"], 
                use_f0=self.use_f0, 
                text_enc_hidden_dim=768 if self.version == "v2" else 256, 
                vocoder=self.vocoder, 
                checkpointing=False, 
                energy=self.energy
            )

            net_g.load_state_dict(model["weight"], strict=False)
            net_g.eval().to(config.device).to(torch.float16 if config.is_half else torch.float32)
            net_g.remove_weight_norm()

            self.net_g = net_g
            self.model = model
            self.suffix = ".pth"
        else:
            metadata_dict = None
            for prop in onnx.load(model_path).metadata_props:
                if prop.key == "model_info":
                    metadata_dict = json.loads(prop.value)
                    break

            self.tgt_sr = metadata_dict.get("sr", 32000)
            self.use_f0 = metadata_dict.get("f0", 1)
            self.version = metadata_dict.get("version", "v1")
            self.energy = metadata_dict.get("energy", False)

            self.net_g = model
            self.model = metadata_dict
            self.suffix = ".onnx"

        return self
    
    def inference(self, feats, p_len, sid, pitch, pitchf, energy):
        output = (
            self.net_g.infer(
                feats, 
                p_len, 
                pitch, 
                pitchf,
                sid,
                energy
            )[0][0, 0]
        ) if self.suffix == ".pth" else (
            torch.as_tensor(
                self.net_g.run(
                    [self.net_g.get_outputs()[0].name], (
                        get_onnx_argument(
                            self.net_g, 
                            feats, 
                            p_len, 
                            sid, 
                            pitch, 
                            pitchf, 
                            energy, 
                            self.use_f0, 
                            self.energy
                        )
                    )
                )[0][0, 0],
                device=config.device
            )
        )

        return torch.clip(output, -1.0, 1.0, out=output)
    
class Pipeline:
    def __init__(self, inference, embedder, predictor = None, rms = None, index = (None, None), f0_method = "rmvpe", sid = 0):
        self.inference = inference
        self.embedder = embedder
        self.predictor = predictor
        self.rms = rms
        self.index = index
        self.use_f0 = inference.use_f0
        self.tgt_sr = inference.tgt_sr
        self.energy = inference.energy
        self.f0_method = f0_method
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.device = config.device
        self.is_half = config.is_half
        self.dtype = torch.float16 if self.is_half else torch.float32
        self.model_window = self.tgt_sr // 100
        self.sid = torch.tensor([sid], device=self.device, dtype=torch.int64)
        self.resamplers = {}
    
    def execute(self, audio, pitch = None, pitchf = None, f0_up_key = 0, index_rate = 0.5, audio_feats_len = 0, silence_front = 0, skip_head = None, return_length = None, protect = 0.5, filter_radius = 3, rms_mix_rate = 1, f0_autotune = False, f0_autotune_strength = 1, proposal_pitch = False, proposal_pitch_threshold = 255.0):
        assert audio.dim() == 1, audio.dim()
        formant_length = int(np.ceil(return_length * 1.0))

        pitch, pitchf = self.predictor.realtime_calculator(audio[silence_front:], self.f0_method, pitch, pitchf, f0_up_key, filter_radius, f0_autotune, f0_autotune_strength, proposal_pitch, proposal_pitch_threshold) if self.use_f0 else (None, None)
        energy = self.rms(audio[silence_front:].to(self.device).unsqueeze(0)) if self.energy else None
        
        feats = extract_features(self.embedder[0], self.embedder[1], audio.view(1, -1), self.inference.version, device=self.device)
        feats = torch.cat((feats, feats[:, -1:, :]), 1)

        feats0 = feats.detach().clone() if protect < 0.5 and self.use_f0 else None

        if (not isinstance(self.index[0], type(None)) and not isinstance(self.index[1], type(None)) and index_rate != 0):
            skip_offset = skip_head // 2
            npy = feats[0][skip_offset :].cpu().numpy()

            if self.is_half: npy = npy.astype(np.float32)

            score, ix = self.index[0].search(npy, k=8)
            weight = np.square(1 / score)

            npy = np.sum(self.index[1][ix] * np.expand_dims(weight / weight.sum(axis=1, keepdims=True), axis=2), axis=1)
            if self.is_half: npy = npy.astype(np.float16)

            feats[0][skip_offset :] = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats[0][skip_offset :])

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        if feats0 is not None:
            pitch = pitch[:, -audio_feats_len:]
            pitchf = pitchf[:, -audio_feats_len:] * (formant_length / return_length)

            pitchff = pitchf.detach().clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)

            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            feats = (feats * pitchff + feats0 * (1 - pitchff)).to(feats0.dtype)

        pitch = pitch if self.use_f0 else None
        pitchf = pitchf.to(self.dtype) if self.use_f0 else None
        energy = energy.to(self.dtype) if self.energy else None

        audio_feats_len = min(audio_feats_len, feats.size(1))
        if self.use_f0: audio_feats_len = min(audio_feats_len, pitch.size(1), pitchf.size(1))
        if self.energy: audio_feats_len = min(audio_feats_len, energy.size(0))

        feats = feats[:, :audio_feats_len, :].to(self.dtype)
        if self.use_f0: pitch, pitchf = pitch[:, :audio_feats_len], pitchf[:, :audio_feats_len]
        if self.energy: energy = energy[:audio_feats_len].unsqueeze(0)

        p_len = torch.tensor([audio_feats_len], device=self.device, dtype=torch.int64)

        out_audio = self.inference.inference(feats, p_len, self.sid, pitch, pitchf, energy).float()
        if rms_mix_rate != 1: out_audio = change_rms(audio, self.predictor.sample_rate, out_audio, self.tgt_sr, rms_mix_rate)

        scaled_window = int(np.floor(1.0 * self.model_window))
    
        if scaled_window != self.model_window:
            if scaled_window not in self.resamplers: self.resamplers[scaled_window] = tat.Resample(orig_freq=scaled_window, new_freq=self.model_window, dtype=torch.float32).to(self.device)
            out_audio = self.resamplers[scaled_window](out_audio[: return_length * scaled_window])

        return out_audio

def create_pipeline(model_path=None, index_path=None, f0_method="rmvpe", f0_onnx=False, embedder_model="hubert_base", embedders_mode="fairseq", sample_rate=16000, hop_length=160):
    inference = Inference()
    inference = inference.get_synthesizer(model_path)

    predictor = Generator(sample_rate=sample_rate, hop_length=hop_length, f0_min=50.0, f0_max=1100.0, is_half=config.is_half, device=config.device, f0_onnx_mode=f0_onnx, del_onnx_model=False) if inference.use_f0 else None
    rms = RMSEnergyExtractor(frame_length=2048, hop_length=160, center=True, pad_mode="reflect").to(config.device).eval() if inference.energy else None

    index, index_reconstruct = load_faiss_index(index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added"))
    embedder, embed_suffix = load_embedders_model(embedder_model, embedders_mode=embedders_mode)
    if embed_suffix != ".onnx": embedder = embedder.to(config.device).to(torch.float16 if config.is_half else torch.float32)  

    pipeline = Pipeline(
        inference,
        (embedder, embed_suffix),
        predictor,
        rms,
        (index, index_reconstruct),
        f0_method
    )

    return pipeline