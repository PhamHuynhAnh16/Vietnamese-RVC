import os
import sys
import torch
import random
import librosa

import numpy as np
import sklearn.metrics.pairwise as pwise

sys.path.append(os.getcwd())

from main.app.variables import config, configs
from main.library.utils import load_embedders_model, check_assets

class AutoPitch:
    def __init__(self, vc, rvc_npz_path, emb_npz_path, pitch_guidance = True, version = "v1", energy_use = False, device = "cpu"):
        self.hubert_mode, self.hubert_base = "fairseq", "hubert_base"
        check_assets("", self.hubert_base, False, self.hubert_mode)
        self.vc = vc
        self.pitch_guidance = pitch_guidance
        self.version = version
        self.energy_use = energy_use
        self.device = device
        self.rvc_loaded = np.load(rvc_npz_path)
        self.emb_loaded = np.load(emb_npz_path)
        self.models = load_embedders_model(self.hubert_base, self.hubert_mode)[0]
        self.embedders = (self.models.half() if config.is_half else self.models.float()).eval().to(self.device)

    def conversion(self, model, net_g, sid, index = None, big_npy = None):
        audio_opt = self.vc.voice_conversion(
            model, 
            net_g, 
            sid, 
            self.rvc_loaded["audio"], 
            torch.tensor(self.rvc_loaded["pitch"]).to(self.device) if self.pitch_guidance else None, 
            torch.tensor(self.rvc_loaded["pitchf"]).to(self.device) if self.pitch_guidance else None, 
            index, 
            big_npy, 
            0.5, 
            self.version, 
            0.5, 
            torch.tensor(self.rvc_loaded["energy"]).to(self.device) if self.energy_use else None
        )[self.vc.t_pad_tgt : -self.vc.t_pad_tgt]

        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1: audio_opt /= audio_max

        return audio_opt
    
    def get_hubert_feature(self, model, feats):
        with torch.no_grad():
            feats = model.extract_features(feats.to(self.device), output_layer=9)[0]
            if feats.dim() == 3: feats = feats.squeeze(0)

        return feats.mean(dim=0).cpu().numpy()
    
    def autopitch(self, model, net_g, sid, index = None, big_npy = None):
        audio_opt = self.conversion(model, net_g, sid, index, big_npy)
        tgt_sr = self.vc.tgt_sr

        if tgt_sr != 16000:
            if audio_opt.ndim > 1: audio_opt = audio_opt[0]

            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=16000)
            audio_opt = torch.from_numpy(audio_opt).to(self.device)

        if audio_opt.ndim > 1 and audio_opt.shape[0] > 1: audio_opt = audio_opt[:1]

        feats = audio_opt.squeeze(0).view(1, -1)
        emb_rvc = self.get_hubert_feature(self.embedders, feats).reshape(1, -1)

        sim_male = pwise.cosine_similarity(emb_rvc, self.emb_loaded["male"])[0][0]
        sim_female = pwise.cosine_similarity(emb_rvc, self.emb_loaded["female"])[0][0]

        if sim_male > sim_female: return round(1127 * np.log(1 + sim_male * 100 / 700), 1) if configs.get("calc_pitch", False) else 155.0, True
        elif sim_male < sim_female: return round(1127 * np.log(1 + sim_female * 10000 / 700) / 10, 1) if configs.get("calc_pitch", False) else 255.0, True
        else: return random.choice([155, 255]), False