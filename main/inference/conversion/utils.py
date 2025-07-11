import os
import gc
import sys
import torch
import librosa

import numpy as np
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library import opencl

def autotune_f0(note_dict, f0, f0_autotune_strength):
    autotuned_f0 = np.zeros_like(f0)

    for i, freq in enumerate(f0):
        autotuned_f0[i] = freq + (min(note_dict, key=lambda x: abs(x - freq)) - freq) * f0_autotune_strength

    return autotuned_f0

def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    rms2 = F.interpolate(torch.from_numpy(librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    return (target_audio * (torch.pow(F.interpolate(torch.from_numpy(librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze(), 1 - rate) * torch.pow(torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6), rate - 1)).numpy())

def clear_gpu_cache():
    gc.collect()

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()
    elif opencl.is_available(): opencl.pytorch_ocl.empty_cache()

def extract_median_f0(f0):
    f0 = np.where(f0 == 0, np.nan, f0)
    return float(np.median(np.interp(np.arange(len(f0)), np.where(~np.isnan(f0))[0], f0[~np.isnan(f0)])))

def proposal_f0_up_key(f0, target_f0 = 155.0, limit = 12):
    return max(-limit, min(limit, int(np.round(12 * np.log2(target_f0 / extract_median_f0(f0))))))

def get_onnx_argument(net_g, feats, p_len, sid, pitch, pitchf, energy, pitch_guidance, energy_use):
    inputs = {
        net_g.get_inputs()[0].name: feats.cpu().numpy().astype(np.float32),
        net_g.get_inputs()[1].name: p_len.cpu().numpy(),
        net_g.get_inputs()[2].name: np.array([sid.cpu().item()], dtype=np.int64),
        net_g.get_inputs()[3].name: np.random.randn(1, 192, p_len).astype(np.float32)
    }

    if energy_use:
        if pitch_guidance:
            inputs.update({
                net_g.get_inputs()[4].name: pitch.cpu().numpy().astype(np.int64),
                net_g.get_inputs()[5].name: pitchf.cpu().numpy().astype(np.float32),
                net_g.get_inputs()[6].name: energy.cpu().numpy().astype(np.float32)
            })
        else:
            inputs.update({
                net_g.get_inputs()[4].name: energy.cpu().numpy().astype(np.float32)
            })
    else:
        if pitch_guidance:
            inputs.update({
                net_g.get_inputs()[4].name: pitch.cpu().numpy().astype(np.int64),
                net_g.get_inputs()[5].name: pitchf.cpu().numpy().astype(np.float32)
            })

    return inputs