import os
import sys
import torch

import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.app.variables import config
from main.library.audio.features import mel

class STFT:
    def __init__(
        self, 
        sr=22050, 
        n_mels=80, 
        n_fft=1024, 
        win_size=1024, 
        hop_length=256, 
        fmin=20, 
        fmax=11025, 
        clip_val=1e-5
    ):
        self.n_fft = n_fft
        self.clip_val = clip_val
        self.win_size = win_size
        self.hop_length = hop_length
        self.hann_window = torch.hann_window(self.win_size).to(config.device)
        self.stftt = self._stft_other_backends if config.device.startswith(("ocl", "privateuseone")) else self._stft_torch
        self.mel_basis = mel(sr=sr, n_fft=self.n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, device=config.device)

    def get_mel(self, y, center=False):
        pad_left = (self.win_size - self.hop_length) // 2
        pad_right = max((self.win_size - self.hop_length + 1) // 2, self.win_size - y.size(-1) - pad_left)

        return (self.mel_basis @ self.stftt(F.pad(y.unsqueeze(1), (pad_left, pad_right), mode="reflect" if pad_right < y.size(-1) else "constant").squeeze(1), center)).clamp(min=self.clip_val).log()
    
    def _stft_other_backends(self, pad, center=True):
        if not hasattr(self, "stft"): 
            from main.library.backends.utils import STFT as _STFT

            self.stft = _STFT(
                filter_length=self.n_fft, 
                hop_length=self.hop_length, 
                win_length=self.win_size,
            ).to(pad.device)

        return self.stft.transform(pad, center=center, eps=1e-5)

    def _stft_torch(self, pad, center=True):
        spec = torch.stft(
            pad, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_size, 
            window=self.hann_window, 
            center=center, 
            pad_mode="reflect", 
            normalized=False, 
            onesided=True, 
            return_complex=True
        )

        return spec.abs().clamp_min_(1e-9)