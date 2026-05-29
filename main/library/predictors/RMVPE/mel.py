import os
import sys
import torch

import torch.nn as nn

sys.path.append(os.getcwd())

from main.app.variables import config
from main.library.audio.features import mel

class MelSpectrogram(nn.Module):
    def __init__(
        self, 
        n_mel_channels, 
        sample_rate, 
        win_length, 
        hop_length, 
        n_fft=None, 
        mel_fmin=0, 
        mel_fmax=None, 
        clamp=1e-5
    ):
        super().__init__()
        self.clamp = clamp
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        self.hann_window = torch.hann_window(win_length).to(config.device)
        self.stftt = self._stft_other_backends if config.device.startswith(("ocl", "privateuseone")) else self._stft_torch
        self.register_buffer("mel_basis", mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True, device=config.device))

    def forward(self, audio, center=True):
        mel_output = self.mel_basis @ self.stftt(audio, center)
        return mel_output.clamp(min=self.clamp).log()
    
    def _stft_other_backends(self, audio, center=True):
        if not hasattr(self, "stft"): 
            from main.library.backends.utils import STFT

            self.stft = STFT(
                filter_length=self.n_fft, 
                hop_length=self.hop_length, 
                win_length=self.win_length,
                center=center
            ).to(audio.device)

        return self.stft.transform(audio, 1e-9)

    def _stft_torch(self, audio, center=True):
        fft = torch.stft(
            audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.hann_window, 
            center=center, 
            return_complex=True
        )
        return fft.abs()