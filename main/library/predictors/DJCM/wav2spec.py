import os
import sys
import torch

import torch.nn as nn

sys.path.append(os.getcwd())

from main.library.backends import opencl

class Wav2Spec(nn.Module):
    def __init__(self, hop_length, window_size):
        super(Wav2Spec, self).__init__()
        self.hop_length = hop_length
        self.window_size = window_size
        self.n_fft = window_size
        self.register_buffer("window", torch.hann_window(window_size), persistent=False)

    def forward(self, audio):
        bs, c, segment_samples = audio.shape
        audio = audio.reshape(bs * c, segment_samples)

        if str(audio.device).startswith("ocl"):
            if not hasattr(self, "stft"): self.stft = opencl.STFT(filter_length=self.n_fft, hop_length=self.hop_length, win_length=self.window_size).to(audio.device)
            magnitude = self.stft.transform(audio, 1e-9)
        else:
            fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.window_size, window=self.window, center=True, return_complex=True, pad_mode='reflect')
            magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))

        return magnitude.unsqueeze(1).transpose(2, 3)