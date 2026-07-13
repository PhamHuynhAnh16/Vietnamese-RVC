import os
import sys
import torch

import numpy as np
import torch.nn as nn

sys.path.append(os.getcwd())

from main.app.variables import config

class Spectrogram(nn.Module):
    """
    A flexible spectrogram extraction module with multi-backend STFT support.

    This module computes the magnitude spectrogram of multi-channel input audio signals.
    It automatically routes computation through native PyTorch STFT or specialized alternative 
    backends (e.g., OpenCL, DirectML via 'ocl' or 'privateuseone' device prefixes) depending on 
    the active execution hardware environment.
    """

    def __init__(
        self, 
        hop_length, 
        win_length, 
        n_fft=None, 
        clamp=1e-10
    ):
        """
        Initializes the Spectrogram module configurations.

        Args:
            hop_length (int): Number of audio samples between successive STFT columns.
            win_length (int): The size of the window signal applied per frame chunk.
            n_fft (int, optional): The length of the windowed signal after zero-padding. Defaults to match `win_length`.
            clamp (float): Minimum absolute threshold value used to prevent numerical underflow. Default is 1e-10.
        """

        super(Spectrogram, self).__init__()
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp
        self.stft = None
        # Pre-calculate a standard Hanning analysis window function and register as non-persistent buffer
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        # Dynamically route backend processing logic depending on runtime hardware acceleration signatures
        self.stftt = self._stft_other_backends if config.device.startswith(("ocl", "privateuseone")) else self._stft_torch

    def forward(self, audio, center=True):
        """
        Extracts the magnitude spectrogram from batched multi-channel raw waveforms.

        Args:
            audio (torch.Tensor): Audio waveform tensor of shape (batch, channels, samples).
            center (bool): Pads waveforms symmetrically so frames align precisely with time points. Default is True.

        Returns:
            torch.Tensor: Magnitude spectrogram block tensor of shape (batch, channels, frequency, time).
        """

        bs, c, segment_samples = audio.shape
        # Flatten the batch and channel axis down together to fit standard 2D signal entry profiles
        audio = audio.reshape(bs * c, segment_samples)
        # Calculate Short-Time Fourier Transform, transpose time/frequency axis, and floor low signals via clamp
        mag = self.stftt(audio, center).transpose(1, 2).clamp(self.clamp, np.inf)
        # Unflatten and reconstruct structured dimensions back out to match incoming shapes
        return mag.reshape(bs, c, mag.shape[1], mag.shape[2])

    def _stft_other_backends(self, audio, center=True):
        """
        Computes Short-Time Fourier Transform utilizing specialized secondary system libraries.

        This branch resolves runtime execution errors encountered on non-native environments 
        such as custom OpenCL integrations by mapping calls out to custom module implementations.
        """

        if self.stft is None: 
            from main.library.backends.utils import STFT

            # Lazy initial loading of custom STFT instance configurations mapped to target calculation device
            self.stft = STFT(
                filter_length=self.n_fft, 
                hop_length=self.hop_length, 
                win_length=self.win_length
            ).to(audio.device)

        return self.stft.transform(audio, 1e-9, center=center)

    def _stft_torch(self, audio, center=True):
        """Computes Magnitude Short-Time Fourier Transform using standard native PyTorch kernels."""

        fft = torch.stft(
            audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.window, 
            center=center, 
            pad_mode="reflect", 
            return_complex=True # Mandated parameter choice for torch compatibility guarantees
        )
        # Extract and return absolute real magnitudes from complex output spectrum fields
        return fft.abs()