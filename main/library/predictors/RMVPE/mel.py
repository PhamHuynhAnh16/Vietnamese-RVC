import os
import sys
import torch

import torch.nn as nn

sys.path.append(os.getcwd())

from main.app.variables import config
from main.library.audio.features import mel

class MelSpectrogram(nn.Module):
    """
    A PyTorch module for extracting Mel-Spectrogram features from raw audio waveforms.

    This module computes the Short-Time Fourier Transform (STFT) of an audio signal
    and projects the linear frequency magnitude spectrum onto the Mel scale using
    a predefined filter bank matrix. It supports dynamic backend switching for alternative
    hardware architectures (e.g., OpenCL, DirectML/PrivateUse1).
    """

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
        """
        Initializes the MelSpectrogram feature extractor.

        Args:
            n_mel_channels (int): Total number of Mel bands to construct.
            sample_rate (int): Target sampling rate of input audio signals.
            win_length (int): Window duration size for frame localized windowing.
            hop_length (int): Step/stride size between consecutive overlapping windows.
            n_fft (int, optional): Length of the FFT window. Defaults to win_length if not provided.
            mel_fmin (float, default=0): Minimum frequency bound for the Mel scale filter bank.
            mel_fmax (float, optional): Maximum frequency bound for the Mel scale filter bank.
            clamp (float, default=1e-5): Floor restriction value utilized before log mapping operations.
        """

        super().__init__()
        self.clamp = clamp
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        # Pre-compute and cache the Hann window function to minimize runtime overhead
        self.hann_window = torch.hann_window(win_length).to(config.device)

        self.stft = None
        # Dynamically dispatch the appropriate STFT calculation routine depending on device backend
        self.stftt = self._stft_other_backends if config.device.startswith(("ocl", "privateuseone")) else self._stft_torch
        # Compute the HTK-scaled Mel filter bank basis matrix and register it as a non-trainable module buffer
        self.register_buffer("mel_basis", mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True, device=config.device))

    def forward(self, audio, center=True):
        """Processes raw audio waveforms into log-compressed Mel-Spectrogram features.

        Args:
            audio (torch.Tensor): Input audio waveform tensor.
            center (bool, default=True): If True, pads input waveform on both sides so that the t-th frame is centered around time t * hop_length.

        Returns:
            torch.Tensor: Log-compressed Mel-spectrogram feature maps.
        """

        # Step 1: Perform STFT magnitude extraction followed by matrix multiplication projection onto the Mel scale matrix
        mel_output = self.mel_basis @ self.stftt(audio, center)
        # Step 2: Apply floor clamping to guarantee mathematical stability, then convert to the logarithmic domain
        return mel_output.clamp(min=self.clamp).log()
    
    def _stft_other_backends(self, audio, center=True):
        """Computes STFT magnitude for specialized hardware backends (e.g., OpenCL/Intel/AMD Extensions)."""

        if self.stft is None: 
            # Lazy import alternative STFT module wrapper to optimize startup performance overhead

            from main.library.backends.utils import STFT

            self.stft = STFT(
                filter_length=self.n_fft, 
                hop_length=self.hop_length, 
                win_length=self.win_length
            ).to(audio.device)

        # Run custom backend transform logic applying a tiny floor constant for division guards
        return self.stft.transform(audio, 1e-9, center=center)

    def _stft_torch(self, audio, center=True):
        """Computes STFT magnitude using the standard native PyTorch implementation wrapper."""

        # Step 1: Calculate the complex frequency representation via native short-time fourier transform
        fft = torch.stft(
            audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.hann_window, 
            center=center, 
            return_complex=True
        )
        # Step 2: Extract the absolute value (magnitude spectrum), discarding phase information
        return fft.abs()