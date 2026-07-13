import os
import sys
import torch

import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.app.variables import config
from main.library.audio.features import mel

class STFT:
    """Short-Time Fourier Transform (STFT) and Mel-Spectrogram extraction processing module.

    Supports native PyTorch STFT backends as well as fallback structures for custom acceleration 
    backends (e.g., OpenCL / privateuseone devices).
    """

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
        """
        Initializes the STFT filterbank and target Mel transformation matrix bases.

        Args:
            sr (int): Sampling rate of input raw waveform audio. Defaults to 22050.
            n_mels (int): Target number of frequency bands. Defaults to 80.
            n_fft (int): Size of the FFT window windowing frames. Defaults to 1024.
            win_size (int): Size of the analytical window. Defaults to 1024.
            hop_length (int): Number of audio samples between successive frames. Defaults to 256.
            fmin (int): Minimum audio frequency bound. Defaults to 20.
            fmax (int): Maximum audio frequency bound. Defaults to 11025.
            clip_val (float): Lower bound floor constant value before log mapping. Defaults to 1e-5.
        """

        self.stft = None
        self.n_fft = n_fft
        self.clip_val = clip_val
        self.win_size = win_size
        self.hop_length = hop_length
        # Instantiate window function and register directly on targeted runtime hardware device
        self.hann_window = torch.hann_window(self.win_size).to(config.device)
        # Route execution pathways conditionally depending on targeted deployment backend environments
        self.stftt = self._stft_other_backends if config.device.startswith(("ocl", "privateuseone")) else self._stft_torch
        # Precompute the fixed weight matrix projecting linear frequencies into log Mel spaces
        self.mel_basis = mel(sr=sr, n_fft=self.n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, device=config.device)

    def get_mel(self, y, center=False):
        """
        Extracts log Mel-Spectrogram features from standard 1D raw waveform signals.

        Args:
            y (torch.Tensor): Audio waveform matrix, expected shape (B, T).
            center (bool, optional): Pads sequence edges symmetry frames internally. Defaults to False.

        Returns:
            torch.Tensor: Log Mel frequency values, shape (B, n_mels, Frames).
        """

        # Calculate strict framing reflection pad constants for non-centered boundary alignments
        pad_left = (self.win_size - self.hop_length) // 2
        pad_right = max((self.win_size - self.hop_length + 1) // 2, self.win_size - y.size(-1) - pad_left)

        # 1. Transform audio into frequency spectrum space using the designated backend
        # 2. Multiply by the Mel basis matrix, bound under floor tolerances, and compute the natural log
        return (self.mel_basis @ self.stftt(F.pad(y.unsqueeze(1), (pad_left, pad_right), mode="reflect" if pad_right < y.size(-1) else "constant").squeeze(1), center)).clamp(min=self.clip_val).log()
    
    def _stft_other_backends(self, pad, center=True):
        """Fallback wrapper performing STFT analysis using custom hardware backends."""

        if self.stft is None: 
            from main.library.backends.utils import STFT as _STFT

            self.stft = _STFT(
                filter_length=self.n_fft, 
                hop_length=self.hop_length, 
                win_length=self.win_size,
            ).to(pad.device)

        return self.stft.transform(pad, center=center, eps=1e-5)

    def _stft_torch(self, pad, center=True):
        """Executes native optimized PyTorch STFT calculations."""

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

        # Extract absolute scalar magnitudes and apply spatial numerical clamps
        return spec.abs().clamp_min_(1e-9)

class Wav2Mel:
    """Convenience pipeline processor mapping raw audio vectors directly to transposed Mel features."""

    def __init__(
        self, 
        device=None, 
        dtype=torch.float32
    ):
        """
        Initializes processing properties and instantiates standard 16kHz STFT targets.

        Args:
            device (str / torch.device, optional): Inference device destination. Defaults to None.
            dtype (torch.dtype, optional): Precision format parsing inputs. Defaults to torch.float32.
        """

        self.sample_rate = 16000
        self.hop_size = 160
        self.device = device
        self.dtype = dtype
        # Instantiate configured instance mapping exact 16kHz audio sample guidelines
        self.stft = STFT(self.sample_rate, 128, 1024, 1024, self.hop_size, 0, 8000)

    def __call__(self, audio):
        """
        Processes raw waveform chunks into aligned frames.

        Args:
            audio (torch.Tensor): Audio waveform tensor.

        Returns:
            torch.Tensor: Padded log Mel frame outputs.
        """

        # Transfer input data variables onto target computational devices and types
        audio = audio.to(self.dtype).to(self.device)
        # Extract features and transpose axis format
        mel = self.stft.get_mel(audio).transpose(1, 2)

        # Calculate expected standard downsampled frame structures
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        # Pad boundary edges manually if sequence elements fall short of frame counts
        mel = (torch.cat((mel, mel[:, -1:, :]), 1) if n_frames > int(mel.shape[1]) else mel)

        # Slice trailing bounds or return formatted tensor matrices uniformly
        return mel[:, :n_frames, :] if n_frames < int(mel.shape[1]) else mel