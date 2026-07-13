import os
import sys
import torch

sys.path.append(os.getcwd())

from main.library.audio.features import mel

def spectral_normalize_torch(magnitudes):
    """
    Applies natural log spectral normalization to a magnitude spectrogram.

    Args:
        magnitudes (torch.Tensor): Input magnitude spectrogram.

    Returns:
        torch.Tensor: Log-scaled normalized spectrogram.
    """

    # Clamp to prevent log(0) and safely compute natural log
    return (magnitudes.clamp(min=1e-5) * 1.0).log()

def spectral_de_normalize_torch(magnitudes):
    """
    Inverts the natural log spectral normalization back to standard magnitudes.

    Args:
        magnitudes (torch.Tensor): Log-scaled normalized spectrogram.

    Returns:
        torch.Tensor: Denormalized magnitude spectrogram.
    """

    return magnitudes.exp() / 1.0

# Cache variables for persistent structures
stft = None
mel_basis, hann_window = {}, {}

def spectrogram_torch(
    y, 
    n_fft, 
    hop_size, 
    win_size, 
    center=False
):
    """
    Computes the linear-frequency magnitude spectrogram from a waveform tensor using STFT.

    Args:
        y (torch.Tensor): Input audio waveform tensor.
        n_fft (int): Size of Fourier transform.
        hop_size (int): Length of hop between STFT windows.
        win_size (int): Size of the windowing function.
        center (bool, optional): Whether to center the waveform frames. Defaults to False.

    Returns:
        torch.Tensor: Magnitude spectrogram on the target audio device.
    """

    global hann_window, stft

    # Create a unique tracking key for the window cache
    wnsize_dtype_device = str(win_size) + "_" + str(y.dtype) + "_" + str(y.device)
    if wnsize_dtype_device not in hann_window: 
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    # Reflectively pad the waveform to handle edge boundaries manually
    pad = torch.nn.functional.pad(
        y.unsqueeze(1), 
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), 
        mode="reflect"
    ).squeeze(1)

    # Use a custom STFT class if running on alternative acceleration backends (OpenCL/PrivateUse1)
    if y.device.type.startswith(("ocl", "privateuseone")):
        if stft is None: 
            from main.library.backends.utils import STFT

            stft = STFT(
                filter_length=n_fft, 
                hop_length=hop_size, 
                win_length=n_fft
            ).to(y.device)

        spec = stft.transform(
            pad.to(y.device), 
            eps=1e-6, 
            center=center
        )
    else:
        # Fallback to the standard native PyTorch execution branch
        spec = torch.stft(
            pad, 
            n_fft, 
            hop_length=hop_size, 
            win_length=win_size, 
            window=hann_window[wnsize_dtype_device].to(pad.device), 
            center=center, 
            pad_mode="reflect", 
            normalized=False, 
            onesided=True, 
            return_complex=True
        )

        # Convert complex output to absolute magnitude safely
        spec = spec.abs().clamp_min_(1e-6)

    return spec.to(y.device)

def spec_to_mel_torch(spec, config):
    """
    Projects a linear magnitude spectrogram onto the Mel scale and applies log normalization.

    Args:
        spec (torch.Tensor): Linear frequency magnitude spectrogram.
        config: Configuration namespace object containing Mel data attributes.

    Returns:
        torch.Tensor: Log-Mel spectrogram.
    """

    global mel_basis

    fmax = config.data.mel_fmax
    fmax_dtype_device = str(fmax) + "_" + str(spec.dtype) + "_" + str(spec.device)
    # Initialize and cache the Mel transformation filter matrix if not present
    if fmax_dtype_device not in mel_basis: 
        mel_basis[fmax_dtype_device] = mel(
            sr=config.data.sample_rate, 
            n_fft=config.data.filter_length, 
            n_mels=config.data.n_mel_channels, 
            fmin=config.data.mel_fmin, 
            fmax=fmax,
            dtype=spec.dtype, 
            device=spec.device
        )
    
    # Apply the Mel filter bank matrix multiplication and normalize
    return spectral_normalize_torch(mel_basis[fmax_dtype_device] @ spec)

def mel_spectrogram_torch(y, config, center=False):
    """
    Wrapper function converting a raw audio waveform directly into a log-Mel spectrogram.

    Args:
        y (torch.Tensor): Raw audio waveform.
        config: Configuration namespace object.
        center (bool, optional): Whether to center frames during STFT. Defaults to False.

    Returns:
        torch.Tensor: Log-Mel spectrogram tensor.
    """

    # Note: Passed positional arguments to spec_to_mel_torch match your layout 
    # but verify config-dependent calls if positional unpacking errors occur.
    return spec_to_mel_torch(
        spectrogram_torch(
            y, 
            config.data.filter_length, 
            config.data.hop_length, 
            config.data.win_length, 
            center
        ), 
        config
    )

class MultiScaleMelSpectrogramLoss(torch.nn.Module):
    """
    Computes Multi-Scale Mel-Spectrogram Loss between reconstructed and target waveforms
    across various STFT window and filter scales to capture audio attributes effectively.
    """

    def __init__(
        self, 
        sample_rate = 24000, 
        n_mels=[5, 10, 20, 40, 80, 160, 320], 
        window_lengths=[32, 64, 128, 256, 512, 1024, 2048], 
        loss_fn=torch.nn.L1Loss()
    ):
        """
        Initializes multi-scale parameters and storage dictionary banks.
        """

        super().__init__()
        self.sample_rate = sample_rate
        self.loss_fn = loss_fn
        self.log_base = torch.tensor(10.0).log() # Cache log(10) for base-10 conversion
        self.stft_params = []
        self.hann_window = {}
        self.mel_banks = {}
        # Combine parameters into paired tuples for the evaluation scale loops
        self.stft_params = [(mel, win) for mel, win in zip(n_mels, window_lengths)]

    def mel_spectrogram(self, wav, n_mels, window_length):
        """
        Extracts a specific scale Mel-spectrogram for multi-scale loss evaluations.

        Args:
            wav (torch.Tensor): Audio waveform tensor.
            n_mels (int): Target count of mel channels.
            window_length (int): Frame length window constraint.

        Returns:
            torch.Tensor: Linear scale mel spectrogram.
        """

        dtype_device = str(wav.dtype) + "_" + str(wav.device)
        win_dtype_device = str(window_length) + "_" + dtype_device
        mel_dtype_device = str(n_mels) + "_" + dtype_device

        # Initialize and track window vectors per device constraint
        if win_dtype_device not in self.hann_window: 
            self.hann_window[win_dtype_device] = torch.hann_window(window_length, device=wav.device, dtype=torch.float32)

        # Ensure correct channel squeezing configuration shapes
        wav = wav.float().squeeze(1)
        # Offload windowing evaluations to CPU if working under alternative backends
        if wav.device.type.startswith(("ocl", "privateuseone")):
            stft = torch.stft(
                wav.cpu(), 
                n_fft=window_length, 
                hop_length=window_length // 4, 
                window=self.hann_window[win_dtype_device].cpu(), 
                return_complex=True
            )

            magnitude = stft.abs().clamp_min_(1e-6).to(wav.device, dtype=torch.float32)
        else:
            stft = torch.stft(
                wav, 
                n_fft=window_length, 
                hop_length=window_length // 4, 
                window=self.hann_window[win_dtype_device], 
                return_complex=True
            )

            magnitude = stft.abs().clamp_min_(1e-6)

        # Lazily instantiate the corresponding scale Mel filter bank
        if mel_dtype_device not in self.mel_banks: 
            self.mel_banks[mel_dtype_device] = mel(
                sr=self.sample_rate, 
                n_mels=n_mels, 
                n_fft=window_length, 
                fmin=0, 
                fmax=None,
                device=wav.device, 
                dtype=torch.float32
            )

        return self.mel_banks[mel_dtype_device] @ magnitude

    def forward(self, real, fake):
        """
        Computes the aggregate multi-scale loss metric sum.

        Args:
            real (torch.Tensor): Ground-truth target audio waveforms.
            fake (torch.Tensor): Synthesized generated model audio waveforms.

        Returns:
            torch.Tensor: Total computed L1/distance error metric scalar.
        """

        loss = 0.0
        # Loop through each scale parameter bundle and calculate individual loss terms
        for p in self.stft_params:
            # Clamp and convert values from natural log to log10 representations
            loss += self.loss_fn(
                self.mel_spectrogram(real, *p).clamp(min=1e-5).log() / self.log_base, 
                self.mel_spectrogram(fake, *p).clamp(min=1e-5).log() / self.log_base
            )

        return loss