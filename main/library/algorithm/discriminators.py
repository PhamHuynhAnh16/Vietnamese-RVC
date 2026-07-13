import os
import sys
import torch
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

sys.path.append(os.getcwd())

from main.library.algorithm.commons import get_padding
from main.library.algorithm.residuals import LRELU_SLOPE

class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-Period Discriminator (MPD) architecture that wraps multiple sub-discriminators 
    including Scale (DiscriminatorS), Period (DiscriminatorP), and Resolution/Multi-Scale 
    Mel-Spectrogram (DiscriminatorR) variants. Used widely in adversarial audio synthesis (e.g., HiFi-GAN).
    """

    def __init__(self, version, use_spectral_norm=False, checkpointing=False):
        """
        Initializes the MultiPeriodDiscriminator ensemble based on a predefined configuration version.

        Args:
            version (str): Configuration version ("v0", "v1", "v2", or "v3").
            use_spectral_norm (bool): If True, applies spectral normalization instead of weight normalization. Defaults to False.
            checkpointing (bool): If True, utilizes gradient checkpointing to save memory during training. Defaults to False.
        """

        super(MultiPeriodDiscriminator, self).__init__()
        self.checkpointing = checkpointing
        # Configure periods and resolutions based on version criteria
        if version == "v0":
            periods = [2, 3, 5, 7, 11]
            resolutions = []
        elif version == "v1":
            periods = [2, 3, 5, 7, 11, 17]
            resolutions = []
        elif version == "v2": 
            periods = [2, 3, 5, 7, 11, 17, 23, 37]
            resolutions = []
        elif version == "v3":
            periods = [2, 3, 5, 7, 11]
            resolutions = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]
        else:
            raise ValueError(f"Unknown MultiPeriodDiscriminator version: {version}")

        # Construct sub-discriminators list dynamically
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)] + 
            [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods] + 
            [DiscriminatorR(r, use_spectral_norm=use_spectral_norm) for r in resolutions]
        )

    def forward(self, y, y_hat):
        """
        Evaluates both real ground-truth audio and generated audio across all sub-discriminators.

        Args:
            y (torch.Tensor): Real audio waveform tensor.
            y_hat (torch.Tensor): Generated/Synthetic audio waveform tensor.

        Returns:
            Tuple containing:
                - y_d_rs (List[torch.Tensor]): Real validation scores from each discriminator.
                - y_d_gs (List[torch.Tensor]): Fake validation scores from each discriminator.
                - fmap_rs (List[List[torch.Tensor]]): Feature maps extracted from real audio inputs.
                - fmap_gs (List[List[torch.Tensor]]): Feature maps extracted from generated audio inputs.
        """

        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for d in self.discriminators:
            # Conditional routing to leverage activation checkpointing during backpropagation memory optimization
            if self.training and self.checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)

            y_d_rs.append(y_d_r); fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g); fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(torch.nn.Module):
    """
    Multi-Scale / Sub-Scale Discriminator (MSD) variant acting on raw 1D audio sequences at full-scale.
    """

    def __init__(self, use_spectral_norm=False):
        """
        Initializes 1D convolutional layers with group configurations for localized pattern matching.

        Args:
            use_spectral_norm (bool): Normalization technique flag switch.
        """

        super(DiscriminatorS, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        # 1D Convolutional blocks with varying kernel lengths and grouped architectures
        self.convs = torch.nn.ModuleList([
            norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)), 
            norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)), 
            norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)), 
            norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)), 
            norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)), 
            norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2))
        ])
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Audio waveform input tensor.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Flattened validity matrix score and intermediate feature map list.
        """

        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        # Flatten features from dimension 1 to the end for categorical classification loss evaluation
        return x.flatten(1, -1), fmap

class DiscriminatorP(torch.nn.Module):
    """
    Period Discriminator that reshapes 1D audio signals into structural 2D planes 
    based on a defined period interval to evaluate periodic patterns.
    """

    def __init__(self, period, kernel_size=5, use_spectral_norm=False):
        """
        Args:
            period (int): Periodic interval slicing window step.
            kernel_size (int): Height dimension kernel size. Defaults to 5.
            use_spectral_norm (bool): Normalization switch indicator.
        """

        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        # 2D Convolutions utilizing a vertical strip kernel to treat periodic frames as image columns
        self.convs = torch.nn.ModuleList([
            norm_f(
                torch.nn.Conv2d(
                    in_ch, 
                    out_ch, 
                    (kernel_size, 1), 
                    (stride, 1), 
                    padding=(get_padding(kernel_size, 1), 0)
                )
            ) 
            for in_ch, out_ch, stride in zip(
                [1, 32, 128, 512, 1024], 
                [32, 128, 512, 1024, 1024], 
                [3, 3, 3, 3, 1]
            )
        ])
        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Audio tensor.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Validated classification tensor and accumulated feature maps.
        """

        fmap = []
        b, c, t = x.shape
        # Pad sequence length to be perfectly divisible by the target period before 2D transformation
        if t % self.period != 0: x = F.pad(x, (0, (self.period - (t % self.period))), "reflect")
        # Reshape 1D data sequence to 2D image matrix
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap

class DiscriminatorR(torch.nn.Module):
    """
    Resolution Discriminator that processes transformed spectrogram magnitudes 
    instead of raw raw time-domain audio sequences.
    """

    def __init__(self, resolution, use_spectral_norm=False):
        """
        Args:
            resolution (List[int]): STFT configuration constants [n_fft, hop_length, win_length].
            use_spectral_norm (bool): Normalization technique selection flag.
        """

        super().__init__()
        self.resolution = resolution
        self.lrelu_slope = LRELU_SLOPE # Specific localized slope modifier for LeakyReLU in Resolution domain
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        # 2D Convolutions utilizing horizontal contextual kernels for time-frequency feature extraction
        self.convs = torch.nn.ModuleList([
            norm_f(torch.nn.Conv2d( 1, 32, (3, 9), padding=(1, 4))), 
            norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))), 
            norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))), 
            norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))), 
            norm_f(torch.nn.Conv2d(32, 32, (3, 3), padding=(1, 1)))
        ])
        self.conv_post = norm_f(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Waveform audio signal.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Flattened evaluation result matrix and spectral feature tracking logs.
        """

        fmap = []
        # Calculate Magnitude Spectrogram and append a dummy channel index
        x = self.spectrogram(x).unsqueeze(1)
        
        for layer in self.convs:
            x = F.leaky_relu(layer(x), self.lrelu_slope)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return x.flatten(1, -1), fmap

    def spectrogram(self, x):
        """
        Helper method computing the short-time Fourier transform (STFT) magnitude map.
        Handles alternative hardware devices (like OpenCL / privateuseone backends) by falling back safely to CPU.

        Args:
            x (torch.Tensor): Raw time-series sequence data tensor.

        Returns:
            torch.Tensor: Linear amplitude spectrogram representation tensor.
        """

        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)

        # Detect non-standard backends to safeguard STFT compatibility errors
        is_not_cuda = x.device.type in ["privateuseone", "ocl"]
        # Execute Short-Time Fourier Transform
        stft = torch.stft(
            F.pad( # Apply structured reflection boundaries and squeeze out the single channel
                x.cpu() if is_not_cuda else x, 
                (pad, pad), 
                mode="reflect"
            ).squeeze(1), 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            window=torch.ones(win_length, device="cpu" if is_not_cuda else x.device), 
            center=False, 
            return_complex=True
        )

        # Extract Euclidean norm magnitude from complex projections and cast back to the native input execution device
        return torch.view_as_real(stft).norm(p=2, dim=-1).to(x.device)