import os
import sys
import math
import torch

import numpy as np
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from torch.nn.utils import remove_weight_norm
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

from main.library.algorithm.commons import init_weights
from main.library.algorithm.residuals import ResBlock, LRELU_SLOPE

class SineGen2(torch.nn.Module):
    """
    Alternative Sine Wave Generator (v2) for Neural Source Filter (NSF).
    Generates multi-harmonic sine wave tracks by matrix multiplying F0 with a harmonic multiplier tensor.
    """

    def __init__(
        self, 
        sampling_rate, 
        harmonic_num=0,
        sine_amp=0.1, 
        noise_std=0.003,
        voiced_threshold=0,
    ):
        """
        Args:
            sampling_rate (int): Target audio sampling rate.
            harmonic_num (int): Number of overtones/harmonics to generate. Default: 0.
            sine_amp (float): Base amplitude factor of the sine waves. Default: 0.1.
            noise_std (float): Standard deviation for voiced random noise. Default: 0.003.
            voiced_threshold (float): F0 threshold value to determine voiced frames. Default: 0.
        """

        super(SineGen2, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        """Maps continuous F0 tracking values into a binary Voiced/Unvoiced mask."""

        return (f0 > self.voiced_threshold).float()

    def _f02sine(self, f0_values):
        """
        Converts matrix-scaled F0 values into continuous sine wave sequences.
        
        Args:
            f0_values (Tensor): Multi-harmonic fundamental frequencies.

        Returns:
            Tensor: Continuous phase-aligned sine matrix.
        """

        # Calculate radiant steps per step and wrap phases within [0, 1]
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], dtype=f0_values.dtype, device=f0_values.device)
        # Lock phase start boundary for the initial index
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # Accumulate wrapped steps over time
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        # Run phase execution using the standard sine function
        return (torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi).sin()

    def forward(self, f0, upp):
        """
        Args:
            f0 (Tensor): Upsampled raw F0 trajectory matrix.
            upp (int): Total upsampling scale factor factor (unused inside v2, kept for interface symmetry).

        Returns:
            Tensor: Combined sine excitation signal matrix.
        """

        with torch.no_grad():
            # Multiply base pitch tracking vector by harmonic indices [1, 2, ..., harmonic_num + 1]
            sine_waves = self._f02sine(
                torch.multiply(
                    f0, 
                    torch.FloatTensor([
                        [range(1, self.harmonic_num + 2)]
                    ]).to(f0.device)
                )
            ) * self.sine_amp

            # Extract Voiced/Unvoiced binary profile mask
            uv = self._f02uv(f0)
            # Blend generated sine waves with noise distributions based on voiced state status
            sine_waves = sine_waves * uv + ((uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves))

        return sine_waves

class SineGen(torch.nn.Module):
    """
    Standard Phase-Continuous Sine Wave Generator (v1) for Neural Source Filter (NSF).
    Performs precise custom step phase accumulation on audio frame strides.
    """

    def __init__(
        self, 
        sampling_rate, 
        harmonic_num=0, 
        sine_amp=0.1, 
        noise_std=0.003, 
        voiced_threshold=0
    ):
        """
        Args:
            sampling_rate (int): Target audio sampling rate.
            harmonic_num (int): Number of overtones/harmonics to generate. Default: 0.
            sine_amp (float): Base amplitude factor of the sine waves. Default: 0.1.
            noise_std (float): Standard deviation for voiced random noise. Default: 0.003.
            voiced_threshold (float): F0 threshold value to determine voiced frames. Default: 0.
        """

        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        """Maps continuous F0 tracking values into a binary Voiced/Unvoiced mask."""

        return (f0 > self.voiced_threshold).float()
    
    def _f02sine(self, f0, upp):
        """
        Generates phase-continuous sine waves matching the audio sample scale.
        
        Args:
            f0 (Tensor): frame-level F0 tracking.
            upp (int): Upsampling factor per audio frame step.

        Returns:
            Tensor: High-resolution phase-continuous multi-harmonic sine waves.
        """

        # Step increment calculation across the frame timeline blocks
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, dtype=f0.dtype, device=f0.device)
        # Compensate step offsets over adjacent windows to maintain smooth continuous phases
        rad += F.pad((torch.fmod(rad[:, :-1, -1:].float() + 0.5, 1.0) - 0.5).cumsum(dim=1).fmod(1.0).to(f0), (0, 0, 1, 0), mode='constant')
        # Reshape to high-resolution time matrix and scale up across individual harmonic channels
        rad = rad.reshape(f0.shape[0], -1, 1)
        rad *= torch.arange(1, self.dim + 1, dtype=f0.dtype, device=f0.device).reshape(1, 1, -1)
        # Append initial randomized start phase bounds to prevent phase locking artifacts
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini

        return (2 * np.pi * rad).sin()
        
    def forward(self, f0, upp):
        """
        Args:
            f0 (Tensor): Base acoustic frame pitch sequence.
            upp (int): Frame-to-sample timeline stretching ratio.

        Returns:
            Tensor: Waveform-rate noise-modulated harmonic matrix.
        """

        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            # Synthesize basic sine wave foundations
            sine_waves = self._f02sine(f0, upp) * self.sine_amp
            # Interpolate Voiced/Unvoiced mask up to high-resolution timeline via nearest neighbor
            uv = F.interpolate(self._f02uv(f0).transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
            # Superimpose multi-band background noise profiles
            sine_waves = sine_waves * uv + ((uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves))

        return sine_waves

class SourceModuleHnNSF(torch.nn.Module):
    """
    Source excitation mixer wrapper routing inputs to either SineGen or SineGen2 pipelines.
    Projects multi-channel tracking branches down into a consolidated 1D source track.
    """

    def __init__(
        self, 
        sample_rate, 
        harmonic_num=0, 
        sine_amp=0.1, 
        add_noise_std=0.003, 
        voiced_threshod=0,
        sinegen_version="v1"
    ):
        """
        Args:
            sample_rate (int): Sampling rate for sound synthesis.
            harmonic_num (int): Upper bound capacity for generated harmonics.
            sine_amp (float): Core output amplitude setting.
            add_noise_std (float): Target gaussian variance.
            voiced_threshod (float): Minimum voiced pitch floor frequency.
            sinegen_version (str): Target variant tag ("v1" or "v2").
        """

        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        # Select target generator engine module instance based on requested version string
        sine_gen_fn = SineGen if sinegen_version == "v1" else SineGen2
        self.l_sin_gen = sine_gen_fn(sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upsample_factor = 1):
        """
        Args:
            x (Tensor): Input fundamental frequency track.
            upsample_factor (int): Audio rate timeline multiplier. Default: 1.

        Returns:
            Tensor: Projective 1D filtered excitation wave track.
        """

        # Generate raw waveforms, mix down dimensions via a linear projection, and bound peaks using Tanh
        return self.l_tanh(
            self.l_linear(
                self.l_sin_gen(
                    x, 
                    upsample_factor
                ).to(dtype=self.l_linear.weight.dtype)
            )
        )

class HiFiGANNSFGenerator(torch.nn.Module):
    """
    HiFi-GAN Generator network variant equipped with Neural Source Filter (NSF) features,
    source-signal injection multi-scale paths, and Gradient Checkpointing support.
    """

    def __init__(
        self, 
        initial_channel, 
        resblock_kernel_sizes, 
        resblock_dilation_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        upsample_kernel_sizes, 
        gin_channels, 
        sr, 
        checkpointing = False,
        harmonic_num = 0
    ):
        """
        Args:
            initial_channel (int): Number of input Mel-spectrogram channels.
            resblock_kernel_sizes (list of int): Receptive field kernel lengths for ResBlocks.
            resblock_dilation_sizes (list of list of int): Dilations nested for internal layers.
            upsample_rates (list of int): Temporal scaling ratios per upsampling layer.
            upsample_initial_channel (int): Number of base hidden channels for pre-convolution.
            upsample_kernel_sizes (list of int): Convolution shapes assigned for upsamplers.
            gin_channels (int): Dimensionality for speaker/style conditioning features.
            sr (int): Target execution sample rate for the sound model.
            checkpointing (bool): Toggle flag to bypass active memory consumption using checkpoint wrappers.
            harmonic_num (int): Target tracking limits for individual pitch branches.
        """

        super(HiFiGANNSFGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = math.prod(upsample_rates)
        self.harmonic_num = harmonic_num
        # Linear interpolator to stretch raw frame F0 data up to high-resolution sample timeline lengths
        self.f0_upsamp = torch.nn.Upsample(scale_factor=self.upp)
        # Instantiate Source Module with version selection depending on harmonic profile bounds
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=harmonic_num, sinegen_version="v1" if harmonic_num == 0 else "v2")
        # Frontend feature map projection block
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.checkpointing = checkpointing

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()

        # Compute layer-by-layer channel dimension changes
        channels = [
            upsample_initial_channel // (2 ** (i + 1)) 
            for i in range(self.num_upsamples)
        ]
        # Calculate sequential scaling coefficients to downscale excitation signals to current block rates
        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < self.num_upsamples else 1 
            for i in range(self.num_upsamples)
        ]

        # Construct upsampling structures paired with matching source conditioning filters
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i), 
                        channels[i], 
                        k, 
                        u, 
                        padding=((k - u) // 2) if u % 2 == 0 else (u // 2 + u % 2), 
                        output_padding=u % 2
                    )
                )
            )

            # Downsample the main high-resolution excitation source to match the current layer's temporal resolution
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2

            self.noise_convs.append(
                torch.nn.Conv1d(
                    1, 
                    channels[i], 
                    kernel_size=kernel, 
                    stride=stride, 
                    padding=0 if stride == 1 else (kernel - stride) // 2
                )
            )

        # Build parallel MRF ResBlock modules across the progressive scaling stages
        self.resblocks = torch.nn.ModuleList([
            ResBlock(channels[i], k, d) 
            for i in range(len(self.ups)) 
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ])

        # Tail synthesis projection layer
        self.conv_post = torch.nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=bool(harmonic_num != 0))
        # Apply weight normalization on input/output endpoints if multi-harmonic modules are activated
        if harmonic_num != 0: self.conv_pre, self.conv_post = weight_norm(self.conv_pre), weight_norm(self.conv_post)

        self.ups.apply(init_weights)
        # Global style profile mapping layer
        if gin_channels != 0: self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g = None):
        """
        Args:
            x (Tensor): Acoustic Mel-spectrogram tensor input.
            f0 (Tensor): F0 tracking matrix.
            g (Tensor, optional): Global features profile tensor.

        Returns:
            Tensor: Synthesized time-domain speech audio track waveform.
        """

        # If running v2 harmonic settings, stretch low-res frame tracks up to match master audio resolution
        if self.harmonic_num != 0: f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        # Synthesize the continuous master source wave excitation track
        har_source = self.m_source(f0, self.upp).transpose(1, 2)

        # Initial transformation from input features to feature maps
        x = self.conv_pre(x)
        if g is not None: x += self.cond(g)

        # Progressively upsample features and fuse downsampled source excitation components
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = F.leaky_relu(x, LRELU_SLOPE)

            # Route through memory checkpoint wrappers during active training phases if toggled
            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False) + noise_convs(har_source)
                xs = sum([checkpoint(resblock, x, use_reentrant=False) for j, resblock in enumerate(self.resblocks) if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)])
            else:
                x = ups(x) + noise_convs(har_source)
                xs = sum([resblock(x) for j, resblock in enumerate(self.resblocks) if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)])

            # Normalize across the multi-kernel residual channels
            x = xs / self.num_kernels

        # Project features to 1D waveform channel and normalize amplitude within [-1.0, 1.0]
        return self.conv_post(F.leaky_relu(x)).tanh()

    def remove_weight_norm(self):
        """Fuses weight norm parametrizations into single weight matrices to accelerate inference speeds."""

        for l in self.ups:
            if hasattr(l, "parametrizations") and "weight" in l.parametrizations: parametrize.remove_parametrizations(l, "weight", leave_parametrized=True)
            else: remove_weight_norm(l)
        
        # Clear normalization pipelines from input and output endpoints if harmonic modes were active
        if self.harmonic_num != 0:
            if hasattr(self.conv_pre, "parametrizations") and "weight" in self.conv_pre.parametrizations: parametrize.remove_parametrizations(self.conv_pre, "weight", leave_parametrized=True)
            else: remove_weight_norm(self.conv_pre)
            if hasattr(self.conv_post, "parametrizations") and "weight" in self.conv_post.parametrizations: parametrize.remove_parametrizations(self.conv_post, "weight", leave_parametrized=True)
            else: remove_weight_norm(self.conv_post)

        # Clear remaining normalization bindings from the nested ResBlock lists
        for l in self.resblocks:
            l.remove_weight_norm()