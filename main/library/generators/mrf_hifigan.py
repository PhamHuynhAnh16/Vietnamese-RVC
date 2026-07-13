import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from torch.nn.utils import remove_weight_norm
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1

class MRFLayer(nn.Module):
    """
    Multi-Receptive Field (MRF) Layer containing two dilated convolutions with a residual connection.
    Processes the features at a specific kernel size and dilation rate.
    """

    def __init__(
        self, 
        channels, 
        kernel_size, 
        dilation
    ):
        """
        Args:
            channels (int): Number of input/output feature channels.
            kernel_size (int): Convolution kernel size.
            dilation (int): Dilation rate for the first convolution layer.
        """

        super().__init__()
        # Dilated convolution layer with weight normalization applied
        self.conv1 = weight_norm(
            nn.Conv1d(
                channels, 
                channels, 
                kernel_size, 
                padding=(kernel_size * dilation - dilation) // 2, 
                dilation=dilation
            )
        )
        # Regular non-dilated convolution layer with weight normalization applied
        self.conv2 = weight_norm(
            nn.Conv1d(
                channels, 
                channels, 
                kernel_size, 
                padding=kernel_size // 2, 
                dilation=1
            )
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature maps.
        Returns:
            Tensor: Output feature maps with residual addition.
        """

        return x + self.conv2(F.leaky_relu(self.conv1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))

    def remove_weight_norm(self):
        """Removes weight normalization parametrization hooks from both convolutions."""

        if hasattr(self.conv1, "parametrizations") and "weight" in self.conv1.parametrizations: parametrize.remove_parametrizations(self.conv1, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv1)

        if hasattr(self.conv2, "parametrizations") and "weight" in self.conv2.parametrizations: parametrize.remove_parametrizations(self.conv2, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv2)

class MRFBlock(nn.Module):
    """
    MRF Block containing multiple MRFLayers arranged sequentially with different dilations.
    """

    def __init__(self, channels, kernel_size, dilations):
        """
        Args:
            channels (int): Number of feature channels.
            kernel_size (int): Common kernel size for internal layers.
            dilations (list of int): List of dilation factors for each sequential layer.
        """

        super().__init__()
        self.layers = nn.ModuleList()
        # Stack multiple MRF layers sequentially
        for dilation in dilations:
            self.layers.append(MRFLayer(channels, kernel_size, dilation))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature maps.
        Returns:
            Tensor: Sequentially processed feature maps.
        """

        for layer in self.layers:
            x = layer(x)

        return x

    def remove_weight_norm(self):
        """Recursively removes weight normalization from all sub-layers."""
        for layer in self.layers:
            layer.remove_weight_norm()

class SineGen(nn.Module):
    """
    Sine wave oscillator for Neural Source Filter (NSF) components.
    Generates deterministic phase-continuous harmonic excitations based on F0 sequence.
    """

    def __init__(
        self, 
        sampling_rate, 
        harmonic_num = 0, 
        sine_amp = 0.1, 
        noise_std = 0.003, 
        voiced_threshold = 0
    ):
        """
        Args:
            sampling_rate (int): Target audio sampling rate.
            harmonic_num (int): Number of upper harmonics to generate. Default: 0.
            sine_amp (float): Base amplitude of the sine waves. Default: 0.1.
            noise_std (float): Standard deviation of random Gaussian noise. Default: 0.003.
            voiced_threshold (float): F0 threshold determining voiced/unvoiced frame status. Default: 0.
        """

        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        """Binarizes F0 trajectory into Voiced/Unvoiced mask."""

        return (f0 > self.voiced_threshold).float()

    def _f02sine(self, f0_values):
        """
        Converts dynamic F0 sequences into continuous sine waves without phase phase-breaks.
        
        Args:
            f0_values (Tensor): F0 tracks for each harmonic.

        Returns:
            Tensor: Synthesized phase-continuous sine waves.
        """

        # Calculate phase steps per temporal step
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], dtype=f0_values.dtype, device=f0_values.device)
        # Ground initial phase for the first time step
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # Accumulate phases and wrap bounds within [0, 1]
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        # Run final mapping via sine function
        return (torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi).sin()

    def forward(self, f0):
        """
        Args:
            f0 (Tensor): Input raw fundamental frequency trajectory.

        Returns:
            Tensor: Combined sine and noise excitation matrix.
        """

        with torch.no_grad():
            # Setup fundamental buffer and seed the root F0 track
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, dtype=f0.dtype, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]

            # Populate harmonic components (F0 * 2, F0 * 3, ...)
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            # Generate structured sine streams
            # The input must be kept as a float; otherwise, the output is prone to becoming NaN.
            sine_waves = self._f02sine(f0_buf.float()) * self.sine_amp
            uv = self._f02uv(f0)
            # Add Gaussian noise into voiced streams and soft white noise into unvoiced sections
            sine_waves = sine_waves * uv + ((uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves))

        return sine_waves

class SourceModuleHnNSF(nn.Module):
    """
    Source wrapper for the Harmonic-plus-Noise Neural Source Filter mechanism.
    Collapses multi-harmonic tracks down into a single composite raw excitation source.
    """

    def __init__(
        self, 
        sampling_rate, 
        harmonic_num = 0, 
        sine_amp = 0.1, 
        add_noise_std = 0.003, 
        voiced_threshold = 0
    ):
        """
        Args:
            sampling_rate (int): Target sampling rate.
            harmonic_num (int): Harmonic counts. Default: 0.
            sine_amp (float): Sine component scale. Default: 0.1.
            add_noise_std (float): Noise standard deviation. Default: 0.003.
            voiced_threshold (float): Voiced threshold frequency. Default: 0.
        """

        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x (Tensor): Fundamental frequency tracking tensor.
        Returns:
            Tensor: Projective 1D excitation signal tensor.
        """

        # Compute harmonics, map down to 1D via linear layer, and clip peaks via Tanh
        return self.l_tanh(
            self.l_linear(
                self.l_sin_gen(x).to(dtype=self.l_linear.weight.dtype)
            )
        )

class HiFiGANMRFGenerator(nn.Module):
    """
    Modified HiFi-GAN Generator equipped with Multi-Receptive Field (MRF) fusion blocks,
    NSF multi-scale excitation conditioning, and optional Gradient Checkpointing for VRAM savings.
    """

    def __init__(
        self, 
        in_channel, 
        upsample_initial_channel, 
        upsample_rates, 
        upsample_kernel_sizes, 
        resblock_kernel_sizes, 
        resblock_dilations, 
        gin_channels, 
        sample_rate, 
        harmonic_num, 
        checkpointing = False
    ):
        """
        Args:
            in_channel (int): Dimensions of input acoustic features (Mel-spec bins).
            upsample_initial_channel (int): Number of base hidden channels for pre-convolution.
            upsample_rates (list of int): Temporal scaling ratios per upsampling layer.
            upsample_kernel_sizes (list of int): Kernel shapes for transposed convolutions.
            resblock_kernel_sizes (list of int): Core kernel sizes utilized for MRF Blocks.
            resblock_dilations (list of list of int): Dilations grouped for sequential MRF layers.
            gin_channels (int): Dimensionality for speaker or style conditioning.
            sample_rate (int): Sampling frequency for source wave generator.
            harmonic_num (int): Harmonics capacity for the source filter.
            checkpointing (bool): Activates PyTorch gradient checkpointing during training. Default: False.
        """

        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.checkpointing = checkpointing
        # Temporal upsampler to stretch sparse F0 vectors to high-resolution raw audio lengths
        self.f0_upsample = nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)
        # Frontend embedding convolution
        self.conv_pre = weight_norm(nn.Conv1d(in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3))
        self.upsamples = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        # Calculate downsampling ratios needed to align master excitation down to each sub-scale block
        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1 
            for i in range(len(upsample_rates))
        ]

        # Allocate transposed conv blocks and secondary source alignment pipelines
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i), 
                        upsample_initial_channel // (2 ** (i + 1)), 
                        kernel_size=k, 
                        stride=u, 
                        padding=((k - u) // 2) if u % 2 == 0 else (u // 2 + u % 2), 
                        output_padding=u % 2
                    )
                )
            )

            # Build matching scale downsampling convolutional layers for source signal addition
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2

            self.noise_convs.append(
                nn.Conv1d(
                    1, 
                    upsample_initial_channel // (2 ** (i + 1)), 
                    kernel_size=kernel, 
                    stride=stride, 
                    padding=0 if stride == 1 else (kernel - stride) // 2
                )
            )

        # Assemble independent MRF Blocks for multi-scale parallel processing
        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                nn.ModuleList([
                    MRFBlock(
                        channel, 
                        kernel_size=k, 
                        dilations=d
                    ) 
                    for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                ])
            )

        # Final projection block mapping down to audio mono track
        self.conv_post = weight_norm(nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3))
        if gin_channels != 0: self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g = None):
        """
        Args:
            x (Tensor): Input Mel-spectrogram tensor.
            f0 (Tensor): Pitch trajectory vector.
            g (Tensor, optional): Conditioning embed input.
        Returns:
            Tensor: Regenerated continuous time audio waveform.
        """

        # Synthesize audio-rate continuous excitation template from pitch track
        har_source = self.m_source(self.f0_upsample(f0[:, None, :]).transpose(-1, -2)).transpose(-1, -2)
        
        # Preprocess features and mix global condition profiles if available
        x = self.conv_pre(x)
        if g is not None: x += self.cond(g)

        # Iterate progressive scale blocks sequentially
        for ups, mrf, noise_conv in zip(self.upsamples, self.mrfs, self.noise_convs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            # Execute forward layers with gradient checkpointing to reduce VRAM if requested
            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False) + noise_conv(har_source)
                xs = sum([checkpoint(layer, x, use_reentrant=False) for layer in mrf])
            else:
                x = ups(x) + noise_conv(har_source)
                xs = sum([layer(x) for layer in mrf])

            # Blend features together from all multi-kernel paths
            x = xs / self.num_kernels

        # Project representations back to audio amplitudes bounded within [-1.0, 1.0]
        return self.conv_post(F.leaky_relu(x)).tanh()

    def remove_weight_norm(self):
        """
        Fuses weight normalization parameters into raw tensor parameters for fast inference deployment.
        """

        if hasattr(self.conv_pre, "parametrizations") and "weight" in self.conv_pre.parametrizations: parametrize.remove_parametrizations(self.conv_pre, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_pre)

        for up in self.upsamples:
            if hasattr(up, "parametrizations") and "weight" in up.parametrizations: parametrize.remove_parametrizations(up, "weight", leave_parametrized=True)
            else: remove_weight_norm(up)

        for mrf in self.mrfs:
            for block in mrf:
                block.remove_weight_norm()

        if hasattr(self.conv_post, "parametrizations") and "weight" in self.conv_post.parametrizations: parametrize.remove_parametrizations(self.conv_post, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_post)