import os
import sys
import torch
import torchaudio

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from torch.utils.checkpoint import checkpoint
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

from main.library.algorithm.commons import init_weights, get_padding

class ResBlock(nn.Module):
    """
    Standard Residual Block containing sequential dilated convolutions.
    Refines features using multiple expansion rates to capture varied context windows.
    """

    def __init__(self, channels, kernel_size = 7, dilation = (1, 3, 5), leaky_relu_slope = 0.2):
        """
        Args:
            channels (int): Number of input/output feature channels.
            kernel_size (int): Convolution kernel width. Default: 7.
            dilation (tuple of int): Dilations applied sequentially to the internal convs. Default: (1, 3, 5).
            leaky_relu_slope (float): Negative slope coefficient for LeakyReLU. Default: 0.2.
        """

        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        # Primary convolution group using scheduled dilations and automated padding matching
        self.convs1 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels, 
                    channels, 
                    kernel_size, 
                    stride=1, 
                    dilation=d, 
                    padding=get_padding(kernel_size, d)
                )
            ) 
            for d in dilation
        ])
        self.convs1.apply(init_weights)

        # Secondary convolution group using fixed dilation of 1
        self.convs2 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels, 
                    channels, 
                    kernel_size, 
                    stride=1, 
                    dilation=1, 
                    padding=get_padding(kernel_size, 1)
                )
            ) 
            for _ in dilation
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        """
        Args:
            x (Tensor): Feature maps.

        Returns:
            Tensor: Refined feature maps after sequential residual accumulations.
        """

        for c1, c2 in zip(self.convs1, self.convs2):
            x = c2(F.leaky_relu(c1(F.leaky_relu(x, self.leaky_relu_slope)), self.leaky_relu_slope)) + x

        return x

    def remove_weight_norm(self):
        """Removes weight normalization parametrizations across both inner conv pipelines."""

        for c1, c2 in zip(self.convs1, self.convs2):
            if hasattr(c1, "parametrizations") and "weight" in c1.parametrizations: parametrize.remove_parametrizations(c1, "weight", leave_parametrized=True)
            else: remove_weight_norm(c1)

            if hasattr(c2, "parametrizations") and "weight" in c2.parametrizations: parametrize.remove_parametrizations(c2, "weight", leave_parametrized=True)
            else: remove_weight_norm(c2)

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization variant.
    Injects scaled Gaussian perturbations as a form of lightweight stochastic conditioning.
    """

    def __init__(
        self, 
        *, 
        channels, 
        leaky_relu_slope = 0.2
    ):
        """
        Args:
            channels (int): Channel capacity for input feature vectors.
            leaky_relu_slope (float): LeakyReLU activation slope factor. Default: 0.2.
        """

        super().__init__()
        # Small learned weight scalar parameter initialized to a base floor scale
        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x):
        """
        Args:
            x (Tensor): Target feature map.
        Returns:
            Tensor: Perturbed feature maps after activation.
        """

        # Apply structured noise multiplication matched across the tensor shape dimensions
        return self.activation(x + (torch.randn_like(x) * self.weight[None, :, None]))
    
class ParallelResBlock(nn.Module):
    """
    Parallel Residual Fusion Block.
    Processes features across distinct multi-scale kernel pathways and fuses their outputs.
    """

    def __init__(
        self, 
        *, 
        in_channels, 
        out_channels, 
        kernel_sizes = (3, 7, 11), 
        dilation = (1, 3, 5), 
        leaky_relu_slope = 0.2
    ):
        """
        Args:
            in_channels (int): Input channel counts.
            out_channels (int): Target output channel counts.
            kernel_sizes (tuple of int): Discrete sizes assigned to separate parallel paths. Default: (3, 7, 11).
            dilation (tuple of int): Common dilations passed down into inner ResBlocks. Default: (1, 3, 5).
            leaky_relu_slope (float): Negative activation gradient rate. Default: 0.2.
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Entrance projection mapping to unify disparate incoming channel splits
        self.input_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=3)
        self.input_conv.apply(init_weights)

        # Parallel block streams consisting of AdaIN wrappers surrounding kernel-specific ResBlocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                AdaIN(
                    channels=out_channels
                ),
                ResBlock(
                    out_channels, 
                    kernel_size=kernel_size, 
                    dilation=dilation, 
                    leaky_relu_slope=leaky_relu_slope
                ), 
                AdaIN(
                    channels=out_channels
                )
            ) 
            for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        """
        Args:
            x (Tensor): Integrated context tensors.

        Returns:
            Tensor: Averaged multi-scale fusion results.
        """
    
        x = self.input_conv(x)
        # Compute parallel paths separately, stack them along a new dimension, and extract the mean
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)

    def remove_weight_norm(self):
        """Unwinds weight norm parametrizations embedded within sub-ResBlocks."""

        for block in self.blocks:
            block[1].remove_weight_norm()

class SineGen(nn.Module):
    """
    Sine Wave Excitation Generator component for Neural Source Filter (NSF) architecture.
    Produces harmonic sine wave excitations and noise models based on a primary F0 trajectory.
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
            sampling_rate (int): Audio processing target sample rate.
            harmonic_num (int): Additional overtones to synthesize. Default: 0.
            sine_amp (float): Output base amplitude. Default: 0.1.
            noise_std (float): Target white noise variance. Default: 0.003.
            voiced_threshold (float): Baseline floor frequency determining unvoiced sections. Default: 0.
        """

        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold
        # Collapses multi-harmonic tracks into a unified 1D excitation track
        self.merge = nn.Sequential(nn.Linear(self.dim, 1, bias=False), nn.Tanh())

    def _f02uv(self, f0):
        """Maps continuous F0 tracking values into a binary Voiced/Unvoiced mask."""

        return (f0 > self.voiced_threshold).float()

    def _f02sine(self, f0_values):
        """
        Maps pitch metrics to phase-continuous sine waves.
        
        Args:
            f0_values (Tensor): Harmonically shifted fundamental frequency trackers.

        Returns:
            Tensor: Generated phase-aligned continuous sine tracking array.
        """

        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], dtype=f0_values.dtype, device=f0_values.device)

        # Seed zero starting phase parameters on target frame zero bounds
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # Accumulate time sequences smoothly across period limits
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0

        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        return (torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi).sin()
    
    def forward(self, f0):
        """
        Args:
            f0 (Tensor): Low-res fundamental frequency tracking matrix.

        Returns:
            Tensor: Projective 1D excitation signal.
        """

        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, dtype=f0.dtype, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]

            # Form overtone channels using consecutive integer multiplications
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            # Synthesize basic sine structures
            # The input must be kept as a float; otherwise, the output is prone to becoming NaN.
            sine_waves = self._f02sine(f0_buf.float()) * self.sine_amp
            uv = self._f02uv(f0)
            sine_waves = sine_waves * uv + ((uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves))

        return self.merge(sine_waves.to(f0.dtype))
    
class RefineGANGenerator(nn.Module):
    """
    RefineGAN Generator network.
    Combines a U-Net style downsampling analyzer for the raw excitation signal
    with a parallel multi-scale upsampling decoder conditioned on Mel-spectrograms.
    """

    def __init__(
        self, 
        *, 
        sample_rate = 44100, 
        upsample_rates = (8, 8, 2, 2), 
        leaky_relu_slope = 0.2, 
        num_mels = 128, 
        start_channels = 16, 
        gin_channels = 256, 
        checkpointing = False, 
        upsample_initial_channel = 512
    ):
        """
        Args:
            sample_rate (int): Base operational frequency used inside sine oscillators. Default: 44100.
            upsample_rates (tuple of int): Multi-stage stride upscale coefficients. Default: (8, 8, 2, 2).
            leaky_relu_slope (float): LeakyReLU negative alpha parameter. Default: 0.2.
            num_mels (int): Dimensions of source Mel bands. Default: 128.
            start_channels (int): Initial filter count for the first raw source encoder stage. Default: 16.
            gin_channels (int): Sizing parameters for speaker or style embeddings. Default: 256.
            checkpointing (bool): Toggle to activate VRAM-saving gradient checkpoint wrappers. Default: False.
            upsample_initial_channel (int): Hidden feature channel depth for decoding branches. Default: 512.
        """

        super().__init__()
        self.upsample_rates = upsample_rates
        self.checkpointing = checkpointing
        self.leaky_relu_slope = leaky_relu_slope
        self.upp = np.prod(upsample_rates)
        self.m_source = SineGen(sample_rate)
        # Front conv mapping 1D excitation track up to base filter space
        self.pre_conv = weight_norm(nn.Conv1d(1, 16, 7, 1, padding=3))
        channels = start_channels
        size = self.upp
        self.downsample_blocks = nn.ModuleList([])
        self.df0 = []

        # Map downscaling steps backwards from master rate down to localized mel steps
        for i, _ in enumerate(upsample_rates):
            new_size = int(size / upsample_rates[-i - 1])
            self.df0.append([size, new_size])
            size = new_size

            new_channels = channels * 2
            self.downsample_blocks.append(weight_norm(nn.Conv1d(channels, new_channels, 7, 1, padding=3)))
            channels = new_channels

        # Mel acoustic map preprocessing layers
        channels = upsample_initial_channel
        self.mel_conv = weight_norm(nn.Conv1d(num_mels, channels // 2, 7, 1, padding=3))
        self.mel_conv.apply(init_weights)

        # Global style conditioning map vector layer
        if gin_channels != 0: self.cond = nn.Conv1d(256, channels // 2, 1)

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])
        # Allocate matching upsamplers along with Parallel Fusion blocks
        for rate in upsample_rates:
            new_channels = channels // 2
            self.upsample_blocks.append(
                nn.Upsample(
                    scale_factor=rate, 
                    mode="linear"
                )
            )
            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    in_channels=channels + channels // 4, 
                    out_channels=new_channels, 
                    kernel_sizes=(3, 7, 11), 
                    dilation=(1, 3, 5), 
                    leaky_relu_slope=leaky_relu_slope
                )
            )
            channels = new_channels

        # Terminal projection flattening down into standard 1D raw waveform output space
        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False))
        self.conv_post.apply(init_weights)

    def forward(self, mel, f0, g = None):
        """
        Args:
            mel (Tensor): Mel-spectrogram features.
            f0 (Tensor): Fundamental frequency trajectory.
            g (Tensor, optional): Speaker/style condition identity maps.

        Returns:
            Tensor: Synthesized high-fidelity target audio wave.
        """

        f0_size = mel.shape[-1]
        # Interpolate pitch sequence to audio rate and generate raw multi-harmonic excitation wave
        har_source = self.m_source(F.interpolate(f0.unsqueeze(1), size=f0_size * self.upp, mode="linear").transpose(1, 2)).transpose(1, 2)
        # Analyze excitation via progressive downsampling steps and collect multi-scale features
        x = self.pre_conv(har_source)
        downs = []

        for block, (old_size, new_size) in zip(self.downsample_blocks, self.df0):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            downs.append(x)

            # High-fidelity resampler using Kaiser-windowed Sinc interpolation
            # Half-precison turns this resampler into a professional audio destroyer.
            x = torchaudio.functional.resample(
                x.float().contiguous(), 
                orig_freq=int(f0_size * old_size), 
                new_freq=int(f0_size * new_size), 
                lowpass_filter_width=64, 
                rolloff=0.9475937167399596, 
                resampling_method="sinc_interp_kaiser", 
                beta=14.769656459379492
            ).to(x.dtype)
            x = block(x)

        # Preprocess acoustic feature blueprints and mix identity descriptors if available
        mel = self.mel_conv(mel)
        if g is not None: mel += self.cond(g)

        # Combine mel features with the lowest-resolution excitation features to begin decoding
        x = torch.cat([mel, x], dim=1)

        # Progressively upscale feature resolution while concatenating corresponding encoder features (skip connections)
        for ups, res, down in zip(self.upsample_blocks, self.upsample_conv_blocks, reversed(downs)):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            # Execute blocks using memory-saving checkpoint steps if training modes request it
            x = checkpoint(res, torch.cat([checkpoint(ups, x, use_reentrant=False), down], dim=1), use_reentrant=False) if self.training and self.checkpointing else res(torch.cat([ups(x), down], dim=1))

        # Perform final mapping back down into amplitude wave steps bounded within [-1.0, 1.0]
        return self.conv_post(F.leaky_relu(x, self.leaky_relu_slope)).tanh()

    def remove_weight_norm(self):
        """Fuses weight norm parametrizations into single weight matrices to accelerate inference speeds."""

        if hasattr(self.pre_conv, "parametrizations") and "weight" in self.pre_conv.parametrizations: parametrize.remove_parametrizations(self.pre_conv, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.pre_conv)

        if hasattr(self.mel_conv, "parametrizations") and "weight" in self.mel_conv.parametrizations: parametrize.remove_parametrizations(self.mel_conv, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.mel_conv)

        if hasattr(self.conv_post, "parametrizations") and "weight" in self.conv_post.parametrizations: parametrize.remove_parametrizations(self.conv_post, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_post)

        for block in self.downsample_blocks:
            if hasattr(block, "parametrizations") and "weight" in block.parametrizations: parametrize.remove_parametrizations(block, "weight", leave_parametrized=True)
            else: remove_weight_norm(block)

        for block in self.upsample_conv_blocks:
            block.remove_weight_norm()