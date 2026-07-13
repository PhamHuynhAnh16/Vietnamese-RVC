import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

class AntiAliasActivation(nn.Module):
    """
    Anti-aliased activation layer using upsampling, SnakeBeta activation, and downsampling.
    This helps prevent high-frequency aliasing artifacts when applying non-linearities.
    """

    def __init__(
        self, 
        channels, 
        up=2, 
        down=2, 
        up_k=12, 
        down_k=12
    ):
        """
        Args:
            channels (int): Number of input/output channels.
            up (int): Upsampling factor. Default: 2.
            down (int): Downsampling factor. Default: 2.
            up_k (int): Kernel size for the upsampling low-pass filter. Default: 12.
            down_k (int): Kernel size for the downsampling low-pass filter. Default: 12.
        """

        super().__init__()
        # Internal modules for anti-aliasing pipeline
        self.up = UpSample1d(up, up_k)
        self.act = SnakeBeta(channels)
        self.down = DownSample1d(down, down_k)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, T).
        Returns:
            Tensor: Activated and filtered tensor of shape (B, C, T).
        """

        # Run through upsample -> non-linear activation -> downsample
        return self.down(
            self.act(self.up(x))
        )

class SnakeBeta(nn.Module):
    """
    SnakeBeta activation function proposed in BigVGAN.
    Introduces learnable periodic parameters (alpha and beta) to better model periodic signals like audio.
    """

    def __init__(
        self, 
        channels
    ):
        """
        Args:
            channels (int): Number of channels for learnable parameters.
        """

        super().__init__()
        # Initialize learnable parameters to zeros
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, T).
        Returns:
            Tensor: SnakeBeta activated tensor of shape (B, C, T).
        """

        # Formula: x + (1 / (e^beta + 1e-9)) * sin^2(x * e^alpha)
        return x + (1.0 / (self.beta.exp() + 1e-9)) * (x * self.alpha.exp()).sin().pow(2)

def kaiser_sinc_filter1d(
    cutoff, 
    half_width, 
    kernel_size
):
    """
    Generates a 1D low-pass sinc filter windowed with a Kaiser window.

    Args:
        cutoff (float): Cutoff frequency relative to Nyquist (0.0 to 0.5).
        half_width (float): Transition band half-width.
        kernel_size (int): Total size of the filter kernel.
    Returns:
        Tensor: Generated filter tensor of shape (1, 1, kernel_size).
    """

    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width

    # Compute empirical Kaiser beta parameter based on stopband attenuation formula
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0

    # Generate Kaiser window
    window = torch.kaiser_window(
        kernel_size, 
        beta=beta, 
        periodic=False
    )

    # Compute time grid depending on kernel parity
    time = (
        torch.arange(-half_size, half_size) + 0.5
    ) if even else (
        torch.arange(kernel_size) - half_size
    )

    # Compute windowed sinc function
    if cutoff == 0:
        filter = torch.zeros_like(time)
    else:
        filter = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        filter /= filter.sum() # Normalize to ensure unity gain at DC

    return filter.view(1, 1, kernel_size)

class UpSample1d(nn.Module):
    """
    1D Upsampling layer utilizing a Kaiser-windowed sinc filter for anti-aliasing during interpolation.
    """

    def __init__(
        self, 
        ratio=2, 
        kernel_size=None
    ):
        """
        Args:
            ratio (int): Upsampling factor. Default: 2.
            kernel_size (int, optional): Size of the low-pass filter kernel. Defaults to automatic calculation.
        """

        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        # Determine kernel size if not provided
        kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        # Calculate padding to ensure proper centering after transposed convolution
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2

        # Design the anti-aliasing low-pass filter
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, 
            half_width=0.6 / ratio, 
            kernel_size=kernel_size
        )
        self.register_buffer("filter", filter)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Upsampled tensor.
        """

        # Apply replicated padding to reduce edge artifacts
        x = self.ratio * F.conv_transpose1d(
            F.pad(
                x, 
                (self.pad, self.pad), 
                mode="replicate"
            ), 
            self.filter.expand(x.size(1), -1, -1), # Expand filter to match input channel size
            stride=self.stride, 
            groups=x.size(1)
        )

        # Slice the tensor to remove extra padding artifacts
        return x[..., self.pad_left : -self.pad_right]

class LowPassFilter1d(nn.Module):
    """
    Standard 1D Low-Pass Filter layer using Kaiser-windowed sinc weights.
    """

    def __init__(
        self, 
        cutoff=0.5, 
        half_width=0.6, 
        stride=1, 
        kernel_size=12
    ):
        """
        Args:
            cutoff (float): Cutoff frequency. Must be between 0.0 and 0.5. Default: 0.5.
            half_width (float): Transition width. Default: 0.6.
            stride (int): Stride for downsampling. Default: 1.
            kernel_size (int): Size of the filter kernel. Default: 12.
        """

        super().__init__()
        if cutoff < -0.0 or cutoff > 0.5:
            raise ValueError("Cutoff frequency must be in the range [0.0, 0.5]")

        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        # Create filter buffer
        filter = kaiser_sinc_filter1d(
            cutoff, 
            half_width, 
            kernel_size
        )
        self.register_buffer("filter", filter)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, T).
        Returns:
            Tensor: Filtered (and potentially downsampled) tensor.
        """

        # Pad edges, expand filter across channels, and apply depthwise convolution
        return F.conv1d(
            F.pad(
                x, 
                (self.pad_left, self.pad_right), 
                mode="replicate"
            ), 
            self.filter.expand(x.size(1), -1, -1), 
            stride=self.stride, 
            groups=x.size(1)
        )

class DownSample1d(nn.Module):
    """
    1D Downsampling layer that applies a low-pass filter before strided decimation to avoid aliasing.
    """

    def __init__(
        self, 
        ratio=2, 
        kernel_size=None
    ):
        """
        Args:
            ratio (int): Downsampling factor. Default: 2.
            kernel_size (int, optional): Filter kernel size.
        """
        super().__init__()
        # Combine low-pass filtering and strided downsampling into a single LowPassFilter1d module
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, T).
        Returns:
            Tensor: Downsampled tensor.
        """

        return self.lowpass(x)

class AMPLayer(nn.Module):
    def __init__(
        self, 
        channels, 
        kernel_size, 
        dilation
    ):
        """
        Args:
            channels (int): Number of feature channels.
            kernel_size (int): Size of the convolution kernel.
            dilation (int): Dilation rate for the first convolution layer.
        """

        super().__init__()
        # Dilated convolution with weight normalization
        self.conv1 = weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size * dilation - dilation) // 2,
                dilation=dilation,
            )
        )
        # Regular convolution with weight normalization
        self.conv2 = weight_norm(
            nn.Conv1d(
                channels, 
                channels, 
                kernel_size, 
                padding=kernel_size // 2, 
                dilation=1
            )
        )

        # Anti-aliased activations instead of standard LeakyReLU/ReLU
        self.act1 = AntiAliasActivation(channels)
        self.act2 = AntiAliasActivation(channels)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Residual output tensor.
        """

        # Forward pass with residual/skip connection
        y = self.conv1(self.act1(x))
        y = self.conv2(self.act2(y))

        return x + y

    def remove_weight_norm(self):
        """
        Removes weight normalization from both convolution layers.
        Supports both modern torch parametrization and legacy weight_norm hooks.
        """

        if hasattr(self.conv1, "parametrizations") and "weight" in self.conv1.parametrizations: parametrize.remove_parametrizations(self.conv1, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv1)

        if hasattr(self.conv2, "parametrizations") and "weight" in self.conv2.parametrizations: parametrize.remove_parametrizations(self.conv2, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv2)

class AMPBlock(nn.Module):
    """
    AMP Block containing multiple AMPLayers with different dilation factors.
    """

    def __init__(
        self, 
        channels, 
        kernel_size, 
        dilations
    ):
        """
        Args:
            channels (int): Number of feature channels.
            kernel_size (int): Kernel size for convolutions.
            dilations (list of int): List of dilations for each AMPLayer.
        """

        super().__init__()
        # Accumulate multiple AMPLayers sequentially
        self.layers = nn.ModuleList([
            AMPLayer(channels, kernel_size, dilation) 
            for dilation in dilations
        ])

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Processed feature map.
        """

        for layer in self.layers:
            x = layer(x)

        return x

    def remove_weight_norm(self):
        """
        Recursively removes weight normalization from all sub-layers.
        """

        for layer in self.layers:
            layer.remove_weight_norm()

class SineGen(nn.Module):
    """
    Sine wave generator for Neural Source Filter (NSF) models.
    Generates excitation signals based on Fundamental Frequency (F0).
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        """
        Args:
            sampling_rate (int): Sampling rate of the target audio.
            harmonic_num (int): Number of harmonics to generate. Default: 0.
            sine_amp (float): Amplitude of sine waves. Default: 0.1.
            noise_std (float): Standard deviation of unvoiced noise. Default: 0.003.
            voiced_threshold (float): F0 threshold to determine voiced/unvoiced frames. Default: 0.
            flag_for_pulse (bool): Unused flag preserved for compatibility. Default: False.
        """

        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        """Calculates Voiced/Unvoiced (U/V) mask based on threshold."""

        return (f0 > self.voiced_threshold).float()

    def _f02sine(self, f0_values):
        """
        Converts continuous F0 trajectories into phase-continuous sine waves.
        
        Args:
            f0_values (Tensor): F0 values for harmonics.

        Returns:
            Tensor: Generated sine wave matrix.
        """

        # Calculate phase increment per sample step
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], dtype=f0_values.dtype, device=f0_values.device)

        # Set first time-step initial phase
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # Handle phase wrapping logic over multiple periods
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0

        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        # Compute continuous sine trajectories
        return (torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi).sin()

    def forward(self, f0):
        """
        Args:
            f0 (Tensor): F0 tensor.

        Returns:
            Tensor: Sine waves mixed with noise excitation.
        """

        with torch.no_grad():
            # Initialize array to contain fundamental frequency and its harmonics
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, dtype=f0.dtype, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]

            # Compute harmonic frequencies (F0 * 2, F0 * 3, ...)
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            # Generate pure sine waves
            # The input must be kept as a float; otherwise, the output is prone to becoming NaN.
            sine_waves = self._f02sine(f0_buf.float()) * self.sine_amp
            uv = self._f02uv(f0)

            # Blend sine waves with Gaussian noise for voiced sections, or pure noise for unvoiced sections
            sine_waves = sine_waves * uv + ((uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves))

        return sine_waves

class SourceModuleHnNSF(nn.Module):
    """
    Source Module for Harmonic-plus-Noise NSF architecture.
    Refines raw excitation signals through linear mixing and non-linear squashing.
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0.0,
    ):
        """
        Args:
            sampling_rate (int): Audio sampling rate.
            harmonic_num (int): Number of harmonics. Default: 0.
            sine_amp (float): Sine amplitude. Default: 0.1.
            add_noise_std (float): Noise standard deviation. Default: 0.003.
            voiced_threshod (float): Threshold for voiced frames. Default: 0.0.
        """

        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        # Merge multi-harmonic signals into a mono channel excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x (Tensor): F0 input tensor.

        Returns:
            Tensor: Blended mono source excitation tensor.
        """

        # Generate, project, and squash excitation through Tanh
        sine_merge = self.l_tanh(
            self.l_linear(
                self.l_sin_gen(x).to(dtype=self.l_linear.weight.dtype)
            )
        )

        return sine_merge

class BigVGANGenerator(nn.Module):
    """
    BigVGAN Generator incorporating Anti-Aliased Multi-Period Blocks (AMP) 
    and Neural Source Filter (NSF) excitation signals for high-fidelity vocoding.
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
    ):
        """
        Args:
            in_channel (int): Number of input channels (e.g., Mel-spectrogram bins).
            upsample_initial_channel (int): Number of hidden channels for the initial convolution.
            upsample_rates (list of int): Upsampling factor per layer.
            upsample_kernel_sizes (list of int): Kernel sizes for transposed convolutions.
            resblock_kernel_sizes (list of int): Kernel sizes for AMP Blocks.
            resblock_dilations (list of list of int): Dilations for layers inside AMP Blocks.
            gin_channels (int): Global conditioning channels (0 if disabled).
            sample_rate (int): Sampling rate for NSF module.
            harmonic_num (int): Harmonic number for NSF module.
        """

        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len (upsample_rates)

        # Temporal upsampling for F0 sequence to match audio resolution
        self.f0_upsample = nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)

        # Initial preprocessing convolution
        self.conv_pre = weight_norm(
            nn.Conv1d(
                in_channel, 
                upsample_initial_channel, 
                kernel_size=7, 
                stride=1, 
                padding=3
            )
        )

        self.amps = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        # Compute internal downsampling strides for multi-scale source conditioning
        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < self.num_upsamples else 1 
            for i in range(self.num_upsamples)
        ]

        # Construct upsampling layers and corresponding noise downsamplers
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            padding = ((k - u) // 2) if u % 2 == 0 else (u // 2 + u % 2)
                
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=padding,
                        output_padding=u % 2,
                    )
                )
            )

            # Match source excitation resolution down to the specific upsampling scale
            stride = stride_f0s[i]
            kernel = (1 if stride == 1 else stride * 2 - stride % 2)
            padding = (0 if stride == 1 else (kernel - stride) // 2)
            
            self.noise_convs.append(
                nn.Conv1d(
                    1,
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

        # Build parallel AMP blocks for multi-scale feature refinement
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))

            self.amps.append(
                nn.ModuleList([
                    AMPBlock(channel, kernel_size=k, dilations=d)
                    for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                ])
            )

        # Final output layers
        self.act_post = AntiAliasActivation(channel)
        self.conv_post = weight_norm(
            nn.Conv1d(
                channel, 
                1, 
                kernel_size=7, 
                stride=1, 
                padding=3
            )
        )

        # Global conditioning layer (e.g. speaker embeddings)
        if gin_channels != 0:
            self.cond = nn.Conv1d(
                gin_channels, 
                upsample_initial_channel, 
                1
            )

    def forward(self, x, f0, g=None):
        """
        Args:
            x (Tensor): Mel-spectrogram features of shape (B, in_channels, T_mel).
            f0 (Tensor): Fundamental frequency trajectory of shape (B, T_mel).
            g (Tensor, optional): Global conditioning feature of shape (B, gin_channels, 1).
        Returns:
            Tensor: Synthesized audio waveform of shape (B, 1, T_audio).
        """

        # Generate full-resolution harmonic excitation source
        har_source = self.m_source(self.f0_upsample(f0[:, None, :]).transpose(-1, -2)).transpose(-1, -2)
        # Initial transformation
        x = self.conv_pre(x)
        if g is not None: x += self.cond(g)  
        
        # Iterative upsampling combined with AMP block refinement
        for up, amp, noise_conv in zip(self.upsamples, self.amps, self.noise_convs):
            xs = 0

            x = up(x) # Scale feature maps up
            x += noise_conv(har_source) # Inject multi-scale source excitation

            # Process through parallel multi-kernel AMP layers
            for layer in amp:
                xs += layer(x)

            # Average parallel block streams
            x = xs / self.num_kernels

        # Final activation and mapping to continuous waveform domain [-1.0, 1.0]
        return self.conv_post(self.act_post(x)).tanh()

    def remove_weight_norm(self):
        """
        Removes weight normalization from all pre-convolutions, post-convolutions, 
        and internal AMP blocks for inference deployment.
        """

        if hasattr(self.conv_pre, "parametrizations") and "weight" in self.conv_pre.parametrizations: parametrize.remove_parametrizations(self.conv_pre, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_pre)

        for up in self.upsamples:
            if hasattr(up, "parametrizations") and "weight" in up.parametrizations: parametrize.remove_parametrizations(up, "weight", leave_parametrized=True)
            else: remove_weight_norm(up)

        for amps in self.amps:
            for amp in amps:
                amp.remove_weight_norm()

        if hasattr(self.conv_post, "parametrizations") and "weight" in self.conv_post.parametrizations: parametrize.remove_parametrizations(self.conv_post, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_post)