import os
import sys
import torch

import torch.nn.utils.parametrize as parametrize

from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

from main.library.algorithm.commons import init_weights
from main.library.algorithm.residuals import ResBlock, LRELU_SLOPE

class HiFiGANGenerator(torch.nn.Module):
    """
    HiFi-GAN generator for synthesizing high-fidelity audio waveforms.

    The generator progressively upsamples frame-level acoustic representations
    through a stack of transposed convolution layers followed by
    Multi-Receptive Field Fusion (MRF) residual blocks. Outputs from multiple
    residual blocks at each upsampling stage are averaged to capture features
    across different receptive fields.
    """

    def __init__(
        self, 
        initial_channel, 
        resblock_kernel_sizes, 
        resblock_dilation_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        upsample_kernel_sizes, 
        gin_channels=0
    ):
        """
        Initialize the HiFi-GAN generator.

        Args:
            initial_channel (int): Number of input feature channels.
            resblock_kernel_sizes (list of int): Kernel sizes used by each residual block.
            resblock_dilation_sizes (Sequencelist of int): Dilation rates for each residual block.
            upsample_rates (list of int): Upsampling factors applied at each stage.
            upsample_initial_channel (int): Number of channels before the first upsampling stage.
            upsample_kernel_sizes (list of int): Kernel sizes for transposed convolution layers.
            gin_channels (int, optional): Number of global conditioning channels. If zero, conditioning is disabled.
        """

        super(HiFiGANGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        # Pre-convolution to map input features to the initial upsampling channel dimension
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        # Lists to hold progressive upsampling layers and corresponding multi-scale residual blocks
        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()

        # Build upsampling modules paired with parallel multi-receptive field ResBlocks
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Transposed convolution to upscale temporal resolution and halve channel dimension
            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i), 
                        upsample_initial_channel // (2 ** (i + 1)), 
                        k, 
                        u, 
                        padding=(k - u) // 2
                    )
                )
            )

            # Construct parallel ResBlocks for the current scale to capture different temporal patterns
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        # Final post-convolution to map hidden features to a 1-channel raw audio waveform
        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # Initialize weights for all upsampling layers using custom initialization scheme
        self.ups.apply(init_weights)
        # Global conditioning projection layer if speaker/style embedding channels are provided
        if gin_channels != 0: self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g = None):
        """
        Args:
            x (Tensor): Input Mel-spectrogram tensor.
            f0 (Tensor): Unused fundamental frequency tensor (maintained for API compatibility).
            g (Tensor, optional): Global conditioning tensor.

        Returns:
            Tensor: Synthesized 1D waveform tensor.
        """
        # Initial transformation from Mel-spectrogram to feature map
        x = self.conv_pre(x)
        # Inject global speaker/style conditioning if available
        if g is not None: x += self.cond(g)

        # Progressively upsample and refine features stage-by-stage
        for i in range(self.num_upsamples):
            # Apply LeakyReLU activation followed by ConvTranspose1d upsampling
            x = self.ups[i](torch.nn.functional.leaky_relu(x, LRELU_SLOPE))
            xs = None

            # Accumulate and average outputs from parallel Multi-Receptive Field ResBlocks
            for j in range(self.num_kernels):
                if xs is None: xs = self.resblocks[i * self.num_kernels + j](x)
                else: xs += self.resblocks[i * self.num_kernels + j](x)
            
            # Normalize by the number of parallel kernels to prevent scale explosion
            x = xs / self.num_kernels

        # Apply final LeakyReLU activation, squeeze via post-convolution, and bound audio to [-1.0, 1.0]
        return self.conv_post(torch.nn.functional.leaky_relu(x)).tanh()
    
    def remove_weight_norm(self):
        """
        Removes weight normalization from all transposed convolutions and residual blocks.
        This optimization reduces computation overhead and speeds up the model during inference.
        """

        # Remove weight norm from upsampling layers
        for l in self.ups:
            if hasattr(l, "parametrizations") and "weight" in l.parametrizations: parametrize.remove_parametrizations(l, "weight", leave_parametrized=True)
            else: remove_weight_norm(l)

        # Delegate weight norm removal to individual ResBlocks
        for l in self.resblocks:
            l.remove_weight_norm()