import os
import sys

import torch.nn as nn

sys.path.append(os.getcwd())

from main.library.predictors.DJCM.utils import ResConvBlock

class ResEncoderBlock(nn.Module):
    """
    Residual Encoder Block combining feature extraction and optional spatial pooling.

    This block applies a series of residual convolutional blocks to increase the feature 
    representation capacity, followed by an optional MaxPool2d layer for temporal/frequency downsampling.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        n_blocks, 
        kernel_size
    ):
        """
        Initializes the ResEncoderBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of target output channels.
            n_blocks (int): Total number of inner residual convolutional blocks.
            kernel_size (tuple or None): Pooling kernel shape (e.g., (1, 2)), or None to skip downsampling.
        """

        super(ResEncoderBlock, self).__init__()
        # Initialize the first residual layer handling channel size adjustments
        self.conv = nn.ModuleList([
            ResConvBlock(
                in_channels, 
                out_channels
            )
        ])

        # Append remaining residual layers keeping the output channels constant
        for _ in range(n_blocks - 1):
            self.conv.append(
                ResConvBlock(
                    out_channels, 
                    out_channels
                )
            )

        # Optional pooling layer for downsampling across frequency/temporal grids
        self.pool = nn.MaxPool2d(kernel_size) if kernel_size is not None else None

    def forward(self, x):
        """
        Executes the forward pass of the block.

        Args:
            x (torch.Tensor): Input feature map tensor.

        Returns:
            torch.Tensor or tuple: 
                - If pooling is active, returns a tuple (unpooled_features, pooled_features) 
                  where unpooled_features is cached for UNet skip-connections.
                - If pooling is None, returns the unpooled feature map tensor directly.
        """

        # Chain consecutive residual blocks sequentially to extract high-level feature maps
        for each_layer in self.conv:
            x = each_layer(x)

        # If pooling is defined, return both the pre-pooled representation (for skip-connection) and the downsampled version
        if self.pool is not None: return x, self.pool(x)
        return x

class Encoder(nn.Module):
    """
    Core UNet-style Encoder backbone of the DJCM model.

    Compiles a multi-stage downsampling ladder that progressively compresses 
    spatial resolution (specifically frequency bins) while scaling up channel capacities.
    """

    def __init__(
        self, 
        in_channels, 
        n_blocks
    ):
        """
        Initializes the Encoder module stack.

        Args:
            in_channels (int): Number of channels in the incoming input audio spectrogram.
            n_blocks (int): Number of internal residual blocks per encoder stage.
        """

        super(Encoder, self).__init__()
        # Stacked configuration expanding channels from base input to deep 384 feature dimensions
        # Stride/Kernel configuration (1, 2) selectively halves the second spatial dimension (frequency)
        self.en_blocks = nn.ModuleList([
            ResEncoderBlock(
                in_channels, 
                32, 
                n_blocks, 
                (1, 2)
            ), 
            ResEncoderBlock(
                32, 
                64, 
                n_blocks, 
                (1, 2)
            ), 
            ResEncoderBlock(
                64, 
                128, 
                n_blocks, 
                (1, 2)
            ), 
            ResEncoderBlock(
                128, 
                256, 
                n_blocks, 
                (1, 2)
            ), 
            ResEncoderBlock(
                256, 
                384, 
                n_blocks, 
                (1, 2)
            ), 
            ResEncoderBlock(
                384, 
                384, 
                n_blocks, 
                (1, 2)
            )
        ])

    def forward(self, x):
        """
        Processes input feature representations through the multi-stage downsampling architecture.

        Args:
            x (torch.Tensor): Input spectrogram feature tensor.

        Returns:
            tuple: (bottleneck_tensor, concat_tensors)
                - bottleneck_tensor (torch.Tensor): Deep latent representation at the narrowest bottleneck.
                - concat_tensors (list of torch.Tensor): Collected high-resolution feature tensors cached at each block stage for decoder skip-connection mirroring.
        """

        concat_tensors = []
        # Progressively extract features, downsample, and store pre-pooled maps for UNet skip-connections
        for layer in self.en_blocks:
            # '_' captures the pre-pooled, high-resolution features; 'x' receives the downsampled result
            _, x = layer(x)
            concat_tensors.append(_)

        return x, concat_tensors