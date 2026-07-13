import os
import sys

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.predictors.FCPE.attentions import SelfAttention
from main.library.predictors.FCPE.utils import calc_same_padding, Transpose, GLU, Swish

class ConformerConvModule_LEGACY(nn.Module):
    """
    Legacy Convolution Module for the Conformer architecture block.

    Processes sequential features using LayerNorm, point-wise 1D convolutions,
    Gated Linear Units (GLU), and explicit manual depth-wise padding/convolutions.
    """

    def __init__(
        self, 
        dim, 
        causal=False, 
        expansion_factor=2, 
        kernel_size=31, 
        dropout=0.0
    ):
        """
        Initializes the legacy Conformer convolutional structure block.

        Args:
            dim (int): Input channel or feature size dimension.
            causal (bool, optional): If True, applies causal asymmetric padding. Defaults to False.
            expansion_factor (int, optional): Bottleneck channel expansion scale. Defaults to 2.
            kernel_size (int, optional): Width of 1D conv filters. Defaults to 31.
            dropout (float, optional): Conv block dropout probability. Defaults to 0.0.
        """

        super().__init__()
        inner_dim = dim * expansion_factor
        # Sequentially map, gate, filter, and project feature frames
        self.net = nn.Sequential(
            nn.LayerNorm(dim), 
            Transpose((1, 2)), 
            nn.Conv1d(dim, inner_dim * 2, 1), # Point-wise expansion
            GLU(dim=1), # Gated Linear Unit split along channel dimension
            DepthWiseConv1d_LEGACY(
                inner_dim, 
                inner_dim, 
                kernel_size=kernel_size, 
                padding=(
                    calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
                )
            ), 
            Swish(), 
            nn.Conv1d(inner_dim, dim, 1), # Point-wise projection back to model dim
            Transpose((1, 2)), 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass through the legacy Conformer convolution module."""

        return self.net(x)

class ConformerConvModule(nn.Module):
    """
    Modernized Convolution Module for the Conformer block architecture.

    Utilizes standardized PyTorch components (`nn.GLU`, `nn.SiLU`) and native 1D
    same-padding depth-wise operations to optimize runtime efficiency.
    """

    def __init__(
        self, 
        dim, 
        expansion_factor=2, 
        kernel_size=31, 
        dropout=0
    ):
        """Initializes the modernized Conformer convolutional block."""

        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(dim), 
            Transpose((1, 2)), 
            nn.Conv1d(dim, inner_dim * 2, 1), 
            nn.GLU(dim=1), # Standardized PyTorch GLU activation
            DepthWiseConv1d(
                inner_dim, 
                inner_dim, 
                kernel_size=kernel_size, 
                padding=calc_same_padding(kernel_size)[0], 
                groups=inner_dim # Forces exact depth-wise channel segregation
            ), 
            nn.SiLU(), # Equivalent to Swish activation block
            nn.Conv1d(inner_dim, dim, 1), 
            Transpose((1, 2)), 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass through the standard Conformer convolution module."""

        return self.net(x)

class DepthWiseConv1d_LEGACY(nn.Module):
    """Legacy Depth-wise 1D Convolution wrapper with explicit functional padding."""

    def __init__(
        self, 
        chan_in, 
        chan_out, 
        kernel_size, 
        padding
    ):
        super().__init__()
        self.padding = padding
        # groups=chan_in applies convolution independently per channel layer
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        """Pads and executes depthwise convolutions across sequential inputs."""

        return self.conv(F.pad(x, self.padding))

class DepthWiseConv1d(nn.Module):
    """Modernized Depth-wise 1D Convolution leveraging internal native padding parameters."""

    def __init__(
        self, 
        chan_in, 
        chan_out, 
        kernel_size, 
        padding, 
        groups
    ):
        super().__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size=kernel_size, padding=padding, groups=groups)

    def forward(self, x):
        """Executes native convolution pass."""

        return self.conv(x)

class EncoderLayer(nn.Module):
    """
    Legacy Conformer Encoder block layer wrapper.

    Applies alternating multi-head self-attention mechanisms alongside structural
    legacy convolution blocks bounded by residual addition loops.
    """

    def __init__(
        self, 
        parent
    ):
        """
        Initializes encoder layer variables using parent properties.

        Args:
            parent (Any): Parent context module containing structural hyper-parameters.
        """

        super().__init__()
        self.conformer = ConformerConvModule_LEGACY(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        self.attn = SelfAttention(dim=parent.dim_model, heads=parent.num_heads)

    def forward(self, phone, mask=None):
        """Applies sequence blocks sequentially wrapped with residual logic lanes.

        Args:
            phone (torch.Tensor): Extracted voice/phone target vectors.
            mask (torch.Tensor, optional): Optional frame sequence masks. Defaults to None.
        """

        # Multi-Head Attention step with residual summation
        phone = phone + (self.attn(self.norm(phone), mask=mask))
        # Convolution step with residual summation
        return phone + (self.conformer(phone))

class ConformerNaiveEncoder(nn.Module):
    """
    A Naive stack of Conformer Encoder Layers (CFNEncoderLayer).

    Acts as the primary text or feature contextual transformer within the system architecture.
    """

    def __init__(
        self, 
        num_layers, 
        num_heads, 
        dim_model, 
        use_norm = False, 
        conv_only = False, 
        conv_dropout = 0, 
        atten_dropout = 0
    ):
        """Initializes the Conformer Naive Encoder Stack."""

        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.use_norm = use_norm
        self.residual_dropout = 0.1  
        self.attention_dropout = 0.1  
        # Instantiate uniform structural transformer layers
        self.encoder_layers = nn.ModuleList([
            CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only, conv_dropout, atten_dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """Sequentially updates latents across all stack layer blocks."""

        for layer in self.encoder_layers:
            x = layer(x, mask)

        return x 
    
class CFNEncoderLayer(nn.Module):
    """
    Modern Naive Conformer Encoder block layer.

    Supports optional self-attention tracking mechanisms (toggled by `conv_only`) 
    and conditional dropout wrapper assignments on top of modern convolution pipelines.
    """

    def __init__(
        self, 
        dim_model, 
        num_heads = 8, 
        use_norm = False, 
        conv_only = False, 
        conv_dropout = 0, 
        atten_dropout = 0
    ):
        super().__init__()
        # Conditionally wrap ConformerConvModule with Dropout if specified
        self.conformer = (
            nn.Sequential(
                ConformerConvModule(dim_model), 
                nn.Dropout(conv_dropout)
            )
        ) if conv_dropout > 0 else (
            ConformerConvModule(dim_model)
        )

        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)  
        # Explicitly skip Self-Attention instantiation if conv_only configuration is True
        self.attn = SelfAttention(
            dim=dim_model, 
            heads=num_heads, 
            use_norm=use_norm, 
            dropout=atten_dropout
        ) if not conv_only else None

    def forward(self, x, mask=None):
        """Forward pass applying structural attention logic and spatial convolutions."""

        # Optional self-attention layer pass with residual summation
        if self.attn is not None: x = x + (self.attn(self.norm(x), mask=mask))
        # Standard convolution block pass with residual summation
        return x + (self.conformer(x)) 