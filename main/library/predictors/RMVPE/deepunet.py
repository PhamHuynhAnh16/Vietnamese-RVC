import os
import sys

import torch

import torch.nn as nn

sys.path.append(os.getcwd())

from main.library.predictors.RMVPE.yolo import YOLO13Encoder, YOLO13FullPADDecoder, HyperACE

class ConvBlockRes(nn.Module):
    """
    Standard 2D Convolutional Block with Residual/Shortcut Connection.

    Applies two sequential blocks of Conv2d-BatchNorm2d-ReLU. If the input channels
    do not match the output channels, an additional 1x1 2D Convolution is initialized
    as a shortcut projector to enable residual identity addition.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        momentum=0.01
    ):
        """
        Initializes the ConvBlockRes layer module.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            momentum (float, default=0.01): BatchNorm2d momentum configuration.
        """

        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=(3, 3), 
                stride=(1, 1), 
                padding=(1, 1), 
                bias=False
            ), 
            nn.BatchNorm2d(
                out_channels, 
                momentum=momentum
            ), 
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=(3, 3), 
                stride=(1, 1), 
                padding=(1, 1), 
                bias=False
            ), 
            nn.BatchNorm2d(
                out_channels, 
                momentum=momentum
            ), 
            nn.ReLU()
        )

        # Dynamic shortcut configuration block
        self.is_shortcut = False
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True

        # Optimize execution path branch by assigning the forward method dynamically
        self.forward = self.forward_shortcut if self.is_shortcut else self.forward_non_shortcut
    
    def forward_shortcut(self, x):
        """Executes convolution with a 1x1 projected skip connection."""

        return self.conv(x) + self.shortcut(x)
    
    def forward_non_shortcut(self, x):
        """Executes convolution with a direct identity skip connection."""

        return self.conv(x) + x

class ResEncoderBlock(nn.Module):
    """Residual Encoder Block with an optional Average Pooling downsampler."""

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        n_blocks=1, 
        momentum=0.01
    ):
        """
        Initializes the ResEncoderBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple or None): Pooling window size. If None, downsampling is skipped.
            n_blocks (int, default=1): Number of stacked ConvBlockRes layers.
            momentum (float, default=0.01): BatchNorm2d momentum.
        """

        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        # First block processes the input channel shift channel space transition
        self.conv.append(
            ConvBlockRes(
                in_channels, 
                out_channels, 
                momentum
            )
        )

        # Subsequent internal blocks retain identical output channel sizes
        for _ in range(n_blocks - 1):
            self.conv.append(
                ConvBlockRes(
                    out_channels, 
                    out_channels, 
                    momentum
                )
            )

        self.kernel_size = kernel_size
        if self.kernel_size is not None: self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        """Processes sequential convolutions and returns skip-connections and pooled representations."""

        for i in range(self.n_blocks):
            x = self.conv[i](x)

        if self.kernel_size is not None: return x, self.pool(x) # Returns features for skip connections along with downsampled maps
        else: return x

class Encoder(nn.Module):
    """The complete multi-stage Downsampling Encoder pipeline for DeepUnet."""

    def __init__(
        self, 
        in_channels, 
        in_size, 
        n_encoders, 
        kernel_size, 
        n_blocks, 
        out_channels=16, 
        momentum=0.01
    ):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()

        for _ in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    n_blocks, 
                    momentum=momentum
                )
            )
            # Channel configurations double while resolution scales halve at each stage sequence boundary
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
            
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        """Performs multi-stage encoding and records spatial features for U-Net skip-connections."""

        concat_tensors = []
        x = self.bn(x)

        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t) # Stashes encoder feature maps for Decoder concatenation bounds

        return x, concat_tensors

class Intermediate(nn.Module):
    """The bottleneck layer structure connecting Encoder and Decoder chains."""

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        n_inters, 
        n_blocks, 
        momentum=0.01
    ):
        super(Intermediate, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(
                in_channels, 
                out_channels, 
                None, 
                n_blocks, 
                momentum
            )
        )

        for _ in range(n_inters - 1):
            self.layers.append(
                ResEncoderBlock(
                    out_channels, 
                    out_channels, 
                    None, 
                    n_blocks, 
                    momentum
                )
            )

    def forward(self, x):
        """Passes features through the sequential non-downsampling bottleneck blocks."""

        for layer in self.layers:
            x = layer(x)

        return x

class ResDecoderBlock(nn.Module):
    """Residual Decoder Block executing Transposed Convolutions and feature map concatenation."""

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        stride, 
        n_blocks=1, 
        momentum=0.01
    ):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        # Up-sampling layer path via Transposed Convolutional operations
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=(3, 3), 
                stride=stride, 
                padding=(1, 1), 
                output_padding=out_padding, 
                bias=False
            ), 
            nn.BatchNorm2d(
                out_channels, 
                momentum=momentum
            ), 
            nn.ReLU()
        )

        self.conv2 = nn.ModuleList()
        # Takes concatenated channels (doubled channel space size) from up-sampled and encoder features
        self.conv2.append(
            ConvBlockRes(
                out_channels * 2, 
                out_channels, 
                momentum
            )
        )

        for _ in range(n_blocks - 1):
            self.conv2.append(
                ConvBlockRes(
                    out_channels, 
                    out_channels, 
                    momentum
                )
            )

    def forward(self, x, concat_tensor):
        """Upsamples and concatenates corresponding encoder feature maps before final convolutions."""

        x = torch.cat((self.conv1(x), concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)

        return x

class Decoder(nn.Module):
    """The multi-stage Upsampling Decoder pipeline matching corresponding Encoder blocks."""

    def __init__(
        self, 
        in_channels, 
        n_decoders, 
        stride, 
        n_blocks, 
        momentum=0.01
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()

        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(
                    in_channels, 
                    out_channels, 
                    stride, 
                    n_blocks, 
                    momentum
                )
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        """Iterates through decoding loops processing stored encoder tensors in reversed order."""

        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])

        return x

class DeepUnet(nn.Module):
    """Standard Baseline Deep U-Net encoder-decoder network used in standard RMVPE."""

    def __init__(
        self, 
        kernel_size, 
        n_blocks, 
        en_de_layers=5, 
        inter_layers=4, 
        in_channels=1, 
        en_out_channels=16
    ):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(
            in_channels, 
            128, 
            en_de_layers, 
            kernel_size, 
            n_blocks, 
            en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, 
            self.encoder.out_channel, 
            inter_layers, 
            n_blocks
        )
        self.decoder = Decoder(
            self.encoder.out_channel, 
            en_de_layers, 
            kernel_size, 
            n_blocks
        )

    def forward(self, x):
        """Main core DeepUnet forward flow execution pipeline."""

        x, concat_tensors = self.encoder(x)

        return self.decoder(
            self.intermediate(x), 
            concat_tensors
        )
    
class HPADeepUnet(nn.Module):
    """
    YOLOv13-enhanced U-Net architecture integrating Hypergraph and FullPAD paradigms.

    This advanced variant replaces standard convolution layers with a YOLOv13 Backbone 
    Encoder, processes high-order correlations via a Hypergraph-based Adaptive Correlation 
    Enhancement (HyperACE) module, and fuses feature hierarchies using a Full-Pipeline 
    Aggregation-and-Distribution (FullPAD) Decoder scheme.
    """

    def __init__(
        self, 
        in_channels=1, 
        en_out_channels=16, 
        base_channels=64, 
        hyperace_k=2, 
        hyperace_l=1, 
        num_hyperedges=16, 
        num_heads=8
    ):
        """
        Initializes the YOLOv13-based HPADeepUnet block.

        Args:
            in_channels (int, default=1): Dimensionality layout of raw spectrogram tracks.
            en_out_channels (int, default=16): Final target channel layout leaving the decoder module.
            base_channels (int, default=64): Multiplier factor channel layer settings for YOLO13 stages.
            hyperace_k (int, default=2): Parameter scale governing HyperACE cross-location clustering boundaries.
            hyperace_l (int, default=1): Level bounds configuration parameters for hypergraph convolution steps.
            num_hyperedges (int, default=16): Total continuous differentiable hyperedges allocation count.
            num_heads (int, default=8): Head splits for the multi-head hypergraph query sub-spaces.
        """

        super().__init__()
        # Step 1: Initialize YOLOv13 Multi-Stage Feature Extraction Backbone Encoder
        self.encoder = YOLO13Encoder(
            in_channels, 
            base_channels
        )

        enc_ch = self.encoder.out_channels
        # Step 2: Initialize Hypergraph-based Adaptive Correlation Enhancement (HyperACE) Module
        self.hyperace = HyperACE(
            in_channels=enc_ch,
            out_channels=enc_ch[-1],
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            k=hyperace_k, 
            l=hyperace_l
        )

        # Step 3: Initialize Full-Pipeline Aggregation-and-Distribution (FullPAD) Decoder
        self.decoder = YOLO13FullPADDecoder(
            encoder_channels=enc_ch,
            hyperace_out_c=enc_ch[-1],
            out_channels_final=en_out_channels
        )

    def forward(self, x):
        """Executes the complete YOLOv13 Hypergraph + FullPAD feature mapping route."""

        # Extract hierarchical features across multiple multi-scale visual layers
        features = self.encoder(x)

        # Dynamically scale spatial dimensional sizes back to perfectly match raw audio shape domains
        return nn.functional.interpolate(
            # Pass multi-level features into HyperACE to construct global high-order correlations,
            # then map and project everything down inside the specialized FullPAD Decoder pipeline.
            self.decoder(
                features, 
                self.hyperace(features)
            ), 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )