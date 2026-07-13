import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.predictors.DJCM.encoder import ResEncoderBlock
from main.library.predictors.DJCM.utils import ResConvBlock, BiGRU, init_bn, init_layer

class ResDecoderBlock(nn.Module):
    """
    Residual Decoder Block using Transposed Convolution for upsampling.

    This block upsamples the feature maps, processes skip-connections (concatenation), 
    and applies a series of residual convolutional blocks to refine the features.
    """

    def __init__(self, in_channels, out_channels, n_blocks, stride):
        """
        Initializes the ResDecoderBlock.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels after upsampling and processing.
            n_blocks (int): Total number of residual convolutional blocks.
            stride (tuple): Stride for the ConvTranspose2d layer (controls upsampling factors).
        """

        super(ResDecoderBlock, self).__init__()
        # Transposed convolution to handle spatial upsampling (usually doubling the frequency dimension)
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, stride, stride, (0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01)
        # The first convolution handles concatenated features, hence input channels = out_channels * 2
        self.conv = nn.ModuleList([ResConvBlock(out_channels * 2, out_channels)])

        # Append remaining residual blocks keeping the channels constant
        for _ in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels))

        self.init_weights()

    def init_weights(self):
        """Initializes weights for Batch Normalization and Transposed Convolution layers."""

        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(self, x, concat):
        """
        Executes the forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder layer.
            concat (torch.Tensor): Skip-connection encoder tensor to concatenate.

        Returns:
            torch.Tensor: Upsampled and refined feature map tensor.
        """

        # Pre-activation standard pipeline: BN -> Inplace ReLU -> Transposed Conv Upsample
        x = self.conv1(F.relu_(self.bn1(x)))
        # Merge high-level abstract features with low-level spatial features from the encoder
        x = torch.cat((x, concat), dim=1)
    
        # Pass the concatenated tensor through the residual convolution sequence
        for each_layer in self.conv:
            x = each_layer(x)
    
        return x

class Decoder(nn.Module):
    """
    Core UNet-style Decoder trunk mapping latent features back to spatial resolutions.

    Chains multiple ResDecoderBlock layers together, popping and merging historical
    encoder feature mirrors sequentially via reverse indexing.
    """

    def __init__(self, n_blocks):
        """
        Initializes the main structural stack of Decoder components.

        Args:
            n_blocks (int): Number of inner residual blocks per decoder stage.
        """

        super(Decoder, self).__init__()
        # Stacked configuration gradually expanding spatial dims while narrowing down feature channels
        self.de_blocks = nn.ModuleList([
            ResDecoderBlock(384, 384, n_blocks, (1, 2)), 
            ResDecoderBlock(384, 384, n_blocks, (1, 2)), 
            ResDecoderBlock(384, 256, n_blocks, (1, 2)), 
            ResDecoderBlock(256, 128, n_blocks, (1, 2)), 
            ResDecoderBlock(128, 64, n_blocks, (1, 2)), 
            ResDecoderBlock(64, 32, n_blocks, (1, 2))
        ])

    def forward(self, x, concat_tensors):
        """
        Sequentially upsamples the representation while feeding encoder mirrors.

        Args:
            x (torch.Tensor): Bottleneck latent tensor representation.
            concat_tensors (list of torch.Tensor): List of accumulated encoder stage outputs.

        Returns:
            torch.Tensor: High-resolution reconstructed output block tensor.
        """

        # Iterate and match features back to corresponding encoder layers in reverse order
        for i, layer in enumerate(self.de_blocks):
            x = layer(x, concat_tensors[-1 - i])

        return x

class PE_Decoder(nn.Module):
    """
    Pitch Estimation (PE) Decoder module.

    Processes the reconstructed decoder maps, applies standard convolutions, and maps 
    the spatial representation into frame-level pitch bin activations using a BiGRU and a Linear classifier.
    """

    def __init__(self, n_blocks, seq_layers=1, window_length = 1024, n_class = 360):
        """
        Initializes the Pitch Estimation Decoder module.

        Args:
            n_blocks (int): Number of residual blocks per decoder stage.
            seq_layers (int): Number of recurrent layers within the BiGRU block. Default is 1.
            window_length (int): Analysis FFT window size constraint. Default is 1024.
            n_class (int): Total number of pitch output target classes. Default is 360.
        """

        super(PE_Decoder, self).__init__()
        self.de_blocks = Decoder(n_blocks)
        self.after_conv1 = ResEncoderBlock(32, 32, n_blocks, None) # Refinement block without pooling
        self.after_conv2 = nn.Conv2d(32, 1, (1, 1)) # Compress feature channels down to 1

        # Temporal smoothing and classification network block sequence
        self.fc = nn.Sequential(
            BiGRU(
                (1, window_length // 2), 
                1, 
                seq_layers
            ), 
            nn.Linear(
                window_length // 2, 
                n_class
            ), 
            nn.Sigmoid() # Normalize outputs to class probabilities/salience maps
        )
        init_layer(self.after_conv2)

    def forward(self, x, concat_tensors):
        """Predicts vocal pitch probabilities from abstract feature maps.

        Args:
            x (torch.Tensor): Bottleneck feature tensor.
            concat_tensors (list): Encoder features for skip connections.

        Returns:
            torch.Tensor: Decoded pitch bin activations squeezed.
        """
    
        # Reconstruct, refine channels, flatten/evaluate temporal GRU sequence, and remove redundant axis
        return self.fc(self.after_conv2(self.after_conv1(self.de_blocks(x, concat_tensors)))).squeeze(1)
    
class SVS_Decoder(nn.Module):
    """
    Singing Voice Separation (SVS) Decoder module.

    Maps bottleneck tensors back into target audio separation masks or complex spectrogram dimensions
    corresponding to separated signal tracks.
    """

    def __init__(self, in_channels, n_blocks):
        """
        Initializes the Singing Voice Separation Decoder module.

        Args:
            in_channels (int): Base audio input channels.
            n_blocks (int): Number of internal block structures.
        """

        super(SVS_Decoder, self).__init__()
        self.de_blocks = Decoder(n_blocks)
        self.after_conv1 = ResEncoderBlock(32, 32, n_blocks, None)
        # Final projecting layer to compute multichannel audio separation targets
        self.after_conv2 = nn.Conv2d(32, in_channels * 4, (1, 1))
        self.init_weights()

    def init_weights(self):
        """Initializes the projection block weights."""

        init_layer(self.after_conv2)

    def forward(self, x, concat_tensors):
        """
        Extracts source-separated mask fields from underlying representations.

        Args:
            x (torch.Tensor): Bottleneck feature representation input.
            concat_tensors (list): Encoder historical cross-connections mapping list.

        Returns:
            torch.Tensor: Decoded separation mask tensor matching input shapes.
        """

        # Execute UNet trunk reconstruction, refine, and project channels back to audio targets
        return self.after_conv2(self.after_conv1(self.de_blocks(x, concat_tensors)))