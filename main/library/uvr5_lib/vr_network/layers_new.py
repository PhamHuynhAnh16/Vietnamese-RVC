import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.uvr5_lib import spec_utils

class Conv2DBNActiv(nn.Module):
    """
    A sequential block consisting of a 2D Convolution, 2D Batch Normalization, 
    and an Activation function.
    """

    def __init__(
        self, 
        nin, 
        nout, 
        ksize=3, 
        stride=1, 
        pad=1, 
        dilation=1, 
        activ=nn.ReLU
    ):
        """
        Args:
            nin (int): Number of input channels.
            nout (int): Number of output channels.
            ksize (int, optional): Kernel size. Defaults to 3.
            stride (int, optional): Stride size. Defaults to 1.
            pad (int, optional): Padding size. Defaults to 1.
            dilation (int, optional): Dilation rate. Defaults to 1.
            activ (nn.Module, optional): Activation function class. Defaults to nn.ReLU.
        """

        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            # Bias is set to False because BatchNorm follows immediately
            nn.Conv2d(
                nin, 
                nout, 
                kernel_size=ksize, 
                stride=stride, 
                padding=pad, 
                dilation=dilation, 
                bias=False
            ), 
            nn.BatchNorm2d(nout), 
            activ()
        )

    def __call__(self, input_tensor):
        """
        Forward pass.
        
        Args:
            input_tensor (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Normalized and activated feature maps.
        """

        return self.conv(input_tensor)

class Encoder(nn.Module):
    """
    Encoder block composed of two successive Conv2DBNActiv layers.
    Typically used to downsample and extract features in a U-Net architecture.
    """

    def __init__(
        self, 
        nin, 
        nout, 
        ksize=3, 
        stride=1, 
        pad=1, 
        activ=nn.LeakyReLU
    ):
        """
        Args:
            nin (int): Number of input channels.
            nout (int): Number of output channels.
            ksize (int, optional): Kernel size. Defaults to 3.
            stride (int, optional): Stride size. Defaults to 1.
            pad (int, optional): Padding size. Defaults to 1.
            activ (nn.Module, optional): Activation function class. Defaults to nn.LeakyReLU.
        """

        super(Encoder, self).__init__()
        # First layer handles downsampling/channel changes via defined stride
        self.conv1 = Conv2DBNActiv(
            nin, 
            nout, 
            ksize, 
            stride, 
            pad, 
            activ=activ
        )
        # Second layer refines the features with a fixed stride of 1
        self.conv2 = Conv2DBNActiv(
            nout, 
            nout, 
            ksize, 
            1, 
            pad, 
            activ=activ
        )

    def __call__(self, input_tensor):
        """
        Forward pass.
        
        Args:
            input_tensor (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Encoded feature maps.
        """

        hidden = self.conv1(input_tensor)
        hidden = self.conv2(hidden)

        return hidden

class Decoder(nn.Module):
    """
    Decoder block that upsamples the input tensor, aligns/concatenates 
    it with skip connections, and processes it through a Conv2DBNActiv layer.
    """

    def __init__(
        self, 
        nin, 
        nout, 
        ksize=3, 
        stride=1, 
        pad=1, 
        activ=nn.ReLU, 
        dropout=False
    ):
        """
        Args:
            nin (int): Number of input channels (including skip connections if any).
            nout (int): Number of output channels.
            ksize (int, optional): Kernel size. Defaults to 3.
            stride (int, optional): Stride size. Defaults to 1.
            pad (int, optional): Padding size. Defaults to 1.
            activ (nn.Module, optional): Activation function class. Defaults to nn.ReLU.
            dropout (bool, optional): Whether to apply 2D Dropout. Defaults to False.
        """

        super(Decoder, self).__init__()
        self.dropout = nn.Dropout2d(0.1) if dropout else None
        self.conv1 = Conv2DBNActiv(
            nin, 
            nout, 
            ksize, 
            1, 
            pad, 
            activ=activ
        )

    def __call__(self, input_tensor, skip=None):
        """
        Forward pass.
        
        Args:
            input_tensor (torch.Tensor): Low-resolution bottleneck features.
            skip (torch.Tensor, optional): High-resolution skip connection tensor from the encoder.
            
        Returns:
            torch.Tensor: Decoded upsampled feature maps.
        """

        # Upsample the lower resolution input tensor spatially by a factor of 2
        input_tensor = F.interpolate(
            input_tensor, 
            scale_factor=2, 
            mode="bilinear", 
            align_corners=True
        )

        # Handle encoder-decoder skip connections
        if skip is not None:
            # Crop the skip connection tensor to match the spatial size of the upsampled input tensor
            skip = spec_utils.crop_center(skip, input_tensor)
            # Concatenate along the channel dimension (dim=1)
            input_tensor = torch.cat([input_tensor, skip], dim=1)

        hidden = self.conv1(input_tensor)

        if self.dropout is not None:
            hidden = self.dropout(hidden)

        return hidden

class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module.
    Captures multi-scale contextual information using parallel dilated convolutions.
    """

    def __init__(
        self, 
        nin, 
        nout, 
        dilations=(4, 8, 12), 
        activ=nn.ReLU, 
        dropout=False
    ):
        """
        Args:
            nin (int): Number of input channels.
            nout (int): Number of output channels for each parallel branch.
            dilations (tuple, optional): Dilation rates for the three dilated branches. Defaults to (4, 8, 12).
            activ (nn.Module, optional): Activation function class. Defaults to nn.ReLU.
            dropout (bool, optional): Whether to apply 2D Dropout. Defaults to False.
        """

        super(ASPPModule, self).__init__()
        # Branch 1: Global Context (Image-level pooling followed by 1x1 Conv)
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)), 
            Conv2DBNActiv(
                nin, 
                nout, 
                1, 
                1, 
                0, 
                activ=activ
            )
        )

        # Branch 2: Standard 1x1 Convolution
        self.conv2 = Conv2DBNActiv(
            nin, 
            nout, 
            1, 
            1, 
            0, 
            activ=activ
        )

        # Branch 3: 3x3 Convolution with first dilation rate
        self.conv3 = Conv2DBNActiv(
            nin, 
            nout, 
            3, 
            1, 
            dilations[0], 
            dilations[0], 
            activ=activ
        )

        # Branch 4: 3x3 Convolution with second dilation rate
        self.conv4 = Conv2DBNActiv(
            nin, 
            nout, 
            3, 
            1, 
            dilations[1], 
            dilations[1], 
            activ=activ
        )

        # Branch 5: 3x3 Convolution with third dilation rate
        self.conv5 = Conv2DBNActiv(
            nin, 
            nout, 
            3, 
            1, 
            dilations[2], 
            dilations[2], 
            activ=activ
        )

        # Bottleneck layer to fuse features from all 5 parallel branches (nout * 5 channels)
        self.bottleneck = Conv2DBNActiv(
            nout * 5, 
            nout, 
            1, 
            1, 
            0, 
            activ=activ
        )

        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, input_tensor):
        """
        Forward pass.
        
        Args:
            input_tensor (torch.Tensor): Input feature maps.
            
        Returns:
            torch.Tensor: Multi-scale aggregated feature maps.
        """

        _, _, h, w = input_tensor.size()
        # Concatenate features from all 5 scales along the channel dimension
        out = self.bottleneck(
            torch.cat((
                # Upsample the global pooling branch back to original spatial dimensions
                F.interpolate(
                    self.conv1(input_tensor), 
                    size=(h, w), 
                    mode="bilinear", 
                    align_corners=True
                ), 
                self.conv2(input_tensor), 
                self.conv3(input_tensor), 
                self.conv4(input_tensor), 
                self.conv5(input_tensor)
            ), dim=1))

        if self.dropout is not None:
            out = self.dropout(out)

        return out

class LSTMModule(nn.Module):
    """
    A hybrid Recurrent-Convolutional module.
    Squeezes multi-channel feature maps into a 1D sequence per frame, 
    processes it using a Bi-directional LSTM, and projects it back with a Dense layer.
    """

    def __init__(
        self, 
        nin_conv, 
        nin_lstm, 
        nout_lstm
    ):
        """
        Args:
            nin_conv (int): Number of input channels for the initial 2D convolution.
            nin_lstm (int): Input size (feature dimension) for the LSTM layer.
            nout_lstm (int): Total output size for the LSTM layer (split equally if bidirectional).
        """

        super(LSTMModule, self).__init__()
        # Compresses the input feature channels down to 1 channel
        self.conv = Conv2DBNActiv(
            nin_conv, 
            1, 
            1, 
            1, 
            0
        )
        # Sequence modeling layer over time/frequency frames
        self.lstm = nn.LSTM(
            input_size=nin_lstm, 
            hidden_size=nout_lstm // 2, 
            bidirectional=True
        )
        # Dense linear projections with Batch Normalization and ReLU activation
        self.dense = nn.Sequential(
            nn.Linear(nout_lstm, nin_lstm), 
            nn.BatchNorm1d(nin_lstm), 
            nn.ReLU()
        )

    def forward(self, input_tensor):
        """
        Forward pass.
        
        Args:
            input_tensor (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Sequence-processed tensor restored back to 4D shape.
        """

        N, _, nbins, nframes = input_tensor.size()

        hidden, _ = self.lstm(self.conv(input_tensor)[:, 0].permute(2, 0, 1))
        hidden = self.dense(hidden.reshape(-1, hidden.size()[-1])).reshape(nframes, N, 1, nbins)

        return hidden.permute(1, 2, 3, 0)