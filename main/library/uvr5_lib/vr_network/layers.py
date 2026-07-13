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

class SeperableConv2DBNActiv(nn.Module):
    """
    A Depthwise Separable 2D Convolutional block followed by Batch Normalization 
    and an Activation function to reduce computational complexity.
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
            activ (Type[nn.Module], optional): Activation function class. Defaults to nn.ReLU.
        """

        super(SeperableConv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            # Depthwise Convolution: filters are applied to each channel independently (groups=nin)
            nn.Conv2d(
                nin,
                nin,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=nin,
                bias=False,
            ),
            # Pointwise Convolution: combines channels linearly using a 1x1 kernel
            nn.Conv2d(
                nin,
                nout,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, input_tensor):
        """
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """

        return self.conv(input_tensor)

class Encoder(nn.Module):
    """
    Encoder block comprising two successive Conv2DBNActiv layers.
    It extracts features and optionally applies downsampling via stride.
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
            stride (int, optional): Downsampling stride size. Defaults to 1.
            pad (int, optional): Padding size. Defaults to 1.
            activ (Type[nn.Module], optional): Activation function class. Defaults to nn.LeakyReLU.
        """

        super(Encoder, self).__init__()
        # First layer preserves the original spatial scale (stride=1) and changes channel capacity
        self.conv1 = Conv2DBNActiv(
            nin, 
            nout, 
            ksize, 
            1, 
            pad, 
            activ=activ
        )
        # Second layer handles the structural downsampling via the provided stride parameter
        self.conv2 = Conv2DBNActiv(
            nout, 
            nout, 
            ksize, 
            stride, 
            pad, 
            activ=activ
        )

    def __call__(self, input_tensor):
        """
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Input feature maps.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - hidden: Downsampled features for the deeper layers.
                - skip: Higher-resolution features saved for skip connections in the decoder.
        """

        skip = self.conv1(input_tensor)
        hidden = self.conv2(skip)

        return hidden, skip

class Decoder(nn.Module):
    """
    Decoder block that upsamples the incoming features, optionally 
    concatenates skip connections from the encoder, and applies convolutions.
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
            nin (int): Number of combined input channels (including skip connections).
            nout (int): Number of output channels.
            ksize (int, optional): Kernel size. Defaults to 3.
            stride (int, optional): Stride size. Defaults to 1.
            pad (int, optional): Padding size. Defaults to 1.
            activ (Type[nn.Module], optional): Activation function class. Defaults to nn.ReLU.
            dropout (bool, optional): Whether to use 2D spatial dropout. Defaults to False.
        """

        super(Decoder, self).__init__()
        self.dropout = nn.Dropout2d(0.1) if dropout else None
        self.conv = Conv2DBNActiv(
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
            input_tensor (torch.Tensor): Lower resolution deep feature map.
            skip (torch.Tensor, optional): Encoder skip feature map for fusion. Defaults to None.

        Returns:
            torch.Tensor: Upsampled and merged feature map.
        """

        # Spatially upsample the low-resolution feature map by a factor of 2
        input_tensor = F.interpolate(
            input_tensor, 
            scale_factor=2, 
            mode="bilinear", 
            align_corners=True
        )

        # Merge structural information from the encoder if provided
        if skip is not None:
            # Crop encoder features to perfectly match upsampled tensor dimensions
            skip = spec_utils.crop_center(skip, input_tensor)
            # Concatenate features along the channel dimension (dim=1)
            input_tensor = torch.cat([input_tensor, skip], dim=1)

        output_tensor = self.conv(input_tensor)
        if self.dropout is not None:
            output_tensor = self.dropout(output_tensor)

        return output_tensor

class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module with variable architecture support.
    Captures multi-scale multi-context features using parallel dilated convolutions.
    """

    def __init__(self, nn_architecture, nin, nout, dilations=(4, 8, 16), activ=nn.ReLU):
        """
        Args:
            nn_architecture (int): Identifier code representing specific architectural configurations.
            nin (int): Number of input channels.
            nout (int): Number of output channels from the bottleneck layer.
            dilations (Tuple[int, int, int], optional): Dilation rates for parallel branches. Defaults to (4, 8, 16).
            activ (Type[nn.Module], optional): Activation function class. Defaults to nn.ReLU.
        """

        super(ASPPModule, self).__init__()
        self.nn_architecture = nn_architecture
        # Hardcoded ID groupings for dynamic architectural adaptations
        self.six_layer = [129605]
        self.seven_layer = [537238, 537227, 33966]

        # Template for additional high-dilation parallel branches
        extra_conv = SeperableConv2DBNActiv(
            nin, 
            nin, 
            3, 
            1, 
            dilations[2], 
            dilations[2], 
            activ=activ
        )

        # Branch 1: Image-level global pooling context
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)), 
            Conv2DBNActiv(
                nin, 
                nin, 
                1, 
                1, 
                0, 
                activ=activ
            )
        )
        # Branch 2: Standard 1x1 Convolution
        self.conv2 = Conv2DBNActiv(
            nin, 
            nin, 
            1, 
            1, 
            0, 
            activ=activ
        )
        # Branch 3: Dilated Depthwise Separable 3x3 Conv (Dilation Rate 1)
        self.conv3 = SeperableConv2DBNActiv(
            nin, 
            nin, 
            3, 
            1, 
            dilations[0], 
            dilations[0], 
            activ=activ
        )
        # Branch 4: Dilated Depthwise Separable 3x3 Conv (Dilation Rate 2)
        self.conv4 = SeperableConv2DBNActiv(
            nin, 
            nin, 
            3, 
            1, 
            dilations[1], 
            dilations[1], 
            activ=activ
        )
        # Branch 5: Dilated Depthwise Separable 3x3 Conv (Dilation Rate 3)
        self.conv5 = SeperableConv2DBNActiv(
            nin, 
            nin, 
            3, 
            1, 
            dilations[2], 
            dilations[2], 
            activ=activ
        )

        # Dynamically append extra parallel branches based on architecture presets
        if self.nn_architecture in self.six_layer:
            self.conv6 = extra_conv
            nin_x = 6 # 6 features to concatenate
        elif self.nn_architecture in self.seven_layer:
            self.conv6 = extra_conv
            self.conv7 = extra_conv
            nin_x = 7 # 7 features to concatenate
        else:
            nin_x = 5 # Standard 5 baseline branches

        # Fuses concatenated multi-scale channels back to the requested `nout` shape
        self.bottleneck = nn.Sequential(
            Conv2DBNActiv(
                nin * nin_x, 
                nout, 
                1, 
                1, 
                0, 
                activ=activ
            ), 
            nn.Dropout2d(0.1)
        )

    def forward(self, input_tensor):
        """
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Feature map tensor.

        Returns:
            torch.Tensor: Context-aware aggregated representation.
        """

        _, _, h, w = input_tensor.size()
        # Rescale the global average pooled branch back to match spatial dimensions
        feat1 = F.interpolate(
            self.conv1(input_tensor), 
            size=(h, w), 
            mode="bilinear", 
            align_corners=True
        )

        # Execute baseline branches
        feat2 = self.conv2(input_tensor)
        feat3 = self.conv3(input_tensor)
        feat4 = self.conv4(input_tensor)
        feat5 = self.conv5(input_tensor)

        # Execute conditional branches and concatenate along channel dimension (dim=1)
        if self.nn_architecture in self.six_layer:
            feat6 = self.conv6(input_tensor)
            out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6), dim=1)
        elif self.nn_architecture in self.seven_layer:
            feat6 = self.conv6(input_tensor)
            feat7 = self.conv7(input_tensor)

            out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6, feat7), dim=1)
        else:
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)

        # Apply spatial fusion and dropout
        bottleneck_output = self.bottleneck(out)
        return bottleneck_output