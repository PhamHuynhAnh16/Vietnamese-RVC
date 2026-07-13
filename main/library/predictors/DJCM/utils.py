import torch

from torch import nn
from einops.layers.torch import Rearrange

def init_layer(layer):
    """
    Initializes a linear or convolutional layer's weights using Xavier Uniform.

    If the layer has a learnable bias, it is reset to zero.

    Args:
        layer (nn.Module): The target layer to initialize.
    """

    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias") and layer.bias is not None: 
        layer.bias.data.fill_(0.0)

def init_bn(bn):
    """
    Initializes Batch Normalization parameters to standard baseline starting values.

    Sets biases to 0.0, scaling weights to 1.0, tracking means to 0.0, and variances to 1.0.

    Args:
        bn (nn.Module): The BatchNorm layer to initialize.
    """

    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.fill_(0.0)
    bn.running_var.data.fill_(1.0)

class BiGRU(nn.Module):
    """
    Bidirectional GRU module that extracts sequential temporal relationships from image patches.

    Flattens spatial 2D macro-patches into sequential tokens using `einops.Rearrange`, 
    then processes them via a Bidirectional GRU layer.
    """

    def __init__(
        self, 
        patch_size, 
        channels, 
        depth
    ):
        """
        Initializes the BiGRU module.

        Args:
            patch_size (tuple): Dimensions of each patch (patch_width, patch_height).
            channels (int): Number of feature channels in the incoming input maps.
            depth (int): The total number of stacked recurrent layers within the GRU.
        """

        super(BiGRU, self).__init__()
        patch_width, patch_height = patch_size
        # Calculate the flattened feature vector dimension per patch step token
        patch_dim = channels * patch_height * patch_width

        # Rearrange 4D tensor maps to sequential patch representations
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (w p1) (h p2) -> b (w h) (p1 p2 c)', 
                p1=patch_width, 
                p2=patch_height
            )
        )

        # Bidirectional GRU splits the target capacity between forward and backward passes (hence hidden_size = patch_dim // 2)
        self.gru = nn.GRU(
            patch_dim, 
            patch_dim // 2, 
            num_layers=depth, 
            batch_first=True, 
            bidirectional=True
        )

    def forward(self, x):
        """
        Executes the forward pass of the patch-based BiGRU.

        Args:
            x (torch.Tensor): 4D spatial feature tensor.

        Returns:
            torch.Tensor: Sequential hidden features matrix from the GRU.
        """

        x = self.to_patch_embedding(x)
        # Gracefully handle cuDNN hidden sequence state mismatches across different GPU architectures
        try:
            return self.gru(x)[0]
        except:
            # Fallback routine: temporarily disable cuDNN to calculate on vanilla backend if an exception occurs
            # Exceptions typically occur when the input is too large.
            torch.backends.cudnn.enabled = False
            return self.gru(x)[0]

class ResConvBlock(nn.Module):
    """
    Pre-activation Residual Convolutional Block.

    Applies BatchNorm -> PReLU -> Conv twice, along with an integrated automatic 
    identity shortcut connection router.
    """

    def __init__(
        self, 
        in_planes, 
        out_planes
    ):
        """
        Initializes the ResConvBlock.

        Args:
            in_planes (int): Number of incoming input channels.
            out_planes (int): Number of target output channels.
        """

        super(ResConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(
            in_planes, 
            momentum=0.01
        )
        self.bn2 = nn.BatchNorm2d(
            out_planes, 
            momentum=0.01
        )
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        # Standard 3x3 padded convolutions that preserve spatial dimensions
        self.conv1 = nn.Conv2d(
            in_planes, 
            out_planes, 
            (3, 3), 
            padding=(1, 1), 
            bias=False
        )
        self.conv2 = nn.Conv2d(
            out_planes, 
            out_planes, 
            (3, 3), 
            padding=(1, 1), 
            bias=False
        )

        # Determine if a 1x1 projection shortcut is needed to align channel dimensions
        self.is_shortcut = False
        if in_planes != out_planes:
            self.shortcut = nn.Conv2d(
                in_planes, 
                out_planes, 
                (1, 1)
            )
            self.is_shortcut = True

        self.init_weights()
        # Performance optimization: pre-assign the forward projection method to bypass conditional execution branches
        self._return = self.return_shortcut if self.is_shortcut else self.return_non_shortcut

    def init_weights(self):
        """Initializes internal weights across all parameters."""

        init_bn(self.bn1)
        init_bn(self.bn2)

        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut: init_layer(self.shortcut)
    
    def return_shortcut(self, x, out):
        """Sums projection mapped input modifications with processed output features."""

        return self.shortcut(x) + out
    
    def return_non_shortcut(self, x, out):
        """Applies basic residual identity additions directly."""

        return out + x

    def forward(self, x):
        """
        Executes the pre-activation forward pass loop.

        Args:
            x (torch.Tensor): Input activation maps.

        Returns:
            torch.Tensor: Accumulated residual output tensor map.
        """

        # First Block: BN -> PReLU -> Conv
        out = self.conv1(
            self.act1(self.bn1(x))
        )
        # Second Block: BN -> PReLU -> Conv
        out = self.conv2(
            self.act2(self.bn2(out))
        )

        # Return combined result using the pre-mapped execution pathway function pointer
        return self._return(x, out)