import os
import sys
import torch

import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from itertools import chain
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

from main.library.algorithm.modules import WaveNet
from main.library.algorithm.commons import get_padding, init_weights

LRELU_SLOPE = 0.1

def create_conv1d_layer(channels, kernel_size, dilation):
    """
    Helper function to instantiate a 1D Convolutional layer wrapped with weight normalization 
    and automatically calculated padding to preserve sequence length.
    """

    return weight_norm(
        torch.nn.Conv1d(
            channels, 
            channels, 
            kernel_size, 
            1, 
            dilation=dilation, 
            padding=get_padding(kernel_size, dilation)
        )
    )

def apply_mask(tensor, mask):
    """
    Applies a binary tracking sequence mask to clear out values in padding positions.
    """

    return tensor * mask if mask is not None else tensor

class ResBlock(torch.nn.Module):
    """
    Standard Residual Block containing a series of dilated 1D Convolutions with 
    LeakyReLU activations, acting as a secondary alternative backbone layer.
    """

    def __init__(
        self, 
        channels, 
        kernel_size=3, 
        dilations=(1, 3, 5)
    ):
        """
        Initializes convolutional block stacks.

        Args:
            channels (int): Input and output channel dimensionality.
            kernel_size (int): Convolution kernel width. Defaults to 3.
            dilations (List[int]): Dilation multipliers for each parallel layer sequence. Defaults to (1, 3, 5).
        """

        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

    @staticmethod
    def _create_convs(channels, kernel_size, dilations):
        """
        Generates weight-normalized conv layers and applies model parameter weight initializations.
        """

        layers = torch.nn.ModuleList([
            create_conv1d_layer(channels, kernel_size, d) 
            for d in dilations
        ])
        layers.apply(init_weights)

        return layers

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input hidden state matrix.
        """

        for conv1, conv2 in zip(self.convs1, self.convs2):
            x = conv2(F.leaky_relu(conv1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE)) + x

        return x

    def remove_weight_norm(self):
        """Removes weight normalization configurations across all internal layers."""

        for conv in chain(self.convs1, self.convs2):
            if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations: parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
            else: remove_weight_norm(conv)

class Flip(torch.nn.Module):
    """
    A bi-directional flow layer that inverts/flips the sequence order along the channel dimension.
    Ensures that alternative parts of the split features interact across sequential flow steps.
    """

    def forward(
        self, 
        x, 
        *args, 
        reverse=False, 
        **kwargs
    ):
        """
        Flips the channel axis order.

        Args:
            x (torch.Tensor): Feature tensor.
            reverse (bool): Execution direction trigger flag. Defaults to False.

        Returns:
            - If forward (reverse=False): Tuple containing the flipped tensor and a zero log-determinant.
            - If backward (reverse=True): Flipped tensor only.
        """

        x = x.flip([1]) # Flip across dimension 1 (Channels)

        if not reverse: return x, torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        else: return x

class ResidualCouplingBlock(torch.nn.Module):
    """
    Multi-layer Normalizing Flow block containing alternating Residual Coupling Layers 
    and Channel-Flipping layers. This handles complex probability distribution mappings 
    invertibly (e.g., from linear specs to latents and back) in VITS architectures.
    """

    def __init__(
        self, 
        channels, 
        hidden_channels, 
        kernel_size, 
        dilation_rate, 
        n_layers, 
        n_flows=4, 
        gin_channels=0
    ):
        """
        Initializes standard flows consisting of sequential Coupling Layers paired with Flips.
        """

        super(ResidualCouplingBlock, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.flows = torch.nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, 
                    hidden_channels, 
                    kernel_size, 
                    dilation_rate, 
                    n_layers, 
                    gin_channels=gin_channels, 
                    mean_only=True
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g = None, reverse = False):
        """
        Propagates sequence transformations through the flow network.

        Args:
            x (torch.Tensor): Input hidden vector matrix.
            x_mask (torch.Tensor): Binary temporal mask padding tensor.
            g (torch.Tensor, optional): Global speaker embedding condition context.
            reverse (bool): If True, executes the inverse mathematical mapping path. Defaults to False.
        """

        if not reverse:
            # Forward propagation passes consecutively from step 0 to end
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            # Inverse backward path processes blocks in strict backwards order
            for flow in reversed(self.flows):
                x = flow.forward(x, x_mask, g=g, reverse=reverse)

        return x

    def remove_weight_norm(self):
        """Removes weight normalization from internal WaveNet layers."""
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()

class ResidualCouplingLayer(torch.nn.Module):
    """
    An affine or additive coupling layer split across the channel axis.
    One half remains unchanged and generates parameters (mean/variance) via WaveNet 
    to transform the remaining half, keeping the transformation entirely invertible.
    """

    def __init__(
        self, 
        channels, 
        hidden_channels, 
        kernel_size, 
        dilation_rate, 
        n_layers, 
        p_dropout=0, 
        gin_channels=0, 
        mean_only=False
    ):
        """Initializes components for the specific Coupling Layer segment."""

        assert channels % 2 == 0
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        # Dimension matching input layer
        self.pre = torch.nn.Conv1d(self.half_channels, hidden_channels, 1)
        # Deep internal processing module
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        # Output projection head: switches size depending on whether variance tracking logs are calculated
        self.post = torch.nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)

        # Initialize final projection parameters to zero so the layer acts as an identity mapping initially
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Executes coupling steps to transform split tensors.

        Args:
            x (torch.Tensor): Audio latent/mel representation tensor of shape (B, C, T).
            x_mask (torch.Tensor): Sequence masking configuration logic matrix (B, 1, T).
            g (torch.Tensor, optional): Conditioning vector matrix. Defaults to None.
            reverse (bool): Direction switch flag (False = forward, True = inverse reconstruction).
        """

        # Split features equally along the channel index dimension
        x0, x1 = x.split([self.half_channels] * 2, 1)
        # Pass the first half through WaveNet blocks to compute transformation parameters
        stats = self.post(self.enc((self.pre(x0) * x_mask), x_mask, g=g)) * x_mask

        # Extract shift (m) and scale scale logs parameters accordingly
        if not self.mean_only: m, logs = stats.split([self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m) # Zero scale logs map implies an additive coupling mode

        if not reverse: 
            # Forward transform
            return torch.cat([x0, (m + x1 * logs.exp() * x_mask)], 1), logs.sum(dim=[1, 2])
        else:
            # Inverse exact reconstruction 
            return torch.cat([x0, ((x1 - m) * (-logs).exp() * x_mask)], 1)

    def remove_weight_norm(self):
        """Propagates normalization stripping commands downstream to WaveNet layers."""

        self.enc.remove_weight_norm()