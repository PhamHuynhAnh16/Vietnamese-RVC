import os
import sys
import torch

import torch.nn.utils.parametrize as parametrize

sys.path.append(os.getcwd())

from main.library.algorithm.commons import fused_add_tanh_sigmoid_multiply

class WaveNet(torch.nn.Module):
    """
    A 1D Dilated Non-Causal WaveNet architecture acting as a residual sequence 
    backbone. It utilizes Gated Activation Units (tanh and sigmoid multiplication) 
    and supports global conditioning (e.g., speaker or noise embeddings).
    """

    def __init__(
        self, 
        hidden_channels, 
        kernel_size, 
        dilation_rate, 
        n_layers, 
        gin_channels=0, 
        p_dropout=0
    ):
        """
        Initializes the WaveNet architecture layer blocks.

        Args:
            hidden_channels (int): Input and output hidden channel size.
            kernel_size (int): Size of the 1D convolution kernel. Must be an odd number.
            dilation_rate (int): Base dilation factor geometric growth multiplier.
            n_layers (int): Total depth number of dilated convolutional layers.
            gin_channels (int): Channels for global conditioning input. Defaults to 0 (disabled).
            p_dropout (float): Dropout probability. Defaults to 0.0.
        """

        super(WaveNet, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        # Kept as a tensor block to serve as a channel delimiter indicator for custom fused operators
        self.n_channels_tensor = torch.IntTensor([hidden_channels])
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)

        # Optional projection layer for global condition embeddings (e.g., Speaker IDs)
        if gin_channels: 
            self.cond_layer = torch.nn.utils.parametrizations.weight_norm(
                torch.nn.Conv1d(
                    gin_channels, 
                    2 * hidden_channels * n_layers, 
                    1
                ), 
                name="weight"
            )

        # Precompute dilation sizes and matching padding limits to keep temporal lengths aligned
        dilations = [dilation_rate**i for i in range(n_layers)]
        paddings = [(kernel_size * d - d) // 2 for d in dilations]

        # Construct cascading WaveNet layer blocks
        for i in range(n_layers):
            # Dilated convolution layer projecting to gated channels
            self.in_layers.append(
                torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(
                        hidden_channels, 
                        2 * hidden_channels, 
                        kernel_size, 
                        dilation=dilations[i], 
                        padding=paddings[i]
                    ), 
                    name="weight"
                )
            )

            # Determine structural channels for residual and skip connection pathways
            # The final layer only needs to output residual features (hidden_channels)
            res_skip_channels = (hidden_channels if i == n_layers - 1 else 2 * hidden_channels)

            self.res_skip_layers.append(
                torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(
                        hidden_channels, 
                        res_skip_channels, 
                        1
                    ), 
                    name="weight"
                )
            )

    def forward(self, x, x_mask, g=None):
        """
        Executes WaveNet forward propagation over sequences.

        Args:
            x (torch.Tensor): Input hidden representation tensor.
            x_mask (torch.Tensor): Temporal tracking binary mask tensor.
            g (torch.Tensor, optional): Global conditioning embedding. Defaults to None.

        Returns:
            torch.Tensor: Aggregated sequence tensor maps.
        """

        # Allocate accumulation storage initialized to zeros
        output = x.clone().zero_()
        # Pre-project global condition embeddings across all layers simultaneously if present
        g = self.cond_layer(g) if g is not None else None

        for i in range(self.n_layers):
            # Pass input data through the dilated 1D convolution layer
            x_in = self.in_layers[i](x)
            # Extract the unique chunk of conditional parameters designated for the current layer block
            g_l = (g[:, i * 2 * self.hidden_channels : (i + 1) * 2 * self.hidden_channels, :] if g is not None else 0)

            # Run dropout regularizer followed by 1x1 projection mapping
            # Apply Gated Activation Unit (GAU): tanh(x_conv + g_tanh) * sigmoid(x_conv + g_sigmoid)
            # Implemented as a highly optimized fused function to minimize GPU memory traffic
            res_skip_acts = self.res_skip_layers[i](self.drop(fused_add_tanh_sigmoid_multiply(x_in, g_l, self.n_channels_tensor)))

            if i < self.n_layers - 1:
                # Mid layer paths: update the core representation for the next block using residual values
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
                # Accumulate the remaining section directly into the global skip connection track
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else: output = output + res_skip_acts # Final layer path: accumulate the remaining output states entirely into the skip connections

        return output * x_mask

    def remove_weight_norm(self):
        """
        Removes weight normalization from all convolution layers. 
        Compatible with both PyTorch modern 'parametrizations' API and legacy hooks 
        to ensure seamless inference execution.
        """

        if self.gin_channels != 0: 
            if hasattr(self.cond_layer, "parametrizations") and "weight" in self.cond_layer.parametrizations: parametrize.remove_parametrizations(self.cond_layer, "weight", leave_parametrized=True)
            else: torch.nn.utils.remove_weight_norm(self.cond_layer)

        for l in self.in_layers:
            if hasattr(l, "parametrizations") and "weight" in l.parametrizations: parametrize.remove_parametrizations(l, "weight", leave_parametrized=True)
            else: torch.nn.utils.remove_weight_norm(l)

        for l in self.res_skip_layers:
            if hasattr(l, "parametrizations") and "weight" in l.parametrizations: parametrize.remove_parametrizations(l, "weight", leave_parametrized=True)
            else: torch.nn.utils.remove_weight_norm(l)