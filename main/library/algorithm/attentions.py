import math
import torch

import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module with support for relative position representations,
    proximal bias, and ONNX-friendly exporting. Designed primarily for 1D sequences 
    (e.g., speech or text processing) using Conv1D operators.
    """

    def __init__(
        self, 
        channels, 
        out_channels, 
        n_heads, 
        p_dropout=0.0, 
        window_size=None, 
        heads_share=True, 
        block_length=None, 
        proximal_bias=False, 
        proximal_init=False, 
        onnx=False
    ):
        """
        Initializes the MultiHeadAttention module.

        Args:
            channels (int): Number of input channels (hidden dimension).
            out_channels (int): Number of output channels.
            n_heads (int): Number of attention heads.
            p_dropout (float): Dropout probability. Defaults to 0.0.
            window_size (int, optional): Window size for relative position embeddings. Defaults to None.
            heads_share (bool): If True, all heads share the same relative embeddings. Defaults to True.
            block_length (int, optional): Maximum distance constraint for block-sparse masking. Defaults to None.
            proximal_bias (bool): If True, applies log-distance penalty to local attention. Defaults to False.
            proximal_init (bool): If True, initializes key weights using query weights. Defaults to False.
            onnx (bool): If True, uses ONNX-compatible tensor operations. Defaults to False.
        """

        super().__init__()
        assert channels % n_heads == 0
        # Attribute assignments
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads # Dimension per head
        self.window_size = window_size
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        # Linear projections implemented via 1x1 convolutions
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        # Dynamic routing of functions based on ONNX export mode
        self._get_relative_embeddings = self._get_relative_embeddings_onnx if onnx else self._get_relative_embeddings_torch
        self._relative_position_to_absolute_position = self._relative_position_to_absolute_position_onnx if onnx else self._relative_position_to_absolute_position_torch
        self._absolute_position_to_relative_position = self._absolute_position_to_relative_position_onnx if onnx else self._absolute_position_to_relative_position_torch

        # Initialize relative position embeddings if window_size is specified
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5 # Scale initialization standard deviation

            # Relative embeddings for keys and values
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

        # Weight initialization using Xavier uniform
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        nn.init.xavier_uniform_(self.conv_o.weight)

        # Initialize Key projection to copy Query projection for localized bias initialization
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        """
        Forward pass for Multi-Head Attention.

        Args:
            x (torch.Tensor): Query input tensor.
            c (torch.Tensor): Key/Value input tensor.
            attn_mask (torch.Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """

        # Project inputs to query, key, and value representations
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        # Calculate attention mechanism
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        # Final linear projection to destination channels
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        """
        Computes scaled dot-product attention with optional relative positioning and masking.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Context tensor and attention probability weights.
        """

        b, d, t_s, t_t = (*key.size(), query.size(2))
        # Reshape and transpose for multi-head attention
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        # Compute base attention scores
        scores = (query / math.sqrt(self.k_channels)) @ key.transpose(-2, -1)
    
        # Incorporate relative position representations for keys
        if self.window_size:
            assert t_s == t_t
            # Extract sliced embeddings, apply relative matrix multiplication and shift to absolute positions
            scores += self._relative_position_to_absolute_position(
                self._matmul_with_relative_keys(
                    query / math.sqrt(self.k_channels), 
                    self._get_relative_embeddings(
                        self.emb_rel_k, 
                        t_s
                    )
                )
            )

        # Apply localized distance-based log penalty
        if self.proximal_bias:
            assert t_s == t_t
            scores += self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)

        # Apply external masking (e.g., sequence padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            # Apply local block-diagonal constraints (banded attention)
            if self.block_length:
                assert (t_s == t_t)
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length) # Keep upper band
                    .tril(self.block_length) # Keep lower band
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)

        # Softmax normalize and apply dropout to obtain final probabilities
        p_attn = self.drop(F.softmax(scores, dim=-1))
        # Compute context vector
        output = p_attn @ value

        # Incorporate relative position representations for values
        if self.window_size: 
            output += self._matmul_with_relative_values(
                self._absolute_position_to_relative_position(
                    p_attn
                ), 
                self._get_relative_embeddings(
                    self.emb_rel_v, 
                    t_s
                )
            )

        # Permute and reshape back to 1D channel format.
        return output.transpose(2, 3).contiguous().view(b, d, t_t), p_attn

    def _matmul_with_relative_values(self, x, y):
        """Helper for relative value matrix multiplication."""

        return x @ y.unsqueeze(0)

    def _matmul_with_relative_keys(self, x, y):
        """Helper for relative key matrix multiplication."""

        return x @ y.unsqueeze(0).transpose(-2, -1)

    def _get_relative_embeddings_onnx(self, relative_embeddings, length):
        """Slices relative embeddings to match current sequence length (ONNX compatible)."""

        pad_length = (length - (self.window_size + 1)).clamp(min=0)
        slice_start_position = ((self.window_size + 1) - length).clamp(min=0)

        # Padding via concatenation to workaround ONNX padding limitations
        x = torch.cat([torch.zeros(relative_embeddings.size(0), pad_length, relative_embeddings.size(2), device=relative_embeddings.device, dtype=relative_embeddings.dtype), relative_embeddings], dim=1)
        relative_embeddings = torch.cat([x, torch.zeros(x.size(0), pad_length, x.size(2), device=x.device, dtype=x.dtype)], dim=1)

        return relative_embeddings[:, slice_start_position:(slice_start_position + 2 * length - 1)]  

    def _get_relative_embeddings_torch(self, relative_embeddings, length):
        """Slices relative embeddings to match current sequence length (PyTorch native)."""

        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        # Native functional padding
        if pad_length > 0:
            relative_embeddings = F.pad(
                relative_embeddings, 
                [0, 0, pad_length, pad_length, 0, 0],
            )

        return relative_embeddings[:, slice_start_position:(slice_start_position + 2 * length - 1)]  

    def _relative_position_to_absolute_position_onnx(self, x):
        """Converts a relative representation matrix to absolute coordinates (ONNX compatible)."""

        batch, heads, length, _ = x.size()
        # Reshaping trick via manual padding to perform coordinate shifting
        pad = self.padding(x, [0, 1]).view([batch, heads, length * 2 * length])
        return self.padding(pad, [0, length - 1]).view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]

    def _relative_position_to_absolute_position_torch(self, x):
        """Converts a relative representation matrix to absolute coordinates using structural padding."""

        batch, heads, length, _ = x.size()
        # Tensor flattening via padding trick to transform relative indices to absolute
        pad = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0]).view([batch, heads, length * 2 * length])
        return F.pad(pad, [0, length - 1, 0, 0, 0, 0]).view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]

    def _absolute_position_to_relative_position_onnx(self, x):
        """Converts absolute coordinate metrics to relative position coordinates (ONNX compatible)."""

        batch, heads, length, _ = x.size()

        pad = self.padding(x, [0, length - 1]).view([batch, heads, length**2 + length * (length - 1)])
        return self.padding(pad, [length, 0]).view([batch, heads, length, 2 * length])[:, :, :, 1:]

    def _absolute_position_to_relative_position_torch(self, x):
        """Converts absolute coordinate metrics to relative position coordinates using structural padding."""

        batch, heads, length, _ = x.size()

        pad = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0]).view([batch, heads, length**2 + length * (length - 1)])
        return F.pad(pad,  [length, 0, 0, 0, 0, 0]).view([batch, heads, length, 2 * length])[:, :, :, 1:]

    def _attention_bias_proximal(self, length):
        """Generates a localized attention bias matrix where penalty increases with distance."""

        r = torch.arange(length, dtype=torch.float32)
        diff = r.unsqueeze(0) - r.unsqueeze(1) # Matrix of structural distance gaps
        return -diff.abs().log1p().unsqueeze(0).unsqueeze(0)

    def padding(self, x, pad):
        """Alternative zero padding function using concatenation to ensure ONNX support."""

        left, right = pad[-2], pad[-1]
        shape = list(x.shape[:-1])

        # Concatenate zeros to the edges of the last dimension
        x = torch.cat([torch.zeros(*shape, left, device=x.device, dtype=x.dtype), x], dim=-1)
        x = torch.cat([x, torch.zeros(*shape, right, device=x.device, dtype=x.dtype)], dim=-1)

        return x

class FFN(nn.Module):
    """
    Feed-Forward Network (FFN) using 1D convolutions with support for 
    causal or identical padding, custom activations, and masking.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        filter_channels, 
        kernel_size, 
        p_dropout=0.0, 
        activation=None, 
        causal=False
    ):
        """
        Initializes the FFN module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            filter_channels (int): Intermediate hidden layer channels.
            kernel_size (int): Size of the convolutional kernel.
            p_dropout (float): Dropout probability. Defaults to 0.0.
            activation (str, optional): Type of activation ("gelu" or "relu"). Defaults to None.
            causal (bool): If True, applies causal padding to prevent looking into future steps. Defaults to False.
        """

        super().__init__()
        # 1D Convolution operations acting as position-wise layers
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)
        self.kernel_size = kernel_size
        self.activation = activation
        # Route padding technique conditionally
        self.padding_fn = self._causal_padding if causal else self._same_padding

    def forward(self, x, x_mask):
        """
        Forward pass for the Feed-Forward Network.

        Args:
            x (torch.Tensor): Input tensor.
            x_mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        # First linear transformation layer with structured padding and sequence masking
        x = self.conv_1(self.padding_fn(x * x_mask))
        # Non-linear activation combined with dropout regularization
        x = self.drop(self._apply_activation(x))
        # Second linear transformation layer
        x = self.conv_2(self.padding_fn(x * x_mask))

        # Enforce zeros on masked elements in the final output
        return x * x_mask

    def _apply_activation(self, x):
        """Applies configured non-linear function (supports custom fast approximate GELU or ReLU)."""
        if self.activation == "gelu": return x * (1.702 * x).sigmoid() # Fast GeLU approximation formula
        return x.relu()

    def _causal_padding(self, x):
        """Pads the input exclusively on the left to enforce temporal causality."""

        return F.pad(
            x, 
            [self.kernel_size, 0, 0, 0, 0, 0]
        )

    def _same_padding(self, x):
        """Pads symmetrically on both edges to preserve equivalent sequence length."""

        return F.pad(
            x,
            [(self.kernel_size - 1) // 2, self.kernel_size // 2, 0, 0, 0, 0],
        )