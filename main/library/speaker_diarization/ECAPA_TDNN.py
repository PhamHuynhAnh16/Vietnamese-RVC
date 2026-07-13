import math
import torch  

import torch.nn as nn
import torch.nn.functional as F

def length_to_mask(length, max_len=None, dtype=None, device=None):
    """
    Converts a 1D tensor of sequence lengths into a 2D binary mask matrix.

    Args:
        length (torch.Tensor): 1D tensor containing lengths of individual sequences.
        max_len (int, optional): The maximum sequence length for the mask. Defaults to the maximum value in `length`.
        dtype (torch.dtype, optional): Desired data type of returned tensor. Defaults to length's dtype.
        device (torch.device, optional): Desired device of returned tensor. Defaults to length's device.

    Returns:
        torch.Tensor: A 2D boolean/numeric mask tensor of shape (batch_size, max_len).
    """

    assert len(length.shape) == 1
    # Determine maximum length if not explicitly provided
    if max_len is None: max_len = length.max().long().item() 

    # Generate a grid of indices [0, 1, ..., max_len - 1] and broadcast to compare with lengths
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)

    # Resolve default data type and device constraints
    if dtype is None: dtype = length.dtype
    if device is None: device = length.device

    return torch.as_tensor(mask, dtype=dtype, device=device)

def get_padding_elem(L_in, stride, kernel_size, dilation):
    """
    Calculates asymmetric or symmetric padding values required for maintaining temporal alignment.

    Args:
        L_in (int): The dimension size of the input channel sequence.
        stride (int): Stride step size of the convolution.
        kernel_size (int): Size of the convolving kernel.
        dilation (int): Spacing between kernel elements.

    Returns:
        List[int]: A list of integers specifying padding values [left, right].
    """

    # For strided convolutions, apply standard half-kernel padding bounds
    if stride > 1: padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]
    else:
        # Calculate strict output sequence lengths to maintain exact internal spatial boundaries
        L_out = (math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1)
        padding = [math.floor((L_in - L_out) / 2), math.floor((L_in - L_out) / 2)]

    return padding

class _BatchNorm1d(nn.Module):
    """
    An internal custom wrapper for PyTorch's BatchNorm1d to easily handle 

    transpositions and flexible multi-dimensional sequential tensors.
    """

    def __init__(
        self, 
        input_shape=None, 
        input_size=None, 
        eps=1e-05, 
        momentum=0.1, 
        affine=True, 
        track_running_stats=True, 
        combine_batch_time=False, 
        skip_transpose=False
    ):
        """Initializes the _BatchNorm1d tracking block."""

        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        # Deduce input features automatically if input_shape is provided instead of input_size
        if input_size is None and skip_transpose: input_size = input_shape[1]
        elif input_size is None: input_size = input_shape[-1]

        self.norm = nn.BatchNorm1d(
            input_size, 
            eps=eps, 
            momentum=momentum, 
            affine=affine, 
            track_running_stats=track_running_stats
        )

    def forward(self, x):
        """Applies normalization over the input sequence."""

        shape_or = x.shape

        if self.combine_batch_time: 
            # Flatten Batch and Time dimensions together to normalize features independently

            x = (
                x.reshape(shape_or[0] * shape_or[1], shape_or[2]) 
            ) if x.ndim == 3 else (
                x.reshape(shape_or[0] * shape_or[1], shape_or[3], shape_or[2])
            )
        elif not self.skip_transpose: x = x.transpose(-1, 1)

        x_n = self.norm(x)
        # Restore original tensor layouts post-normalization
        if self.combine_batch_time: x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose: x_n = x_n.transpose(1, -1)

        return x_n

class _Conv1d(nn.Module):
    """
    An internal core 1D Convolution class supporting dynamic padding strategies and flexible initialization methods.
    """

    def __init__(
        self, 
        out_channels, 
        kernel_size, 
        input_shape=None, 
        in_channels=None, 
        stride=1, 
        dilation=1, 
        padding="same", 
        groups=1, 
        bias=True, 
        padding_mode="reflect", 
        skip_transpose=False, 
        weight_norm=False,
        conv_init=None, 
        default_padding=0
    ):
        """Initializes custom configurations for 1D convolution layer layers."""

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None: raise ValueError("Either input_shape or in_channels must be defined.")
        if in_channels is None: in_channels = self._check_input_shape(input_shape)

        self.in_channels = in_channels
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            self.kernel_size, 
            stride=self.stride, 
            dilation=self.dilation, 
            padding=default_padding, 
            groups=groups, 
            bias=bias
        )

        # Initialize weight structures based on standard initialization algorithms
        if conv_init == "kaiming": nn.init.kaiming_normal_(self.conv.weight)
        elif conv_init == "zero": nn.init.zeros_(self.conv.weight)
        elif conv_init == "normal": nn.init.normal_(self.conv.weight, std=1e-6)

        # Apply standard weight normalization transformations
        if weight_norm: self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        """Applies spatial padding configurations and forward convolution pass."""

        if not self.skip_transpose: x = x.transpose(1, -1)
        if self.unsqueeze: x = x.unsqueeze(1)

        # Evaluate contextual padding configurations
        if self.padding == "same": 
            x = self._manage_padding(
                x, 
                self.kernel_size, 
                self.dilation, 
                self.stride
            )
        elif self.padding == "causal": 
            # Apply causal padding to prevent information leakage from future frames
            x = F.pad(
                x, 
                ((self.kernel_size - 1) * self.dilation, 0)
            )
        elif self.padding == "valid": pass
        else: raise ValueError(f"Unsupported padding strategy: {self.padding}")

        wx = self.conv(x)

        if self.unsqueeze: wx = wx.squeeze(1)
        if not self.skip_transpose: wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(self, x, kernel_size, dilation, stride):
        """Wraps functional padding operations with dynamically computed bounds."""

        return F.pad(
            x, 
            get_padding_elem(
                self.in_channels, 
                stride, 
                kernel_size, 
                dilation
            ), 
            mode=self.padding_mode
        )

    def _check_input_shape(self, shape):
        """Validates shape semantics and determines operational input channel counts."""

        if len(shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose: in_channels = shape[1]
        elif len(shape) == 3: in_channels = shape[2]
        else: raise ValueError(f"Unsupported input shape dimensions: {len(shape)}")

        if not self.padding == "valid" and self.kernel_size % 2 == 0: raise ValueError("Even kernel sizes are only compatible with 'valid' padding.")
        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization from the internal convolution layer."""

        self.conv = nn.utils.remove_weight_norm(self.conv)

class Linear(torch.nn.Module):
    """
    Custom Linear projection block handling higher-order dimension flattening 

    and weight normalization (max-norm clipping) techniques.
    """

    def __init__(
        self, 
        n_neurons, 
        input_shape=None, 
        input_size=None, 
        bias=True, 
        max_norm=None, 
        combine_dims=False
    ):
        """Initializes projection weights and flattening options."""

        super().__init__()
        self.max_norm = max_norm
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None: raise ValueError
        if input_size is None:
            input_size = input_shape[-1]
            # Compress spatial dimensions if specific 4D processing conditions are requested
            if len(input_shape) == 4 and self.combine_dims: input_size = input_shape[2] * input_shape[3]

        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        """Executes forward projection passing after checking structural norm criteria."""

        if x.ndim == 4 and self.combine_dims: x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        # Enforce max-norm constraints by renormalizing row weights inline
        if self.max_norm is not None: self.w.weight.data = torch.renorm(self.w.weight.data, p=2, dim=0, maxnorm=self.max_norm)

        return self.w(x)

class Conv1d(_Conv1d):
    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        """An explicit 1D Convolution wrapper subclass defaulting to channel-first layout formatting."""

        super().__init__(
            skip_transpose=True, 
            *args, 
            **kwargs
        )

class BatchNorm1d(_BatchNorm1d):
    """An explicit 1D Batch Normalization wrapper subclass defaulting to channel-first layout formatting."""

    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(
            skip_transpose=True, 
            *args, 
            **kwargs
        )

class TDNNBlock(nn.Module):
    """
    Time Delay Neural Network (TDNN) base block grouping 1D Convolution, 

    activation function, normalization layers, and dropout steps.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        dilation, 
        activation=nn.ReLU, 
        groups=1, 
        dropout=0.0
    ):
        """Builds standard structural TDNN sequence operations."""

        super().__init__()
        self.conv = Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            groups=groups
        )

        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        """Passes sequential representations through standard composite layer blocks."""

        return self.dropout(self.norm(self.activation(self.conv(x))))

class Res2NetBlock(torch.nn.Module):
    """
    Res2Net component module partitioning channel profiles into multi-stage residual scales

    to establish hierarchical multi-scale receptive processing paths.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        scale=8, 
        kernel_size=3, 
        dilation=1, 
        dropout=0.0
    ):
        """Constructs sub-block pipelines according to target scale values."""

        super().__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        # Instantiate recursive processing blocks across segmented groups
        self.blocks = nn.ModuleList([
            TDNNBlock(
                in_channel, 
                hidden_channel, 
                kernel_size=kernel_size, 
                dilation=dilation, 
                dropout=dropout
            ) 
            for _ in range(scale - 1)
        ])
        self.scale = scale

    def forward(self, x):
        """Applies sequential split operations and sums forward channels hierarchically."""

        y = []
        # Split along channels dimension into equal chunk groups
        for i, x_i in enumerate(x.chunk(self.scale, dim=1)):
            if i == 0: y_i = x_i # The first subset passes through unmodified without operation
            elif i == 1: y_i = self.blocks[i - 1](x_i)
            else: y_i = self.blocks[i - 1](x_i + y_i) # Accumulate current chunks with the previous sub-block output to form local context links

            y.append(y_i)

        return torch.cat(y, dim=1)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block customized to track context limits 

    dynamically using optional length masks.
    """

    def __init__(
        self, 
        in_channels, 
        se_channels, 
        out_channels
    ):
        """Creates compression bottlenecks for feature re-weighting pipelines."""

        super().__init__()
        self.conv1 = Conv1d(
            in_channels=in_channels, 
            out_channels=se_channels, 
            kernel_size=1
        )
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = Conv1d(
            in_channels=se_channels, 
            out_channels=out_channels, 
            kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lengths=None):
        """Performs masked statistics collection and rescales channel profiles."""

        L = x.shape[-1]
        # Calculate localized temporal means based on input lengths if available
        if lengths is not None:
            mask = length_to_mask(
                lengths * L, 
                max_len=L, 
                device=x.device
            ).unsqueeze(1)

            s = (x * mask).sum(dim=2, keepdim=True) / mask.sum(dim=2, keepdim=True)
        else: s = x.mean(dim=2, keepdim=True)

        # Generate excitation scalars and scale the input activations
        return self.sigmoid(self.conv2(self.relu(self.conv1(s)))) * x

class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP) layer designed to compute weighted 

    means and variances across variable-length frame sequences using self-attention mechanisms.
    """

    def __init__(
        self, 
        channels, 
        attention_channels=128, 
        global_context=True
    ):
        """Initializes internal attention models and multi-scale feature tracking parameters."""

        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        # Determine projection paths using global sequence contexts if requested
        self.tdnn = (
            TDNNBlock(
                channels * 3, 
                attention_channels, 
                1, 
                1
            )
        ) if global_context else (
            TDNNBlock(
                channels, 
                attention_channels, 
                1, 
                1
            )
        )
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, 
            out_channels=channels, 
            kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Aggregates temporal frame sequences into fixed-dimensional statistical vectors."""

        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            # Internal helper function to extract attention-weighted mean and standard deviation
            mean = (m * x).sum(dim)
            return mean, ((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)).sqrt()

        if lengths is None: lengths = torch.ones(x.shape[0], device=x.device)
        mask = length_to_mask(lengths * L, max_len=L, device=x.device).unsqueeze(1)

        if self.global_context:
            # Squeeze time dimension and merge global mean/std vectors alongside raw sequence frames
            mean, std = _compute_statistics(
                x, 
                mask / mask.sum(dim=2, keepdim=True).float()
            )

            attn = torch.cat([
                x, 
                mean.unsqueeze(2).repeat(1, 1, L), 
                std.unsqueeze(2).repeat(1, 1, L)
            ], dim=1)
        else: attn = x

        # Compute the final attention-weighted mean and standard deviation vectors
        mean, std = _compute_statistics(
            x, 
            # Compute raw energy scores, mask invalid sequence padding elements, and normalize with Softmax
            F.softmax(
                self.conv(
                    self.tanh(self.tdnn(attn))
                ).masked_fill(mask == 0, float("-inf")), 
                dim=2
            )
        )

        # Concatenate mean and variance descriptors into a unified spatial pooling vector
        return torch.cat((mean, std), dim=1).unsqueeze(2)

class SERes2NetBlock(nn.Module):
    """
    A unified block combining Multi-scale Res2Net features with Squeeze-and-Excitation 

    gating logic and identity residual shortcuts.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        res2net_scale=8, 
        se_channels=128, 
        kernel_size=1, 
        dilation=1, 
        activation=torch.nn.ReLU, 
        groups=1, 
        dropout=0.0
    ):
        """Initializes internal structural component blocks."""

        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            dilation=1, 
            activation=activation, 
            groups=groups, dropout=dropout
        )
        self.res2net_block = Res2NetBlock(
            out_channels, 
            out_channels, 
            res2net_scale, 
            kernel_size, 
            dilation
        )
        self.tdnn2 = TDNNBlock(
            out_channels, 
            out_channels, 
            kernel_size=1, 
            dilation=1, 
            activation=activation, 
            groups=groups, 
            dropout=dropout
        )
        self.se_block = SEBlock(
            out_channels, 
            se_channels, 
            out_channels
        )
        # Set up a projection linear shortcut if channel dimension sizes mismatch
        self.shortcut = None
        if in_channels != out_channels: 
            self.shortcut = Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=1
            )

    def forward(self, x, lengths=None):
        """Executes full block transformations and returns combined shortcut residuals."""

        residual = x
        if self.shortcut: residual = self.shortcut(x)
        # Pipeline flow: TDNN -> Res2Net -> TDNN -> SE -> Residual Add
        return self.se_block(
            self.tdnn2(self.res2net_block(self.tdnn1(x))), 
            lengths
        ) + residual

class ECAPA_TDNN(torch.nn.Module):
    """
    The complete ECAPA-TDNN architecture tailored for robust speaker embedding extraction

    and audio classification tasks.
    """

    def __init__(
        self, 
        input_size, 
        device="cpu", 
        lin_neurons=192, 
        activation=torch.nn.ReLU, 
        channels=[512, 512, 512, 512, 1536], 
        kernel_sizes=[5, 3, 3, 3, 1], 
        dilations=[1, 2, 3, 4, 1], 
        attention_channels=128, 
        res2net_scale=8, 
        se_channels=128, 
        global_context=True, 
        groups=[1, 1, 1, 1, 1], 
        dropout=0.0
    ):
        """Constructs consecutive layers and multi-layer feature aggregation (MFA) blocks."""

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)

        self.channels = channels
        self.blocks = nn.ModuleList()
        # Layer 1: Initial feature processing TDNN block
        self.blocks.append(
            TDNNBlock(
                input_size, 
                channels[0], 
                kernel_sizes[0], 
                dilations[0], 
                activation, 
                groups[0], 
                dropout
            )
        )

        # Layers 2 to 4: Intermediate multi-scale SERes2Net blocks
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1], 
                    channels[i], 
                    res2net_scale=res2net_scale, 
                    se_channels=se_channels, 
                    kernel_size=kernel_sizes[i], 
                    dilation=dilations[i], 
                    activation=activation, 
                    groups=groups[i], 
                    dropout=dropout
                )
            )

        # Layer 5: Multi-Stage Feature Aggregation block (MFA)
        self.mfa = TDNNBlock(
            channels[-2] * (len(channels) - 2), 
            channels[-1], 
            kernel_sizes[-1], 
            dilations[-1], 
            activation, 
            groups=groups[-1], 
            dropout=dropout
        )
        # Layer 6: Attentive Statistics Pooling (ASP) segment
        self.asp = AttentiveStatisticsPooling(
            channels[-1], 
            attention_channels=attention_channels, 
            global_context=global_context
        )

        # Layer 7: Final linear bottleneck projections and normalizations
        self.asp_bn = BatchNorm1d(
            input_size=channels[-1] * 2
        )
        self.fc = Conv1d(
            in_channels=channels[-1] * 2, 
            out_channels=lin_neurons, 
            kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Processes raw acoustic inputs and yields optimized latent identity representations."""

        x = x.transpose(1, 2)
        xl = []
        # Forward step through foundational layer hierarchies
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x) # Fallback if block does not support sequence length parameters

            xl.append(x)

        return self.fc(
            self.asp_bn(self.asp(self.mfa(torch.cat(xl[1:], dim=1)), lengths=lengths))
        ).transpose(1, 2)

class Classifier(torch.nn.Module):
    """
    Cosine-similarity classifier layer equipped with customizable linear 

    projection blocks and Xavier weight initializations.
    """

    def __init__(
        self, 
        input_size, 
        device="cpu", 
        lin_blocks=0, 
        lin_neurons=192, 
        out_neurons=1211
    ):
        """Constructs projection paths and initializes prediction weight weights."""
        super().__init__()
        self.blocks = nn.ModuleList()
        # Interleave additional linear scaling projections if requested
        for _ in range(lin_blocks):
            self.blocks.extend([
                _BatchNorm1d(input_size=input_size), 
                Linear(input_size=input_size, n_neurons=lin_neurons)
            ])
            input_size = lin_neurons

        # Linear classification matrix initialized via Xavier distribution mapping
        self.weight = nn.Parameter(
            torch.FloatTensor(
                out_neurons, 
                input_size, 
                device=device
            )
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Computes normalized cosine logit scores over feature embeddings."""

        for layer in self.blocks:
            x = layer(x)

        # Apply Cosine similarity metric normalization across activations and targets
        return F.linear(
            F.normalize(x.squeeze(1)), 
            F.normalize(self.weight)
        ).unsqueeze(1)