import torch

import torch.nn.functional as F

class LayerNorm(torch.nn.Module):
    """
    Custom Layer Normalization module designed for channel-first tensors.
    It dynamically selects between a standard PyTorch execution path and an 
    ONNX-friendly path with static shape definitions to facilitate model export.
    """

    def __init__(self, channels, eps=1e-5, onnx=False):
        """
        Initializes the LayerNorm module and sets the appropriate forward path.

        Args:
            channels (int): Expected number of channels in the input tensor.
            eps (float): A value added to the denominator for numerical stability. Defaults to 1e-5.
            onnx (bool): If True, configures the module to use static shape constraints for ONNX export.
        """

        super().__init__()
        self.channels = channels
        self.eps = eps
        # Learnable scale (gamma) and shift (beta) parameters initialized to 1s and 0s
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))
        # Dynamic forward routing based on targeted runtime compiler environment
        self.forward = self._forward_onnx if onnx else self._forward_torch

    def _forward_onnx(self, x):
        """
        ONNX-compatible forward path using a explicit static channel length constraint.
        """

        # Swap axes to channel-last representation
        x = x.transpose(1, -1)

        # Normalize across the fixed static channel dimension parameter
        return F.layer_norm(
            x, 
            (self.channels,), 
            self.gamma, 
            self.beta, 
            self.eps
        ).transpose(1, -1) # Restore axis format back

    def _forward_torch(self, x):
        """
        Standard PyTorch forward path tracking shape dimensions dynamically.
        """
    
        # Swap axes to channel-last representation
        x = x.transpose(1, -1)

        # Normalize across the runtime-evaluated dynamic dimension length
        return F.layer_norm(
            x, 
            (x.size(-1),), 
            self.gamma, 
            self.beta, 
            self.eps
        ).transpose(1, -1) # Restore axis format back

class Fp32LayerNorm(torch.nn.LayerNorm):
    """
    A Layer Normalization wrapper that forces operations to execute in Float32 precision.
    Prevents underflow/overflow artifacts when running models under low-precision mix (AMP/FP16).
    """

    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        """Initializes the base torch.nn.LayerNorm wrapper module."""

        super().__init__(*args, **kwargs)

    def forward(self, input):
        """
        Casts inputs to full FP32 resolution before evaluating layer norm criteria.

        Args:
            input (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Normalized tensor matched back to the input's original data type.
        """

        # Upscale weights, biases, and activation features to 32-bit floating point matrix maps
        output = F.layer_norm(
            input.float(), 
            self.normalized_shape, 
            self.weight.float() if self.weight is not None else None, 
            self.bias.float() if self.bias is not None else None, 
            self.eps
        )

        # Downscale precision state safely back to the tensor format passed by the caller
        return output.type_as(input)

class Fp32GroupNorm(torch.nn.GroupNorm):
    """
    A Group Normalization wrapper that forces operations to execute in Float32 precision.
    Ensures mathematical stability across separate split feature sub-channels under mixed-precision.
    """

    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        """Initializes the base torch.nn.GroupNorm wrapper module."""

        super().__init__(*args, **kwargs)

    def forward(self, input):
        """
        Casts inputs to full FP32 resolution before evaluating group norm criteria.

        Args:
            input (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Group-normalized tensor recast back to the input's native format.
        """

        # Execute Group Normalization calculations at 32-bit resolution to protect scaling logic
        output = F.group_norm(
            input.float(), 
            self.num_groups, 
            self.weight.float() if self.weight is not None else None, 
            self.bias.float() if self.bias is not None else None, 
            self.eps
        )

        return output.type_as(input)

class Fp32GroupNormTranspose(torch.nn.GroupNorm):
    """
    A specialized Group Normalization wrapper executing at Float32 precision 
    tailored specifically for channel-last sequence formats.
    Swap the internal dimensions to comply with PyTorch's default GroupNorm constraint.
    """

    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        """Initializes the underlying GroupNorm module infrastructure."""

        super().__init__(
            *args, 
            **kwargs
        )

    def forward(self, input):
        """
        Transposes input sequence steps, computes group norms in FP32, and restores original axes.

        Args:
            input (torch.Tensor): Channel-last sequence tensor.

        Returns:
            torch.Tensor: Transposed and normalized sequence tensor matching shape.
        """

        # Rearrange layout to match standard channel-first expectations
        input = input.transpose(1, 2)

        # Cast tensors to full 32-bit float mappings and run Group Norm functions
        output = F.group_norm(
            input.float(), 
            self.num_groups, 
            self.weight.float() if self.weight is not None else None, 
            self.bias.float() if self.bias is not None else None, 
            self.eps
        ).transpose(1, 2) # Revert output structures back to original axis layout

        return output.type_as(input)

class LayerScale(torch.nn.Module):
    def __init__(
        self, 
        channels, 
        init = 0, 
        channel_last=False
    ):
        super().__init__()
        self.channel_last = channel_last
        self.scale = torch.nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last: return self.scale * x
        else: return self.scale[:, None] * x