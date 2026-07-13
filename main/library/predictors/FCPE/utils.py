import os
import torch

from torch import nn
from io import BytesIO
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

def decrypt_model(configs, input_path):
    """
    Decrypts a model checkpoint file encrypted using AES-CBC 256/128 encryption.

    Extracts an initialization vector (IV) from the first 16 bytes of the file,
    decrypts the remaining payload, and strips the cryptographic padding.

    Args:
        configs (Dict[str, Any]): Dictionary containing configuration keys, 
            specifically 'binary_path' pointing to the decryption key directory.
        input_path (str | os.PathLike): File path to the encrypted model binary file.

    Returns:
        bytes: Raw decrypted binary buffer ready to be loaded by torch.load.
    """

    # 1. Read the encrypted binary payload from disk
    with open(input_path, "rb") as f:
        data = f.read()

    # 2. Load the raw shared symmetric key matrix configuration
    with open(
        os.path.join(configs["binary_path"], "decrypt.bin"), 
        "rb"
    ) as f:
        key = f.read()

    # 4. Remove standard block padding and return data inside an in-memory stream wrapper
    return BytesIO(
        unpad(
            AES.new(
                key, 
                AES.MODE_CBC, 
                data[:16]
            ).decrypt(data[16:]), 
            AES.block_size
        )
    ).read()

def calc_same_padding(kernel_size):
    """Calculates asymmetric padding size parameters required for 1D convolutions.

    Maintains output sequences of matching lengths across standard 'same' configurations.

    Args:
        kernel_size (int): Size of the 1D convolution filter window.

    Returns:
        Tuple[int, int]: Explicit (left_padding, right_padding) integers context.
    """

    pad = kernel_size // 2
    # Adjust right pad value if the kernel size is an even number
    return (pad, pad - (kernel_size + 1) % 2)

def torch_interp(x, xp, fp):
    """
    Performs 1D piecewise linear interpolation for PyTorch tensors.

    Mimics the behavior of `numpy.interp` natively on the target device context.

    Args:
        x (torch.Tensor): Coordinates at which to evaluate the interpolated values.
        xp (torch.Tensor): 1D sequence of float data points defining the x-coordinates.
        fp (torch.Tensor): 1D sequence of float data points defining the y-coordinates.

    Returns:
        torch.Tensor: Interpolated scalar values of the same shape as `x`.
    """

    # 1. Sort the reference baseline vectors to guarantee monotonically increasing coordinates
    sort_idx = xp.argsort()
    xp = xp[sort_idx]
    fp = fp[sort_idx]

    # 2. Search for index positions using bin boundary conditions
    right_idxs = torch.searchsorted(xp, x).clamp(max=len(xp) - 1)
    left_idxs = (right_idxs - 1).clamp(min=0)
    x_left = xp[left_idxs]
    y_left = fp[left_idxs]

    # 3. Calculate slopes and apply standard linear extrapolation intervals
    # Prevent divide-by-zero errors on flat duplicate coordinate values
    interp_vals = y_left + ((x - x_left) * (fp[right_idxs] - y_left) / (xp[right_idxs] - x_left))
    # 4. Clamp out-of-bounds inputs to boundary endpoints explicitly (constant boundary padding)
    interp_vals[x < xp[0]] = fp[0]
    interp_vals[x > xp[-1]] = fp[-1]

    return interp_vals

class DotDict(dict):
    """
    Custom dictionary wrapper allowing attribute-style dot access notation.

    Enables accessing standard dictionary fields using `obj.key` syntax while 
    recursively handling nested dictionary nodes.
    """

    def __getattr__(*args):
        """Retrieves dictionary attributes, wrapping nested dicts on the fly."""

        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    # Directly map native internal attribute overrides to dictionary methods
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Swish(nn.Module):
    """
    Swish (or SiLU) activation function wrapper module.

    Computes the standard equation element-wise.
    """

    def forward(self, x):
        """Applies the Swish activation function to the input tensor."""

        return x * x.sigmoid()

class Transpose(nn.Module):
    """Utility wrapper for swapping arbitrary dimension pairs in a tensor."""

    def __init__(self, dims):
        """
        Initializes the layer configuration.

        Args:
            dims (Tuple[int, int]): Pair of dimension indices target to transpose.
        """

        super().__init__()
        assert len(dims) == 2
        self.dims = dims

    def forward(self, x):
        """Transposes the input tensor along the pre-configured axis targets."""

        return x.transpose(*self.dims)

class GLU(nn.Module):
    """
    Custom Gated Linear Unit (GLU) implementation block.

    Splits the input tensor along a target axis and applies a Sigmoid gate activation 
    to regulate the output values.
    """

    def __init__(self, dim):
        """
        Initializes the GLU layer.

        Args:
            dim (int): Axis dimension index along which to split and evaluate features.
        """

        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Executes splitting gating calculations."""

        # Split the feature map evenly into content structures and filter channels
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()