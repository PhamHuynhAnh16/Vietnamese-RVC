import torch
import functools

def init_weights(m, mean=0.0, std=0.01):
    """
    Applies standard normal initialization parameters across target convolution layer elements.

    Args:
        m (nn.Module): Layer target instance node context.
        mean (float, optional): Mean initialization factor. Defaults to 0.0.
        std (float, optional): Variance scale standard deviation. Defaults to 0.01.
    """

    if m.__class__.__name__.find("Conv") != -1: m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    """
    Calculates exact causal zero padding width to maintain temporal spatial resolution.

    Args:
        kernel_size (int): Size of the convolution kernel.
        dilation (int, optional): Dilation factor coefficient rate. Defaults to 1.

    Returns:
        int: Required symmetric padding width.
    """

    return int((kernel_size * dilation - dilation) / 2)

def slice_segments(x, ids_str, segment_size = 4, dim = 2):
    """
    Slices segments of a fixed size from a tensor based on given start indices.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T) for dim=2 or (B, C, T) for dim=3.
        ids_str (torch.Tensor): Start indices for each batch element, shape (B,).
        segment_size (int, optional): The length of the sliced segment. Defaults to 4.
        dim (int, optional): Dimension variant support. Must be 2 or 3. Defaults to 2.

    Returns:
        torch.Tensor: The sliced tensor with segment length equal to `segment_size`.
    """

    # Initialize the output tensor template based on the chosen dimension format
    if dim == 2: ret = torch.zeros_like(x[:, :segment_size])
    elif dim == 3: ret = torch.zeros_like(x[:, :, :segment_size]) # dim == 3

    # Slice each batch element individually using the specific start index
    for i in range(x.size(0)):
        idx_str = ids_str[i].item()
        idx_end = idx_str + segment_size

        if dim == 2: ret[i] = x[i, idx_str:idx_end]
        else: ret[i] = x[i, :, idx_str:idx_end]

    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """
    Randomly selects and slices contiguous segments from a batched sequence tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, T).
        x_lengths (torch.Tensor or int, optional): Actual lengths of sequences in the batch. 
            Defaults to the total time length T.
        segment_size (int, optional): Desired segment length. Defaults to 4.

    Returns:
        tuple: (ret, ids_str)
            - ret (torch.Tensor): Randomly sliced segments of shape (B, C, segment_size).
            - ids_str (torch.Tensor): Generated random starting indices of shape (B,).
    """

    b, _, t = x.size()
    if x_lengths is None: x_lengths = t

    # Calculate the upper bound for the random start indices
    ids_str_max = x_lengths - segment_size + 1
    # Generate random start indices uniformly across the valid window range
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    # Perform the slice operation using the 3D tensor path
    ret = slice_segments(x, ids_str, segment_size, dim=3)

    return ret, ids_str

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """
    Gated activation unit combining element-wise addition, Tanh, and Sigmoid layers.

    Designed for TorchScript compilation to achieve high-performance fusion kernels.

    Args:
        input_a (torch.Tensor): First input activation map tensor.
        input_b (torch.Tensor): Second input activation map tensor (e.g., condition/bias).
        n_channels (torch.Tensor): A tensor containing the split channel index at index 0.

    Returns:
        torch.Tensor: Gated multiplied activations.
    """

    n_channels_int = n_channels[0]
    # Fuse element-wise addition to save memory bandwidth
    in_act = input_a + input_b

    # Split channels into main activation branch and gating branch
    t_act = in_act[:, :n_channels_int, :].tanh()
    s_act = in_act[:, n_channels_int:, :].sigmoid()

    # Apply gating mechanism via multiplication
    acts = t_act * s_act
    return acts

def sequence_mask(length, max_length = None):
    """
    Generates a binary mask tensor for batches of variable-length sequences.

    Args:
        length (torch.Tensor): Tensor indicating actual lengths of sequences, shape (B,).
        max_length (int, optional): Upper bound ceiling for the mask grid. Defaults to length.max().

    Returns:
        torch.Tensor: Boolean mask tensor of shape (B, max_length), where valid positions are True.
    """

    if max_length is None: max_length = length.max()
    # Construct mask via broadcasting comparison: [max_length] matrix vs [B, 1] row matrix
    return torch.arange(max_length, dtype=length.dtype, device=length.device).unsqueeze(0) < length.unsqueeze(1)

def clip_grad_value(parameters, clip_value=None, norm_type=2):
    """
    Clips the gradients of an iterable of parameters and returns the accumulated norm.

    Args:
        parameters (Iterable[torch.Tensor] or torch.Tensor): Model weights/parameters.
        clip_value (float, optional): Maximum allowed absolute gradient value. Clips in-place to [-clip_value, clip_value] if provided.
        norm_type (float or int, optional): Type of the p-norm to compute. Defaults to 2.

    Returns:
        float: Total computed p-norm of the parameters' gradients prior to clipping.
    """

    if isinstance(parameters, torch.Tensor): parameters = [parameters]
    norm_type = float(norm_type)

    if clip_value is not None: clip_value = float(clip_value)
    total_norm = 0

    for p in list(filter(lambda p: p.grad is not None, parameters)): # Filter parameter list to extract only those with active gradients
        # Accumulate total gradient norm using step-by-step un-averaged sum powers
        total_norm += (p.grad.data.norm(norm_type)).item() ** norm_type
        # Perform absolute value clamping in-place on target gradient
        if clip_value is not None: p.grad.data.clamp_(min=-clip_value, max=clip_value)

    # Return the global root of total accumulated norms
    return total_norm ** (1.0 / norm_type)

def grad_norm(parameters, norm_type = 2.0):
    """
    Computes the total vector p-norm over gradients of all parameters treated as a single vector.

    Args:
        parameters (Iterable[torch.Tensor] or torch.Tensor): Model weights/parameters.
        norm_type (float or int, optional): Type of the p-norm to compute. Defaults to 2.0.

    Returns:
        float: The calculated global gradient vector norm. Returns 0.0 if no gradients are found.
    """

    if isinstance(parameters, torch.Tensor): parameters = [parameters]
    # Extract active gradient tensors
    parameters = [p for p in parameters if p.grad is not None]

    if not parameters: return 0.0
    # Stack norms of individual parameter tensors and compute overall vector norm safely
    # Uses PyTorch native linalg engine to prevent underflow/overflow scaling issues
    return torch.linalg.vector_norm(torch.stack([p.grad.norm(norm_type) for p in parameters]), ord=norm_type).item()

def rescale_conv(conv, reference):
    """
    Rescales the weights and biases of a convolutional layer based on a reference standard deviation.

    This function normalizes the weight tensor of a convolutional layer so that
    its standard deviation matches a target reference scale, adjusted by a square root.
    It prevents exploding or vanishing gradients during initialization.

    Args:
        conv (torch.nn.modules.conv._ConvNd): The convolutional layer to rescale.
        reference (float): The target reference value (e.g., standard deviation of a baseline layer).
    """

    # Calculate the scaling factor based on the ratio of the current weight's standard deviation to the reference.
    # The square root (** 0.5) is used here to dampen the scaling adjustment.
    scale = (conv.weight.std().detach() / reference) ** 0.5
    # In-place division of weights to adjust the variance/standard deviation
    conv.weight.data /= scale
    # If the layer uses a bias, scale it by the exact same factor to maintain relative magnitude
    if conv.bias is not None: conv.bias.data /= scale

def rescale_module(module, reference):
    """
    Recursively finds and rescales all convolutional layers within a PyTorch module.

    Args:
        module (torch.nn.Module): The root PyTorch module containing sub-modules to rescale.
        reference (float): The target reference standard deviation passed to `rescale_conv`.
    """

    # Iterate through all child and sub-child modules recursively
    for sub in module.modules():
        # Check if the sub-module is a 1D or 2D standard or transposed convolutional layer
        if isinstance(sub, (torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.Conv2d, torch.nn.ConvTranspose2d)): rescale_conv(sub, reference)

def capture_init(init):
    """
    A decorator to capture the positional and keyword arguments passed to a class `__init__` method.

    This is highly useful for serialization, logging, or re-instantiating the object
    later with its exact initial configuration.

    Args:
        init (callable): The original `__init__` method of a class.

    Returns:
        callable: The wrapped `__init__` method that stores arguments before executing.
    """

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        # Store a copy of the initialization arguments as a tuple and dictionary in the instance self
        self._init_args_kwargs = (args, kwargs)
        # Call the original __init__ constructor to proceed with normal object setup
        init(self, *args, **kwargs)

    return __init__