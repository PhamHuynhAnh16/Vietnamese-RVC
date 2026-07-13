import os
import sys
import torch
import ctypes

# Use these backends only if the library is installed.
try:
    import pytorch_ocl
except:
    pytorch_ocl = None

sys.path.append(os.getcwd())

from main.library.backends.utils import GRU, DeviceProperties

# Global variables tracking OpenCL state and devices
devices = []
torch_available = pytorch_ocl != None
adaptive_orig, softmax_orig, opencl = None, None, None

# Backup original PyTorch operations for future reference or fallbacks
if torch_available: 
    adaptive_orig = torch.nn.AdaptiveAvgPool2d
    softmax_orig = torch.nn.functional.softmax

def get_opencl_lib():
    """
    Loads the system-specific OpenCL dynamic link library (.dll/.so).

    This function updates the global `opencl` variable. It handles cross-platform 
    loading for Windows and Linux architectures.

    Raises:
        OSError: If the OpenCL runtime library cannot be located or loaded on the host system.
    """

    global opencl

    if sys.platform == "win32": 
        try:
            opencl = ctypes.CDLL("OpenCL.dll")
        except:
            raise OSError("Failed to load OpenCL.dll. Ensure OpenCL drivers are installed.")
    elif sys.platform == "linux":
        for path in ["libOpenCL.so", "libOpenCL.so.1", "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1"]:
            try: 
                opencl = ctypes.CDLL(path)
                break
            except OSError: 
                continue

        if not opencl: raise OSError("OpenCL shared library (.so) not found in standard Linux paths.")
    else: return

def get_opencl_device():
    """
    Queries and populates the list of available OpenCL compute devices.

    Uses ctypes to interface directly with the OpenCL runtime API, extracting 
    vendor metadata, device names, board names (for AMD devices), and available VRAM.
    """

    global devices

    # Short-circuit if devices are already cached, torch is missing, or library fails to load
    if len(devices) >= 1: return
    if not torch_available: return
    if opencl is None: get_opencl_lib()

    # Define strict argument signatures for C-functions to prevent memory corruption
    opencl.clGetPlatformIDs.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32)]
    opencl.clGetPlatformInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    opencl.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32)]
    opencl.clGetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]

    # Fetch total number of platforms available
    num_platforms = ctypes.c_uint32(0)
    opencl.clGetPlatformIDs(0, None, ctypes.byref(num_platforms))
    if num_platforms.value == 0: return

    idx = 0
    platforms = (ctypes.c_void_p * num_platforms.value)()
    opencl.clGetPlatformIDs(num_platforms.value, platforms, None)

    # Loop through each discovered platform
    for p in platforms:
        if p is None: continue

        # Query platform Vendor Name (Parameter ID: 0x0903)
        param_size = ctypes.c_size_t(0)
        opencl.clGetPlatformInfo(p, 0x0903, 0, None, ctypes.byref(param_size))
        v_name = ctypes.create_string_buffer(param_size.value)
        opencl.clGetPlatformInfo(p, 0x0903, param_size.value, v_name, None)
        vendor_str = v_name.value.decode('utf-8', errors='ignore').lower()

        # Query devices belonging to this specific platform (0xFFFFFFFF handles CL_DEVICE_TYPE_ALL)
        num_devices = ctypes.c_uint32(0)
        res = opencl.clGetDeviceIDs(p, 0xFFFFFFFF, 0, None, ctypes.byref(num_devices))
        if res != 0 or num_devices.value == 0: continue

        ocl_devices = (ctypes.c_void_p * num_devices.value)()
        opencl.clGetDeviceIDs(p, 0xFFFFFFFF, num_devices.value, ocl_devices, None)

        # Loop through each device on the platform
        for d in ocl_devices:
            if d is None: continue

            # Query individual Device Name (Parameter ID: 0x102B)
            opencl.clGetDeviceInfo(d, 0x102B, 0, None, ctypes.byref(param_size))
            d_name = ctypes.create_string_buffer(param_size.value)
            opencl.clGetDeviceInfo(d, 0x102B, param_size.value, d_name, None)
            d_name = d_name.value.decode('utf-8', errors='ignore').strip()

            # AMD-specific extension to fetch the precise Board Name (Parameter ID: 0x4038)
            if "advanced micro devices" in vendor_str or "amd" in vendor_str:
                try:
                    opencl.clGetDeviceInfo(d, 0x4038, 0, None, ctypes.byref(param_size))

                    if param_size.value > 0:
                        board = ctypes.create_string_buffer(param_size.value)
                        opencl.clGetDeviceInfo(d, 0x4038, param_size.value, board, None)
                        board = board.value.decode('utf-8', errors='ignore').strip()
                        if board: d_name = f"{board} ({d_name})"
                except:
                    pass # Non-critical error fallback if extensions fail

            # Query global device VRAM size in bytes (Parameter ID: 0x101F)
            vram_bytes = ctypes.c_uint64(0)
            opencl.clGetDeviceInfo(d, 0x101F, ctypes.sizeof(vram_bytes), ctypes.byref(vram_bytes), None)

            # Add the GPU descriptor to the device list.
            devices.append(DeviceProperties(idx, d_name, float(vram_bytes.value)))
            idx += 1

def device_count():
    """
    Returns the total number of detected OpenCL devices.

    Returns:
        int: The number of tracked OpenCL devices if PyTorch backend is available, otherwise 0.
    """

    if opencl is None and len(devices) == 0: get_opencl_device()

    return len(devices) if torch_available else 0

def get_device_name(device_id = 0):
    """
    Retrieves the human-readable string name of a specific OpenCL device.

    Args:
        device_id (int, optional): Zero-indexed identifier of the target device. Defaults to 0.

    Returns:
        str: The designated name of the requested compute device.

    Raises:
        RuntimeError: If no hardware devices are registered in the current context.
        ValueError: If the `device_id` falls outside the bounds of discovered devices.
    """

    if opencl is None and len(devices) == 0: get_opencl_device()

    if len(devices) == 0:
        raise RuntimeError("No OpenCL devices found on the system.")

    if device_id >= 0 and device_id < device_count():
        return devices[device_id].name
    else:
        raise ValueError(f"Device ID {device_id} is out of bounds. Available count: {device_count()}.")

def get_device_properties(device_id = 0):
    if opencl is None and len(devices) == 0: get_opencl_device()

    if len(devices) == 0:
        raise RuntimeError("No OpenCL devices found on the system.")

    if device_id >= 0 and device_id < device_count():
        return devices[device_id]
    else:
        raise ValueError(f"Device ID {device_id} is out of bounds. Available count: {device_count()}.")

def is_available():
    """
    Checks whether OpenCL acceleration is active and usable.

    Returns:
        bool: True if at least one device is found and the backend is ready; otherwise False.
    """

    return (device_count() > 0) if torch_available else False

def empty_cache():
    """
    Triggers the underlying PyTorch OpenCL backend memory cache garbage collection.
    Flushes unused allocations to reclaim hardware memory segments.
    """

    if opencl is None and len(devices) == 0: get_opencl_device()

    if torch_available:
        pytorch_ocl.empty_cache()
    else:
        return

def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    """
    Applies Group Normalization over a mini-batch of inputs.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, *).
        num_groups (int): Number of separate groups to split the channels into.
        weight (torch.Tensor, optional): Scale parameter tensor of shape (C). Defaults to None.
        bias (torch.Tensor, optional): Offset parameter tensor of shape (C). Defaults to None.
        eps (float, optional): Value added to the denominator for numerical stability. Defaults to 1e-5.

    Returns:
        torch.Tensor: The normalized tensor with identical dimensions to the input `x`.
    """

    N, C = x.shape[:2]
    assert C % num_groups == 0

    shape = (N, num_groups, C // num_groups) + x.shape[2:]
    x_reshaped = x.view(shape)
    # Calculate mean and variance over the spatial and intra-group channel axis dimensions
    dims = (2,) + tuple(range(3, x_reshaped.dim()))
    mean = x_reshaped.mean(dim=dims, keepdim=True)
    var = x_reshaped.var(dim=dims, keepdim=True, unbiased=False)

    x_norm = (x_reshaped - mean) / (var + eps).sqrt()
    x_norm = x_norm.view_as(x)

    if weight is not None: # Apply scaling transformation weights if supplied
        weight = weight.view(1, C, *([1] * (x.dim() - 2)))
        x_norm = x_norm * weight

    if bias is not None: # Apply positional translation bias adjustments if supplied
        bias = bias.view(1, C, *([1] * (x.dim() - 2)))
        x_norm = x_norm + bias

    return x_norm

def script(f, *_, **__):
    """
    Mock decorator implementation replacing standard JIT compiling scripts.
    """

    f.graph = pytorch_ocl.torch._C.Graph()
    return f

def AdaptiveAvgPool2d(input):
    """
    Wrapper for ``torch.nn.AdaptiveAvgPool2d`` that accepts tuple inputs.

    Args:
        input (Any | tuple): Output size or a tuple containing the output size as its first element.

    Returns:
        torch.nn.AdaptiveAvgPool2d: Configured adaptive average pooling module.
    """

    input = input[0] if isinstance(input, tuple) else input
    return adaptive_orig(input)

def softmax_cpu(input, dim = None, _stacklevel = 3, dtype = None):
    """softmax is executed on the CPU because the OpenCL implementation may produce incorrect results on NVIDIA GPUs and potentially other GPUs as well."""

    return softmax_orig(input.cpu(), dim, _stacklevel, dtype).to(input.device)

# Patch classes and functions to ensure everything operates correctly
if torch_available:
    torch.nn.GRU = GRU # The GRU layer does not work on this backend, so I switched to a GRU Wrapper layer.
    torch.jit.script = script
    torch.inference_mode = torch.no_grad # Using `inference_mode` causes errors in some cases; therefore, I switched to using `no_grad`
    torch.nn.functional.softmax = softmax_cpu # The softmax function was not working correctly on NVIDIA hardware, so it has been replaced.
    torch.nn.functional.group_norm = group_norm # GroupNorm was not working correctly, so I replaced it.
    torch.nn.AdaptiveAvgPool2d = AdaptiveAvgPool2d