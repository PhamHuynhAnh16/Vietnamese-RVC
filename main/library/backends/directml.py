import os
import gc
import sys
import torch
import subprocess

sys.path.append(os.getcwd())

from main.library.embedders import fairseq
from main.library.backends.utils import GRU

try:
    import torch_directml
except:
    torch_directml = None

torch_available = torch_directml != None

def device_count():
    return torch_directml.device_count() if torch_available else 0

def device_name(device_id = 0):
    return torch_directml.device_name(device_id) if torch_available else ""

def is_available():
    return torch_directml.is_available() if torch_available else False

def empty_cache():
    empty_cache_path = os.path.join(
        "main", 
        "library", 
        "backends", 
        "dml_empty_cache", 
        "empty_cache.exe"
    )

    if torch_available and os.path.exists(empty_cache_path):
        subprocess.run([empty_cache_path], capture_output=True, text=True)
        gc.collect()

def forward_dml(ctx, x, scale):
    ctx.scale = scale
    res = x.clone().detach()
    return res

orig_conv_transpose1d = torch.nn.functional.conv_transpose1d

def cpu_conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    return orig_conv_transpose1d(
        input.cpu(), 
        weight.cpu(), 
        bias.cpu() if bias is not None else bias, 
        stride, 
        padding, 
        output_padding, 
        groups, 
        dilation
    ).to(input.device)

if torch_available: 
    torch.nn.GRU = GRU
    fairseq.GradMultiply.forward = forward_dml
    torch.nn.functional.conv_transpose1d = cpu_conv_transpose1d