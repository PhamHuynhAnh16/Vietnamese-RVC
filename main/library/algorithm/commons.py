import torch

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1: m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def slice_segments(x, ids_str, segment_size = 4, dim = 2):
    if dim == 2: ret = torch.zeros_like(x[:, :segment_size])
    elif dim == 3: ret = torch.zeros_like(x[:, :, :segment_size])

    for i in range(x.size(0)):
        idx_str = ids_str[i].item()
        idx_end = idx_str + segment_size

        if dim == 2: ret[i] = x[i, idx_str:idx_end]
        else: ret[i] = x[i, :, idx_str:idx_end]

    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, _, t = x.size()
    if x_lengths is None: x_lengths = t

    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size, dim=3)

    return ret, ids_str

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b

    t_act = in_act[:, :n_channels_int, :].tanh()
    s_act = in_act[:, n_channels_int:, :].sigmoid()

    acts = t_act * s_act
    return acts

def sequence_mask(length, max_length = None):
    if max_length is None: max_length = length.max()
    return torch.arange(max_length, dtype=length.dtype, device=length.device).unsqueeze(0) < length.unsqueeze(1)

def clip_grad_value(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor): parameters = [parameters]
    norm_type = float(norm_type)

    if clip_value is not None: clip_value = float(clip_value)
    total_norm = 0

    for p in list(filter(lambda p: p.grad is not None, parameters)):
        total_norm += (p.grad.data.norm(norm_type)).item() ** norm_type

        if clip_value is not None: p.grad.data.clamp_(min=-clip_value, max=clip_value)

    return total_norm ** (1.0 / norm_type)

def grad_norm(parameters, norm_type = 2.0):
    if isinstance(parameters, torch.Tensor): parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    if not parameters: return 0.0
    return torch.linalg.vector_norm(torch.stack([p.grad.norm(norm_type) for p in parameters]), ord=norm_type).item()