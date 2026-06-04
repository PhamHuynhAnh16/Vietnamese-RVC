import os
import sys
import math
import torch
import einops
import librosa

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from inspect import isfunction
from abc import abstractmethod
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from scipy.signal import butter, cheby1, cheby2, ellip, bessel, sosfiltfilt, resample_poly

sys.path.append(os.getcwd())

from main.library.audio.features import mel

def align_length(x, y):
    Lx = len(x)
    Ly = len(y)

    if Lx == Ly: return y
    elif Lx > Ly: return np.pad(y, (0, Lx - Ly), mode="constant")
    else: return y[:Lx]

def subsampling(data, lowpass_ratio, fs_ori=44100, upsample_to_original=True):
    assert len(data.shape) == 1

    fs_down = int(lowpass_ratio * fs_ori)
    y = resample_poly(data, fs_down, fs_ori)

    if upsample_to_original:
        y = resample_poly(y, fs_ori, fs_down)
        if len(y) != len(data): y = align_length(data, y)

    return y

def lowpass_filter(x, highcutoff_freq, fs, order, ftype, upsample_to_original = True):
    nyq = 0.5 * fs
    hi = highcutoff_freq / nyq

    if ftype == "butter": sos = butter(order, hi, btype="low", output="sos")
    elif ftype == "cheby1": sos = cheby1(order, 0.1, hi, btype="low", output="sos")
    elif ftype == "cheby2": sos = cheby2(order, 60, hi, btype="low", output="sos")
    elif ftype == "ellip": sos = ellip(order, 0.1, 60, hi, btype="low", output="sos")
    elif ftype == "bessel": sos = bessel(order, hi, btype="low", output="sos")
    else: raise ValueError

    y = sosfiltfilt(sos, x)
    if len(y) != len(x): y = align_length(x, y)

    y = subsampling(y, lowpass_ratio=highcutoff_freq / int(fs / 2), fs_ori=fs, upsample_to_original=upsample_to_original)
    return y

def lowpass(audio, sr, filter_name, filter_order, cutoff_freq, upsample_to_original = True):
    assert len(audio.shape) == 1 or (len(audio.shape) == 2 and (audio.shape[0] == 1 or audio.shape[0] == 2))
    if filter_name == "cheby": filter_name = "cheby1"
    assert filter_order >= 2 and filter_order <= 10
    if cutoff_freq == sr: cutoff_freq -= 1

    if len(audio.shape) == 2:
        lowpassed_audio = np.zeros_like(audio)
    
        for i in range(audio.shape[0]):
            lowpassed_audio[i] = lowpass_filter(
                x=audio[i], 
                highcutoff_freq=int(cutoff_freq), 
                fs=sr, 
                order=filter_order, 
                ftype=filter_name, 
                upsample_to_original=upsample_to_original
            )
    else:
        lowpassed_audio = lowpass_filter(
            x=audio, 
            highcutoff_freq=int(cutoff_freq), 
            fs=sr, 
            order=filter_order, 
            ftype=filter_name, 
            upsample_to_original=upsample_to_original
        )

    if upsample_to_original: assert lowpassed_audio.shape == audio.shape
    return lowpassed_audio.copy()

def find_cutoff(x, percentile=0.95):
    percentile = x[-1] * percentile

    for i in range(1, x.shape[0]):
        if x[-i] < percentile: return x.shape[0] - i

    return 0

def locate_cutoff_freq(stft, percentile=0.985):
    return find_cutoff(stft, percentile)

def find_cutoff_freq(audio):
    device = audio.device
    if not audio.device.type.startswith(("cpu", "cuda", "xpu")): audio = audio.cpu()

    stft_spec = torch.stft(audio, n_fft=2048, hop_length=480, win_length=2048, window=torch.hann_window(2048).to(audio.device), center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    stft_spec = stft_spec[0].T.abs().sum(dim=0).cumsum(dim=0).float()

    cutoff_freq = (locate_cutoff_freq(stft_spec.to(device), percentile=0.983) / 1024) * 24000
    if cutoff_freq < 1000: cutoff_freq = 24000

    return cutoff_freq

def get_diffusers_output_type_name(ddpm_module):
    output_type_dict = {"v_prediction": "v_prediction", "noise": "epsilon", "x_start": "sample"}
    return output_type_dict[ddpm_module.model_output_type]

def get_diffusers_scheduler_config(ddpm_module, scheduler_args):
    config = {"num_train_timesteps": ddpm_module.timesteps, "trained_betas": ddpm_module.betas.to("cpu"), "prediction_type": get_diffusers_output_type_name(ddpm_module)}
    config.update(scheduler_args)

    return config

def freeze_param(model):
    model = model.eval()
    model.train = lambda self: self

    for param in model.parameters():
        param.requires_grad = False

    return model

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = (-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).exp().to(device=timesteps.device)

        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else: embedding = einops.repeat(timesteps, "b -> b d", d=dim)

    return embedding

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def register_buffer(model, variable_name, value, dtype = torch.float32):
    if type(value) != torch.Tensor: value = torch.tensor(value, dtype=dtype)
    model.register_buffer(variable_name, value)
    return getattr(model,variable_name)

def checkpoint(func, inputs, params, flag):
    return CheckpointFunction.apply(func, len(inputs), *(tuple(inputs) + tuple(params))) if flag else func(*inputs)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val): return val
    return d() if isfunction(d) else d

def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1: m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width

    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0

    window = torch.kaiser_window(
        kernel_size, 
        beta=beta, 
        periodic=False
    )

    time = (
        torch.arange(-half_size, half_size) + 0.5
    ) if even else (
        torch.arange(kernel_size) - half_size
    )

    if cutoff == 0:
        filter = torch.zeros_like(time)
    else:
        filter = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        filter /= filter.sum()

    return filter.view(1, 1, kernel_size)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return F.group_norm(
            x,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps
        )

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)

        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)

        input_grads = torch.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, output_grads, allow_unused=True)
        del ctx.input_tensors, ctx.input_params, output_tensors

        return (None, None) + input_grads

class UtilAudioMelSpec:
    def __init__(self, nfft, hop_size, sample_rate, mel_size, frequency_min, frequency_max, device):
        self.nfft = nfft
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.mel_size = mel_size
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max if frequency_max is not None else (sample_rate // 2)
        self.hann_window = torch.hann_window(self.nfft)
        self.mel_frequncies = librosa.mel_frequencies(n_mels=self.mel_size, fmin=self.frequency_min, fmax=self.frequency_max)
        self.mel_basis_tensor = mel(sr=self.sample_rate, n_fft=self.nfft, n_mels=self.mel_size, fmin=self.frequency_min, fmax=self.frequency_max, device=device, dtype=torch.float32)

    def stft_torch(self, audio_torch):
        assert(len(audio_torch.shape) <= 3)
        if (len(audio_torch.shape) == 1): audio_torch = audio_torch.unsqueeze(0)
        shape_is_three = len(audio_torch.shape) == 3

        if shape_is_three:
            batch_size, channels_num, segment_samples = audio_torch.shape
            audio_torch = audio_torch.reshape(batch_size * channels_num, segment_samples)
        
        spec_dict = dict()
        audio_torch = F.pad(audio_torch.unsqueeze(1), (int((self.nfft - self.hop_size) / 2), int((self.nfft - self.hop_size) / 2)), mode='reflect').squeeze(1)

        device = audio_torch.device
        if not audio_torch.device.type.startswith(("cpu", "cuda", "xpu")): audio_torch = audio_torch.cpu()

        spec_dict['stft'] = torch.stft(audio_torch, self.nfft, hop_length=self.hop_size, window=self.hann_window.to(audio_torch.device), center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec_dict['mag'] = spec_dict['stft'].abs().to(device)
        spec_dict['angle'] = spec_dict['stft'].angle().to(device)

        if shape_is_three:
            _, time_steps, freq_bins = spec_dict['stft'].shape
            for feature_name in spec_dict:
                spec_dict[feature_name] = spec_dict[feature_name].reshape(batch_size, channels_num, time_steps, freq_bins)

        return spec_dict

    def get_mel_spec(self, audio):
        while len(audio.shape) < 2: audio = audio.unsqueeze(0)
        return ((self.mel_basis_tensor @ self.stft_torch(audio)["mag"]).clamp(min=1e-5) * 1.0).log()

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0: self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut: self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else: self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x

        h = self.norm1(h)
        h = self.conv1(h * h.sigmoid())

        if temb is not None: h = h + self.temb_proj(temb * temb.sigmoid())[:, :, None, None]

        h = self.norm2(h)
        h = self.conv2(self.dropout(h * h.sigmoid()))

        if self.in_channels != self.out_channels: x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape

        return x + self.proj_out(
            torch.bmm(
                v.reshape(b, c, h * w).contiguous(), 
                F.softmax(
                    torch.bmm(
                        q.reshape(b, c, h * w).contiguous().permute(0, 2, 1).contiguous(), 
                        k.reshape(b, c, h * w).contiguous()
                    ).contiguous() * (int(c) ** (-0.5)), 
                    dim=2
                ).permute(0, 2, 1).contiguous()
            ).contiguous().reshape(b, c, h, w).contiguous()
        )

class _Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv: self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0)) if self.with_conv else F.avg_pool2d(x, kernel_size=2, stride=2)

class _Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv: self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv: x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=[], dropout=0.0, resamp_with_conv=True, in_channels=1, resolution=256, z_channels=16, double_z=True, attn_type="vanilla"):
        super().__init__()
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions: attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = _Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2

            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0: h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if i_level != self.num_resolutions - 1: hs.append(self.down[i_level].downsample(hs[-1]))

        h = self.norm_out(self.mid.block_2(self.mid.attn_1(self.mid.block_1(hs[-1], temb)), temb))
        return self.conv_out(h * h.sigmoid())

class Decoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=[], dropout=0.0, resamp_with_conv=True, resolution=256, z_channels=16, attn_type="vanilla"):
        super().__init__()
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.up = nn.ModuleList()
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions: attn.append(AttnBlock(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                up.upsample = _Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2

            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None

        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z), temb)), temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0: h = self.up[i_level].attn[i_block](h)

            if i_level != 0: h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        return self.conv_out(h * h.sigmoid())

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = (self.logvar).clamp(-30.0, 20.0)
        self.std = (0.5 * self.logvar).exp()

    def sample(self):
        return self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)

class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim=None):
        super().__init__()
        self.encoder = Encoder(ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=[], dropout=0.1, resamp_with_conv=True, in_channels=1, resolution=256, z_channels=16, double_z=True, attn_type="vanilla")
        self.decoder = Decoder(ch=128, out_ch=1, ch_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=[], dropout=0.1, resamp_with_conv=True, resolution=256, z_channels=16, attn_type="vanilla")
        self.quant_conv = nn.Conv2d(2 * 16, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, 16, 1)
        self.embed_dim = embed_dim

    def encode(self, x): 
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        return None

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)

        context = default(context, x)

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = einops.repeat(einops.rearrange(mask, "b ... -> b (...)"), "b j -> (b h) () j", h=h)
            sim.masked_fill_(~(mask == 1), -torch.finfo(sim.dtype).max)

        return self.to_out(einops.rearrange(torch.einsum("b i j, b j d -> b i d", sim.softmax(dim=-1), v), "(b h) n d -> b n (h d)", h=h))

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None):
        return checkpoint(self._forward, (x,), self.parameters(), self.checkpoint) if context is None else checkpoint(self._forward, (x, context, mask), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        return self.ff(self.norm3(x)) + x

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        context_dim = context_dim
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim) for _ in range(depth)])
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None, mask=None):
        _, _, h, w = x.shape

        x_in = x
        x = einops.rearrange(self.proj_in(self.norm(x)), "b c h w -> b (h w) c")

        for block in self.transformer_blocks:
            x = block(x, context=context, mask=mask)

        x = self.proj_out(einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w))
        return x + x_in

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context_list=None, mask_list=None):
        spatial_transformer_id = 0
        context_list = [None] + context_list
        mask_list = [None] + mask_list

        for layer in self:
            if isinstance(layer, TimestepBlock): x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                context, mask = (None, None) if spatial_transformer_id >= len(context_list) else (context_list[spatial_transformer_id], mask_list[spatial_transformer_id])
                x = layer(x, context, mask=mask)
                spatial_transformer_id += 1
            else: x = layer(x)

        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)

        if use_conv:
            conv_fn = nn.Conv2d if dims == 2 else nn.Conv3d
            self.op = conv_fn(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            avg_fn = nn.AvgPool2d if dims == 2 else nn.AvgPool3d
            self.op = avg_fn(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv: 
            conv_fn = nn.Conv2d if dims == 2 else nn.Conv3d
            self.conv = conv_fn(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest") if self.dims == 3 else F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv: x = self.conv(x)

        return x

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(GroupNorm32(32, channels), nn.SiLU(), nn.Conv2d(channels, self.out_channels, 3, padding=1))
        self.updown = up or down

        if up: self.h_upd, self.x_upd = Upsample(channels, False, dims), Upsample(channels, False, dims)
        elif down: self.h_upd, self.x_upd = Downsample(channels, False, dims), Downsample(channels, False, dims)
        else: self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(GroupNorm32(32, self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)))

        if self.out_channels == channels: self.skip_connection = nn.Identity()
        elif use_conv: self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else: self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]

            h = self.h_upd(in_rest(x))
            x = self.x_upd(x)
            h = in_conv(h)
        else: h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_rest(out_norm(h) * (1 + scale) + shift)
        else: h = self.out_layers(h + emb_out)

        return self.skip_connection(x) + h

class AudioSRUnet(nn.Module):
    def __init__(self, image_size = 64, in_channels = 32, model_channels = 128, out_channels = 16, num_res_blocks = 2, attention_resolutions = [8, 4, 2], dropout=0, channel_mult=[1, 2, 3, 5], conv_resample=True, num_classes=None, extra_film_condition_dim=None, use_checkpoint=False, num_heads=-1, num_head_channels=32, num_heads_upsample=-1, use_scale_shift_norm=False, transformer_depth=1, context_dim=None, n_embed=None):
        super().__init__()

        if num_heads_upsample == -1: num_heads_upsample = num_heads
        if num_heads == -1: assert num_head_channels != -1
        if num_head_channels == -1: assert num_heads != -1

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.extra_film_condition_dim = extra_film_condition_dim
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        time_embed_dim = model_channels * 4
        self.use_extra_film_by_concat = self.extra_film_condition_dim is not None
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim))

        if self.num_classes is not None: self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        if self.extra_film_condition_dim is not None: self.film_emb = nn.Linear(self.extra_film_condition_dim, time_embed_dim)
        if context_dim is not None and not isinstance(context_dim, list): context_dim = [context_dim]
        elif context_dim is None: context_dim = [None]

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim if not self.use_extra_film_by_concat else (time_embed_dim * 2), dropout, out_channels=mult * model_channels, dims=2, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = mult * model_channels

                if ds in attention_resolutions:
                    if num_head_channels == -1: dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    dim_head = ch // num_heads
                    layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=None))

                    for context_dim_id in range(len(context_dim)):
                        layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim[context_dim_id]))

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample, dims=2, out_channels=out_ch)))
                ch = out_ch

                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1: dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        dim_head = ch // num_heads
        middle_layers = [ResBlock(ch, time_embed_dim if not self.use_extra_film_by_concat else (time_embed_dim * 2), dropout, dims=2, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,)]
        middle_layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=None))

        for context_dim_id in range(len(context_dim)):
            middle_layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim[context_dim_id]))

        middle_layers.append(ResBlock(ch, time_embed_dim if not self.use_extra_film_by_concat else (time_embed_dim * 2), dropout, dims=2, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        self.middle_block = TimestepEmbedSequential(*middle_layers)

        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim if (not self.use_extra_film_by_concat) else time_embed_dim * 2, dropout, out_channels=model_channels * mult, dims=2, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]

                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1: dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    dim_head = ch // num_heads
                    layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=None))

                    for context_dim_id in range(len(context_dim)):
                        layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim[context_dim_id]))

                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(Upsample(ch, conv_resample, dims=2, out_channels=out_ch))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(GroupNorm32(32, ch), nn.SiLU(), zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)))
        if self.predict_codebook_ids: self.id_predictor = nn.Sequential(GroupNorm32(32, ch), nn.Conv2d(model_channels, n_embed, 1))

    def forward(self, x, timesteps=None, y=None, context_list=list(), context_attn_mask_list=list(), **kwargs):
        x = torch.concat([x, y], dim=1)
        y = None

        assert (y is not None) == (
            self.num_classes is not None or
            self.extra_film_condition_dim is not None
        )
        
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.to(next(self.parameters()).dtype))

        emb = self.time_embed(t_emb)
        if self.use_extra_film_by_concat: emb = torch.cat([emb, self.film_emb(y)], dim=-1)

        h = x.to(next(self.parameters()).dtype)
        for module in self.input_blocks:
            h = module(h, emb, context_list, context_attn_mask_list)
            hs.append(h)

        h = self.middle_block(h, emb, context_list, context_attn_mask_list)
        for module in self.output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb, context_list, context_attn_mask_list)

        h = h.type(x.dtype)
        return self.id_predictor(h) if self.predict_codebook_ids else self.out(h)

class UpSample1d(nn.Module):
    def __init__(
        self, 
        ratio=2, 
        kernel_size=None
    ):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size

        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2

        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, 
            half_width=0.6 / ratio, 
            kernel_size=kernel_size
        )
        self.register_buffer("filter", filter)

    def forward(self, x):
        x = self.ratio * F.conv_transpose1d(
            F.pad(
                x, 
                (self.pad, self.pad), 
                mode="replicate"
            ), 
            self.filter.expand(x.size(1), -1, -1), 
            stride=self.stride, 
            groups=x.size(1)
        )

        return x[..., self.pad_left : -self.pad_right]

class LowPassFilter1d(nn.Module):
    def __init__(
        self, 
        cutoff=0.5, 
        half_width=0.6, 
        stride=1, 
        kernel_size=12
    ):
        super().__init__()
        if cutoff < -0.0 or cutoff > 0.5:
            raise ValueError

        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride

        filter = kaiser_sinc_filter1d(
            cutoff, 
            half_width, 
            kernel_size
        )
        self.register_buffer("filter", filter)

    def forward(self, x):
        return F.conv1d(
            F.pad(
                x, 
                (self.pad_left, self.pad_right), 
                mode="replicate"
            ), 
            self.filter.expand(x.size(1), -1, -1), 
            stride=self.stride, 
            groups=x.size(1)
        )

class DownSample1d(nn.Module):
    def __init__(
        self, 
        ratio=2, 
        kernel_size=None
    ):
        super().__init__()
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size,
        )

    def forward(self, x):
        return self.lowpass(x)

class Activation1d(nn.Module):
    def __init__(self, activation, up_ratio = 2, down_ratio = 2, up_kernel_size = 12, down_kernel_size = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)

        if self.alpha_logscale:
            alpha = alpha.exp()
            beta = beta.exp()

        return x + (1.0 / (beta + self.no_div_by_zero)) * pow((x * alpha).sin(), 2)

class AMPBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), snake_logscale = 'snakebeta'):
        super(AMPBlock1, self).__init__()
        self.convs1 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList([Activation1d(activation=SnakeBeta(channels, alpha_logscale=snake_logscale)) for _ in range(self.num_layers)])

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.activations[::2], self.activations[1::2]):
            x = c2(a2(c1(a1(x)))) + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            if hasattr(l, "parametrizations") and "weight" in l.parametrizations: parametrize.remove_parametrizations(l, "weight", leave_parametrized=True)
            else: remove_weight_norm(l)
        for l in self.convs2:
            if hasattr(l, "parametrizations") and "weight" in l.parametrizations: parametrize.remove_parametrizations(l, "weight", leave_parametrized=True)
            else: remove_weight_norm(l)

class SRVocoder(nn.Module):
    def __init__(self, num_mels = 256, upsample_initial_channel = 1536, resblock_kernel_sizes = [3, 7, 11], resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]], upsample_rates = [10, 6, 2, 2, 2], upsample_kernel_sizes = None, snake_logscale = True):
        super(SRVocoder, self).__init__()
        if upsample_kernel_sizes is None: upsample_kernel_sizes = [upsample_rate * 2 for upsample_rate in upsample_rates]

        self.audio_block = nn.ModuleDict()
        self.audio_block["downsamples"] = nn.ModuleList()
        self.audio_block["emb"] = nn.Conv1d(1, upsample_initial_channel // (2 ** len(upsample_rates)), 7, bias=True, padding=(7 - 1) // 2)

        for i in reversed(range(len(upsample_kernel_sizes))):
            self.audio_block["downsamples"] += [nn.Sequential(nn.Conv1d(upsample_initial_channel // (2 ** (i + 1)), upsample_initial_channel // (2 ** i), upsample_kernel_sizes[i], upsample_rates[i], padding=upsample_rates[i] - (upsample_kernel_sizes[i] % 2 == 0), bias=True), nn.LeakyReLU(negative_slope = 0.1))]

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([weight_norm(nn.ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2))]))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock1(ch, k, d))

        activation_post = SnakeBeta(ch, alpha_logscale=snake_logscale)
        self.activation_post = Activation1d(activation=activation_post)
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

        self.conv_post.apply(init_weights)
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)

    def forward(self, mel_spec, lr_audio):
        audio_emb = self.audio_block["emb"](lr_audio.unsqueeze(1))
        audio_emb_list = [audio_emb]

        for i in range(self.num_upsamples - 1):
            audio_emb = self.audio_block["downsamples"][i](audio_emb)
            audio_emb_list += [audio_emb]

        x = self.conv_pre(mel_spec)

        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x) + audio_emb_list[-1-i]

            xs = None
            for j in range(self.num_kernels):
                if xs is None: xs = self.resblocks[i * self.num_kernels + j](x)
                else: xs += self.resblocks[i * self.num_kernels + j](x)

            x = xs / self.num_kernels

        return {'pred_hr_audio': self.conv_post(self.activation_post(x)).tanh().squeeze(1)}

    def remove_weight_norm(self):
        if hasattr(self.conv2, "parametrizations") and "weight" in self.conv2.parametrizations: parametrize.remove_parametrizations(self.conv2, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv2)

        for l in self.ups:
            for l_i in l:
                if hasattr(l_i, "parametrizations") and "weight" in l_i.parametrizations: parametrize.remove_parametrizations(l_i, "weight", leave_parametrized=True)
                else: remove_weight_norm(l_i)

        for l in self.resblocks:
            l.remove_weight_norm()

        if hasattr(self.conv_pre, "parametrizations") and "weight" in self.conv_pre.parametrizations: parametrize.remove_parametrizations(self.conv_pre, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_pre)
        if hasattr(self.conv_post, "parametrizations") and "weight" in self.conv_post.parametrizations: parametrize.remove_parametrizations(self.conv_post, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_post)

class BetaSchedule:
    @staticmethod
    def linear(timesteps, start=1e-4, end=2e-2):
        return np.linspace(start, end, timesteps)
        
    @staticmethod
    def cosine(timesteps, s=0.008):
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)

        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        return np.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), a_min=0, a_max=0.999)

class DDPM(nn.Module):
    def __init__(self, model_output_type = 'noise', timesteps = 1000, loss_func = F.mse_loss, betas = None, beta_schedule_type = 'cosine', beta_arg_dict = dict(), unconditional_prob = 0, cfg_scale = None):
        super().__init__()
        self.model = AudioSRUnet()
        self.model_output_type = model_output_type
        self.loss_func = loss_func
        self.timesteps = timesteps
        self.set_noise_schedule(betas=betas, beta_schedule_type=beta_schedule_type, beta_arg_dict=beta_arg_dict, timesteps=timesteps)
        self.unconditional_prob = unconditional_prob
        self.cfg_scale = cfg_scale
    
    def set_noise_schedule(self, betas = None, beta_schedule_type = 'linear', beta_arg_dict = dict(), timesteps = 1000):
        if betas is None:
            beta_arg_dict.update({'timesteps':timesteps})
            betas = getattr(BetaSchedule, beta_schedule_type)(**beta_arg_dict)
        
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.betas = register_buffer(model = self, variable_name = 'betas', value = betas)
        self.alphas_cumprod = register_buffer(model = self, variable_name = 'alphas_cumprod', value = alphas_cumprod)
        self.alphas_cumprod_prev = register_buffer(model = self, variable_name = 'alphas_cumprod_prev', value = alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = register_buffer(model = self, variable_name = 'sqrt_alphas_cumprod', value = np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = register_buffer(model = self, variable_name = 'sqrt_one_minus_alphas_cumprod', value = np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = register_buffer(model = self, variable_name = 'log_one_minus_alphas_cumprod', value = np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = register_buffer(model = self, variable_name = 'sqrt_recip_alphas_cumprod', value = np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = register_buffer(model = self, variable_name = 'sqrt_recipm1_alphas_cumprod', value = np.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = register_buffer(model = self, variable_name = 'posterior_variance', value = posterior_variance)
        self.posterior_log_variance_clipped = register_buffer(model = self, variable_name = 'posterior_log_variance_clipped', value = np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = register_buffer(model = self, variable_name = 'posterior_mean_coef1', value = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = register_buffer(model = self, variable_name = 'posterior_mean_coef2', value = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

    def apply_model(self, x, t, cond, is_cond_unpack, cfg_scale = None):
        if cfg_scale is None or cfg_scale == 1.0:
            if cond is None: return self.model(x, t)
            elif is_cond_unpack: return self.model(x, t, **cond)
            else: return self.model(x, t, cond)
        else:
            model_conditioned_output = self.model(x, t, **cond) if is_cond_unpack else self.model(x, t, cond)
            unconditional_conditioning = self.get_unconditional_condition(cond=cond)
            model_unconditioned_output = self.model(x, t, **unconditional_conditioning) if is_cond_unpack else self.model(x, t, unconditional_conditioning)
            return model_unconditioned_output + cfg_scale * (model_conditioned_output - model_unconditioned_output)
    
    def get_unconditional_condition(self, cond = None, cond_shape = None, condition_device = None):
        if cond_shape is None: cond_shape = cond.shape
        if cond is not None and isinstance(cond, torch.Tensor): condition_device = cond.device
        return (-11.4981 + torch.zeros(cond_shape)).to(condition_device)

class DPMSolverMultistepScheduler:
    def __init__(
        self,
        num_train_timesteps,
        trained_betas,
        prediction_type="epsilon",
        solver_order=2,
        **kwargs
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.solver_order = solver_order

        betas = trained_betas.float()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.alphas_cumprod = alphas_cumprod
        self.alpha_t = alphas_cumprod.sqrt()
        self.sigma_t = (1 - alphas_cumprod).sqrt()

        self.model_outputs = []
        self.timesteps = None

    def set_timesteps(self, num_inference_steps, device=None):
        timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
            dtype=torch.long,
            device=device
        )

        self.timesteps = timesteps
        self.model_outputs = []

    def scale_model_input(self, sample, timestep):
        return sample

    def convert_model_output(self, model_output, sample, timestep):
        alpha_t = self.alpha_t[timestep].to(sample.device)
        sigma_t = self.sigma_t[timestep].to(sample.device)

        while alpha_t.ndim < sample.ndim:
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        if self.prediction_type == "epsilon": x0 = (sample - sigma_t * model_output) / alpha_t
        elif self.prediction_type == "sample": x0 = model_output
        elif self.prediction_type == "v_prediction": x0 = alpha_t * sample - sigma_t * model_output
        else: raise ValueError

        return x0

    def dpm_solver_first_order_update(
        self,
        model_output,
        sample,
        timestep,
        prev_timestep
    ):
        alpha_t = self.alpha_t[timestep].to(sample.device)
        sigma_t = self.sigma_t[timestep].to(sample.device)

        alpha_s = self.alpha_t[prev_timestep].to(sample.device)
        sigma_s = self.sigma_t[prev_timestep].to(sample.device)

        while alpha_t.ndim < sample.ndim:
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
            alpha_s = alpha_s.unsqueeze(-1)
            sigma_s = sigma_s.unsqueeze(-1)

        return (
            alpha_s * model_output +
            sigma_s * (
                sample - alpha_t * model_output
            ) / sigma_t
        )

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list,
        sample,
        timestep,
        prev_timestep
    ):
        m0 = model_output_list[-1]
        m1 = model_output_list[-2]

        alpha_t = self.alpha_t[timestep]
        sigma_t = self.sigma_t[timestep]

        alpha_s = self.alpha_t[prev_timestep]
        sigma_s = self.sigma_t[prev_timestep]

        while alpha_t.ndim < sample.ndim:
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
            alpha_s = alpha_s.unsqueeze(-1)
            sigma_s = sigma_s.unsqueeze(-1)

        D0 = m0
        D1 = m0 - m1

        x0 = (
            alpha_s * D0 +
            sigma_s * (
                sample - alpha_t * D0
            ) / sigma_t
        )

        return x0 + 0.5 * D1

    def step(
        self,
        model_output,
        timestep,
        sample,
        return_dict=False
    ):
        step_index = (self.timesteps == timestep).nonzero()[0].item()
        prev_timestep = 0 if (step_index == len(self.timesteps) - 1) else self.timesteps[step_index + 1]

        model_output = self.convert_model_output(
            model_output,
            sample,
            timestep
        )

        self.model_outputs.append(model_output)
        if len(self.model_outputs) > self.solver_order: self.model_outputs.pop(0)

        if len(self.model_outputs) == 1:
            prev_sample = self.dpm_solver_first_order_update(
                model_output,
                sample,
                timestep,
                prev_timestep
            )
        else:
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs,
                sample,
                timestep,
                prev_timestep
            )

        if return_dict:
            return {"prev_sample": prev_sample}

        return (prev_sample,)

class FlashSR:
    def __init__(self, model_path, scale_factor_z = 0.3342, model_output_type = 'v_prediction', beta_schedule_type = 'cosine', device = None, is_half = False, **kwargs):
        super().__init__()
        self.scale_factor_z = scale_factor_z
        self.device = device
        self.is_half = is_half

        self.sr_vocoder = SRVocoder()
        self.autoencoder = AutoencoderKL(embed_dim=16)
        self.ddpm = DDPM(model_output_type=model_output_type, beta_schedule_type=beta_schedule_type, **kwargs)
        self.util_mel_spec = UtilAudioMelSpec(nfft=2048, hop_size=480, sample_rate=48000, mel_size=256, frequency_min=20, frequency_max=24000, device=device)

        model = torch.load(model_path, map_location="cpu", weights_only=True)

        self.ddpm.load_state_dict(model["ldm"])
        self.sr_vocoder.load_state_dict(model["vocoder"])
        self.autoencoder.load_state_dict(model["vae"])
        self.autoencoder = freeze_param(self.autoencoder)

        self.ddpm.to(device).eval()
        self.sr_vocoder.to(device).eval()
        self.autoencoder.to(device).eval()

        self.ddpm.to(torch.float16 if is_half else torch.float32)
        self.sr_vocoder.to(torch.float16 if is_half else torch.float32)
        self.autoencoder.to(torch.float16 if is_half else torch.float32)

    def upscaler(self, lr_audio, num_steps = 1, lowpass_input = True, lowpass_cutoff_freq = None):
        if lowpass_input:
            if lowpass_cutoff_freq is None: lowpass_cutoff_freq = find_cutoff_freq(lr_audio)

            lr_audio = torch.from_numpy(
                lowpass(
                    lr_audio.cpu().numpy(), 
                    48000, 
                    filter_name='cheby', 
                    filter_order=8, 
                    cutoff_freq=lowpass_cutoff_freq
                )
            ).to(lr_audio.device)

        with torch.no_grad():
            pred_hr_audio = self.infer(
                lr_audio,
                is_cond_unpack=False,
                num_steps=num_steps,
                device=lr_audio.device
            )

            return pred_hr_audio[...,:lr_audio.shape[-1]]
    
    def infer(self, cond, is_cond_unpack = False, num_steps = 20, scheduler_args = {'timestep_spacing': 'trailing'}, cfg_scale = None, device = None):
        noise_scheduler = DPMSolverMultistepScheduler(**get_diffusers_scheduler_config(self.ddpm, scheduler_args))
        _, cond, additional_data_dict = self.preprocess(x_start=None, cond=cond)

        x_shape = cond.shape
        noise_scheduler.set_timesteps(num_steps)

        x = torch.randn(x_shape, device=device) * 1.0

        for t in noise_scheduler.timesteps:
            denoiser_input = noise_scheduler.scale_model_input(x, t)
            denoiser_input, cond = denoiser_input.to(torch.float16 if self.is_half else torch.float32), cond.to(torch.float16 if self.is_half else torch.float32)

            model_output = self.ddpm.apply_model(denoiser_input, torch.full((x_shape[0],), t, device=device, dtype=torch.long), cond, is_cond_unpack, cfg_scale=self.ddpm.cfg_scale if cfg_scale is None else cfg_scale)
            x = noise_scheduler.step(model_output, t, x, return_dict=False)[0]
        
        return self.postprocess(x, additional_data_dict)

    def preprocess(self, x_start, cond = None):
        x_dict = dict()
        cond_dict = self.encode_to_z(cond)

        if x_start is not None:
            x_dict = self.encode_to_z(
                x_start, 
                scale_dict={
                    'mean_scale_factor': cond_dict['mean_scale_factor'], 
                    'var_scale_factor': cond_dict['var_scale_factor']
                }
            )

        return x_dict.get('z', None), cond_dict['z'], cond_dict

    def postprocess(self, x, additional_data_dict):
        return self.denormalize_wav(
            self.sr_vocoder(
                self.z_to_mel(x).squeeze(1).transpose(1, 2), 
                additional_data_dict['norm_wav']
            )['pred_hr_audio'], 
            additional_data_dict
        )
    
    def denormalize_wav(self, waveform, scale_dict):
        return ((waveform * 2.0) * (scale_dict['var_scale_factor'] + 1e-8)) + scale_dict['mean_scale_factor']

    @torch.no_grad()
    def encode_to_z(self, audio, normalize = True, scale_dict = None,):
        assert len(audio.shape) == 2
        result_dict = {'wav': audio}

        if normalize: 
            audio, scale_dict = self.normalize_wav(audio, scale_dict=scale_dict)
            result_dict['norm_wav'] = audio
            result_dict.update(scale_dict)

        mel_spec = self.audio_to_mel(audio).to(torch.float16 if self.is_half else torch.float32)
        result_dict['mel_spec'] = mel_spec
        result_dict['z'] = self.autoencoder.encode(mel_spec).sample() * self.scale_factor_z

        return result_dict
        
    def normalize_wav(self, waveform, scale_dict):
        mean_scale_factor = waveform.mean(dim=1, keepdim=True) if scale_dict is None else scale_dict['mean_scale_factor']
        waveform -= mean_scale_factor

        var_scale_factor = waveform.abs().max(dim=1, keepdim=True)[0] if scale_dict is None else scale_dict['var_scale_factor']
        waveform /= (var_scale_factor + 1e-8)

        return waveform * 0.5, {'mean_scale_factor': mean_scale_factor, 'var_scale_factor': var_scale_factor}
    
    @torch.no_grad()
    def audio_to_mel(self, audio):
        mel_spec = self.util_mel_spec.get_mel_spec(audio).to(self.device)
        if len(mel_spec.shape) == 3: mel_spec = mel_spec.unsqueeze(1)
        return mel_spec.permute(0, 1, 3, 2)
    
    def z_to_mel(self, z): 
        with torch.no_grad():
            x = (1.0 / self.scale_factor_z) * z
            return self.autoencoder.decode(x.to(torch.float16 if self.is_half else torch.float32))

def get_window(window_size, fade_size):
    window = torch.ones(window_size)
    window[-fade_size:] *= torch.linspace(1, 0, fade_size)
    window[:fade_size] *= torch.linspace(0, 1, fade_size)
    return window

def upscaler(audio, model, pbar = None, device = "cpu", C = 245760, step = 122880, fade_size = 24576, border = 122880):
    if not torch.is_tensor(audio): audio = torch.from_numpy(audio.flatten()).to(device).float()

    if len(audio.shape) == 1: audio = audio.unsqueeze(0)
    if audio.shape[1] > 2 * border and (border > 0): audio = F.pad(audio, (border, border), mode="reflect")

    result = torch.zeros((1,) + tuple(audio.shape), device=audio.device, dtype=torch.float32)
    counter = torch.zeros((1,) + tuple(audio.shape), device=audio.device, dtype=torch.float32)

    i = 0
    pbar.total = pbar.total + math.ceil(audio.size(1) / step)
    pbar.refresh()

    while i < audio.shape[1]:
        part = audio[:, i : i + C]
        length = part.shape[-1]
        if length < C: part = F.pad(part, pad=(0, C - length), mode="reflect") if length > C // 2 + 1 else F.pad(part, pad=(0, C - length, 0, 0), mode="constant", value=0.0)

        out = model.upscaler(part, lowpass_input=True)
        window = get_window(C, fade_size).to(audio.device)

        if i == 0: window[:fade_size] = 1
        elif i + C >= audio.shape[1]: window[-fade_size:] = 1

        result[..., i : i + length] += out[..., :length] * window[..., :length]
        counter[..., i : i + length] += window[..., :length]

        i += step
        pbar.update(1)

    final_output = (result / counter).squeeze(0)
    if audio.shape[1] > 2 * border and (border > 0): final_output = final_output[..., border:-border]

    return final_output.float().flatten().cpu().detach().numpy()