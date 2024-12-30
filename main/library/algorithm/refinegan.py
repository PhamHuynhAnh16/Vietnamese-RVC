import torch
import numpy as np
import torch.utils.checkpoint as checkpoint

from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from .commons import get_padding

class ResBlock(torch.nn.Module):
    def __init__(self, *, in_channels, out_channels, kernel_size = 7, dilation = (1, 3, 5), leaky_relu_slope = 0.2):
        super(ResBlock, self).__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convs1 = torch.nn.ModuleList([weight_norm(torch.nn.Conv1d(in_channels=in_channels if idx == 0 else out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d))) for idx, d in enumerate(dilation)])
        self.convs1.apply(self.init_weights)
        self.convs2 = torch.nn.ModuleList([weight_norm(torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d))) for _, d in enumerate(dilation)])
        self.convs2.apply(self.init_weights)

    def forward(self, x):
        for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            xt = c2(torch.nn.functional.leaky_relu(c1(torch.nn.functional.leaky_relu(x, self.leaky_relu_slope)), self.leaky_relu_slope))
            x = (xt + x) if idx != 0 or self.in_channels == self.out_channels else xt
        return x

    def remove_parametrizations(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_parametrizations(c1)
            remove_parametrizations(c2)

    def init_weights(self, m):
        if type(m) == torch.nn.Conv1d:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.fill_(0.0)

class AdaIN(torch.nn.Module):
    def __init__(self, *, channels, leaky_relu_slope = 0.2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(channels))
        self.activation = torch.nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x):
        return self.activation(x + (torch.randn_like(x) * self.weight[None, :, None]))

class ParallelResBlock(torch.nn.Module):
    def __init__(self, *, in_channels, out_channels, kernel_sizes = (3, 7, 11), dilation = (1, 3, 5), leaky_relu_slope = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=3)
        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(AdaIN(channels=out_channels), ResBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, leaky_relu_slope=leaky_relu_slope), AdaIN(channels=out_channels)) for kernel_size in kernel_sizes])

    def forward(self, x):
        x = self.input_conv(x)
        return torch.mean(torch.stack([block(x) for block in self.blocks]), dim=0)

    def remove_parametrizations(self):
        for block in self.blocks:
            block[1].remove_parametrizations()

class SineGenerator(torch.nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        return torch.ones_like(f0) * (f0 > self.voiced_threshold)

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        return torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]

            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            uv = self._f02uv(f0)
            sine_waves = sine_waves * uv + ((uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves)) * (1 - uv)
        return sine_waves

class SourceModuleHnNSF(torch.nn.Module):
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGenerator(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        return self.l_tanh(self.l_linear(self.l_sin_gen(x).to(dtype=self.l_linear.weight.dtype)))

class RefineGANGenerator(torch.nn.Module):
    def __init__(self, *, sample_rate = 44100, downsample_rates = (2, 2, 8, 8), upsample_rates = (8, 8, 2, 2), leaky_relu_slope = 0.2, num_mels = 128, start_channels = 16, gin_channels = 256, checkpointing = False):
        super().__init__()
        self.downsample_rates = downsample_rates
        self.upsample_rates = upsample_rates
        self.checkpointing = checkpointing
        self.leaky_relu_slope = leaky_relu_slope
        self.f0_upsample = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num=8)
        self.source_conv = weight_norm(torch.nn.Conv1d(in_channels=1, out_channels=start_channels, kernel_size=7, stride=1, padding=3))
        channels = start_channels
        self.downsample_blocks = torch.nn.ModuleList([])

        for rate in downsample_rates:
            new_channels = channels * 2
            self.downsample_blocks.append(torch.nn.Sequential(torch.nn.Upsample(scale_factor=1 / rate, mode="linear"), ResBlock(in_channels=channels, out_channels=new_channels, kernel_size=7, dilation=(1, 3, 5), leaky_relu_slope=leaky_relu_slope)))
            channels = new_channels

        self.mel_conv = weight_norm(torch.nn.Conv1d(in_channels=num_mels, out_channels=channels, kernel_size=7, stride=1, padding=3))
        if gin_channels != 0: self.cond = torch.nn.Conv1d(256, channels, 1)

        channels *= 2
        self.upsample_blocks = torch.nn.ModuleList([])
        self.upsample_conv_blocks = torch.nn.ModuleList([])

        for rate in upsample_rates:
            new_channels = channels // 2
            self.upsample_blocks.append(torch.nn.Upsample(scale_factor=rate, mode="linear"))
            self.upsample_conv_blocks.append(ParallelResBlock(in_channels=channels + channels // 4, out_channels=new_channels, kernel_sizes=(3, 7, 11), dilation=(1, 3, 5), leaky_relu_slope=leaky_relu_slope))
            channels = new_channels

        self.conv_post = weight_norm(torch.nn.Conv1d(in_channels=channels, out_channels=1, kernel_size=7, stride=1, padding=3))

    def forward(self, mel, f0, g = None):
        har_source = self.m_source(self.f0_upsample(f0[:, None, :]).transpose(-1, -2)).transpose(-1, -2)
        x = self.source_conv(har_source)
        downs = []

        for _, block in enumerate(self.downsample_blocks):
            x = torch.nn.functional.leaky_relu(x, self.leaky_relu_slope, inplace=True)
            downs.append(x)
            x = checkpoint.checkpoint(block, x, use_reentrant=False) if self.training and self.checkpointing else block(x)

        mel = self.mel_conv(mel)
        if g is not None: x = x + self.cond(g)

        x = torch.cat([x, mel], dim=1)
        for ups, res, down in zip(self.upsample_blocks, self.upsample_conv_blocks, reversed(downs)):
            x = torch.nn.functional.leaky_relu(x, self.leaky_relu_slope, inplace=True)
            x = checkpoint.checkpoint(res, torch.cat([checkpoint.checkpoint(ups, x, use_reentrant=False), down], dim=1), use_reentrant=False) if self.training and self.checkpointing else res(torch.cat([ups(x), down], dim=1))
        return torch.tanh(self.conv_post(torch.nn.functional.leaky_relu(x, self.leaky_relu_slope, inplace=True)))

    def remove_parametrizations(self):
        remove_parametrizations(self.source_conv)
        remove_parametrizations(self.mel_conv)
        remove_parametrizations(self.conv_post)
        for block in self.downsample_blocks:
            block[1].remove_parametrizations()
        for block in self.upsample_conv_blocks:
            block.remove_parametrizations()