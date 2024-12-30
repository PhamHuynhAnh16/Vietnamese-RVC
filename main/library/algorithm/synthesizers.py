import os
import sys
import math
import torch

import torch.utils.checkpoint as checkpoint

from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

from .modules import WaveNet
from .residuals import ResidualCouplingBlock, ResBlock, LRELU_SLOPE
from .commons import init_weights, slice_segments, rand_slice_segments, sequence_mask, convert_pad_shape

class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups_and_resblocks = torch.nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups_and_resblocks.append(weight_norm(torch.nn.ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2)))
            ch = upsample_initial_channel // (2 ** (i + 1))

            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.ups_and_resblocks.append(ResBlock(ch, k, d))

        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups_and_resblocks.apply(init_weights)
        if gin_channels != 0: self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        def forward(self, x, g = None):
            x = self.conv_pre(x)
            if g is not None: x = x + self.cond(g)
            resblock_idx = 0

            for _ in range(self.num_upsamples):
                x = self.ups_and_resblocks[resblock_idx](torch.nn.functional.leaky_relu(x, LRELU_SLOPE))
                resblock_idx += 1
                xs = 0

                for _ in range(self.num_kernels):
                    xs += self.ups_and_resblocks[resblock_idx](x)
                    resblock_idx += 1

                x = xs / self.num_kernels

            return torch.tanh(self.conv_post(torch.nn.functional.leaky_relu(x)))

    def __prepare_scriptable__(self):
        for l in self.ups_and_resblocks:
            for hook in l._forward_pre_hooks.values():
                if (hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm"): torch.nn.utils.remove_weight_norm(l)

        return self
    
    def remove_weight_norm(self):
        for l in self.ups_and_resblocks:
            remove_weight_norm(l)

class SineGen(torch.nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0, flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sample_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        return torch.ones_like(f0) * (f0 > self.voiced_threshold)

    def forward(self, f0, upp):
        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2)
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            f0_buf[:, :, 1:] = (f0_buf[:, :, 0:1] * torch.arange(2, self.harmonic_num + 2, device=f0.device)[None, None, :])
            rad_values = (f0_buf / float(self.sample_rate)) % 1
            rand_ini = torch.rand(f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device)
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
            tmp_over_one = torch.cumsum(rad_values, 1)
            tmp_over_one *= upp
            tmp_over_one = torch.nn.functional.interpolate(tmp_over_one.transpose(2, 1), scale_factor=float(upp), mode="linear", align_corners=True).transpose(2, 1)
            rad_values = torch.nn.functional.interpolate(rad_values.transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
            tmp_over_one %= 1
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = ((tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0) * -1.0
            sine_waves = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * torch.pi) * self.sine_amp
            uv = torch.nn.functional.interpolate(self._f02uv(f0).transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
            sine_waves = sine_waves * uv + ((uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves))
            
        return sine_waves

class SourceModuleHnNSF(torch.nn.Module):
    def __init__(self, sample_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upsample_factor = 1):
        return self.l_tanh(self.l_linear(self.l_sin_gen(x, upsample_factor).to(dtype=self.l_linear.weight.dtype)))

class GeneratorNSF(torch.nn.Module):
    def __init__(self, initial_channel, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels, sr, checkpointing = False):
        super(GeneratorNSF, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.checkpointing = checkpointing
        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()
        channels = [upsample_initial_channel // (2 ** (i + 1)) for i in range(len(upsample_rates))]
        stride_f0s = [math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1 for i in range(len(upsample_rates))]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(torch.nn.ConvTranspose1d(upsample_initial_channel // (2**i), channels[i], k, u, padding=(k - u) // 2)))
            self.noise_convs.append(torch.nn.Conv1d(1, channels[i], kernel_size=(stride_f0s[i] * 2 if stride_f0s[i] > 1 else 1), stride=stride_f0s[i], padding=(stride_f0s[i] // 2 if stride_f0s[i] > 1 else 0)))

        self.resblocks = torch.nn.ModuleList([ResBlock(channels[i], k, d) for i in range(len(self.ups)) for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)])
        self.conv_post = torch.nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if gin_channels != 0: self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x, f0, g = None):
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None: x = x + self.cond(g)

        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            x = checkpoint.checkpoint(ups, x, use_reentrant=False) if self.training and self.checkpointing else ups(x)      
            x += noise_convs(har_source)

            def resblock_forward(x, blocks):
                return sum(block(x) for block in blocks) / len(blocks)
            
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            x = checkpoint.checkpoint(resblock_forward, x, blocks, use_reentrant=False)if self.training and self.checkpointing else resblock_forward(x, blocks)

        return torch.tanh(self.conv_post(torch.nn.functional.leaky_relu(x)))

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)

        for l in self.resblocks:
            l.remove_weight_norm()

class LayerNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        return torch.nn.functional.layer_norm(x, (x.size(-1),), self.gamma, self.beta, self.eps).transpose(1, -1)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0, window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None
        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = torch.nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = torch.nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert (t_s == t_t), "(t_s == t_t)"
            scores = scores + self._relative_position_to_absolute_position(self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), self._get_relative_embeddings(self.emb_rel_k, t_s)))

        if self.proximal_bias:
            assert t_s == t_t, "t_s == t_t"
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert (t_s == t_t), "(t_s == t_t)"
                scores = scores.masked_fill((torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)) == 0, -1e4)

        p_attn = self.drop(torch.nn.functional.softmax(scores, dim=-1)  )
        output = torch.matmul(p_attn, value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3))

        if self.window_size is not None: output = output + self._matmul_with_relative_values(self._absolute_position_to_relative_position(p_attn), self._get_relative_embeddings(self.emb_rel_v, t_s))
        return (output.transpose(2, 3).contiguous().view(b, d, t_t)), p_attn

    def _matmul_with_relative_values(self, x, y):
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_with_relative_keys(self, x, y):
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        return (torch.nn.functional.pad(relative_embeddings, convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]])) if pad_length > 0 else relative_embeddings)[:, slice_start_position:(slice_start_position + 2 * length - 1)]

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        return torch.nn.functional.pad(torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]])).view([batch, heads, length * 2 * length]), convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])).view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        return torch.nn.functional.pad(torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])).view([batch, heads, length**2 + length * (length - 1)]), convert_pad_shape([[0, 0], [0, 0], [length, 0]])).view([batch, heads, length, 2 * length])[:, :, :, 1:]

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs((torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)))), 0), 0)

class FFN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0, activation=None, causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.padding = self._causal_padding if causal else self._same_padding
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        return self.conv_2(self.padding(self.drop(((x * torch.sigmoid(1.702 * x)) if self.activation == "gelu" else torch.relu(x))) * x_mask)) * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1: return x
        return torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [(self.kernel_size - 1), 0]]))

    def _same_padding(self, x):
        if self.kernel_size == 1: return x
        return torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [((self.kernel_size - 1) // 2), (self.kernel_size // 2)]]))

class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, window_size=10, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            x = self.norm_layers_1[i](x + self.drop(self.attn_layers[i](x, x, attn_mask)))
            x = self.norm_layers_2[i](x + self.drop(self.ffn_layers[i](x, x_mask)))

        return x * x_mask

class TextEncoder(torch.nn.Module):
    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, embedding_dim, f0=True):
        super(TextEncoder, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        if f0: self.emb_pitch = torch.nn.Embedding(256, hidden_channels)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths):
        x = self.emb_phone(phone) if pitch is None else (self.emb_phone(phone) + self.emb_pitch(pitch))
        x = torch.transpose(self.lrelu((x * math.sqrt(self.hidden_channels))), 1, -1) 
        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        m, logs = torch.split((self.proj(self.encoder(x * x_mask, x_mask)) * x_mask), self.out_channels, dim=1)
        return m, logs, x_mask

class PosteriorEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super(PosteriorEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.pre = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g = None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        m, logs = torch.split((self.proj(self.enc((self.pre(x) * x_mask), x_mask, g=g)) * x_mask), self.out_channels, dim=1)
        return ((m + torch.randn_like(m) * torch.exp(logs)) * x_mask), m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

class Synthesizer(torch.nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, use_f0, text_enc_hidden_dim=768, vocoder="Default", checkpointing = False, **kwargs):
        super(Synthesizer, self).__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.use_f0 = use_f0
        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), text_enc_hidden_dim, f0=use_f0)

        if use_f0:
            if vocoder == "RefineGAN": 
                from .refinegan import RefineGANGenerator
                self.dec = RefineGANGenerator(sample_rate=sr, downsample_rates=upsample_rates[::-1], upsample_rates=upsample_rates, start_channels=16, num_mels=inter_channels, checkpointing=checkpointing)
            elif vocoder == "MRF HiFi-GAN": 
                from .mrf_hifigan import HiFiGANMRFGenerator
                self.dec = HiFiGANMRFGenerator(in_channel=inter_channels, upsample_initial_channel=upsample_initial_channel, upsample_rates=upsample_rates, upsample_kernel_sizes=upsample_kernel_sizes, resblock_kernel_sizes=resblock_kernel_sizes, resblock_dilations=resblock_dilation_sizes, gin_channels=gin_channels, sample_rate=sr, harmonic_num=8, checkpointing=checkpointing)
            else: self.dec = GeneratorNSF(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, sr=sr, checkpointing=checkpointing)
        else: self.dec = Generator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)

        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = torch.nn.Embedding(self.spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    @torch.jit.ignore
    def forward(self, phone, phone_lengths, pitch = None, pitchf = None, y = None, y_lengths = None, ds = None):
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        if y is not None:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
            return (self.dec(z_slice, slice_segments(pitchf, ids_slice, self.segment_size, 2), g=g) if self.use_f0 else self.dec(z_slice, g=g)), ids_slice, x_mask, y_mask, (z, self.flow(z, y_mask, g=g), m_p, logs_p, m_q, logs_q)
        else: return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(self, phone, phone_lengths, pitch = None, nsff0 = None, sid = None, rate = None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask

        if rate is not None:
            assert isinstance(rate, torch.Tensor)
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
            if self.use_f0: nsff0 = nsff0[:, head:]

        if self.use_f0:
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            o = self.dec(z * x_mask, nsff0, g=g)
        else:
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            o = self.dec(z * x_mask, g=g)

        return o, x_mask, (z, z_p, m_p, logs_p)