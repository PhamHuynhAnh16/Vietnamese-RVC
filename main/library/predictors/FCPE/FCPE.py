import os
import sys
import torch

import numpy as np
import torch.nn as nn
import onnxruntime as ort

from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())
os.environ["LRU_CACHE_CAPACITY"] = "3"

from main.app.variables import config, configs
from main.library.predictors.FCPE.wav2mel import Wav2Mel
from main.library.predictors.FCPE.utils import decrypt_model, DotDict
from main.library.predictors.FCPE.encoder import EncoderLayer, ConformerNaiveEncoder

@torch.no_grad()
def cent_to_f0(cent):
    return 10 * 2 ** (cent / 1200)

@torch.no_grad()
def f0_to_cent(f0):
    return 1200 * (f0 / 10).log2()

@torch.no_grad()
def latent2cents_local_decoder(cent_table, out_dims, y, threshold = 0.05, mask = True):
    B, N, _ = y.size()
    ci = cent_table[None, None, :].expand(B, N, -1)
    confident, max_index = y.max(dim=-1, keepdim=True)

    local_argmax_index = torch.arange(0, 9).to(max_index.device) + (max_index - 4)
    local_argmax_index[local_argmax_index < 0] = 0
    local_argmax_index[local_argmax_index >= out_dims] = out_dims - 1

    y_l = y.gather(-1, local_argmax_index)
    rtn = (ci.gather(-1, local_argmax_index) * y_l).sum(dim=-1, keepdim=True) / y_l.sum(dim=-1, keepdim=True) 

    if mask:
        confident_mask = torch.ones_like(confident)
        confident_mask[confident <= threshold] = float("-INF")
        rtn = rtn * confident_mask

    return rtn

@torch.no_grad()
def latent2cents_local_decoder_cpu(cent_table, out_dims, y, threshold = 0.05, mask = True):
    cent_table, y = cent_table.cpu(), y.cpu()

    B, N, _ = y.size()
    ci = cent_table[None, None, :].expand(B, N, -1)
    confident, max_index = y.max(dim=-1, keepdim=True)

    local_argmax_index = torch.arange(0, 9).to(max_index.device) + (max_index - 4)
    local_argmax_index[local_argmax_index < 0] = 0
    local_argmax_index[local_argmax_index >= out_dims] = out_dims - 1

    y_l = y.gather(-1, local_argmax_index)
    rtn = (ci.gather(-1, local_argmax_index) * y_l).sum(dim=-1, keepdim=True) / y_l.sum(dim=-1, keepdim=True) 

    if mask:
        confident_mask = torch.ones_like(confident)
        confident_mask[confident <= threshold] = float("-INF")
        rtn = rtn * confident_mask

    return rtn

def cents_local_decoder(cent_table, y, n_out, confidence, threshold = 0.05, mask=True):
    B, N, _ = y.size()
    confident, max_index = y.max(dim=-1, keepdim=True)
    local_argmax_index = (torch.arange(0, 9).to(max_index.device) + (max_index - 4)).clamp(0, n_out - 1)
    y_l = y.gather(-1, local_argmax_index)
    rtn = (cent_table[None, None, :].expand(B, N, -1).gather(-1, local_argmax_index) * y_l).sum(dim=-1, keepdim=True) / y_l.sum(dim=-1, keepdim=True)

    if mask:
        confident_mask = torch.ones_like(confident)
        confident_mask[confident <= threshold] = float("-INF")
        rtn = rtn * confident_mask

    return (rtn, confident) if confidence else rtn

def cents_local_decoder_cpu(cent_table, y, n_out, confidence, threshold = 0.05, mask=True):
    cent_table, y = cent_table.cpu(), y.cpu()

    B, N, _ = y.size()
    confident, max_index = y.max(dim=-1, keepdim=True)
    local_argmax_index = (torch.arange(0, 9).to(max_index.device) + (max_index - 4)).clamp(0, n_out - 1)
    y_l = y.gather(-1, local_argmax_index)
    rtn = (cent_table[None, None, :].expand(B, N, -1).gather(-1, local_argmax_index) * y_l).sum(dim=-1, keepdim=True) / y_l.sum(dim=-1, keepdim=True)

    if mask:
        confident_mask = torch.ones_like(confident)
        confident_mask[confident <= threshold] = float("-INF")
        rtn = rtn * confident_mask

    return (rtn, confident) if confidence else rtn

class PCmer(nn.Module):
    def __init__(
        self, 
        num_layers, 
        num_heads, 
        dim_model, 
        dim_keys, 
        dim_values, 
        residual_dropout, 
        attention_dropout
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self._layers = nn.ModuleList([EncoderLayer(self) for _ in range(num_layers)])

    def forward(self, phone, mask=None):
        for layer in self._layers:
            phone = layer(phone, mask)

        return phone

class CFNaiveMelPE(nn.Module):
    def __init__(
        self, 
        input_channels, 
        out_dims, 
        hidden_dims = 512, 
        n_layers = 6, 
        n_heads = 8, 
        use_fa_norm = False, 
        conv_only = False, 
        conv_dropout = 0, 
        atten_dropout = 0, 
    ):
        super().__init__()
        self.input_channels = input_channels
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_fa_norm = use_fa_norm
        self.input_stack = nn.Sequential(
            nn.Conv1d(
                input_channels, 
                hidden_dims, 
                3, 
                1, 
                1
            ), 
            nn.GroupNorm(
                4, 
                hidden_dims
            ), 
            nn.LeakyReLU(), 
            nn.Conv1d(
                hidden_dims, 
                hidden_dims, 
                3, 
                1, 
                1
            )
        )
        self.net = ConformerNaiveEncoder(
            num_layers=n_layers, 
            num_heads=n_heads, 
            dim_model=hidden_dims, 
            use_norm=use_fa_norm, 
            conv_only=conv_only, 
            conv_dropout=conv_dropout, 
            atten_dropout=atten_dropout
        )
        self.norm = nn.LayerNorm(hidden_dims)
        self.output_proj = weight_norm(
            nn.Linear(
                hidden_dims, 
                out_dims
            )
        )

        self.cent_table_b = torch.linspace(
            f0_to_cent(torch.Tensor([32.70]))[0], 
            f0_to_cent(torch.Tensor([1975.5]))[0], 
            out_dims
        ).detach()
        self.gaussian_blurred_cent_mask_b = (
            1200 * torch.Tensor([197.55]).log2()
        )[0].detach()

        self.register_buffer("cent_table", self.cent_table_b)
        self.register_buffer("gaussian_blurred_cent_mask", self.gaussian_blurred_cent_mask_b)
        self.latent2cents_local_decoder = latent2cents_local_decoder_cpu if config.device.startswith("privateuseone") else latent2cents_local_decoder

    def forward(self, mel, threshold = 0.006):
        with torch.no_grad():
            mels = rearrange(torch.stack([mel], -1), "B T C K -> (B K) T C")

            x = self.input_stack(mels.transpose(-1, -2)).transpose(-1, -2)
            latent = self.output_proj(self.norm(self.net(x))).sigmoid()

            x = cent_to_f0(
                self.latent2cents_local_decoder(
                    self.cent_table, 
                    self.out_dims, 
                    latent, 
                    threshold=threshold
                )
            )

            f0 = rearrange(x, "(B K) T 1 -> B T (K 1)", K=1)
            f0 = f0 * (1 - (f0 < 32.70).type(f0.dtype))
            f0[f0 > 1975.5] = 1975.5

        return f0

class CFNaiveMelPE_LEGACY(nn.Module):
    def __init__(
        self, 
        input_channel=128, 
        out_dims=360, 
        n_layers=12, 
        n_chans=512, 
        confidence=False
    ):
        super().__init__()
        self.n_out = out_dims
        self.confidence = confidence

        self.cent_table_b = torch.Tensor(
            np.linspace(
                f0_to_cent(torch.Tensor([32.70]))[0], 
                f0_to_cent(torch.Tensor([1975.5]))[0], 
                out_dims
            )
        )
        self.register_buffer("cent_table", self.cent_table_b)

        self.stack = nn.Sequential(
            nn.Conv1d(
                input_channel, 
                n_chans, 
                3, 
                1, 
                1
            ), 
            nn.GroupNorm(
                4, 
                n_chans
            ), 
            nn.LeakyReLU(), 
            nn.Conv1d(
                n_chans, 
                n_chans, 
                3, 
                1, 
                1
            )
        )
        self.decoder = PCmer(
            num_layers=n_layers, 
            num_heads=8, 
            dim_model=n_chans, 
            dim_keys=n_chans, 
            dim_values=n_chans, 
            residual_dropout=0.1, 
            attention_dropout=0.1
        )
        self.norm = nn.LayerNorm(n_chans)
        self.dense_out = weight_norm(
            nn.Linear(
                n_chans, 
                self.n_out
            )
        )

        self.cents_local_decoder = cents_local_decoder_cpu if config.device.startswith("privateuseone") else cents_local_decoder

    def forward(self, mel, threshold=0.05):
        x = self.decoder(self.stack(mel.transpose(1, 2)).transpose(1, 2))
        x = self.dense_out(self.norm(x)).sigmoid()

        x = cent_to_f0(
            self.cents_local_decoder(
                self.cent_table, 
                x, 
                self.n_out, 
                self.confidence, 
                threshold=threshold, 
                mask=True
            )
        )

        return x

class FCPE:
    def __init__(
        self, 
        model_path, 
        device=None, 
        threshold=0.05, 
        providers=None, 
        onnx=False, 
        legacy=False,
        compile_model=False,
        compile_mode=None
    ):
        self.device = device
        self.threshold = threshold
        self.wav2mel = Wav2Mel(device=self.device, dtype=torch.float32)

        if onnx:
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            model = ort.InferenceSession(decrypt_model(configs, model_path), sess_options=sess_options, providers=providers)
        elif legacy:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            self.args = DotDict(ckpt["config"])

            model = CFNaiveMelPE_LEGACY(
                input_channel=self.args.model.input_channel, 
                out_dims=self.args.model.out_dims, 
                n_layers=self.args.model.n_layers, 
                n_chans=self.args.model.n_chans, 
                confidence=self.args.model.confidence
            )

            model.load_state_dict(ckpt["model"])
            if compile_model: model.decoder, model.stack = torch.compile(model.decoder, mode=compile_mode), torch.compile(model.stack, mode=compile_mode)
            model = model.to(self.device).eval()
        else:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            self.args = DotDict(ckpt["config_dict"])

            model = CFNaiveMelPE(
                input_channels=self.args.mel.num_mels, 
                out_dims=self.args.model.out_dims, 
                hidden_dims=self.args.model.hidden_dims, 
                n_layers=self.args.model.n_layers, 
                n_heads=self.args.model.n_heads, 
                use_fa_norm=self.args.model.use_fa_norm, 
                conv_only=self.args.model.conv_only
            )

            model.load_state_dict(ckpt["model"])
            if compile_model: model.net, model.input_stack = torch.compile(model.net, mode=compile_mode), torch.compile(model.input_stack, mode=compile_mode)
            model = model.to(self.device).eval()
        
        self.model = model
        self.infer = self._infer_onnx if onnx else self._infer_torch
    
    def compute_f0(self, wav):
        if not torch.is_tensor(wav): wav = torch.from_numpy(wav).float().to(self.device)

        with torch.no_grad():
            f0 = self.infer(self.wav2mel(audio=wav[None, :]), self.threshold)
            f0 = f0[:] if f0.dim() == 1 else f0[0, :, 0]

        return f0
            
    def _infer_onnx(self, mel, threshold):
        return torch.as_tensor(
            self.model.run(
                [self.model.get_outputs()[0].name], 
                {
                    self.model.get_inputs()[0].name: mel.detach().cpu().numpy(), 
                    self.model.get_inputs()[1].name: np.array(threshold, dtype=np.float32)
                }
            )[0], 
            dtype=torch.float32, 
            device=self.device
        ) 

    def _infer_torch(self, mel, threshold):
        return self.model(mel, threshold=threshold)