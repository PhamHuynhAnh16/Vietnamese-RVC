import os
import sys
import torch

import numpy as np
import torch.nn as nn
import onnxruntime as ort
import torch.nn.functional as F

from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())
os.environ["LRU_CACHE_CAPACITY"] = "3"

from main.library.predictors.FCPE.wav2mel import spawn_wav2mel, Wav2Mel
from main.library.predictors.FCPE.encoder import EncoderLayer, ConformerNaiveEncoder
from main.library.predictors.FCPE.utils import l2_regularization, ensemble_f0, batch_interp_with_replacement_detach, decrypt_model, DotDict

class PCmer(nn.Module):
    def __init__(self, num_layers, num_heads, dim_model, dim_keys, dim_values, residual_dropout, attention_dropout):
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
    def __init__(self, input_channels, out_dims, hidden_dims = 512, n_layers = 6, n_heads = 8, f0_max = 1975.5, f0_min = 32.70, use_fa_norm = False, conv_only = False, conv_dropout = 0, atten_dropout = 0, use_harmonic_emb = False):
        super().__init__()
        self.input_channels = input_channels
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.use_fa_norm = use_fa_norm
        self.residual_dropout = 0.1  
        self.attention_dropout = 0.1  
        self.harmonic_emb = nn.Embedding(9, hidden_dims) if use_harmonic_emb else None
        self.input_stack = nn.Sequential(nn.Conv1d(input_channels, hidden_dims, 3, 1, 1), nn.GroupNorm(4, hidden_dims), nn.LeakyReLU(), nn.Conv1d(hidden_dims, hidden_dims, 3, 1, 1))
        self.net = ConformerNaiveEncoder(num_layers=n_layers, num_heads=n_heads, dim_model=hidden_dims, use_norm=use_fa_norm, conv_only=conv_only, conv_dropout=conv_dropout, atten_dropout=atten_dropout)
        self.norm = nn.LayerNorm(hidden_dims)
        self.output_proj = weight_norm(nn.Linear(hidden_dims, out_dims))
        self.cent_table_b = torch.linspace(self.f0_to_cent(torch.Tensor([f0_min]))[0], self.f0_to_cent(torch.Tensor([f0_max]))[0], out_dims).detach()
        self.register_buffer("cent_table", self.cent_table_b)
        self.gaussian_blurred_cent_mask_b = (1200 * torch.log2(torch.Tensor([self.f0_max / 10.])))[0].detach()
        self.register_buffer("gaussian_blurred_cent_mask", self.gaussian_blurred_cent_mask_b)

    def forward(self, x, _h_emb=None):
        x = self.input_stack(x.transpose(-1, -2)).transpose(-1, -2)
        if self.harmonic_emb is not None: x = x + self.harmonic_emb(torch.LongTensor([0]).to(x.device)) if _h_emb is None else x + self.harmonic_emb(torch.LongTensor([int(_h_emb)]).to(x.device))
        return torch.sigmoid(self.output_proj(self.norm(self.net(x))))

    @torch.no_grad()
    def latent2cents_decoder(self, y, threshold = 0.05, mask = True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)  

        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= threshold] = float("-INF")
            rtn = rtn * confident_mask

        return rtn  

    @torch.no_grad()
    def latent2cents_local_decoder(self, y, threshold = 0.05, mask = True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        confident, max_index = torch.max(y, dim=-1, keepdim=True)

        local_argmax_index = torch.arange(0, 9).to(max_index.device) + (max_index - 4)
        local_argmax_index[local_argmax_index < 0] = 0
        local_argmax_index[local_argmax_index >= self.out_dims] = self.out_dims - 1

        y_l = torch.gather(y, -1, local_argmax_index)
        rtn = torch.sum(torch.gather(ci, -1, local_argmax_index) * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True) 

        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= threshold] = float("-INF")
            rtn = rtn * confident_mask

        return rtn  

    @torch.no_grad()
    def infer(self, mel, decoder = "local_argmax", threshold = 0.05):
        latent = self.forward(mel)
        if decoder == "argmax": cents = self.latent2cents_local_decoder
        elif decoder == "local_argmax": cents = self.latent2cents_local_decoder

        return self.cent_to_f0(cents(latent, threshold=threshold))  

    @torch.no_grad()
    def cent_to_f0(self, cent: torch.Tensor) -> torch.Tensor:
        return 10 * 2 ** (cent / 1200)

    @torch.no_grad()
    def f0_to_cent(self, f0):
        return 1200 * torch.log2(f0 / 10)

class FCPE_LEGACY(nn.Module):
    def __init__(self, input_channel=128, out_dims=360, n_layers=12, n_chans=512, loss_mse_scale=10, loss_l2_regularization=False, loss_l2_regularization_scale=1, loss_grad1_mse=False, loss_grad1_mse_scale=1, f0_max=1975.5, f0_min=32.70, confidence=False, threshold=0.05, use_input_conv=True):
        super().__init__()
        self.loss_mse_scale = loss_mse_scale
        self.loss_l2_regularization = loss_l2_regularization
        self.loss_l2_regularization_scale = loss_l2_regularization_scale
        self.loss_grad1_mse = loss_grad1_mse
        self.loss_grad1_mse_scale = loss_grad1_mse_scale
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.confidence = confidence
        self.threshold = threshold
        self.use_input_conv = use_input_conv
        self.cent_table_b = torch.Tensor(np.linspace(self.f0_to_cent(torch.Tensor([f0_min]))[0], self.f0_to_cent(torch.Tensor([f0_max]))[0], out_dims))
        self.register_buffer("cent_table", self.cent_table_b)
        self.stack = nn.Sequential(nn.Conv1d(input_channel, n_chans, 3, 1, 1), nn.GroupNorm(4, n_chans), nn.LeakyReLU(), nn.Conv1d(n_chans, n_chans, 3, 1, 1))
        self.decoder = PCmer(num_layers=n_layers, num_heads=8, dim_model=n_chans, dim_keys=n_chans, dim_values=n_chans, residual_dropout=0.1, attention_dropout=0.1)
        self.norm = nn.LayerNorm(n_chans)
        self.n_out = out_dims
        self.dense_out = weight_norm(nn.Linear(n_chans, self.n_out))

    def forward(self, mel, infer=True, gt_f0=None, return_hz_f0=False, cdecoder="local_argmax", output_interp_target_length=None):
        if cdecoder == "argmax": self.cdecoder = self.cents_decoder
        elif cdecoder == "local_argmax": self.cdecoder = self.cents_local_decoder

        x = torch.sigmoid(self.dense_out(self.norm(self.decoder((self.stack(mel.transpose(1, 2)).transpose(1, 2) if self.use_input_conv else mel)))))

        if not infer:
            loss_all = self.loss_mse_scale * F.binary_cross_entropy(x, self.gaussian_blurred_cent(self.f0_to_cent(gt_f0)))
            if self.loss_l2_regularization: loss_all = loss_all + l2_regularization(model=self, l2_alpha=self.loss_l2_regularization_scale)
            x = loss_all
        else:
            x = self.cent_to_f0(self.cdecoder(x))
            x = (1 + x / 700).log() if not return_hz_f0 else x

        if output_interp_target_length is not None: 
            x = F.interpolate(torch.where(x == 0, float("nan"), x).transpose(1, 2), size=int(output_interp_target_length), mode="linear").transpose(1, 2)
            x = torch.where(x.isnan(), float(0.0), x)

        return x

    def cents_decoder(self, y, mask=True):
        B, N, _ = y.size()
        rtn = torch.sum(self.cent_table[None, None, :].expand(B, N, -1) * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)

        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask

        return (rtn, confident) if self.confidence else rtn

    def cents_local_decoder(self, y, mask=True):
        B, N, _ = y.size()

        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = torch.clamp(torch.arange(0, 9).to(max_index.device) + (max_index - 4), 0, self.n_out - 1)
        y_l = torch.gather(y, -1, local_argmax_index)
        rtn = torch.sum(torch.gather(self.cent_table[None, None, :].expand(B, N, -1), -1, local_argmax_index) * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True)

        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask

        return (rtn, confident) if self.confidence else rtn

    def cent_to_f0(self, cent):
        return 10.0 * 2 ** (cent / 1200.0)

    def f0_to_cent(self, f0):
        return 1200.0 * torch.log2(f0 / 10.0)

    def gaussian_blurred_cent(self, cents):
        B, N, _ = cents.size()
        return torch.exp(-torch.square(self.cent_table[None, None, :].expand(B, N, -1) - cents) / 1250) * (cents > 0.1) & (cents < (1200.0 * np.log2(self.f0_max / 10.0))).float()

class InferCFNaiveMelPE(torch.nn.Module):
    def __init__(self, args, state_dict, device="cpu"):
        super().__init__()
        self.wav2mel = spawn_wav2mel(args, device=device)
        self.model = CFNaiveMelPE(input_channels=args.mel.num_mels, out_dims=args.model.out_dims, hidden_dims=args.model.hidden_dims, n_layers=args.model.n_layers, n_heads=args.model.n_heads, f0_max=args.model.f0_max, f0_min=args.model.f0_min, use_fa_norm=args.model.use_fa_norm, conv_only=args.model.conv_only, conv_dropout=args.model.conv_dropout, atten_dropout=args.model.atten_dropout, use_harmonic_emb=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.args_dict = dict(args)
        self.register_buffer("tensor_device_marker", torch.tensor(1.0).float(), persistent=False)

    def forward(self, wav, sr, decoder_mode = "local_argmax", threshold = 0.006, key_shifts = [0]):
        with torch.no_grad():
            mels = rearrange(torch.stack([self.wav2mel(wav.to(self.tensor_device_marker.device), sr, keyshift=keyshift) for keyshift in key_shifts], -1), "B T C K -> (B K) T C")
            f0s = rearrange(self.model.infer(mels, decoder=decoder_mode, threshold=threshold), "(B K) T 1 -> B T (K 1)", K=len(key_shifts))

        return f0s 

    def infer(self, wav, sr, decoder_mode = "local_argmax", threshold = 0.006, f0_min = None, f0_max = None, interp_uv = False, output_interp_target_length = None, return_uv = False, test_time_augmentation = False, tta_uv_penalty = 12.0, tta_key_shifts = [0, -12, 12], tta_use_origin_uv=False):
        if test_time_augmentation:
            assert len(tta_key_shifts) > 0
            flag = 0
            if tta_use_origin_uv:
                if 0 not in tta_key_shifts:
                    flag = 1
                    tta_key_shifts.append(0)

            tta_key_shifts.sort(key=lambda x: (x if x >= 0 else -x / 2))
            f0s = self.__call__(wav, sr, decoder_mode, threshold, tta_key_shifts)
            f0 = ensemble_f0(f0s[:, :, flag:], tta_key_shifts[flag:], tta_uv_penalty)
            f0_for_uv = f0s[:, :, [0]] if tta_use_origin_uv else f0
        else:
            f0 = self.__call__(wav, sr, decoder_mode, threshold)
            f0_for_uv = f0

        if f0_min is None: f0_min = self.args_dict["model"]["f0_min"]
        uv = (f0_for_uv < f0_min).type(f0_for_uv.dtype)
        f0 = f0 * (1 - uv)

        if interp_uv: f0 = batch_interp_with_replacement_detach(uv.squeeze(-1).bool(), f0.squeeze(-1)).unsqueeze(-1)
        if f0_max is not None: f0[f0 > f0_max] = f0_max
        if output_interp_target_length is not None: 
            f0 = F.interpolate(torch.where(f0 == 0, float("nan"), f0).transpose(1, 2), size=int(output_interp_target_length), mode="linear").transpose(1, 2)
            f0 = torch.where(f0.isnan(), float(0.0), f0)

        if return_uv: return f0, F.interpolate(uv.transpose(1, 2), size=int(output_interp_target_length), mode="nearest").transpose(1, 2)
        else: return f0

class FCPEInfer_LEGACY:
    def __init__(self, configs, model_path, device=None, dtype=torch.float32, providers=None, onnx=False, f0_min=50, f0_max=1100):
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.onnx = onnx
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.wav2mel = Wav2Mel(device=self.device, dtype=self.dtype)

        if self.onnx:
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(decrypt_model(configs, model_path), sess_options=sess_options, providers=providers)
        else:
            ckpt = torch.load(model_path, map_location=torch.device(self.device), weights_only=True)
            self.args = DotDict(ckpt["config"])
            model = FCPE_LEGACY(input_channel=self.args.model.input_channel, out_dims=self.args.model.out_dims, n_layers=self.args.model.n_layers, n_chans=self.args.model.n_chans, loss_mse_scale=self.args.loss.loss_mse_scale, loss_l2_regularization=self.args.loss.loss_l2_regularization, loss_l2_regularization_scale=self.args.loss.loss_l2_regularization_scale, loss_grad1_mse=self.args.loss.loss_grad1_mse, loss_grad1_mse_scale=self.args.loss.loss_grad1_mse_scale, f0_max=self.f0_max, f0_min=self.f0_min, confidence=self.args.model.confidence)
            model.to(self.device).to(self.dtype)
            model.load_state_dict(ckpt["model"])
            model.eval()
            self.model = model

    @torch.no_grad()
    def __call__(self, audio, sr, threshold=0.05, p_len=None):
        if not self.onnx: self.model.threshold = threshold
        if not hasattr(self, "numpy_threshold") and self.onnx: self.numpy_threshold = np.array(threshold, dtype=np.float32)

        mel = self.wav2mel(audio=audio[None, :], sample_rate=sr).to(self.dtype)

        if self.onnx:
            return torch.as_tensor(
                self.model.run(
                    [self.model.get_outputs()[0].name], 
                    {
                        self.model.get_inputs()[0].name: mel.detach().cpu().numpy(), 
                        self.model.get_inputs()[1].name: self.numpy_threshold
                    }
                )[0], 
                dtype=self.dtype, 
                device=self.device
            )
        else: 
            return self.model(
                mel=mel, 
                infer=True, 
                return_hz_f0=True, 
                output_interp_target_length=p_len
            )

class FCPEInfer:
    def __init__(self, configs, model_path, device=None, dtype=torch.float32, providers=None, onnx=False, f0_min=50, f0_max=1100):
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.onnx = onnx
        self.f0_min = f0_min
        self.f0_max = f0_max

        if self.onnx:
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(decrypt_model(configs, model_path), sess_options=sess_options, providers=providers)
        else:
            ckpt = torch.load(model_path, map_location=torch.device(device), weights_only=True)
            ckpt["config_dict"]["model"]["conv_dropout"] = ckpt["config_dict"]["model"]["atten_dropout"] = 0.0
            self.args = DotDict(ckpt["config_dict"])
            model = InferCFNaiveMelPE(self.args, ckpt["model"], device)
            model = model.to(device).to(self.dtype)
            model.eval()
            self.model = model

    @torch.no_grad()
    def __call__(self, audio, sr, threshold=0.05, p_len=None):
        if not hasattr(self, "wav2mel") and self.onnx: self.wav2mel = Wav2Mel(device=self.device, dtype=self.dtype)
        if not hasattr(self, "numpy_threshold") and self.onnx: self.numpy_threshold = np.array(threshold, dtype=np.float32)

        if self.onnx:
            return torch.as_tensor(
                self.model.run(
                    [self.model.get_outputs()[0].name], 
                    {
                        self.model.get_inputs()[0].name: self.wav2mel(audio=audio[None, :], sample_rate=sr).to(self.dtype).detach().cpu().numpy(), 
                        self.model.get_inputs()[1].name: self.numpy_threshold
                    }
                )[0], 
                dtype=self.dtype, 
                device=self.device
            ) 
        else: 
            return self.model.infer(
                audio[None, :], 
                sr, 
                threshold=threshold, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                output_interp_target_length=p_len
            )

class FCPE:
    def __init__(self, configs, model_path, hop_length=512, f0_min=50, f0_max=1100, dtype=torch.float32, device=None, sample_rate=16000, threshold=0.05, providers=None, onnx=False, legacy=False):
        self.model = FCPEInfer_LEGACY if legacy else FCPEInfer
        self.fcpe = self.model(configs, model_path, device=device, dtype=dtype, providers=providers, onnx=onnx, f0_min=f0_min, f0_max=f0_max)
        self.hop_length = hop_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.legacy = legacy

    def compute_f0(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        p_len = (x.shape[0] // self.hop_length) if p_len is None else p_len

        f0 = self.fcpe(x, sr=self.sample_rate, threshold=self.threshold, p_len=p_len)
        f0 = f0[:] if f0.dim() == 1 else f0[0, :, 0]

        if torch.all(f0 == 0): return f0.cpu().numpy() if p_len is None else np.zeros(p_len), (f0.cpu().numpy() if p_len is None else np.zeros(p_len))
        return f0.cpu().numpy()