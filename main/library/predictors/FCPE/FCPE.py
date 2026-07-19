import os
import sys
import torch

import numpy as np
import torch.nn as nn
import onnxruntime as ort

from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

# Limit Least Recently Used (LRU) cache capacity for specific backend operations
os.environ["LRU_CACHE_CAPACITY"] = "3"

from main.app.variables import config, configs
from main.library.predictors.FCPE.stft import Wav2Mel
from main.library.predictors.FCPE.utils import decrypt_model, DotDict
from main.library.predictors.FCPE.encoder import EncoderLayer, ConformerNaiveEncoder

@torch.no_grad()
def cent_to_f0(cent):
    """
    Convert pitch values from cents to Frequency (Hz).

    Args:
        cent (torch.Tensor): Pitch values in cents.

    Returns:
        torch.Tensor: Pitch frequency values in Hz.
    """

    return 10 * 2 ** (cent / 1200)

@torch.no_grad()
def f0_to_cent(f0):
    """
    Convert pitch values from Frequency (Hz) to cents.

    Args:
        f0 (torch.Tensor): Pitch frequency values in Hz.

    Returns:
        torch.Tensor: Pitch values in cents.
    """

    return 1200 * (f0 / 10).log2()

@torch.no_grad()
def latent2cents_local_decoder(cent_table, out_dims, y, threshold = 0.05):
    """
    Decodes latent probabilities into cent values using a local neighborhood (GPU-optimized).

    This decodes the model output distribution into a continuous pitch value by calculating
    a weighted average around the peak index (argmax) within a window of 9 bins.

    Args:
        cent_table (torch.Tensor): A reference table containing cent mapping values.
        out_dims (int): The number of output dimensions (bins) in the probability distribution.
        y (torch.Tensor): Model prediction probabilities, shape (B, N, out_dims).
        threshold (float, optional): Confidence threshold. Values below this are masked. Defaults to 0.05.

    Returns:
        torch.Tensor: Continuous cent values masked by confidence, shape (B, N, 1).
    """

    B, N, _ = y.size()
    # Expand cent table to match batch and time frames dimensions
    ci = cent_table[None, None, :].expand(B, N, -1)
    confident, max_index = y.max(dim=-1, keepdim=True)

    # Define a local window of 9 bins around the argmax position [max_index - 4, max_index + 4]
    local_argmax_index = torch.arange(0, 9).to(max_index.device) + (max_index - 4)
    # Clip indices to prevent index out of bounds error
    local_argmax_index[local_argmax_index < 0] = 0
    local_argmax_index[local_argmax_index >= out_dims] = out_dims - 1

    # Gather local probabilities and their corresponding cent values
    y_l = y.gather(-1, local_argmax_index)
    # Perform a center-of-mass weighted average within the local window
    rtn = (ci.gather(-1, local_argmax_index) * y_l).sum(dim=-1, keepdim=True) / y_l.sum(dim=-1, keepdim=True) 

    # Mask unconfident predictions by assigning negative infinity
    confident_mask = torch.ones_like(confident)
    confident_mask[confident <= threshold] = float("-INF")
    rtn = rtn * confident_mask

    return rtn

def cents_local_decoder(cent_table, y, n_out, threshold = 0.05):
    """
    Alternative implementation of local center-of-mass decoding for cent values.

    Args:
        cent_table (torch.Tensor): Reference cent mapping values.
        y (torch.Tensor): Model output logits/probabilities.
        n_out (int): Total number of output frequency bins.
        threshold (float, optional): Confidence threshold. Defaults to 0.05.

    Returns:
        torch.Tensor: Calculated continuous cent values.
    """

    B, N, _ = y.size()
    confident, max_index = y.max(dim=-1, keepdim=True)
    # Define and clamp the local window indices to safety constraints
    local_argmax_index = (torch.arange(0, 9).to(max_index.device) + (max_index - 4)).clamp(0, n_out - 1)
    y_l = y.gather(-1, local_argmax_index)
    # Compute center of mass mapping
    rtn = (cent_table[None, None, :].expand(B, N, -1).gather(-1, local_argmax_index) * y_l).sum(dim=-1, keepdim=True) / y_l.sum(dim=-1, keepdim=True)

    # Filter out low-confidence frame predictions
    confident_mask = torch.ones_like(confident)
    confident_mask[confident <= threshold] = float("-INF")
    rtn = rtn * confident_mask

    return rtn

def latent2cents_local_decoder_cpu(cent_table, out_dims, y, threshold = 0.05):
    """Wrapper to force latent2cents decoding on the CPU (for specialized hardware backends)."""

    cent_table, y = cent_table.cpu(), y.cpu()
    return latent2cents_local_decoder(cent_table, out_dims, y, threshold)

def cents_local_decoder_cpu(cent_table, y, n_out, threshold = 0.05):
    """Wrapper to force cents decoding on the CPU (for specialized hardware backends)."""

    cent_table, y = cent_table.cpu(), y.cpu()
    return cents_local_decoder(cent_table, y, n_out, threshold)

class PCmer(nn.Module):
    """Parallel Conformer-based module used for processing frame representations."""

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
        """
        Initializes PCmer layer configurations.

        Args:
            num_layers (int): Number of Conformer Encoder blocks.
            num_heads (int): Number of attention heads.
            dim_model (int): Hidden size dimension.
            dim_keys (int): Feature dimensionality for key projections.
            dim_values (int): Feature dimensionality for value projections.
            residual_dropout (float): Dropout probability for residual connections.
            attention_dropout (float): Dropout probability for attention weights.
        """

        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        # Instantiate sequence of sequential encoder layers
        self._layers = nn.ModuleList([EncoderLayer(self) for _ in range(num_layers)])

    def forward(self, phone, mask=None):
        """
        Forward pass through the PCmer block layers.

        Args:
            phone (torch.Tensor): Feature representations.
            mask (torch.Tensor, optional): Optional padding masks. Defaults to None.
        """

        for layer in self._layers:
            phone = layer(phone, mask)

        return phone

class CFNaiveMelPE(nn.Module):
    """Conformer-based Naive Mel-spectrogram Pitch Estimator (Modern Architecture)."""

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
        atten_dropout = 0
    ):
        super().__init__()
        self.input_channels = input_channels
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_fa_norm = use_fa_norm
        # Front-end 1D convolution processing blocks
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
        # Core encoder network based on Conformer architecture
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
        # Output projection utilizing Weight Normalization for stabilization
        self.output_proj = weight_norm(
            nn.Linear(
                hidden_dims, 
                out_dims
            )
        )

        # Pre-calculate constant buffers for mapping index predictions back to Hz
        self.cent_table_b = torch.linspace(
            f0_to_cent(torch.Tensor([32.70]))[0], 
            f0_to_cent(torch.Tensor([1975.5]))[0], 
            out_dims
        ).detach()
        self.gaussian_blurred_cent_mask_b = (
            1200 * torch.Tensor([197.55]).log2()
        )[0].detach()

        # Register non-trainable state variables
        self.register_buffer("cent_table", self.cent_table_b)
        self.register_buffer("gaussian_blurred_cent_mask", self.gaussian_blurred_cent_mask_b)
        # Adapt decoding strategy dynamically according to hardware limitations
        self.latent2cents_local_decoder = latent2cents_local_decoder_cpu if config.device.startswith("privateuseone") else latent2cents_local_decoder

    def forward(self, mel, threshold = 0.006):
        """
        Infers F0 tracks natively given extracted Mel spectrogram vectors.

        Args:
            mel (torch.Tensor): Input mel-spectrogram.
            threshold (float, optional): Confidence decoding threshold. Defaults to 0.006.

        Returns:
            torch.Tensor: Extracted continuous fundamental frequency values (Hz).
        """

        with torch.no_grad():
            # Reorganize batch sequence structures safely
            mels = rearrange(torch.stack([mel], -1), "B T C K -> (B K) T C")

            # Extract internal intermediate latents via spatial blocks
            x = self.input_stack(mels.transpose(-1, -2)).transpose(-1, -2)
            # Keep it as Float; using Half will cause NaN issues and crash RVC/SVC
            latent = self.output_proj(self.norm(self.net(x))).float().sigmoid()

            # Local probabilistic center-of-mass cent mapping decoding
            x = cent_to_f0(
                self.latent2cents_local_decoder(
                    self.cent_table, 
                    self.out_dims, 
                    latent, 
                    threshold=threshold
                )
            )

            # Post-process dimensions and clamp extreme boundaries
            f0 = rearrange(x, "(B K) T 1 -> B T (K 1)", K=1)
            f0 = f0 * (1 - (f0 < 32.70).type(f0.dtype)) # Zero out lower noise floor
            f0[f0 > 1975.5] = 1975.5

        return f0

class CFNaiveMelPE_LEGACY(nn.Module):
    """Legacy Conformer-based Naive Mel-spectrogram Pitch Estimator architecture."""

    def __init__(
        self, 
        input_channel=128, 
        out_dims=360, 
        n_layers=12, 
        n_chans=512
    ):
        super().__init__()
        self.n_out = out_dims

        # Pre-computed linear space table for cent conversions
        self.cent_table_b = torch.linspace(
            f0_to_cent(torch.Tensor([32.70]))[0], 
            f0_to_cent(torch.Tensor([1975.5]))[0], 
            out_dims
        ).detach()

        self.register_buffer("cent_table", self.cent_table_b)
        # Processing stack for structural feature downsampling/upsampling
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
        # Dynamic strategy alignment for target device backends
        self.cents_local_decoder = cents_local_decoder_cpu if config.device.startswith("privateuseone") else cents_local_decoder

    def forward(self, mel, threshold=0.05):
        """Infers F0 tracks via the legacy model forward routine pipeline."""

        x = self.decoder(self.stack(mel.transpose(1, 2)).transpose(1, 2))
        x = self.dense_out(self.norm(x)).sigmoid()

        # Unpack indices to frequency
        x = cent_to_f0(
            self.cents_local_decoder(
                self.cent_table, 
                x, 
                self.n_out, 
                threshold=threshold
            )
        )

        return x

class FCPE:
    """
    FCPE: A Fast Context-based Pitch Estimation Model wrapper class.

    This class serves as the production-ready wrapper for the FCPE model architecture,
    supporting high-performance inference pipelines including PyTorch Native (FP32/FP16),
    compiled Torchscript/Dynamo execution, and optimized ONNX Runtime evaluation 
    leveraging zero-copy memory IO-Binding features.
    """

    def __init__(
        self, 
        model_path, 
        device=None, 
        threshold=0.05, 
        providers=None, 
        onnx=False, 
        legacy=False,
        is_half=False,
        compile_model=False,
        compile_mode=None
    ):
        """
        Initializes the Fast Context-based Pitch Estimation (FCPE) pipeline backend.

        Args:
            model_path (str): Path leading to the local model checkpoint file.
            device (str/torch.device, optional): Device context execution binding (e.g., 'cuda', 'cpu').
            threshold (float, optional): Local extraction confidence window thresholds. Defaults to 0.05.
            providers (list, optional): Execution engine providers specified for ONNX Runtime (e.g., ['CUDAExecutionProvider']).
            onnx (bool, optional): Toggles ONNX Runtime inference usage instead of PyTorch. Defaults to False.
            legacy (bool, optional): Fallback flag for loading old-format model structures. Defaults to False.
            is_half (bool, optional): Enables FP16 precision calculations. Defaults to False.
            compile_model (bool, optional): Calls torch.compile over submodules for enhanced speed. Defaults to False.
            compile_mode (str, optional): Target compilation strategy modes parameterization (e.g., 'reduce-overhead').
        """

        self.device = device
        self.threshold = threshold
        self.wav2mel = Wav2Mel(device=self.device, dtype=torch.float32)

        if onnx:
            # Setup configuration optimized execution boundaries for ONNX runtimes
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            if providers[0][0].startswith("Tensorrt"): sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Decrypt parameters safely and initialize session state contexts
            # My FCPE ONNX models previously encountered issues with Hugging Face, so I chose to encode the models to ensure compatibility.
            # I haven't tried it again since then to see if the problem persists.
            model = ort.InferenceSession(decrypt_model(configs, model_path), sess_options=sess_options, providers=providers)
        elif legacy:
            # Parse classic architecture format checkpoints safely
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            self.args = DotDict(ckpt["config"])

            model = CFNaiveMelPE_LEGACY(
                input_channel=self.args.model.input_channel, 
                out_dims=self.args.model.out_dims, 
                n_layers=self.args.model.n_layers, 
                n_chans=self.args.model.n_chans
            )

            model.load_state_dict(ckpt["model"])
            if compile_model: model.decoder, model.stack = torch.compile(model.decoder, mode=compile_mode), torch.compile(model.stack, mode=compile_mode)
            model = model.to(self.device).eval()
            if is_half: model = model.half()
        else:
            # Load modern architecture configurations
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
            if is_half: model = model.half()
        
        self.model = model
        # Identify standard processing target strings for system contexts
        self._device = "cuda" if providers[0][0].startswith(("Tensorrt", "CUDA")) else "cpu"
        # Preallocate memory buffers depending on execution driver features
        self._threshold = (torch.zeros((), device=self.device, dtype=torch.float32) if providers[0][0].startswith(("Tensorrt", "CUDA", "CPU")) else np.empty((), dtype=np.float32)) if onnx else None
        # Explicitly assign function execution routing pathways to minimize checking costs
        self.infer = (self._infer_onnx_io if providers[0][0].startswith(("Tensorrt", "CUDA", "CPU")) else self._infer_onnx_non_io) if onnx else (self._infer_torch_fp16 if is_half else self._infer_torch_fp32)
    
    def compute_f0(self, wav):
        """
        Extract fundamental frequency (F0) tracking directly from raw waveform arrays.

        Args:
            wav (np.ndarray | torch.Tensor): Input audio waveform tensor or array.

        Returns:
            torch.Tensor: Frame-level F0 series estimations.
        """

        if not torch.is_tensor(wav): wav = torch.from_numpy(wav).float().to(self.device)

        with torch.inference_mode():
            # Extract mel configurations and pass tensors down to active routing backends
            f0 = self.infer(self.wav2mel(audio=wav[None, :]), self.threshold)
            # Squeeze output to present simple 1D structures
            f0 = f0[:] if f0.dim() == 1 else f0[0, :, 0]

        return f0
            
    def _infer_onnx_non_io(self, mel, threshold):
        """Inference routing utilizing conventional NumPy array transitions (Slower)."""

        self._threshold.fill(threshold)

        return torch.as_tensor(
            self.model.run(
                ["pitchf"], {"mel": mel.detach().cpu().numpy(), "threshold": self._threshold}
            )[0], 
            dtype=torch.float32, 
            device=self.device
        ) 

    def _infer_onnx_io(self, mel, threshold):
        """High-performance ONNX Inference using device IO-Binding to avoid CPU host copies."""

        mel = mel.contiguous()
        self._threshold.fill_(threshold)
        device_idx = mel.device.index or 0

        # Bind hardware native allocations directly to bypass host-device copy roundtrips
        io_binding = self.model.io_binding()
        io_binding.bind_input(name="mel", device_type=self._device, device_id=device_idx, element_type=np.float32, shape=tuple(mel.shape), buffer_ptr=mel.data_ptr())
        io_binding.bind_input(name="threshold", device_type=self._device, device_id=device_idx, element_type=np.float32, shape=tuple(self._threshold.shape), buffer_ptr=self._threshold.data_ptr())

        io_binding.bind_output(name="pitchf", device_type=self._device, device_id=device_idx)
        self.model.run_with_iobinding(io_binding)

        return torch.as_tensor(
            io_binding.get_outputs()[0].numpy(), 
            dtype=torch.float32, 
            device=self.device
        ) 

    def _infer_torch_fp32(self, mel, threshold):
        """Standard PyTorch inference runner using FP32 precision."""

        return self.model(mel, threshold=threshold)

    def _infer_torch_fp16(self, mel, threshold):
        """Standard PyTorch inference runner using FP16 half-precision."""

        return self.model(mel.half(), threshold=threshold)