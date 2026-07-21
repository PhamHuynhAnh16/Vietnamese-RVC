import os
import sys
import torch
import librosa
import scipy.stats

import numpy as np

sys.path.append(os.getcwd())

from main.library.algorithm.viterbi import viterbi

CENTS_PER_BIN, PITCH_BINS, SAMPLE_RATE, WINDOW_SIZE = 20, 360, 16000, 1024

class CREPE:
    """
    CREPE (Convolutional Representation for Pitch Estimation) model wrapper.

    This class handles preprocessing, inference (via PyTorch or ONNX Runtime),
    Viterbi decoding, and postprocessing to estimate fundamental frequency (F0)
    and optional periodicity from raw audio waveforms.
    """

    def __init__(
        self, 
        model_path, 
        model_size="full", 
        hop_length=512, 
        batch_size=None, 
        f0_min=50, 
        f0_max=1100, 
        device=None, 
        sample_rate=16000, 
        providers=None, 
        onnx=False, 
        is_half=False,
        return_periodicity=False,
        compile_model = False,
        compile_mode = None
    ):
        """Initializes the CREPE model instance with specified configurations.

        Args:
            model_path (str): Path to the model weight file (.pth or .onnx).
            model_size (str): Size of the CREPE architecture ('full', 'tiny', etc.). Default is "full".
            hop_length (int, optional): Number of samples between frames. Defaults to 512.
            batch_size (int, optional): Number of frames processed at once. Defaults to None (all frames).
            f0_min (float): Minimum frequency bound for pitch filtering in Hz. Default is 50.
            f0_max (float): Maximum frequency bound for pitch filtering in Hz. Default is 1100.
            device (str or torch.device): Device execution target (e.g., 'cuda', 'cpu').
            sample_rate (int): Input audio sampling rate. Default is 16000.
            providers (list, optional): ONNX Execution Providers list (e.g., ['CUDAExecutionProvider']).
            onnx (bool): If True, uses ONNX runtime inference. Default is False.
            is_half (bool): If True, runs PyTorch model in FP16 precision. Default is False.
            return_periodicity (bool): If True, returns confidence scores along with F0. Default is False.
            compile_model (bool): If True, compiles PyTorch model using torch.compile(). Default is False.
            compile_mode (str, optional): Optimization mode for torch.compile().
        """

        self.device = device
        self.hop_length = hop_length
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.return_periodicity = return_periodicity

        # Map minimum and maximum frequency constraints from Hz to CREPE's cent-based bins
        self.f0_min_bin = (((1200 * torch.tensor(f0_min / 10).log2()) - 1997.3794084376191) / CENTS_PER_BIN).floor().int()
        self.f0_max_bin = (((1200 * torch.tensor(f0_max / 10).log2()) - 1997.3794084376191) / CENTS_PER_BIN).ceil().int()
        # Small constant to avoid division-by-zero during normalization
        self.eps = torch.tensor(1e-10, device=device)
        self.transition = None

        # Load and configure model based on backend (ONNX vs PyTorch)
        if onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3 # Suppress verbose warnings
            if providers[0][0].startswith("Tensorrt"): sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.CREPE.model import CREPEE

            model = CREPEE(model_size)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.to(device).eval()
            # Apply performance optimizations if requested
            # Warning: Not recommended, as it degrades the model's quality and introduces noise during silent segments.
            if is_half: model = model.half()
            if compile_model: model = torch.compile(model, mode=compile_mode)

        self.model = model
        # Route execution to the correct private inference function based on configuration
        self._device = "cuda" if providers[0][0].startswith(("Tensorrt", "CUDA")) else "cpu"
        self.infer = (self._infer_onnx_io if providers[0][0].startswith(("Tensorrt", "CUDA", "CPU")) else self._infer_onnx_non_io) if onnx else (self._infer_torch_fp16 if is_half else self._infer_torch_fp32)

    def bins_to_frequency(self, bins):
        """
        Converts model bin indices back to frequency in Hz.

        Args:
            bins (torch.Tensor): Tensor containing predicted bin indices.

        Returns:
            torch.Tensor: Predicted fundamental frequencies in Hz.
        """

        # Ensure compatibility with specialized hardware accelerators
        if str(bins.device).startswith(("ocl", "privateuseone")): bins = bins.to(torch.float32)

        # Reconstruct pitch in cents from bins and add triangular noise for quantization smoothing
        cents = CENTS_PER_BIN * bins + 1997.3794084376191
        cents = (
            cents + cents.new_tensor(
                scipy.stats.triang.rvs(
                    c=0.5, 
                    loc=-CENTS_PER_BIN, 
                    scale=2 * CENTS_PER_BIN, 
                    size=cents.size()
                )
            )
        ) / 1200

        # Convert cents back to Hz frequency
        return 10 * 2 ** cents

    def viterbi(self, logits):
        """
        Applies Viterbi decoding algorithm on the activation logits to find the optimal path.

        Args:
            logits (torch.Tensor): Model activation outputs before softmax.

        Returns:
            tuple: (bins, frequency)
                - bins (torch.Tensor): Decoded bin indices.
                - frequency (torch.Tensor): Corresponding frequencies in Hz.
        """

        # Lazy initialization of the transition matrix for Viterbi decoding
        if self.transition is None:
            idx = torch.arange(360, device=logits.device, dtype=logits.dtype)
            # Create a symmetric penalty matrix based on absolute distance between bins
            transition = (12 - (idx[:, None] - idx[None, :]).abs()).clamp(min=0)
            # Normalize to form valid probability distribution per row
            self.transition = transition / transition.sum(dim=1, keepdim=True)

        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=1)
            bins = viterbi(probs, self.transition)

        return bins, self.bins_to_frequency(bins)
    
    def preprocess(self, audio, pad=True):
        """Resamples, frames, and normalizes raw audio into model-ready chunks.

        Args:
            audio (torch.Tensor): Input audio tensor with shape (1, samples).
            pad (bool): Whether to pad the input to preserve center-aligned frames. Default is True.

        Yields:
            torch.Tensor: Standardized, normalized audio frames batch by batch.
        """

        # Fallback to default hop length if not explicitly defined
        hop_length = (self.sample_rate // 100) if self.hop_length is None else self.hop_length

        # Perform High-Quality Resampling if the input sample rate doesn't match CREPE's 16kHz
        if self.sample_rate != SAMPLE_RATE:
            audio = torch.tensor(
                librosa.resample(
                    audio.detach().cpu().numpy().squeeze(0), 
                    orig_sr=self.sample_rate, 
                    target_sr=SAMPLE_RATE, 
                    res_type="soxr_vhq"
                ), 
                device=audio.device
            ).unsqueeze(0)
            # Adjust hop length proportionally to the new sample rate
            hop_length = int(hop_length * SAMPLE_RATE / self.sample_rate)

        # Pad or compute total frames based on padding strategy
        if pad:
            total_frames = 1 + int(audio.size(1) // hop_length)
            audio = torch.nn.functional.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        else: total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)

        # Batch slicing initialization
        batch_size = total_frames if self.batch_size is None else self.batch_size

        # Slide over audio array using `unfold` to slice out overlapping windows
        for i in range(0, total_frames, batch_size):
            frames = torch.nn.functional.unfold(
                audio[:, None, None, max(0, i * hop_length):min(audio.size(1), (i + batch_size - 1) * hop_length + WINDOW_SIZE)], 
                kernel_size=(1, WINDOW_SIZE), 
                stride=(1, hop_length)
            ).transpose(1, 2)
            
            if self.device.startswith(("ocl", "privateuseone")): frames = frames.contiguous()
            frames = frames.reshape(-1, WINDOW_SIZE).to(self.device)
            # Z-score Standardization (Zero mean, unit variance normalization per frame)
            frames -= frames.mean(dim=1, keepdim=True)
            frames /= self.eps.max(frames.std(dim=1, keepdim=True))

            yield frames

    def periodicity(self, probabilities, bins):
        """
        Extracts model confidence (periodicity) scores from active pitch bins.

        Args:
            probabilities (torch.Tensor): Softmax probabilities from model.
            bins (torch.Tensor): Decoded optimal bin tracks.

        Returns:
            torch.Tensor: Extracted periodicity mapping.
        """
        probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)
        periodicity = probs_stacked.gather(1, bins.reshape(-1, 1).to(torch.int64))
        
        return periodicity.reshape(probabilities.size(0), probabilities.size(2))

    def postprocess(self, probabilities):
        """
        Filters logits by frequency constraints, triggers Viterbi decoding, and extracts periodicity.

        Args:
            probabilities (torch.Tensor): Output activation tensor from the model inference.

        Returns:
            torch.Tensor or tuple: Pitch trajectory vector, or (pitch, periodicity) tuple.
        """

        probabilities = probabilities.detach()
        # Suppress predictions out of the configured valid F0 boundaries
        probabilities[:, :self.f0_min_bin] = -float('inf')
        probabilities[:, self.f0_max_bin:] = -float('inf')

        bins, pitch = self.viterbi(probabilities)

        if not self.return_periodicity: return pitch
        return pitch, self.periodicity(probabilities, bins)

    def compute_f0(self, audio, pad=True):
        """
        Main entry point to execute the complete pitch track extraction pipeline.

        Args:
            audio (torch.Tensor): Raw input audio signal tensor.
            pad (bool): Toggles temporal center padding. Default is True.

        Returns:
            torch.Tensor or tuple: Combined pitch values or a tuple of (pitch, periodicity).
        """

        results = []
        with torch.no_grad():
            for frames in self.preprocess(audio, pad):
                # Infer -> Reshape/Transpose -> Post-process
                result = self.postprocess(self.infer(frames).reshape(audio.size(0), -1, PITCH_BINS).transpose(1, 2))
                # Manage data transfers securely to match incoming tensor's home device
                results.append((result[0].to(audio.device), result[1].to(audio.device)) if isinstance(result, tuple) else result.to(audio.device))
        
        # Aggregate chunked sequence tracks into contiguous output
        if self.return_periodicity:
            pitch, periodicity = zip(*results)
            return torch.cat(pitch, 1), torch.cat(periodicity, 1)
        
        return torch.cat(results, 1)
    
    def _infer_torch_fp32(self, frames):
        """Performs PyTorch inference in full 32-bit floating point precision."""

        return self.model(
            frames, 
            embed=False
        )

    def _infer_torch_fp16(self, frames):
        """Performs PyTorch inference in half 16-bit floating point precision."""

        return self.model(
            frames.half(), 
            embed=False
        ).float() # It needs to be converted to Float32; otherwise, the quality will be worse.

    def _infer_onnx_non_io(self, frames):
        """Performs traditional ONNX runtime inference with CPU/GPU memory copying overhead."""

        return torch.tensor(
            self.model.run(
                ["f0"], {"frames": frames.cpu().numpy()}
            )[0],
            device=self.device
        )

    def _infer_onnx_io(self, frames):
        """Performs zero-copy optimized ONNX runtime inference using explicit I/O binding."""

        frames = frames.float().contiguous()
        device_idx = frames.device.index or 0

        io_binding = self.model.io_binding()
        # Bind input tensor memory pointer directly to ONNX runtime
        io_binding.bind_input(name="frames", device_type=self._device, device_id=device_idx, element_type=np.float32, shape=tuple(frames.shape), buffer_ptr=frames.data_ptr())

        # Bind output allocation to prevent redundant host-to-device transfers
        io_binding.bind_output(name="f0", device_type=self._device, device_id=device_idx)
        self.model.run_with_iobinding(io_binding)

        return torch.from_dlpack(io_binding.get_outputs()[0]).to(device=self.device)