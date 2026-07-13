import os
import sys
import torch
import torchaudio

import numpy as np
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.algorithm.viterbi import viterbi
from main.library.predictors.PENN.core import bins_to_cents, cents_to_frequency
from main.library.predictors.PENN.core import PITCH_BINS, CENTS_PER_BIN, frequency_to_bins, entropy, interpolate

SAMPLE_RATE, WINDOW_SIZE = 16000, 1024

class Viterbi:
    """
    Viterbi Decoder Module for hidden Markov state pitch sequence tracking.

    Smooths sequential pitch contours using a Gaussian transition penalty matrix
    and estimates localized cent frequencies via windowed centroid interpolation.
    """

    def __init__(
        self, 
        pitch_bins=1440, 
        hop_length=80, 
        sample_rate=16000, 
        local_pitch_window_size=19, 
        max_octaves_per_second=32, 
        cents_per_bin=5
    ):
        """
        Initializes Viterbi decoder parameters.

        Args:
            pitch_bins (int): Discretized frequency tracking bin count. Defaults to 1440.
            hop_length (int): Distance increment separating audio frames. Defaults to 80.
            sample_rate (int): Frequency sample tracking configuration. Defaults to 16000.
            local_pitch_window_size (int): Local window width for fine cent adjustments. Defaults to 19.
            max_octaves_per_second (float): Tracking velocity constraint bound. Defaults to 32.0.
            cents_per_bin (int): Cent logarithmic width per discrete index. Defaults to 5.
        """

        self.transition = None
        self.pitch_bins = pitch_bins
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.cents_per_bin = cents_per_bin
        self.window_size = local_pitch_window_size
        self.max_octaves_per_second = max_octaves_per_second

    def __call__(self, logits):
        """
        Decodes logit tensors into absolute continuous frequencies in Hertz.

        Args:
            logits (torch.Tensor): Output model prediction matrix.

        Returns:
            torch.Tensor: Tracked pitch tensor.
        """

        # Lazily instantiate the Gaussian probability transition matrix based on runtime constraints
        if self.transition is None:
            idx = torch.arange(self.pitch_bins, device=logits.device, dtype=logits.dtype)
            # Formulate tracking standard deviations based on time steps and bin resolution
            # Compute Gaussian log probability transitions: exp(-0.5 * (diff / sigma)^2)
            transition = (-0.5 * ((idx[:, None] - idx[None, :]).abs() / ((self.max_octaves_per_second * 1200 * (self.hop_length / self.sample_rate)) / self.cents_per_bin)) ** 2).exp()
            self.transition = transition / transition.sum(dim=1, keepdim=True)

        # 1. Transform raw logit fields into normalized probabilities across frame paths
        distributions = F.softmax(logits, dim=1)
        # 2. Extract global optimum state tracks via the classic Viterbi dynamic programming block
        bins = viterbi(distributions, self.transition)

        # 3. Apply a localized center-of-mass expected value function to retrieve high-resolution pitches
        pitch = self.local_expected_value_from_bins(bins.T, logits)
        return pitch
    
    def local_expected_value_from_bins(self, bins, logits):
        """
        Calculates fine-grained frequency estimations via center-of-mass cent pooling.

        Restricts calculations to a localized window centered around decoded discrete peak bins.
        """

        batch, pitch_bins, frames = logits.shape
        logits_flat = logits.reshape(-1, pitch_bins)
        bins_flat = bins.reshape(-1)

        # 1. Construct a local evaluation index map bounded around structural centroids
        half_win = self.window_size // 2
        steps = torch.arange(-half_win, half_win + 1, device=bins.device)
        indices = (bins_flat[:, None] + steps[None, :]).clamp(0, pitch_bins - 1)

        # 2. Gather regional logits and normalize them locally into a probability distribution
        # 3. Weight localized cent representations to get expected values, then map back to Hertz
        return cents_to_frequency((F.softmax(logits_flat.gather(1, indices), dim=1) * bins_to_cents(indices)).sum(dim=1, keepdim=True)).reshape(batch, frames)

class PENN:
    """
    Pitch Estimating Neural Network (PENN) Wrapper class.

    Handles audio pre-processing, chunk framing, inference routing 
    (supporting PyTorch, compiled torch, and zero-copy ONNX formats), 
    and decoding backends.
    """

    def __init__(
        self, 
        model_path, 
        hop_length = 80, 
        batch_size = None, 
        f0_min = 31, 
        f0_max = 1984, 
        sample_rate = 8000,
        interp_unvoiced_at = None, 
        device = None, 
        providers = None, 
        onnx = False,
        is_half = False,
        compile_model = False,
        compile_mode = None
    ):
        """
        Initializes and builds the complete neural pitch tracker execution module.

        Args:
            model_path (str): File path location pointing to model weight binaries or ONNX files.
            hop_length (int): Step distance between adjacent frames. Defaults to 80.
            batch_size (int, optional): Parallel evaluation batch split. Defaults to 384.
            f0_min (float): Floor limit tracking threshold parameter in Hertz. Defaults to 31.0.
            f0_max (float): Ceiling limit tracking threshold parameter in Hertz. Defaults to 1984.
            sample_rate (int): Raw source sample rate property configuration. Defaults to 8000.
            interp_unvoiced_at (float, optional): Threshold for interpolating over unvoiced frames. Defaults to None.
            device (str / torch.device, optional): Destination hardware context targets. Defaults to None.
            providers (List, optional): Backends configuring ONNX execution layers. Defaults to None.
            onnx (bool): Toggles ONNX runtime acceleration structures. Defaults to False.
            is_half (bool): Forces FP16 half-precision execution loops. Defaults to False.
            compile_model (bool): Enables torch.compile graph optimizations. Defaults to False.
            compile_mode (str, optional): Optimization profiles matching torch compile. Defaults to None.
        """

        self.device = device
        self.hop_length = hop_length
        self.batch_size = batch_size or 384
        # Compute bound margins in bin space indices
        self.f0_min_bin = frequency_to_bins(torch.tensor(f0_min))
        self.f0_max_bin = frequency_to_bins(torch.tensor(f0_max), torch.ceil)
        self.sample_rate = sample_rate
        self.interp_unvoiced_at = interp_unvoiced_at
        # Instantiate tracking decoders
        self.decoder = Viterbi(
            PITCH_BINS, 
            hop_length, 
            SAMPLE_RATE, 
            19, 
            32, 
            CENTS_PER_BIN
        )

        # Precompute resampling pipelines to convert non-standard inputs down to 8kHz vectors
        self.resample = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=8000, dtype=torch.float32).to(device) if self.sample_rate != 8000 else None

        if onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.PENN.fcn import FCN

            model = FCN(256, PITCH_BINS, (2, 2))
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True)['model'])
            model.to(device).eval()
            if is_half: model = model.half()
            if compile_model: model = torch.compile(model, mode=compile_mode)

        self.model = model
        # Choose specific device configuration mappings
        self._device = "cuda" if providers[0][0].startswith("CUDA") else "cpu"
        # Select target inference function routes based on deployment settings
        self.infer = (self._infer_onnx_io if providers[0][0].startswith(("CUDA", "CPU")) else self._infer_onnx_non_io) if onnx else (self._infer_torch_fp16 if is_half else self._infer_torch_fp32)
    
    def preprocess(self, audio):
        """Applies resampling, context padding, and matrix unfolding to generate overlapping audio frames."""

        if self.resample is not None: audio = self.resample(audio.float()).to(audio.dtype)
        # Apply padding symmetrically to ensure frame centers align with the original audio boundaries
        audio = F.pad(audio.to(self.device), (WINDOW_SIZE // 2, WINDOW_SIZE // 2))

        # Fragment continuous vectors into standalone overlapping contextual matrices
        frames = audio.unfold(-1, WINDOW_SIZE, self.hop_length).squeeze(0)
        frames = frames.unsqueeze(1).contiguous()

        return frames

    def postprocess(self, logits):
        """Filters frequency boundaries and decodes logit fields into pitch and periodicity maps."""

        # Enforce analytical masking penalties over frequency bounds
        logits[:, :self.f0_min_bin] = -float('inf')
        logits[:, self.f0_max_bin:] = -float('inf')

        # Extract absolute continuous pitch tracks and pitch stability tracking matrices
        pitch = self.decoder(logits)
        periodicity = entropy(logits)

        return pitch.T, periodicity.T
    
    def compute_f0(self, audio):
        """Main pipeline orchestration function evaluating fundamental frequencies from raw waveforms."""

        with torch.inference_mode():
            frames = self.preprocess(audio)
            # Execute inference chunks in batch steps to prevent out-of-memory errors
            pitch, periodicity = self.postprocess(torch.cat([self.infer(frames[i:i + self.batch_size]).detach() for i in range(0, frames.shape[0], self.batch_size)], dim=0))

            # Interpolate over unvoiced sections if an interpolation threshold is provided
            if self.interp_unvoiced_at is not None:
                pitch = interpolate(pitch, periodicity, self.interp_unvoiced_at)
                return pitch

            return pitch, periodicity
    
    def _infer_onnx_non_io(self, frames):
        """Executes traditional ONNX inference copies onto Host CPU memory blocks."""

        return torch.tensor(
            self.model.run(
                ["f0"], {"frames": frames.cpu().numpy()}
            )[0],
            device=self.device,
            dtype=torch.float32
        )

    def _infer_onnx_io(self, frames):
        """Executes optimized zero-copy inference leveraging ONNX IO-Binding features."""

        frames = frames.float().contiguous()
        device_idx = frames.device.index or 0

        io_binding = self.model.io_binding()
        # Bind the input tensor directly to its memory address on the target device
        io_binding.bind_input(name="frames", device_type=self._device, device_id=device_idx, element_type=np.float32, shape=tuple(frames.shape), buffer_ptr=frames.data_ptr())
        # Allocate memory for output buffers directly on the target device execution context
        io_binding.bind_output(name="f0", device_type=self._device, device_id=device_idx)
        self.model.run_with_iobinding(io_binding)

        return torch.tensor(
            io_binding.get_outputs()[0].numpy(),
            device=self.device,
            dtype=torch.float32
        )

    def _infer_torch_fp32(self, frames):
        """Executes standard single-precision FP32 PyTorch forward inference passes."""

        return self.model(frames)
    
    def _infer_torch_fp16(self, frames):
        """Executes half-precision FP16 PyTorch inference, upcasting outputs back to FP32."""

        return self.model(frames.half()).float()