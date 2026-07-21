import os
import sys
import torch

import numpy as np

sys.path.append(os.getcwd())

from main.library.predictors.DJCM.spec import Spectrogram

SAMPLE_RATE, N_CLASS = 16000, 360

class DJCM:
    """
    DJCM (A Deep Joint Cascade Model for Singing Voice Separation and Vocal Pitch Estimation) wrapper.

    This class handles feature extraction (Spectrogram), overlapped audio segmentation, 
    multi-backend inference (PyTorch/ONNX), and local-average decoding to convert
    salience maps into absolute fundamental frequencies (F0).
    """

    def __init__(
        self, 
        model_path, 
        device = "cpu", 
        is_half = False, 
        onnx = False, 
        svs = False, 
        providers = ["CPUExecutionProvider"], 
        batch_size = 1, 
        segment_len = 5.12, 
        compile_model = False,
        compile_mode = None,
        return_tensor = False,
        f0_min=50, 
        f0_max=1100
    ):
        """
        Initializes the DJCM prediction engine.

        Args:
            model_path (str): Path to model weights (.pth or .onnx).
            device (str): Device hardware target ('cpu', 'cuda', etc.). Default is "cpu".
            is_half (bool): Toggles FP16 precision for PyTorch models. Default is False.
            onnx (bool): Toggles ONNX Runtime session usage. Default is False.
            svs (bool): Singing Voice Separation/Synthesis flag adjustments. Default is False.
            providers (list): Execution providers for ONNX Runtime. Default is ["CPUExecutionProvider"].
            batch_size (int): Segment processing batch count. Default is 1.
            segment_len (float): Duration in seconds for sliced segments. Default is 5.12.
            compile_model (bool): Toggles torch.compile optimization. Default is False.
            compile_mode (str, optional): Compilation mode setting for PyTorch.
            return_tensor (bool): Determines if decoding returns PyTorch Tensors or NumPy arrays. Default is False.
            f0_min (float): Frequency floor in Hz for filtering predictions. Default is 50.
            f0_max (float): Frequency ceiling in Hz for filtering predictions. Default is 1100.
        """

        super(DJCM, self).__init__()
        # Adjust analysis window size dynamically depending on the task environment
        window_length = 2048 if svs else 1024

        # Model Loading Architecture
        if onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3 # Disable verbose logs
            if providers[0][0].startswith("Tensorrt"): sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.DJCM.model import DJCMM

            model = DJCMM(1, 1, 1, svs=svs, window_length=window_length, n_class=N_CLASS)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.to(device).eval()

            if is_half: model = model.half()
            if compile_model: model = torch.compile(model, mode=compile_mode)

        # Instance Variable Configuration
        self.model = model
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device
        self.batch_size = batch_size
        self.seg_len = int(segment_len * SAMPLE_RATE)
        self.dtype = torch.float16 if is_half else torch.float32
        self.seg_frames = int(self.seg_len // int(SAMPLE_RATE // 100))

        # Cents mapping vector definition along with safety pads for local group window boundaries
        self.cents_mapping = np.pad(20 * np.arange(N_CLASS) + 1997.3794084376191, (4, 4))
        self.spec_extractor = Spectrogram(int(SAMPLE_RATE // 100), window_length).to(device)
        # Structure data types according to backend pipeline specifications (Tensor vs NumPy Array)
        if return_tensor: self.cents_mapping = torch.as_tensor(self.cents_mapping, dtype=self.dtype, device=device)
        self.offsets = torch.arange(-4, 5, device=device) if return_tensor else np.arange(-4, 5)

        # Method routers map to corresponding efficient execution branches
        self._device = "cuda" if providers[0][0].startswith(("Tensorrt", "CUDA")) else "cpu"
        self.to_local_average_cents = self._to_local_average_cents_tensor if return_tensor else self._to_local_average_cents_array
        self.infer = (self._infer_onnx_io if providers[0][0].startswith(("Tensorrt", "CUDA", "CPU")) else self._infer_onnx_non_io) if onnx else (self._infer_torch_fp16 if is_half else self._infer_torch_fp32)

    def infer_from_audio(self, audio, thred=0.03):
        """
        Extracts pitch predictions directly from a raw audio waveform tensor/array.

        Args:
            audio (torch.Tensor or numpy.ndarray): Time-domain audio vector.
            thred (float): Activation confidence filtering threshold. Default is 0.03.

        Returns:
            torch.Tensor or numpy.ndarray: Unfiltered F0 trajectory track.
        """

        if not torch.is_tensor(audio): audio = torch.from_numpy(audio).to(self.device)
        if audio.ndim > 1: audio = audio.squeeze()

        with torch.inference_mode():
            # Generate overlapping chunks of specific durations
            segments = self.pad_audio(audio)

            # Batched spectrogram extraction, model forward pass, and center-frame stitching
            hidden = torch.cat([
                seg[self.seg_frames // 4: int(self.seg_frames * 0.75)] 
                for seg in torch.cat([
                    self.infer(self.spec_extractor(segments[i:i + self.batch_size].float())) 
                    for i in range(0, len(segments), self.batch_size)
                ], dim=0) 
            ], dim=0)[:(audio.shape[-1] // int(SAMPLE_RATE // 100) + 1)].squeeze(0)

            return self.decode(hidden, thred)
        
    def infer_from_audio_with_pitch(self, audio, thred=0.03):
        """
        Extracts F0 tracks from raw audio and automatically applies boundary filters.

        Args:
            audio (torch.Tensor or numpy.ndarray): Time-domain audio vector.
            thred (float): Activation confidence threshold. Default is 0.03.

        Returns:
            torch.Tensor or numpy.ndarray: Bounded F0 track where unvoiced/out-of-bounds frames are 0.
        """

        f0 = self.infer_from_audio(audio, thred)
        # Clamp out-of-bounds frequencies down to zero indicating unvoiced regions
        f0[(f0 < self.f0_min) | (f0 > self.f0_max)] = 0

        return f0

    def decode(self, hidden, thred=0.03):
        """
        Converts model activation salience states into frequencies (Hz).

        Args:
            hidden (torch.Tensor): Raw matrix output from inference.
            thred (float): Salience masking threshold. Default is 0.03.

        Returns:
            torch.Tensor or numpy.ndarray: Estimated F0 pitch matrix.
        """

        # Convert cents array estimates out to regular frequency values
        f0 = 10 * (2 ** (self.to_local_average_cents(hidden, thred=thred) / 1200))
        # Mask out trivial floor configurations representing silence or invalid bounds
        f0[f0 == 10] = 0
        return f0

    def pad_audio(self, audio):
        """
        Applies symmetrical padding and unrolls audio array into overlapping 2D segment windows.

        Args:
            audio (torch.Tensor): Unprocessed 1D input audio vector.

        Returns:
            torch.Tensor: Windowed structured frame blocks.
        """

        seg_len = self.seg_len
        hop = seg_len // 2

        audio_len = audio.shape[-1]
        left_pad = seg_len // 4

        # Compute remaining overlap tail size to match perfect integer windows before flattening layout
        return torch.nn.functional.pad(
            audio, 
            (left_pad, (((audio_len + seg_len - 1) // seg_len + 1) * seg_len + hop) - audio_len - left_pad)
        ).unfold(0, seg_len, hop).unsqueeze(1).contiguous()

    def _infer_torch_fp32(self, spec):
        """Performs PyTorch inference in standard 32-bit floating point precision."""

        return self.model(spec)

    def _infer_torch_fp16(self, spec):
        """Performs PyTorch inference in half 16-bit floating point precision."""

        return self.model(spec.half())

    def _infer_onnx_non_io(self, spec):
        """Performs legacy ONNX runtime execution with CPU conversion/copy overhead."""

        spec = spec.cpu().numpy().astype(np.float32)

        return torch.as_tensor(
            self.model.run(
                ["f0"], {"spec": spec}
            )[0], 
            device=self.device,
            dtype=self.dtype
        )

    def _infer_onnx_io(self, spec):
        """Executes zero-copy optimized ONNX runtime inference via strict explicit I/O allocation."""

        spec = spec.float().contiguous()
        device_idx = spec.device.index or 0

        io_binding = self.model.io_binding()
        # Direct raw pointer bindings skip duplicate memory allocations across systems
        io_binding.bind_input(name="spec", device_type=self._device, device_id=device_idx, element_type=np.float32, shape=tuple(spec.shape), buffer_ptr=spec.data_ptr())

        io_binding.bind_output(name="f0", device_type=self._device, device_id=device_idx)
        self.model.run_with_iobinding(io_binding)

        return torch.from_dlpack(io_binding.get_outputs()[0]).to(device=self.device, dtype=self.dtype)

    def _to_local_average_cents_array(self, salience, thred=0.05):
        """Decodes pitch tracking salience into smooth cents via NumPy arrays."""

        salience = salience.cpu().numpy().astype(np.float32)

        # Extract argmax activation positions
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4 # Offset position map adjustment due to padding shift

        # Slice out a neighborhood window of 9 elements around the argmax peak position
        idx = center[:, None] + self.offsets[None, :]
        local_salience = salience[np.arange(salience.shape[0])[:, None], idx]

        # Calculate localized center of mass (weighted average) to achieve sub-bin frequency interpolation
        devided = np.sum(local_salience * self.cents_mapping[idx], axis=1) / np.sum(local_salience, axis=1)
        # Suppress noisy frames falling below confidence threshold constraints
        devided[np.max(salience, axis=1) <= thred] = 0

        return devided

    def _to_local_average_cents_tensor(self, salience, thred=0.05):
        """Decodes pitch tracking salience into smooth cents via PyTorch Tensors on GPU/CPU."""

        center = torch.argmax(salience, dim=1)
        salience = torch.nn.functional.pad(salience, (4, 4))
        center += 4
        
        # Vectorized gathering operation across tensor indices
        idx = center[:, None] + self.offsets[None, :]
        local_salience = salience[torch.arange(salience.shape[0], device=salience.device)[:, None], idx]

        # Parallel math matrix computation for weighted frequency centroid decoding
        devided = (local_salience * self.cents_mapping[idx]).sum(dim=1) / local_salience.sum(dim=1)
        devided = torch.where(salience.max(dim=1).values <= thred, torch.zeros_like(devided), devided)

        return devided