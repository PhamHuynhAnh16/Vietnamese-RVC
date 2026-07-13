import os
import sys
import torch

import numpy as np
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.predictors.RMVPE.mel import MelSpectrogram

N_MELS, N_CLASS = 128, 360

class RMVPE:
    """
    Robust Model for Vocal Pitch Estimation in Polyphonic Music (RMVPE).

    This class handles feature extraction (MelSpectrogram), model initialization 
    (PyTorch or ONNX Runtime), inference execution, and posterior decoding 
    into frequency cents for robust fundamental frequency (F0) estimation.
    """

    def __init__(
        self, 
        model_path, 
        is_half, 
        device=None, 
        providers=None, 
        onnx=False, 
        hpa=False, 
        compile_model=False, 
        compile_mode=None, 
        enable_chunk = False, 
        chunk_size = 8000, 
        return_tensor = False, 
        f0_min = 50, 
        f0_max = 1100
    ):
        """
        Initializes the RMVPE predictor instance.

        Args:
            model_path (str): File path to the weights or ONNX model checkpoint.
            is_half (bool): Whether to use FP16 half precision initialization.
            device (str or torch.device, optional): Device context execution handler.
            providers (list, optional): ONNX Execution Providers list configuration.
            onnx (bool, default=False): Flag to trigger ONNX backend instead of PyTorch.
            hpa (bool, default=False): Flag to trigger the YOLOv13-based Hypergraph and FullPAD backbone variant.
            compile_model (bool, default=False): Triggers torch.compile optimization.
            compile_mode (str, optional): Mode passed directly to torch.compile backend.
            enable_chunk (bool, default=False): Process long audio segments using chunk sliding.
            chunk_size (int, default=8000): Target frame window size for chunked inference.
            return_tensor (bool, default=False): Decodes via GPU/CPU Tensors instead of NumPy.
            f0_min (float, default=50): Minimum allowable voice boundary.
            f0_max (float, default=1100): Maximum allowable voice boundary.
        """

        if onnx:
            import onnxruntime as ort

            # Configure basic logging level constraints to suppress verbose initialization logs
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.RMVPE.e2e import E2E

            # Setup the End-to-End network architecture and load checkpoint weights
            model = E2E(4, 1, (2, 2), 5, 4, 1, 16, hpa=hpa)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.to(device).eval()

            # Apply runtime optimization transformations (Precision / Compilation)
            if is_half: model = model.half()
            if compile_model: model = torch.compile(model, mode=compile_mode)

        # Set foundational class object configurations
        self.model = model
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device
        self.chunk_size = chunk_size
        self.dtype = torch.float16 if is_half else torch.float32

        # Cents mapping vector definition representing localized pitch classes + center padding adjustments
        self.cents_mapping = np.pad(20 * np.arange(N_CLASS) + 1997.3794084376191, (4, 4))
        # Initialize native MelSpectrogram layer module on specified hardware target
        self.mel_extractor = MelSpectrogram(N_MELS, 16000, 1024, 160, 1024, 30, 8000).to(device)
        # If tensor tracking is requested, push tracking weights data array to target hardware execution context
        if return_tensor: self.cents_mapping = torch.as_tensor(self.cents_mapping, dtype=self.dtype, device=device)

        # Dynamically map processing handles and dynamic dispatch interfaces based on parameters
        self._device = "cuda" if providers[0][0].startswith("CUDA") else "cpu"
        self.mel2hidden = self._mel2hidden_chunk if enable_chunk else self._mel2hidden
        self.offsets = torch.arange(-4, 5, device=device) if return_tensor else np.arange(-4, 5)
        self.to_local_average_cents = self._to_local_average_cents_tensor if return_tensor else self._to_local_average_cents_array
        # Determine the runtime execution function depending on ONNX/PyTorch and Precision flags
        self.infer = (self._infer_onnx_io if providers[0][0].startswith(("CUDA", "CPU")) else self._infer_onnx_non_io) if onnx else (self._infer_torch_fp16 if is_half else self._infer_torch_fp32)

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

    def infer_from_audio(self, audio, thred=0.03):
        """
        Extracts F0 values directly from raw audio waveforms.

        Args:
            audio (numpy.ndarray or torch.Tensor): Input audio signal buffer.
            thred (float, default=0.03): Threshold boundary for activation suppression.

        Returns:
            torch.Tensor or numpy.ndarray: Predicted fundamental frequency array.
        """

        # Convert audio buffer to Torch tensor if it is provided as a NumPy array
        audio = torch.from_numpy(audio).float().to(self.device) if not torch.is_tensor(audio) else audio
        hidden = self.mel2hidden(self.mel_extractor(audio.unsqueeze(0), center=True)).squeeze(0)

        return self.decode(hidden, thred=thred)
    
    def infer_from_audio_with_pitch(self, audio, thred=0.03):
        """
        Extracts F0 values directly from audio waveforms with post-filtered frequency clipping.

        Args:
            audio (numpy.ndarray or torch.Tensor): Input audio signal buffer.
            thred (float, default=0.03): Threshold boundary for activation suppression.

        Returns:
            torch.Tensor or numpy.ndarray: Post-filtered fundamental frequency array.
        """

        # Call standard inference mapping sequence
        f0 = self.infer_from_audio(audio, thred)
        # Zero out values that fall outside the explicitly requested minimum and maximum F0 range
        f0[(f0 < self.f0_min) | (f0 > self.f0_max)] = 0  

        return f0

    def _mel2hidden(self, mel):
        """Processes complete mel spectrogram through the inference pipeline with reflection padding."""

        with torch.inference_mode():
            n_frames = mel.shape[-1]
            # Ensure the frame length is an exact multiple of 32 via reflection padding configuration
            # Slice and discard extra padding tail values out after model inference execution finishes
            return self.infer(F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect"))[:, :n_frames]

    def _mel2hidden_chunk(self, mel):
        """Processes long mel spectrogram configurations using chunk-by-chunk slicing strategy."""

        with torch.inference_mode():
            n_frames = mel.shape[-1]
            # Ensure the frame length matches block padding alignment multiples
            mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect")
            # Slice out input buffers into sequential blocks, run inference, and concatenate outcomes
            hidden = torch.cat([self.infer(mel[..., start:min(start + self.chunk_size, mel.shape[-1])]) for start in range(0, mel.shape[-1], self.chunk_size)], dim=1)

            # Strip padding frames before returning internal hidden state representations
            return hidden[:, :n_frames]

    def _infer_torch_fp32(self, mel):
        """Runs PyTorch inference in standard FP32 floating point mode."""

        return self.model(mel)
    
    def _infer_torch_fp16(self, mel):
        """Runs PyTorch inference casting target input structures to FP16 half precision."""

        return self.model(mel.half())

    def _infer_onnx_non_io(self, mel):
        """Executes ONNX Runtime inference using standard CPU/GPU memory copy operations."""

        # Convert PyTorch Tensor input layout to native NumPy structures required by standard ONNX bindings
        mel = mel.cpu().numpy().astype(np.float32)
        # Run model session execution mapping arrays back onto standard Torch Tensor objects
        return torch.as_tensor(
            self.model.run(
                ["f0"], {"mel": mel}
            )[0], 
            device=self.device,
            dtype=self.dtype
        )

    def _infer_onnx_io(self, mel):
        """Executes optimized ONNX Runtime inference using low-overhead I/O binding allocations."""
        mel = mel.float().contiguous()
        device_idx = mel.device.index or 0

        # Create I/O binding instance context to pin memory references directly onto hardware device
        io_binding = self.model.io_binding()
        # Bind input layer pointers bypassing intermediate CPU-GPU memory context copies
        io_binding.bind_input(name="mel", device_type=self._device, device_id=device_idx, element_type=np.float32, shape=tuple(mel.shape), buffer_ptr=mel.data_ptr())

        # Bind output structure memory allocations directly
        io_binding.bind_output(name="f0", device_type=self._device, device_id=device_idx)
        # Run optimized graph inference execution
        self.model.run_with_iobinding(io_binding)

        # Wrap unmanaged internal memory pointer buffer objects into standard PyTorch abstractions
        return torch.as_tensor(
            io_binding.get_outputs()[0].numpy(), 
            device=self.device,
            dtype=self.dtype
        )

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
        salience = F.pad(salience, (4, 4))
        center += 4
        
        # Vectorized gathering operation across tensor indices
        idx = center[:, None] + self.offsets[None, :]
        local_salience = salience[torch.arange(salience.shape[0], device=salience.device)[:, None], idx]

        # Parallel math matrix computation for weighted frequency centroid decoding
        devided = (local_salience * self.cents_mapping[idx]).sum(dim=1) / local_salience.sum(dim=1)
        devided = torch.where(salience.max(dim=1).values <= thred, torch.zeros_like(devided), devided)

        return devided