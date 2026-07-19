import os
import sys
import torch

import numpy as np

sys.path.append(os.getcwd())

class PESTO:
    """
    PESTO (Pitch Estimation with Self-supervised Transposition-equivariant Objective) Pipeline Wrapper.

    Handles continuous fundamental pitch (F0) estimation across streaming audio 
    tensors. Supports dynamic windowed chunk slicing, raw PyTorch execution graphs, 
    compiled PyTorch configurations, and zero-copy ONNX IO-Binding optimizations.
    """

    def __init__(
        self, 
        model_path, 
        step_size=10, 
        reduction="alwa", 
        sample_rate=16000, 
        device=None, 
        providers=None, 
        onnx=False,
        is_half=False,
        compile_model=False,
        compile_mode=None,
        chunk_size = None
    ):
        """
        Initializes processing properties and compiles the Transposition-equivariant model graph.

        Args:
            model_path (str | os.PathLike): File path pointing to checkpoint weights or ONNX models.
            step_size (int): Hop size factor step between sequential frame outputs. Defaults to 10.
            reduction (str): Network temporal resolution grouping policy. Defaults to "alwa".
            sample_rate (int): Sampling frequency of targeted raw waveform files. Defaults to 16000.
            device (str / torch.device, optional): Destination computational device. Defaults to None.
            providers (List, optional): Execution backends managing ONNX nodes. Defaults to None.
            onnx (bool): If True, activates ONNX runtime execution paths. Defaults to False.
            is_half (bool): Forces FP16 precision computation pipelines. Defaults to False.
            compile_model (bool): Compiles PyTorch modules via torch.compile. Defaults to False.
            compile_mode (str, optional): Compilation profile parameters. Defaults to None.
            chunk_size (int, optional): Sample count limit for block slicing. Defaults to None.
        """

        self.device = device
        self.step_size = step_size
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        if onnx:
            import onnxruntime as ort

            # Configure runtime engine sessions to suppress non-critical logging
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            if providers[0][0].startswith("Tensorrt"): sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.PESTO.model import PPESTO, Resnet1d
            from main.library.predictors.PESTO.preprocessor import Preprocessor

            # Load model parameters and state weights via standard PyTorch loaders
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            # Dynamically instantiate structural submodules using saved hyperparameters
            model = PPESTO(
                Resnet1d(
                    **ckpt["hparams"]["encoder"]
                ), 
                preprocessor=Preprocessor(
                    hop_size=step_size, 
                    sampling_rate=sample_rate, 
                    **ckpt["hcqt_params"]
                ), 
                crop_kwargs=ckpt["hparams"]["pitch_shift"], 
                reduction=reduction or ckpt["hparams"]["reduction"]
            )
            model.load_state_dict(ckpt["state_dict"], strict=False)
            model.to(self.device).eval()
            if is_half: model = model.half()
            if compile_model: model = torch.compile(model, mode=compile_mode)
        
        self.model = model
        # 1. Establish hardware string values to map execution devices correctly
        self._device = "cuda" if providers[0][0].startswith(("Tensorrt", "CUDA")) else "cpu"
        # 2. Select the orchestration strategy based on chunk constraints
        self.compute_f0 = self._compute_f0_chunk if self.chunk_size else self._compute_f0
        # 3. Route structural execution methods based on backend and datatype configurations
        self.infer = (self._infer_onnx_io if providers[0][0].startswith(("Tensorrt", "CUDA", "CPU")) else self._infer_onnx_non_io) if onnx else (self._infer_torch_fp16 if is_half else self._infer_torch_fp32)

    def _compute_f0_chunk(self, x):
        """Slices incoming waveforms into sequential blocks to manage memory consumption."""

        with torch.inference_mode():
            assert x.ndim <= 2

            preds, confidence = [], []
            total_samples = x.shape[-1]

            # Forward the raw data directly if it fits within a single chunk length boundary
            if total_samples <= self.chunk_size: return self.infer(x)

            # Iteratively process the signal step-by-step using slice windows
            for i in range(0, total_samples, self.chunk_size):
                # Maintain consistent tensor dimension structures for both mono and multi-channel audio
                pred_chunk, conf_chunk = self.infer(x[i : i + self.chunk_size] if x.ndim == 1 else x[:, i : i + self.chunk_size])
                
                preds.append(pred_chunk)
                confidence.append(conf_chunk)

            return torch.cat(preds, dim=-1), torch.cat(confidence, dim=-1)
    
    def _compute_f0(self, x):
        """Passes full waveform arrays directly into the target execution engine."""

        with torch.inference_mode():
            assert x.ndim <= 2
            return self.infer(x)

    def _infer_onnx_non_io(self, x):
        """Runs fallback ONNX inference by copying raw numpy arrays over the host memory."""

        ouput = self.model.run(
            ["preds", "conf"], {"chunk": x.cpu().numpy()}
        )

        return torch.tensor(ouput[0], device=self.device), torch.tensor(ouput[1], device=self.device)

    def _infer_onnx_io(self, x):
        """Runs accelerated ONNX inference using zero-copy IO-Binding structures."""

        x = x.float().contiguous()
        device_idx = x.device.index or 0

        io_binding = self.model.io_binding()
        # Bind the input data matrix pointer address directly onto the device context
        io_binding.bind_input(name="chunk", device_type=self._device, device_id=device_idx, element_type=np.float32, shape=tuple(x.shape), buffer_ptr=x.data_ptr())

        # Pre-allocate corresponding output arrays directly on the destination hardware
        io_binding.bind_output(name="preds", device_type=self._device, device_id=device_idx)
        io_binding.bind_output(name="conf", device_type=self._device, device_id=device_idx)
        self.model.run_with_iobinding(io_binding)

        return torch.tensor(io_binding.get_outputs()[0].numpy(), device=self.device), torch.tensor(io_binding.get_outputs()[1].numpy(), device=self.device)

    def _infer_torch_fp32(self, x):
        """Executes native single-precision FP32 PyTorch inference evaluations."""

        return self.model(
            x, 
            sr=self.sample_rate, 
            convert_to_freq=True, 
            return_activations=False
        )
    
    def _infer_torch_fp16(self, x):
        """Executes half-precision FP16 PyTorch inference evaluations."""

        return self.model(
            x.half(), 
            sr=self.sample_rate, 
            convert_to_freq=True, 
            return_activations=False
        )