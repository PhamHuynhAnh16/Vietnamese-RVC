import json
import torch
import onnxruntime

import numpy as np

class ONNXRVC:
    """
    An inference wrapper for Retrieval-based Voice Conversion (RVC) models exported to ONNX format.
    
    This class supports runtime optimization via ONNX Runtime I/O Binding to achieve zero-copy memory 
    transfers between PyTorch tensors and the underlying ONNX Runtime execution providers (e.g., CUDA).
    """

    def __init__(
        self, 
        model_path, 
        providers, 
        log_severity_level = 3
    ):
        """
        Initializes the ONNXRVC instance, reads embed metadata, and setups execution routes.

        Args:
            model_path (str): File path to the target .onnx model.
            providers (List): List of registered execution providers (e.g., ['CUDAExecutionProvider']).
            log_severity_level (int, optional): Log level threshold (3 = ERROR). Defaults to 3.
        """

        # Configure the Session options to limit log pollution
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = log_severity_level
        # Initialize the actual core ONNX computational runtime session
        self.net_g = onnxruntime.InferenceSession(
            model_path, 
            sess_options=sess_options, 
            providers=providers
        )

        # Retrieve and parse serialized architectural settings attached to the ONNX graph metadata block
        metadata_dict = json.loads(self.net_g.get_modelmeta().custom_metadata_map["model_info"])
        self.use_f0 = metadata_dict.get("f0", 1)
        # Route logic according to whether the model requires F0 pitch parameters
        self.get_argument = self.get_onnx_argument_pitch if self.use_f0 else self.get_onnx_argument
        self.cpt = {"tgt_sr": metadata_dict.get("sr", 32000), "use_f0": self.use_f0, "version": metadata_dict.get("version", "v1")}

        # Allocate reusable placeholder arrays for traditional NumPy CPU runtime fallback paths
        self.sid = np.empty(1, dtype=np.int64)
        self.rate = np.ones(1, dtype=np.float32)

        # Identify execution pipeline (Optimize utilizing direct pointers if backed by CUDA or CPU)
        self._device = "cuda" if providers[0][0].startswith("CUDA") else "cpu"
        self.infer = self.infer_io if providers[0][0].startswith(("CUDA", "CPU")) else self.infer_non_io

    def get_onnx_argument_pitch(self, feats, p_len, sid, pitch = None, pitchf = None, rate = None):
        """
        Constructs standard raw NumPy dictionary inputs including pitch values (F0 enabled).

        Args:
            feats (torch.Tensor): Audio features/phone tensor.
            p_len (torch.Tensor): Length array of the features.
            sid (torch.Tensor): Speaker identity token tensor.
            pitch (torch.Tensor, optional): Integer pitch (F0) contour values.
            pitchf (torch.Tensor, optional): Floating-point normalized pitch values.
            rate (torch.Tensor, optional): Speed/stretching factor.
        """

        self.sid[0] = sid.item() 
        self.rate[0] = 1.0 if rate is None else rate.item()

        return {"phone": feats.cpu().numpy().astype(np.float32), "phone_lengths": p_len.cpu().numpy(), "sid": self.sid, "rate": self.rate, "pitch": pitch.cpu().numpy().astype(np.int64), "nsff0": pitchf.cpu().numpy().astype(np.float32)}

    def get_onnx_argument(self, feats, p_len, sid, pitch = None, pitchf = None, rate = None):
        """
        Constructs standard raw NumPy dictionary inputs excluding pitch values (F0 disabled).
        """

        self.sid[0] = sid.item() 
        self.rate[0] = 1.0 if rate is None else rate.item()
        return {"phone": feats.cpu().numpy().astype(np.float32), "phone_lengths": p_len.cpu().numpy(), "sid": self.sid, "rate": self.rate}
    
    def to(self, device = "cpu"):
        """
        Sets the target system device where output synthesized audio tensors will be stored.

        Args:
            device (str or torch.device): Target device path (e.g., 'cuda:0', 'cpu').

        Returns:
            ONNXRVC: The instance itself for method chaining.
        """

        self.device = device
        return self

    def infer_non_io(
        self, 
        feats = None, 
        p_len = None, 
        pitch = None, 
        pitchf = None, 
        sid = None, 
        rate = None
    ):
        """
        Runs classic inference by mapping device matrices back into CPU NumPy arrays.
        High data copying overhead, acts as fallback.

        Returns:
            torch.Tensor: The resulting synthesized raw audio waveform tensor.
        """

        # Execute model using classic key-value Python dictionaries
        output = self.net_g.run(
            ["audio"], (
                self.get_argument(
                    feats, 
                    p_len, 
                    sid, 
                    pitch, 
                    pitchf,
                    rate
                )
            )
        )

        # Wrap array response back into PyTorch and push to the assigned destination device
        return torch.from_numpy(output[0]).to(self.device)
    
    def infer_io(
        self, 
        feats = None, 
        p_len = None, 
        pitch = None, 
        pitchf = None, 
        sid = None, 
        rate = None
    ):
        """
        Runs highly efficient execution passing physical GPU/CPU memory addresses via I/O bindings.
        This cuts out all conversion lags from crossing frameworks.

        Returns:
            torch.Tensor: The resulting synthesized raw audio waveform tensor.
        """

        # Construct and format parameters directly inside native tensor layers
        rate = torch.FloatTensor([1.0]).to(self.device) if rate is None else rate.float()
        feats, p_len = feats.float().contiguous(), p_len.contiguous()
        device_idx = feats.device.index or 0

        # Construct individual runtime binding instance
        io_binding = self.net_g.io_binding()
        io_binding.bind_input(name="phone", device_type=self._device, device_id=device_idx, element_type=np.float32, shape=tuple(feats.shape), buffer_ptr=feats.data_ptr())
        io_binding.bind_input(name="phone_lengths", device_type=self._device, device_id=device_idx, element_type=np.int64, shape=tuple(p_len.shape), buffer_ptr=p_len.data_ptr())
        io_binding.bind_input(name="sid", device_type=self._device, device_id=device_idx, element_type=np.int64, shape=tuple(sid.shape), buffer_ptr=sid.data_ptr())
        io_binding.bind_input(name="rate", device_type=self._device, device_id=device_idx, element_type=np.float32, shape=tuple(rate.shape), buffer_ptr=rate.data_ptr())

        if self.use_f0: # Include F0 frequency layers conditionally if flagged
            pitch, pitchf = pitch.contiguous(), pitchf.float().contiguous()
            io_binding.bind_input(name="pitch", device_type=self._device, device_id=device_idx, element_type=np.int64, shape=tuple(pitch.shape), buffer_ptr=pitch.data_ptr())
            io_binding.bind_input(name="nsff0", device_type=self._device, device_id=device_idx, element_type=np.float32, shape=tuple(pitchf.shape), buffer_ptr=pitchf.data_ptr())

        # Bind output receiver allocation pointing directly to target execution space block
        io_binding.bind_output(name="audio", device_type=self._device, device_id=device_idx)
        # Direct zero-copy inference execution step
        self.net_g.run_with_iobinding(io_binding)

        # Collect target array and safely encapsulate inside PyTorch tensor structures
        return torch.from_numpy(io_binding.get_outputs()[0].numpy()).to(self.device)