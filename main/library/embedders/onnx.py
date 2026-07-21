import torch
import onnxruntime as ort

import numpy as np

class HubertModelONNX:
    """
    A wrapper class for running Hubert audio embedding models using ONNX Runtime.
    
    This class supports both standard NumPy-based inference and optimized 
    I/O binding for CUDA/CPU providers to minimize data transfer overhead 
    between PyTorch and ONNX Runtime.
    """

    def __init__(
        self, 
        embedder_model_path, 
        providers, 
        device
    ):
        """
        Initializes the HubertModelONNX instance.

        Args:
            embedder_model_path (str): Path to the ONNX model file (.onnx).
            providers (List): List of execution providers for ONNX Runtime.
            device (str or torch.device): PyTorch device to use for the output tensors.
        """

        # Configure session options to minimize logging verbosity (3 = ERROR level)
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        if providers[0][0].startswith("Tensorrt"): sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Initialize the ONNX Runtime Inference Session
        self.model = ort.InferenceSession(
            embedder_model_path, 
            sess_options=sess_options, 
            providers=providers
        )

        self.device = device
        self._final_proj = False # Control which output to return (False: 'feats', True: 'feats_proj')
        # Determine device type string based on the primary execution provider
        self._device = "cuda" if providers[0][0].startswith(("Tensorrt", "CUDA")) else "cpu"
        # Optimize performance: use I/O binding if running on CUDA or CPU, else fallback
        self.extract_features = self.extract_features_io if providers[0][0].startswith(("Tensorrt", "CUDA", "CPU")) else self.extract_features_non_io
        # Pre-allocate output_layer buffer based on the execution path (Torch Tensor for I/O binding, NumPy for standard)
        self.output_layer = torch.zeros((), device=self.device, dtype=torch.int64) if providers[0][0].startswith(("Tensorrt", "CUDA", "CPU")) else np.empty((), dtype=np.int64)

    def final_proj(self, source):
        """
        A placeholder identity function for final projection manipulation.

        Args:
            source (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: The unmodified source tensor.
        """
    
        return source
    
    def extract_features_non_io(self, source, output_layer = None):
        """
        Extracts features using standard ONNX Runtime inference without I/O binding.
        Data is copied from GPU to CPU to run in ONNX, then converted back.

        Args:
            source (torch.Tensor): Input audio tensor.
            output_layer (int, optional): The target layer index to extract features from.

        Returns:
            List[torch.Tensor]: A list containing the requested output tensor.
        """

        # Fill the pre-allocated NumPy array with the target layer index
        self.output_layer.fill(output_layer)
        dtype = source.dtype

        # Run inference via standard ONNX CPU/GPU data copying pipeline
        logits = self.model.run(
            # During the model export, I inadvertently gave the input and output the same name without realizing it, which is why it ended up as 'feats.1'.
            ["feats", "feats_proj"], {"feats.1": source.float().detach().cpu().numpy(), "output_layer": self.output_layer}
        )

        # Convert the chosen output back to a PyTorch tensor on the destination device
        return [
            torch.as_tensor(
                logits[int(self._final_proj)], 
                dtype=dtype, 
                device=self.device
            )
        ]

    def extract_features_io(self, source, output_layer = None):
        """
        Extracts features efficiently using ONNX Runtime I/O Binding.
        This avoids expensive data copying between PyTorch tensors and NumPy arrays.

        Args:
            source (torch.Tensor): Input audio tensor residing on CPU/GPU.
            output_layer (int, optional): The target layer index to extract features from.

        Returns:
            List[torch.Tensor]: A list containing the requested output tensor.
        """

        # In-place fill for the PyTorch buffer tensor
        self.output_layer.fill_(output_layer)
        device_idx = source.device.index or 0

        # The exported models are in Float format, so the input must also be Float.
        # And since I/O Binding requires contiguous tensors, they need to be converted before being fed in.
        dtype = source.dtype
        source = source.float().contiguous()

        # Instantiate I/O binding object
        io_binding = self.model.io_binding()
        # Bind inputs using direct memory pointers (buffer_ptr)
        io_binding.bind_input(name="feats.1", device_type=self._device, device_id=device_idx, element_type=np.float32, shape=tuple(source.shape), buffer_ptr=source.data_ptr())
        io_binding.bind_input(name="output_layer", device_type=self._device, device_id=device_idx, element_type=np.int64, shape=tuple(self.output_layer.shape), buffer_ptr=self.output_layer.data_ptr())

        # Bind outputs to device memory to prevent automatic fallback copying to CPU host
        io_binding.bind_output(name="feats", device_type=self._device, device_id=device_idx)
        io_binding.bind_output(name="feats_proj", device_type=self._device, device_id=device_idx)
        # Execute model inference with bindings
        self.model.run_with_iobinding(io_binding)

        return [torch.from_dlpack(io_binding.get_outputs()[int(self._final_proj)]).to(device=self.device, dtype=dtype)]