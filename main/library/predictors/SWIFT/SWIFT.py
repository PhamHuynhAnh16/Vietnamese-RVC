import onnxruntime

import numpy as np

SAMPLE_RATE, HOP_LENGTH, FRAME_LENGTH = 16000, 256, 1024

class SWIFT:
    """
    SWIFT: Fast and Accurate Monophonic Pitch Detection ONNX Inference Wrapper class.

    Provides accelerated, lightweight fundamental frequency (F0) estimation pipeline
    execution. Supports standard sequential Numpy matrix extraction alongside advanced 
    zero-copy IO-Binding features optimized for hardware deployment loops.
    """

    def __init__(
        self, 
        model_path, 
        fmin = 50, 
        fmax = 1100, 
        confidence_threshold = 0.9, 
        providers = ["CPUExecutionProvider"]
    ):
        """
        Initializes runtime environments and execution graph nodes for the SwiftF0 wrapper.

        Args:
            model_path (str): File path pointing to the serialized standalone .onnx model asset.
            fmin (float, optional): Floor frequency boundary value filter threshold. Defaults to 50.0.
            fmax (float, optional): Ceiling frequency boundary value filter threshold. Defaults to 1100.0.
            confidence_threshold (float, optional): Scalar criteria for continuous voicing filtering. Defaults to 0.9.
            providers (List[str], optional): List of target execution engines. Defaults to ["CPUExecutionProvider"].
        """

        self.fmin = fmin
        self.fmax = fmax
        self.confidence_threshold = confidence_threshold

        # 1. Parameterize runtime execution threads to enforce deterministic serialization benchmarks
        session_options = onnxruntime.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1
        session_options.log_severity_level = 3 # Suppress non-critical warning alerts
        if providers[0][0].startswith("Tensorrt"): session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 2. Track device context variables depending on target hardware strings
        self._device = "cuda" if providers[0][0].startswith(("Tensorrt", "CUDA")) else "cpu"
        # 3. Compile backend runtime graph engine structures
        self.model = onnxruntime.InferenceSession(model_path, session_options, providers=providers)
        # 4. Bind dynamic pointer configurations for execution loops based on device types
        self._extract_pitch_and_confidence = self._infer_onnx_io if providers[0][0].startswith(("Tensorrt", "CUDA", "CPU")) else self._infer_onnx_non_io

    def _infer_onnx_non_io(self, audio_16k):
        """Runs inference via standard runtime paths using copy-heavy array inputs."""

        # Add singleton dimension tracking to fulfill structural (Batch, Audio_samples) mandates
        outputs = self.model.run(
            ["pitch_hz", "confidence"], {"input_audio": audio_16k[None, :].astype(np.float32)}
        )

        return outputs[0][0], outputs[1][0]

    def _infer_onnx_io(self, audio_16k):
        """Runs accelerated model inference leveraging Zero-Copy IO-Binding setups."""

        io_binding = self.model.io_binding()
        # Bind input tensor memory addresses directly onto the processing pipeline
        io_binding.bind_cpu_input("input_audio", audio_16k[None, :].astype(np.float32))
        # Pre-allocate output buffers explicitly on the target execution device contexts
        io_binding.bind_output(name="pitch_hz", device_type=self._device)
        io_binding.bind_output(name="confidence", device_type=self._device)
        # Execute model path optimization passes
        self.model.run_with_iobinding(io_binding)

        # Unpack structural tensor data pointers back into basic NumPy format arrays
        return io_binding.get_outputs()[0].numpy()[0], io_binding.get_outputs()[1].numpy()[0]

    def _compute_voicing(self, pitch_hz, confidence):
        """Evaluates boolean masks pinpointing unvoiced and voiced frames across threshold ranges."""

        return (confidence > self.confidence_threshold) & (pitch_hz >= self.fmin) & (pitch_hz <= self.fmax)

    def _calculate_timestamps(self, n_frames):
        """Calculates precise temporal center locations corresponding to generated spectral frames.

        Returns:
            np.ndarray: Vector array mapping sample positions out in seconds.
        """

        # Formulate framing offsets based on window sizes and framing shifts
        frame_centers = np.arange(n_frames) * HOP_LENGTH + ((FRAME_LENGTH - 1) / 2 - ((FRAME_LENGTH - HOP_LENGTH) // 2))
        return frame_centers / SAMPLE_RATE

    def detect_from_array(self, audio_array):
        """
        Extracts fundamental pitches, tracking matrices, and time sequences from an audio array.

        Args:
            audio_array (np.ndarray): Target raw float voice data matrix array.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - pitch_hz: Continuous pitch predictions in Hertz (F0).
                - voicing_mask: Boolean indicator flags tracking valid voiced audio segments.
                - timestamps: Sequence positions mapping estimated points out in seconds.
        """

        # 1. Downmix multi-channel stereo arrays down to uniform mono format if required
        if audio_array.ndim > 1: audio_array = np.mean(audio_array, axis=-1)
        # 2. Enforce minimum sample boundaries to prevent engine runtime crashes
        if len(audio_array) < 256: audio_array = np.pad(audio_array, (0, max(0, 256 - len(audio_array))), mode="constant")
        # 3. Extract core prediction matrices using the configured device pipeline
        pitch_hz, confidence = self._extract_pitch_and_confidence(audio_array)

        return pitch_hz, self._compute_voicing(pitch_hz, confidence), self._calculate_timestamps(len(pitch_hz))