import os
import sys
import onnx
import torch
import platform
import warnings
import onnx2torch

import numpy as np
import onnxruntime as ort

from tqdm import tqdm

sys.path.append(os.getcwd())

from main.app.variables import config
from main.library.uvr5_lib import spec_utils
from main.library.uvr5_lib.common_separator import CommonSeparator

if not config.debug_mode: warnings.filterwarnings("ignore")

class MDXSeparator(CommonSeparator):
    """
    Audio separator specialized for MDX (Multi-band Demixing) models.
    Supports native ONNX Runtime inference using optimized IO-bindings or 
    fallback dynamic graph execution using onnx2torch wrappers.
    """

    def __init__(self, common_config, arch_config):
        """
        Initializes the MDX model runtime environment, parsing windowing specifications,
        Fourier transform targets, and establishing either an ONNX or PyTorch session.

        Args:
            common_config (dict): Core configuration values passed to the base separator.
            arch_config (dict): Architecture-specific hyperparameters for MDX block processing.
        """

        super().__init__(config=common_config)
        # Load hyperparameter constraints from the structural config dictionary
        self.enable_denoise = arch_config.get("enable_denoise")
        self.segment_size = arch_config.get("segment_size")
        self.batch_size = arch_config.get("batch_size", 1)
        self.hop_length = arch_config.get("hop_length")
        self.overlap = arch_config.get("overlap")
        # Load specific metadata parameters packed inside model definitions
        self.config_yaml = self.model_data.get("config_yaml", None)
        self.n_fft = self.model_data["mdx_n_fft_scale_set"]
        self.dim_t = 2 ** self.model_data["mdx_dim_t_set"]
        self.compensate = self.model_data["compensate"]
        self.dim_f = self.model_data["mdx_dim_f_set"]
        # Optimized Pipeline: Direct ONNX execution via shared hardware pointer memory map
        if self.segment_size == self.dim_t:
            ort_session_options = ort.SessionOptions()
            ort_session_options.log_severity_level = 3 # Keep logging minimal (errors only)
            if self.onnx_execution_provider[0][0].startswith("Tensorrt"): ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Instantiate the execution session mapping to user hardware targets
            ort_inference_session = ort.InferenceSession(self.model_path, providers=self.onnx_execution_provider, sess_options=ort_session_options)
            device = "cuda" if self.onnx_execution_provider[0][0].startswith(("Tensorrt", "CUDA")) else "cpu"

            def run_io(spek):
                """
                Binds tensor data pointers directly to ONNX runtime buffers to eliminate CPU/GPU transfer overhead.
                """

                spek = spek.float().contiguous()
                device_idx = spek.device.index or 0

                # Instantiate and allocate bindings matching the active context device
                io_binding = ort_inference_session.io_binding()
                io_binding.bind_input(name="input", device_type=device, device_id=device_idx, element_type=np.float32, shape=tuple(spek.shape), buffer_ptr=spek.data_ptr())
                io_binding.bind_output(name="output", device_type=device, device_id=device_idx)
                # Execute engine evaluation directly on target hardware buffer
                ort_inference_session.run_with_iobinding(io_binding)
                return io_binding.get_outputs()[0].numpy()

            # Dynamic assignment of the execution lambda depending on hardware compatibility
            self.model_run = lambda spek: run_io(spek) if self.onnx_execution_provider[0][0].startswith(("Tensorrt", "CUDA", "CPU")) else ort_inference_session.run(None, {"input": spek.cpu().numpy()})[0]
        else:
            # Fallback Pipeline: Convert ONNX graphs to native PyTorch modules for flexible segment window slicing
            self.model_run = onnx2torch.convert(onnx.load(self.model_path)) if platform.system() == 'Windows' else onnx2torch.convert(self.model_path)
            self.model_run.to(self.torch_device).eval()

        # Placeholders for STFT parameter mappings
        self.trim = 0
        self.n_bins = 0
        self.gen_size = 0
        self.chunk_size = 0

        self.stft = None
        self.primary_source = None
        self.audio_file_path = None
        self.audio_file_base = None
        self.secondary_source = None

    def separate(self, audio_file_path):
        """
        Executes structural source separation on an target audio file using MDX parameters.

        Args:
            audio_file_path (str): Target filesystem location of the audio file to split.

        Returns:
            list: Paths to the exported primary and secondary stem audio tracks.
        """

        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]
        self.audio_file_path = audio_file_path
        # Normalize whole track mix to prevent quantization noise artifacts
        mix = spec_utils.normalize(wave=self.prepare_mix(self.audio_file_path), max_peak=self.normalization_threshold)
        # Infer target sources and rescale values relative to original waveform amplitude limits
        source = self.demix(mix) * np.abs(mix).max()
        output_files = []

        if not isinstance(self.primary_source, np.ndarray): self.primary_source = source.T
        if not isinstance(self.secondary_source, np.ndarray):
            if self.invert_using_spec: self.secondary_source = spec_utils.invert_stem(self.demix(mix, is_match_mix=True), source) # Frequency-domain algorithmic inversion matching phase properties
            else: self.secondary_source = (-self.primary_source * self.compensate) + mix.T # Standard time-domain phase subtraction method

        # Compose output filenames incorporating track metadata definitions
        self.secondary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.secondary_stem_name})_{self.model_name}.{self.output_format.lower()}")
        self.primary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.primary_stem_name})_{self.model_name}.{self.output_format.lower()}")

        # Run file-saving IO loops
        self.final_process(self.secondary_stem_output_path, self.secondary_source, self.secondary_stem_name)
        self.final_process(self.primary_stem_output_path, self.primary_source, self.primary_stem_name)

        output_files.append(self.secondary_stem_output_path)
        output_files.append(self.primary_stem_output_path)

        return output_files

    def initialize_model_settings(self):
        """
        Calculates window sizes, processing chunk layouts, and initializes the internal STFT helper module.
        """

        self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop_length * (self.segment_size - 1)
        self.gen_size = self.chunk_size - 2 * self.trim
        self.stft = STFT(self.n_fft, self.hop_length, self.dim_f, self.torch_device)

    def demix(self, mix, is_match_mix=False):
        """
        Splits full waveforms using an overlapping chunk overlap loop method to run inference.

        Args:
            mix (np.ndarray): Target audio waveform track matrix.
            is_match_mix (bool, optional): Override parameter to lock window sizes during phase matching steps.

        Returns:
            np.ndarray: Evaluated target audio waveform array.
        """

        self.initialize_model_settings()
        tar_waves_list = []

        # Configure overlapping window geometries
        if is_match_mix:
            chunk_size = self.hop_length * (self.segment_size - 1)
            overlap = 0.02
        else:
            chunk_size = self.chunk_size
            overlap = self.overlap

        gen_size = chunk_size - 2 * self.trim
        # Zero-pad outer boundaries of the audio array block to handle edge window fading smoothly
        mixture = np.concatenate((np.zeros((2, self.trim), dtype=np.float32), mix, np.zeros((2, gen_size + self.trim - ((mix.shape[-1]) % gen_size)), dtype=np.float32)), 1)
        step = int((1 - overlap) * chunk_size)

        # Allocate matching memory matrices to compute overlapped window reconstructions
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        total = 0

        # Segment iteration loop over audio length slices
        for i in tqdm(range(0, mixture.shape[-1], step), ncols=100, unit="f"):
            total += 1
            start = i
            end = min(i + chunk_size, mixture.shape[-1])

            chunk_size_actual = end - start
            window = None

            # Calculate a Hanning cross-fade window function to avoid audio clicking artifacts at block edges
            if overlap != 0: window = np.tile(np.hanning(chunk_size_actual)[None, None, :], (1, 2, 1))
            mix_part = mixture[:, start:end]
            # Zero-pad incomplete trailing context frames
            if end != i + chunk_size: mix_part = np.concatenate((mix_part, np.zeros((2, (i + chunk_size) - end), dtype=np.float32)), axis=-1)
            mix_waves = torch.from_numpy(np.array([mix_part])).float().to(self.torch_device).split(self.batch_size)

            with torch.no_grad():
                batches_processed = 0
                
                for mix_wave in mix_waves:
                    batches_processed += 1
                    tar_waves = self.run_model(mix_wave, is_match_mix=is_match_mix)

                    # Accumulate calculations scaled relative to localized window fading weight properties
                    if window is not None:
                        tar_waves[..., :chunk_size_actual] *= window
                        divider[..., start:end] += window
                    else: divider[..., start:end] += 1

                    result[..., start:end] += tar_waves[..., : end - start]

        # Normalize cumulative multi-pass windows, strip bounding pad blocks, and crop to exact sample length
        tar_waves_list.append(result / (divider + 1e-8))
        source = np.concatenate(np.vstack(tar_waves_list)[:, :, self.trim : -self.trim], axis=-1)[:, : mix.shape[-1]][:, 0:None]

        return source

    def run_model(self, mix, is_match_mix=False):
        """
        Converts waveforms to spectrogram representations, passes them to target execution engines, 
        and applies denoising.

        Args:
            mix (torch.Tensor): Pytorch signal batch tensor to process.
            is_match_mix (bool, optional): Skip execution flag during phase inversion alignment.

        Returns:
            np.ndarray: Reconstructed source audio waveform track.
        """

        spek = self.stft(mix.to(self.torch_device))
        # Zero-out ultra-low frequency sub-bass noise components to clean processing artifacts
        spek[:, :, :3, :] *= 0

        if is_match_mix: spec_pred = spek.cpu().numpy()
        else: spec_pred = ((self.model_run(-spek) * -0.5) + (self.model_run(spek) * 0.5)) if self.enable_denoise else self.model_run(spek) # Denoise trick: Evaluate raw signals alongside phase-inverted counterparts to balance distortion artifacts

        # Reconstruct output tracking matrices back into time-domain waveforms
        result = self.stft.inverse((torch.from_numpy(spec_pred) if spec_pred.dtype == np.float32 else spec_pred).to(self.torch_device)).cpu().detach().numpy()
        return result

class STFT:
    """
    An optimized Short-Time Fourier Transform (STFT) wrapper handling multi-channel structures, 
    frequency padding tricks, and hardware target device shifts.
    """

    def __init__(self, n_fft, hop_length, dim_f, device):
        """
        Initializes windowing parameters and pre-allocates standard Hann window curves on hardware.

        Args:
            n_fft (int): Size of Fourier Transform context windows.
            hop_length (int): Frame shift distance stepping properties.
            dim_f (int): Target maximum frequency bin to slice during execution.
            device (torch.device): Host execution target for tensor shifts.
        """

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim_f = dim_f
        self.device = device
        self.hann_window = torch.hann_window(window_length=self.n_fft, periodic=True)

    def __call__(self, input_tensor):
        """
        Executes a forward STFT, merging stereo data dimensions into target model layout shapes.

        Args:
            input_tensor (torch.Tensor): Raw 1D/2D audio signal matrices.

        Returns:
            torch.Tensor: Permuted spectrogram tensor.
        """

        # Fall back to CPU processing if handling custom non-standard accelerators
        is_non_standard_device = not input_tensor.device.type in ["cuda", "xpu", "cpu"]
        if is_non_standard_device: input_tensor = input_tensor.cpu()

        batch_dimensions = input_tensor.shape[:-2]
        channel_dim, time_dim = input_tensor.shape[-2:]

        # Extract complex spectrogram frames and convert real/imaginary components into explicit tensor axes
        stft_output = torch.view_as_real(torch.stft(input_tensor.reshape([-1, time_dim]), n_fft=self.n_fft, hop_length=self.hop_length, window=self.hann_window.to(input_tensor.device), center=True, return_complex=True)).permute([0, 3, 1, 2])
        # Interleave channel and real/imag dimensions to meet MDX expected model structural shape constraints
        final_output = stft_output.reshape([*batch_dimensions, channel_dim, 2, -1, stft_output.shape[-1]]).reshape([*batch_dimensions, channel_dim * 2, -1, stft_output.shape[-1]])

        if is_non_standard_device: final_output = final_output.to(self.device)
        return final_output[..., : self.dim_f, :]

    def pad_frequency_dimension(self, input_tensor, batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins):
        """
        Pads stripped high-frequency spectrogram matrices with zeros back to standard size for correct inverse parsing.
        """

        return torch.cat([input_tensor, torch.zeros([*batch_dimensions, channel_dim, num_freq_bins - freq_dim, time_dim]).to(input_tensor.device)], -2)

    def calculate_inverse_dimensions(self, input_tensor):
        """
        Decomposes dimensions and computes required window shapes to perform an Inverse Short-Time Fourier Transform.
        """

        channel_dim, freq_dim, time_dim = input_tensor.shape[-3:]

        return input_tensor.shape[:-3], channel_dim, freq_dim, time_dim, self.n_fft // 2 + 1

    def prepare_for_istft(self, padded_tensor, batch_dimensions, channel_dim, num_freq_bins, time_dim):
        """
        Reconstructs interleaved real/imaginary feature matrices back into native torch-compliant complex number arrays.
        """

        permuted_tensor = padded_tensor.reshape([*batch_dimensions, channel_dim // 2, 2, num_freq_bins, time_dim]).reshape([-1, 2, num_freq_bins, time_dim]).permute([0, 2, 3, 1])

        return permuted_tensor[..., 0] + permuted_tensor[..., 1] * 1.0j

    def inverse(self, input_tensor):
        """
        Transforms a multi-channel spectrogram tensor back into a standard time-domain waveform track.

        Args:
            input_tensor (torch.Tensor): Processed source spectrogram tensor.

        Returns:
            torch.Tensor: Reconstructed time-domain audio waveform track.
        """

        is_non_standard_device = not input_tensor.device.type in ["cuda", "xpu", "cpu"]
        if is_non_standard_device: input_tensor = input_tensor.cpu()

        # Unpack dimensional geometry configurations
        batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins = self.calculate_inverse_dimensions(input_tensor)
        # Execute padding, combine complex values, and run the native torch ISTFT operation
        final_output = torch.istft(
            self.prepare_for_istft(
                self.pad_frequency_dimension(
                    input_tensor, 
                    batch_dimensions, 
                    channel_dim, 
                    freq_dim, 
                    time_dim, 
                    num_freq_bins
                ), 
                batch_dimensions, 
                channel_dim, 
                num_freq_bins, 
                time_dim
            ), 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.hann_window.to(input_tensor.device), 
            center=True
        ).reshape([*batch_dimensions, 2, -1])

        if is_non_standard_device: final_output = final_output.to(self.device)
        return final_output