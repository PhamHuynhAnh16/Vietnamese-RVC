import os
import sys
import math
import torch
import librosa
import audioread

import numpy as np

from tqdm import tqdm

sys.path.append(os.getcwd())

from main.library.uvr5_lib import spec_utils
from main.app.variables import configs, config
from main.library.uvr5_lib.vr_network import nets
from main.library.uvr5_lib.vr_network import nets_new
from main.library.uvr5_lib.common_separator import CommonSeparator
from main.library.uvr5_lib.vr_network.model_param_init import ModelParameters

class VRSeparator(CommonSeparator):
    """
    Inference runner for the VR (Vocal Remover) Architecture models within UVR5.
    Handles multi-band processing, dynamic neural network layer capacity allocation, 
    Test-Time Augmentation (TTA), and high-end frequency reconstruction.
    """

    def __init__(self, common_config, arch_config):
        """
        Initializes the VR model architecture, determines channel configurations, 
        and maps training hyper-parameters from binary parameter graphs.

        Args:
            common_config (dict): Global shared parameter mappings from the controller.
            arch_config (dict): Architecture-specific configuration keys for VR processing.
        """

        super().__init__(config=common_config)
        self.model_capacity = (32, 128) # Default baseline channel dimensions (nout, nout_lstm)
        self.is_vr_51_model = False
        self.input_high_end_h = None
        self.input_high_end = None

        # Verify if model metadata contains modern UVR v5.1 feature keys
        if "nout" in self.model_data.keys() and "nout_lstm" in self.model_data.keys():
            self.model_capacity = (self.model_data["nout"], self.model_data["nout_lstm"])
            self.is_vr_51_model = True

        # Parse structural parameter guidelines from local binary map
        self.model_params = ModelParameters(os.path.join(configs["binary_path"], "vr_params.bin"), f"{self.model_data['vr_model_param']}.json")
        self.aggressiveness = {"value": float(int(arch_config.get("aggression", 5)) / 100), "split_bin": self.model_params.param["band"]["1"]["crop_stop"], "aggr_correction": self.model_params.param.get("aggr_correction")}
        self.post_process_threshold = arch_config.get("post_process_threshold", 0.2)
        # Bind structural audio runtime thresholds
        self.enable_post_process = arch_config.get("enable_post_process", False)
        self.model_samplerate = self.sample_rate = self.model_params.param["sr"]
        self.high_end_process = arch_config.get("high_end_process", False)
        self.enable_tta = arch_config.get("enable_tta", False)
        self.window_size = arch_config.get("window_size", 512)
        self.batch_size = arch_config.get("batch_size", 1)
        # Match model precision standards against hardware constraints
        self.dtype = torch.float16 if config.is_half else torch.float32
        # Reverse-engineer the target network capacity by scanning total filesystem byte size
        nn_arch_size = min([31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227], key=lambda x: abs(x - math.ceil(os.stat(self.model_path).st_size / 1024)))
        if nn_arch_size in [56817, 218409] or self.is_vr_51_model:
            self.model_run = nets_new.CascadedNet(self.model_params.param["bins"] * 2, nn_arch_size, nout=self.model_capacity[0], nout_lstm=self.model_capacity[1])
            self.is_vr_51_model = True
        else: self.model_run = nets.determine_model_capacity(self.model_params.param["bins"] * 2, nn_arch_size)

        # Inject checkpoint state keys into instance memory mappings
        self.model_run.load_state_dict(torch.load(self.model_path, map_location="cpu", weights_only=True))
        self.model_run.to(self.torch_device).to(self.dtype).eval()

    def separate(self, audio_file_path):
        """
        Orchestrates full VR pipeline separation, mapping raw mixed streams to distinct 
        primary and secondary stem audio outputs.

        Args:
            audio_file_path (str): Target filesystem file location of the source audio.

        Returns:
            list: Paths to the isolated generated audio track stems.
        """

        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]
        self.primary_source, self.secondary_source = None, None
        self.audio_file_path = audio_file_path
        output_files = []

        # Execute core neural forward propagation loop
        y_spec, v_spec = self.inference_vr(self.loading_mix(), self.torch_device)

        # Convert the secondary mask track back to a standard time-domain waveform
        if not isinstance(self.secondary_source, np.ndarray):
            self.secondary_source = self.spec_to_wav(np.nan_to_num(v_spec, nan=0.0, posinf=0.0, neginf=0.0)).T
            if not self.model_samplerate == 44100: self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T

        # Convert the primary mask track back to a standard time-domain waveform
        if not isinstance(self.primary_source, np.ndarray):
            self.primary_source = self.spec_to_wav(np.nan_to_num(y_spec, nan=0.0, posinf=0.0, neginf=0.0)).T
            if not self.model_samplerate == 44100: self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T

        # Establish standardized nomenclature file naming hierarchies
        self.secondary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.secondary_stem_name})_{self.model_name}.{self.output_format.lower()}")
        self.primary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.primary_stem_name})_{self.model_name}.{self.output_format.lower()}")

        # Export raw numpy data arrays out into actual file storage layouts
        self.final_process(self.secondary_stem_output_path, self.secondary_source, self.secondary_stem_name)
        self.final_process(self.primary_stem_output_path, self.primary_source, self.primary_stem_name)

        output_files.append(self.secondary_stem_output_path)
        output_files.append(self.primary_stem_output_path)

        return output_files

    def loading_mix(self):
        """
        Loads the source audio track, slices it across variable model frequency bands, 
        and extracts an integrated combined full master spectrogram.

        Returns:
            np.ndarray: Integrated multi-band complex master spectrogram layout.
        """

        X_wave, X_spec_s = {}, {}
        bands_n = len(self.model_params.param["band"])

        audio_file = spec_utils.write_array_to_mem(self.audio_file_path, subtype="PCM_16")
        is_mp3 = audio_file.endswith(".mp3") if isinstance(audio_file, str) else False

        # Loop through available target parameters in reverse order (High frequency bands down to base bands)
        for d in tqdm(range(bands_n, 0, -1)):
            bp = self.model_params.param["band"][str(d)]

            if d == bands_n:
                # Initialize base track extraction from target physical source file
                X_wave[d], _ = librosa.load(audio_file, sr=bp["sr"], mono=False, dtype=np.float32, res_type="soxr_vhq")
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp["hl"], bp["n_fft"], self.model_params, band=d, is_v51_model=self.is_vr_51_model)

                # Execute dynamic file error recovery routine if the stream returns blank data vectors
                if not np.any(X_wave[d]) and is_mp3: X_wave[d] = rerun_mp3(audio_file, bp["sr"])
                if X_wave[d].ndim == 1: X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else:
                # Down-sample high-band audio components to map down to lower band structural rules
                X_wave[d] = librosa.resample(X_wave[d + 1], orig_sr=self.model_params.param["band"][str(d + 1)]["sr"], target_sr=bp["sr"], res_type="soxr_vhq")
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp["hl"], bp["n_fft"], self.model_params, band=d, is_v51_model=self.is_vr_51_model)

            # Preserve top-end high-frequency components for post-inference mirroring
            if d == bands_n and self.high_end_process:
                self.input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (self.model_params.param["pre_filter_stop"] - self.model_params.param["pre_filter_start"])
                self.input_high_end = X_spec_s[d][:, bp["n_fft"] // 2 - self.input_high_end_h : bp["n_fft"] // 2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.model_params, is_v51_model=self.is_vr_51_model)
        del X_wave, X_spec_s, audio_file
        return X_spec

    def inference_vr(self, X_spec, device):
        """
        Executes sliding-window neural inference across the combined spectrogram magnitude graph.

        Args:
            X_spec (np.ndarray): Complex mixture spectrogram.
            device (torch.device): Device target allocation instance (CPU or CUDA).

        Returns:
            tuple: (primary_complex_spectrogram, secondary_complex_spectrogram)
        """

        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size

            # Slice the full padded spectrogram into regular chunk subsets
            for i in tqdm(range(patches)):
                start = i * roi_size
                X_dataset.append(X_mag_pad[:, :, start : start + self.window_size])

            X_dataset = np.asarray(X_dataset)
            # Pass sliced patches through the neural network to construct the prediction mask
            with torch.no_grad():
                mask = [np.concatenate(self.model_run.predict_mask(torch.from_numpy(X_dataset[i : i + self.batch_size]).to(device).to(self.dtype)).detach().cpu().numpy(), axis=2) for i in tqdm(range(0, patches, self.batch_size))]
                if len(mask) == 0: raise ValueError("Neural framework execution yielded empty mask frames.")
                mask = np.concatenate(mask, axis=2)

            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = False
            for stem in CommonSeparator.NON_ACCOM_STEMS:
                if stem == self.primary_stem_name: is_non_accom_stem = True

            # Scale mask tracking values against aggressiveness parameters
            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, self.aggressiveness)
            if self.enable_post_process: mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)

            # Apply softmask ratios back onto phase fields to isolate tracks
            return mask * X_mag * np.exp(1.0j * X_phase), (1 - mask) * X_mag * np.exp(1.0j * X_phase)

        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        # Pad boundaries to handle neural network convolutional offset trimming lengths
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
        mask = _execute(X_mag_pad / X_mag_pad.max(), roi_size)

        # Handle Test-Time Augmentation (TTA) using half-window phase shifts
        if self.enable_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

            # Average standard and phase-shifted predictions to reduce transient artifacts
            mask = (mask[:, :, :n_frame] + _execute(X_mag_pad / X_mag_pad.max(), roi_size)[:, :, roi_size // 2 :][:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        return postprocess(mask, X_mag, X_phase)

    def spec_to_wav(self, spec):
        """
        Converts the processed spectrogram back to a time-domain wave signal. 
        Reconstructs high frequencies via lower frequency mirroring if enabled.

        Args:
            spec (np.ndarray): Target complex spectrogram array.

        Returns:
            np.ndarray: Reassembled audio waveform signal matrix.
        """

        return (
            spec_utils.cmb_spectrogram_to_wave(spec, self.model_params, self.input_high_end_h, spec_utils.mirroring("mirroring", spec, self.input_high_end, self.model_params), is_v51_model=self.is_vr_51_model) 
            if self.high_end_process and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h else 
            spec_utils.cmb_spectrogram_to_wave(spec, self.model_params, is_v51_model=self.is_vr_51_model)
        )

def rerun_mp3(audio_file, sample_rate=44100):
    """
    Fallback recovery reader that forces alternative librosa streaming 
    loops if an MP3 file returns null data vectors during initial parsing.

    Args:
        audio_file (str): Target physical file path location on disk.
        sample_rate (int): Target resampling frequency bound parameters. Defaults to 44100.

    Returns:
        np.ndarray: Fully recovered raw multi-channel audio data waveform matrix.
    """

    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return librosa.load(audio_file, duration=track_length, mono=False, sr=sample_rate)[0]