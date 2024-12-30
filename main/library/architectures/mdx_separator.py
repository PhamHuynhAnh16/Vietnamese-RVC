import os
import sys
import onnx
import torch
import platform
import onnx2torch

import numpy as np
import onnxruntime as ort

from tqdm import tqdm

sys.path.append(os.getcwd())

from main.configs.config import Config
from main.library.uvr5_separator import spec_utils
from main.library.uvr5_separator.common_separator import CommonSeparator

translations = Config().translations

class MDXSeparator(CommonSeparator):
    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)
        self.segment_size = arch_config.get("segment_size")
        self.overlap = arch_config.get("overlap")
        self.batch_size = arch_config.get("batch_size", 1)
        self.hop_length = arch_config.get("hop_length")
        self.enable_denoise = arch_config.get("enable_denoise")
        self.logger.debug(translations["mdx_info"].format(batch_size=self.batch_size, segment_size=self.segment_size))
        self.logger.debug(translations["mdx_info_2"].format(overlap=self.overlap, hop_length=self.hop_length, enable_denoise=self.enable_denoise))
        self.compensate = self.model_data["compensate"]
        self.dim_f = self.model_data["mdx_dim_f_set"]
        self.dim_t = 2 ** self.model_data["mdx_dim_t_set"]
        self.n_fft = self.model_data["mdx_n_fft_scale_set"]
        self.config_yaml = self.model_data.get("config_yaml", None)
        self.logger.debug(f"{translations['mdx_info_3']}: compensate = {self.compensate}, dim_f = {self.dim_f}, dim_t = {self.dim_t}, n_fft = {self.n_fft}")
        self.logger.debug(f"{translations['mdx_info_3']}: config_yaml = {self.config_yaml}")
        self.load_model()
        self.n_bins = 0
        self.trim = 0
        self.chunk_size = 0
        self.gen_size = 0
        self.stft = None
        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None

    def load_model(self):
        self.logger.debug(translations["load_model_onnx"])

        if self.segment_size == self.dim_t:
            ort_session_options = ort.SessionOptions()
            ort_session_options.log_severity_level = 3 if self.log_level > 10 else 0
            ort_inference_session = ort.InferenceSession(self.model_path, providers=self.onnx_execution_provider, sess_options=ort_session_options)
            self.model_run = lambda spek: ort_inference_session.run(None, {"input": spek.cpu().numpy()})[0]
            self.logger.debug(translations["load_model_onnx_success"])
        else:
            self.model_run = onnx2torch.convert(onnx.load(self.model_path)) if platform.system() == 'Windows' else onnx2torch.convert(self.model_path)
            self.model_run.to(self.torch_device).eval()
            self.logger.debug(translations["onnx_to_pytorch"])

    def separate(self, audio_file_path):
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]
        self.logger.debug(translations["mix"].format(audio_file_path=self.audio_file_path))
        mix = self.prepare_mix(self.audio_file_path)
        self.logger.debug(translations["normalization_demix"])
        mix = spec_utils.normalize(wave=mix, max_peak=self.normalization_threshold)
        source = self.demix(mix)
        self.logger.debug(translations["mix_success"])
        output_files = []
        self.logger.debug(translations["process_output_file"])

        if not isinstance(self.primary_source, np.ndarray):
            self.logger.debug(translations["primary_source"])
            self.primary_source = spec_utils.normalize(wave=source, max_peak=self.normalization_threshold).T

        if not isinstance(self.secondary_source, np.ndarray):
            self.logger.debug(translations["secondary_source"])
            raw_mix = self.demix(mix, is_match_mix=True)

            if self.invert_using_spec:
                self.logger.debug(translations["invert_using_spec"])
                self.secondary_source = spec_utils.invert_stem(raw_mix, source)
            else:
                self.logger.debug(translations["invert_using_spec_2"])
                self.secondary_source = mix.T - source.T

        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem_name.lower():
            self.secondary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.secondary_stem_name})_{self.model_name}.{self.output_format.lower()}")
            self.logger.info(translations["save_secondary_stem_output_path"].format(stem_name=self.secondary_stem_name, stem_output_path=self.secondary_stem_output_path))
            self.final_process(self.secondary_stem_output_path, self.secondary_source, self.secondary_stem_name)
            output_files.append(self.secondary_stem_output_path)

        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem_name.lower():
            self.primary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.primary_stem_name})_{self.model_name}.{self.output_format.lower()}")
            if not isinstance(self.primary_source, np.ndarray): self.primary_source = source.T

            self.logger.info(translations["save_secondary_stem_output_path"].format(stem_name=self.primary_stem_name, stem_output_path=self.primary_stem_output_path))
            self.final_process(self.primary_stem_output_path, self.primary_source, self.primary_stem_name)
            output_files.append(self.primary_stem_output_path)

        return output_files

    def initialize_model_settings(self):
        self.logger.debug(translations["starting_model"])

        self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft // 2

        self.chunk_size = self.hop_length * (self.segment_size - 1)
        self.gen_size = self.chunk_size - 2 * self.trim

        self.stft = STFT(self.logger, self.n_fft, self.hop_length, self.dim_f, self.torch_device)

        self.logger.debug(f"{translations['input_info']}: n_fft = {self.n_fft} hop_length = {self.hop_length} dim_f = {self.dim_f}")
        self.logger.debug(f"{translations['model_settings']}: n_bins = {self.n_bins}, Trim = {self.trim}, chunk_size = {self.chunk_size}, gen_size = {self.gen_size}")

    def initialize_mix(self, mix, is_ckpt=False):
        self.logger.debug(translations["initialize_mix"].format(is_ckpt=is_ckpt, shape=mix.shape))

        if mix.shape[0] != 2:
            error_message = translations["!=2"].format(shape=mix.shape[0])
            self.logger.error(error_message)
            raise ValueError(error_message)

        if is_ckpt:
            self.logger.debug(translations["process_check"])
            pad = self.gen_size + self.trim - (mix.shape[-1] % self.gen_size)
            self.logger.debug(f"{translations['cache']}: {pad}")

            mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, pad), dtype="float32")), 1)

            num_chunks = mixture.shape[-1] // self.gen_size
            self.logger.debug(translations["shape"].format(shape=mixture.shape, num_chunks=num_chunks))

            mix_waves = [mixture[:, i * self.gen_size : i * self.gen_size + self.chunk_size] for i in range(num_chunks)]
        else:
            self.logger.debug(translations["process_no_check"])
            mix_waves = []
            n_sample = mix.shape[1]

            pad = self.gen_size - n_sample % self.gen_size
            self.logger.debug(translations["n_sample_or_pad"].format(n_sample=n_sample, pad=pad))

            mix_p = np.concatenate((np.zeros((2, self.trim)), mix, np.zeros((2, pad)), np.zeros((2, self.trim))), 1)
            self.logger.debug(f"{translations['shape_2']}: {mix_p.shape}")

            i = 0
            while i < n_sample + pad:
                mix_waves.append(np.array(mix_p[:, i : i + self.chunk_size]))

                self.logger.debug(translations["process_part"].format(mix_waves=len(mix_waves), i=i, ii=i + self.chunk_size))
                i += self.gen_size

        mix_waves_tensor = torch.tensor(mix_waves, dtype=torch.float32).to(self.torch_device)
        self.logger.debug(translations["mix_waves_to_tensor"].format(shape=mix_waves_tensor.shape))

        return mix_waves_tensor, pad

    def demix(self, mix, is_match_mix=False):
        self.logger.debug(f"{translations['demix_is_match_mix']}: {is_match_mix}...")
        self.initialize_model_settings()
        self.logger.debug(f"{translations['mix_shape']}: {mix.shape}")
        tar_waves_ = []

        if is_match_mix:
            chunk_size = self.hop_length * (self.segment_size - 1)
            overlap = 0.02
            self.logger.debug(translations["chunk_size_or_overlap"].format(chunk_size=chunk_size, overlap=overlap))
        else:
            chunk_size = self.chunk_size
            overlap = self.overlap
            self.logger.debug(translations["chunk_size_or_overlap_standard"].format(chunk_size=chunk_size, overlap=overlap))

        gen_size = chunk_size - 2 * self.trim
        self.logger.debug(f"{translations['calc_size']}: {gen_size}")

        mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, gen_size + self.trim - ((mix.shape[-1]) % gen_size)), dtype="float32")), 1)
        self.logger.debug(f"{translations['mix_cache']}: {mixture.shape}")

        step = int((1 - overlap) * chunk_size)
        self.logger.debug(translations["step_or_overlap"].format(step=step, overlap=overlap))

        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)

        total = 0
        total_chunks = (mixture.shape[-1] + step - 1) // step
        self.logger.debug(f"{translations['all_process_part']}: {total_chunks}")

        for i in tqdm(range(0, mixture.shape[-1], step), ncols=100, unit="f"):
            total += 1
            start = i
            end = min(i + chunk_size, mixture.shape[-1])
            self.logger.debug(translations["process_part_2"].format(total=total, total_chunks=total_chunks, start=start, end=end))

            chunk_size_actual = end - start
            window = None

            if overlap != 0:
                window = np.hanning(chunk_size_actual)
                window = np.tile(window[None, None, :], (1, 2, 1))
                self.logger.debug(translations["window"])

            mix_part_ = mixture[:, start:end]
            
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype="float32")), axis=-1)

            mix_waves = torch.tensor([mix_part_], dtype=torch.float32).to(self.torch_device).split(self.batch_size)

            total_batches = len(mix_waves)
            self.logger.debug(f"{translations['mix_or_batch']}: {total_batches}")

            with torch.no_grad():
                batches_processed = 0
                
                for mix_wave in mix_waves:
                    batches_processed += 1
                    self.logger.debug(f"{translations['mix_wave']} {batches_processed}/{total_batches}")

                    tar_waves = self.run_model(mix_wave, is_match_mix=is_match_mix)

                    if window is not None:
                        tar_waves[..., :chunk_size_actual] *= window
                        divider[..., start:end] += window
                    else: divider[..., start:end] += 1

                    result[..., start:end] += tar_waves[..., : end - start]


        self.logger.debug(translations["normalization_2"])
        tar_waves = result / divider
        tar_waves_.append(tar_waves)

        tar_waves = np.concatenate(np.vstack(tar_waves_)[:, :, self.trim : -self.trim], axis=-1)[:, : mix.shape[-1]]

        source = tar_waves[:, 0:None]
        self.logger.debug(f"{translations['tar_waves']}: {tar_waves.shape}")

        if not is_match_mix:
            source *= self.compensate
            self.logger.debug(translations["mix_match"])

        self.logger.debug(translations["mix_success"])
        return source

    def run_model(self, mix, is_match_mix=False):
        spek = self.stft(mix.to(self.torch_device))
        self.logger.debug(translations["stft_2"].format(shape=spek.shape))

        spek[:, :, :3, :] *= 0

        if is_match_mix:
            spec_pred = spek.cpu().numpy()
            self.logger.debug(translations["is_match_mix"])
        else:
            if self.enable_denoise:
                spec_pred_neg = self.model_run(-spek)  
                spec_pred_pos = self.model_run(spek)
                spec_pred = (spec_pred_neg * -0.5) + (spec_pred_pos * 0.5)
                self.logger.debug(translations["enable_denoise"])
            else:
                spec_pred = self.model_run(spek)
                self.logger.debug(translations["no_denoise"])

        result = self.stft.inverse(torch.tensor(spec_pred).to(self.torch_device)).cpu().detach().numpy()
        self.logger.debug(f"{translations['stft']}: {result.shape}")

        return result

class STFT:
    def __init__(self, logger, n_fft, hop_length, dim_f, device):
        self.logger = logger
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim_f = dim_f
        self.device = device
        self.hann_window = torch.hann_window(window_length=self.n_fft, periodic=True)

    def __call__(self, input_tensor):
        is_non_standard_device = not input_tensor.device.type in ["cuda", "cpu"]

        if is_non_standard_device: input_tensor = input_tensor.cpu()

        batch_dimensions = input_tensor.shape[:-2]
        channel_dim, time_dim = input_tensor.shape[-2:]

        permuted_stft_output = torch.stft(input_tensor.reshape([-1, time_dim]), n_fft=self.n_fft, hop_length=self.hop_length, window=self.hann_window.to(input_tensor.device), center=True, return_complex=False).permute([0, 3, 1, 2])
        final_output = permuted_stft_output.reshape([*batch_dimensions, channel_dim, 2, -1, permuted_stft_output.shape[-1]]).reshape([*batch_dimensions, channel_dim * 2, -1, permuted_stft_output.shape[-1]])

        if is_non_standard_device: final_output = final_output.to(self.device)
        return final_output[..., : self.dim_f, :]

    def pad_frequency_dimension(self, input_tensor, batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins):
        return torch.cat([input_tensor, torch.zeros([*batch_dimensions, channel_dim, num_freq_bins - freq_dim, time_dim]).to(input_tensor.device)], -2)

    def calculate_inverse_dimensions(self, input_tensor):
        channel_dim, freq_dim, time_dim = input_tensor.shape[-3:]

        return input_tensor.shape[:-3], channel_dim, freq_dim, time_dim, self.n_fft // 2 + 1

    def prepare_for_istft(self, padded_tensor, batch_dimensions, channel_dim, num_freq_bins, time_dim):
        permuted_tensor = padded_tensor.reshape([*batch_dimensions, channel_dim // 2, 2, num_freq_bins, time_dim]).reshape([-1, 2, num_freq_bins, time_dim]).permute([0, 2, 3, 1])

        return permuted_tensor[..., 0] + permuted_tensor[..., 1] * 1.0j

    def inverse(self, input_tensor):
        is_non_standard_device = not input_tensor.device.type in ["cuda", "cpu"]
        if is_non_standard_device: input_tensor = input_tensor.cpu()

        batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins = self.calculate_inverse_dimensions(input_tensor)
        final_output = torch.istft(self.prepare_for_istft(self.pad_frequency_dimension(input_tensor, batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins), batch_dimensions, channel_dim, num_freq_bins, time_dim), n_fft=self.n_fft, hop_length=self.hop_length, window=self.hann_window.to(input_tensor.device), center=True).reshape([*batch_dimensions, 2, -1])

        if is_non_standard_device: final_output = final_output.to(self.device)

        return final_output