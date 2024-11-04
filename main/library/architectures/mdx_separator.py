import os
import onnx
import torch
import platform
import onnx2torch

import numpy as np
import onnxruntime as ort

from tqdm import tqdm

from main.library.uvr5_separator import spec_utils
from main.library.uvr5_separator.common_separator import CommonSeparator


class MDXSeparator(CommonSeparator):
    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)

        self.segment_size = arch_config.get("segment_size")
        self.overlap = arch_config.get("overlap")
        self.batch_size = arch_config.get("batch_size", 1)
        self.hop_length = arch_config.get("hop_length")
        self.enable_denoise = arch_config.get("enable_denoise")

        self.logger.debug(f"Thông số MDX: Kích thước lô = {self.batch_size}, Kích thước phân đoạn = {self.segment_size}")
        self.logger.debug(f"Thông số MDX: Chồng chéo = {self.overlap}, Hop_length = {self.hop_length}, Kích hoạt khữ nhiễu = {self.enable_denoise}")

        self.compensate = self.model_data["compensate"]
        self.dim_f = self.model_data["mdx_dim_f_set"]
        self.dim_t = 2 ** self.model_data["mdx_dim_t_set"]
        self.n_fft = self.model_data["mdx_n_fft_scale_set"]
        self.config_yaml = self.model_data.get("config_yaml", None)

        self.logger.debug(f"Thông số MDX: compensate = {self.compensate}, dim_f = {self.dim_f}, dim_t = {self.dim_t}, n_fft = {self.n_fft}")
        self.logger.debug(f"Thông số MDX: Cấu hình yaml = {self.config_yaml}")

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
        self.logger.debug("Đang tải mô hình ONNX để suy luận...")

        if self.segment_size == self.dim_t:
            ort_session_options = ort.SessionOptions()

            if self.log_level > 10: ort_session_options.log_severity_level = 3
            else: ort_session_options.log_severity_level = 0

            ort_inference_session = ort.InferenceSession(self.model_path, providers=self.onnx_execution_provider, sess_options=ort_session_options)
            self.model_run = lambda spek: ort_inference_session.run(None, {"input": spek.cpu().numpy()})[0]
            self.logger.debug("Đã tải mô hình thành công bằng phiên suy luận ONNXruntime.")
        else:
            if platform.system() == 'Windows':
                onnx_model = onnx.load(self.model_path)
                self.model_run = onnx2torch.convert(onnx_model)
            else: self.model_run = onnx2torch.convert(self.model_path)
   
            self.model_run.to(self.torch_device).eval()
            self.logger.warning("Mô hình được chuyển đổi từ onnx sang pytorch do kích thước phân đoạn không khớp với dim_t, quá trình xử lý có thể chậm hơn.")

    def separate(self, audio_file_path):
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        self.logger.debug(f"Đang chuẩn bị trộn cho tệp âm thanh đầu vào {self.audio_file_path}...")
        mix = self.prepare_mix(self.audio_file_path)

        self.logger.debug("Chuẩn hóa hỗn hợp trước khi khử trộn...")
        mix = spec_utils.normalize(wave=mix, max_peak=self.normalization_threshold)

        source = self.demix(mix)
        self.logger.debug("Quá trình trộn hoàn tất.")

        output_files = []
        self.logger.debug("Xử lý tập tin đầu ra...")

        if not isinstance(self.primary_source, np.ndarray):
            self.logger.debug("Bình thường hóa nguồn chính...")
            self.primary_source = spec_utils.normalize(wave=source, max_peak=self.normalization_threshold).T
        if not isinstance(self.secondary_source, np.ndarray):
            self.logger.debug("Sản xuất nguồn thứ cấp: Trộn ở chế độ trộn phù hợp")
            raw_mix = self.demix(mix, is_match_mix=True)

            if self.invert_using_spec:
                self.logger.debug("Đảo ngược thân thứ cấp bằng cách sử dụng quang phổ khi invert_USE_spec được đặt thành True")
                self.secondary_source = spec_utils.invert_stem(raw_mix, source)
            else:
                self.logger.debug("Đảo ngược thân thứ cấp bằng cách trừ đi thân cây được chuyển đổi từ hỗn hợp ban đầu được chuyển đổi")
                self.secondary_source = mix.T - source.T

        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem_name.lower():
            self.secondary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.secondary_stem_name})_{self.model_name}.{self.output_format.lower()}")
            self.logger.info(f"Đang lưu phần gốc {self.secondary_stem_name} vào {self.secondary_stem_output_path}...")
            self.final_process(self.secondary_stem_output_path, self.secondary_source, self.secondary_stem_name)
            output_files.append(self.secondary_stem_output_path)

        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem_name.lower():
            self.primary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.primary_stem_name})_{self.model_name}.{self.output_format.lower()}")

            if not isinstance(self.primary_source, np.ndarray): self.primary_source = source.T

            self.logger.info(f"Đang lưu phần gốc {self.primary_stem_name} vào {self.primary_stem_output_path}...")
            self.final_process(self.primary_stem_output_path, self.primary_source, self.primary_stem_name)
            output_files.append(self.primary_stem_output_path)

        return output_files

    def initialize_model_settings(self):
        self.logger.debug("Đang khởi tạo cài đặt mô hình...")

        self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft // 2

        self.chunk_size = self.hop_length * (self.segment_size - 1)
        self.gen_size = self.chunk_size - 2 * self.trim

        self.stft = STFT(self.logger, self.n_fft, self.hop_length, self.dim_f, self.torch_device)

        self.logger.debug(f"Thông số đầu vào của mô hình: n_fft = {self.n_fft} hop_length = {self.hop_length} dim_f = {self.dim_f}")
        self.logger.debug(f"Cài đặt mô hình: n_bins = {self.n_bins}, Trim = {self.trim}, chunk_size = {self.chunk_size}, gen_size = {self.gen_size}")

    def initialize_mix(self, mix, is_ckpt=False):
        self.logger.debug(f"Đang khởi tạo kết hợp với is_ckpt = {is_ckpt}. Hình dạng trộn ban đầu: {mix.shape}")

        if mix.shape[0] != 2:
            error_message = f"Dự kiến có tín hiệu âm thanh 2 kênh nhưng lại có {mix.shape[0]} kênh"
            self.logger.error(error_message)
            raise ValueError(error_message)

        if is_ckpt:
            self.logger.debug("Xử lý ở chế độ điểm kiểm tra...")
            pad = self.gen_size + self.trim - (mix.shape[-1] % self.gen_size)
            self.logger.debug(f"Khoảng đệm được tính toán: {pad}")

            mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, pad), dtype="float32")), 1)

            num_chunks = mixture.shape[-1] // self.gen_size
            self.logger.debug(f"Hình dạng hỗn hợp sau khi đệm: {mixture.shape}, Số phần: {num_chunks}")

            mix_waves = [mixture[:, i * self.gen_size : i * self.gen_size + self.chunk_size] for i in range(num_chunks)]
        else:
            self.logger.debug("Xử lý ở chế độ không có điểm kiểm tra...")
            mix_waves = []
            n_sample = mix.shape[1]

            pad = self.gen_size - n_sample % self.gen_size
            self.logger.debug(f"Số lượng mẫu: {n_sample}, Đã tính đệm: {pad}")

            mix_p = np.concatenate((np.zeros((2, self.trim)), mix, np.zeros((2, pad)), np.zeros((2, self.trim))), 1)
            self.logger.debug(f"Hình dạng hỗn hợp sau khi đệm: {mix_p.shape}")

            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + self.chunk_size])
                mix_waves.append(waves)

                self.logger.debug(f"Đoạn đã xử lý {len(mix_waves)}: Bắt đầu {i}, Kết thúc {i + self.chunk_size}")
                i += self.gen_size

        mix_waves_tensor = torch.tensor(mix_waves, dtype=torch.float32).to(self.torch_device)
        self.logger.debug(f"Đã chuyển đổi mix_waves thành tensor. Hình dạng tensor: {mix_waves_tensor.shape}")

        return mix_waves_tensor, pad

    def demix(self, mix, is_match_mix=False):
        self.logger.debug(f"Bắt đầu quá trình hủy trộn với is_match_mix: {is_match_mix}...")
        self.initialize_model_settings()

        org_mix = mix
        self.logger.debug(f"Hỗn hợp phần gốc được lưu trữ. Hình dạng: {org_mix.shape}")

        tar_waves_ = []

        if is_match_mix:
            chunk_size = self.hop_length * (self.segment_size - 1)
            overlap = 0.02
            self.logger.debug(f"Kích thước đoạn để trộn phù hợp: {chunk_size}, Chồng chéo: {overlap}")
        else:
            chunk_size = self.chunk_size
            overlap = self.overlap
            self.logger.debug(f"Kích thước phần tiêu chuẩn: {chunk_size}, Chồng chéo: {overlap}")


        gen_size = chunk_size - 2 * self.trim
        self.logger.debug(f"Kích thước được tạo được tính toán: {gen_size}")


        pad = gen_size + self.trim - ((mix.shape[-1]) % gen_size)

        mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, pad), dtype="float32")), 1)
        self.logger.debug(f"Hỗn hợp được chuẩn bị với lớp đệm. Hình dạng hỗn hợp: {mixture.shape}")

        step = int((1 - overlap) * chunk_size)
        self.logger.debug(f"Kích thước bước để xử lý các phần: {step} khi chồng chéo được đặt thành {overlap}.")

        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)

        total = 0
        total_chunks = (mixture.shape[-1] + step - 1) // step
        self.logger.debug(f"Tổng số phần cần xử lý: {total_chunks}")

        for i in tqdm(range(0, mixture.shape[-1], step)):
            total += 1
            start = i
            end = min(i + chunk_size, mixture.shape[-1])
            self.logger.debug(f"Đang xử lý đoạn {total}/{total_chunks}: Bắt đầu {start}, Kết thúc {end}")

            chunk_size_actual = end - start
            window = None

            if overlap != 0:
                window = np.hanning(chunk_size_actual)
                window = np.tile(window[None, None, :], (1, 2, 1))
                self.logger.debug("Cửa sổ được áp dụng cho đoạn này.")

            mix_part_ = mixture[:, start:end]
            
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype="float32")), axis=-1)

            mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(self.torch_device)

            mix_waves = mix_part.split(self.batch_size)
            total_batches = len(mix_waves)
            self.logger.debug(f"Trộn phần chia thành từng đợt. Số lượng lô: {total_batches}")

            with torch.no_grad():
                batches_processed = 0
                for mix_wave in mix_waves:
                    batches_processed += 1
                    self.logger.debug(f"Đang xử lý lô mix_wave {batches_processed}/{total_batches}")

                    tar_waves = self.run_model(mix_wave, is_match_mix=is_match_mix)

                    if window is not None:
                        tar_waves[..., :chunk_size_actual] *= window
                        divider[..., start:end] += window
                    else:
                        divider[..., start:end] += 1

                    result[..., start:end] += tar_waves[..., : end - start]

        self.logger.debug("Chuẩn hóa kết quả bằng cách chia kết quả cho số chia.")
        tar_waves = result / divider
        tar_waves_.append(tar_waves)

        tar_waves_ = np.vstack(tar_waves_)[:, :, self.trim : -self.trim]
        tar_waves = np.concatenate(tar_waves_, axis=-1)[:, : mix.shape[-1]]

        source = tar_waves[:, 0:None]
        self.logger.debug(f"Sóng tar_waves. Hình dạng: {tar_waves.shape}")

        if not is_match_mix:
            source *= self.compensate
            self.logger.debug("Chế độ trộn Match; áp dụng hệ số bù.")

        self.logger.debug("Quá trình trộn đã hoàn tất.")
        return source

    def run_model(self, mix, is_match_mix=False):
        spek = self.stft(mix.to(self.torch_device))
        self.logger.debug(f"STFT được áp dụng trên hỗn hợp. Hình dạng quang phổ: {spek.shape}")

        spek[:, :, :3, :] *= 0

        if is_match_mix:
            spec_pred = spek.cpu().numpy()
            self.logger.debug("is_match_mix: dự đoán phổ thu được trực tiếp từ đầu ra STFT.")
        else:
            if self.enable_denoise:
                spec_pred_neg = self.model_run(-spek)  
                spec_pred_pos = self.model_run(spek)
                spec_pred = (spec_pred_neg * -0.5) + (spec_pred_pos * 0.5)
                self.logger.debug("Mô hình chạy trên cả phổ âm và dương để khử nhiễu.")
            else:
                spec_pred = self.model_run(spek)
                self.logger.debug("Mô hình chạy trên quang phổ mà không khử nhiễu.")

        result = self.stft.inverse(torch.tensor(spec_pred).to(self.torch_device)).cpu().detach().numpy()
        self.logger.debug(f"STFT nghịch đảo được áp dụng. Trả về kết quả có hình dạng: {result.shape}")

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

        stft_window = self.hann_window.to(input_tensor.device)

        batch_dimensions = input_tensor.shape[:-2]
        channel_dim, time_dim = input_tensor.shape[-2:]

        reshaped_tensor = input_tensor.reshape([-1, time_dim])
        stft_output = torch.stft(reshaped_tensor, n_fft=self.n_fft, hop_length=self.hop_length, window=stft_window, center=True, return_complex=False)

        permuted_stft_output = stft_output.permute([0, 3, 1, 2])

        final_output = permuted_stft_output.reshape([*batch_dimensions, channel_dim, 2, -1, permuted_stft_output.shape[-1]]).reshape([*batch_dimensions, channel_dim * 2, -1, permuted_stft_output.shape[-1]])

        if is_non_standard_device: final_output = final_output.to(self.device)

        return final_output[..., : self.dim_f, :]

    def pad_frequency_dimension(self, input_tensor, batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins):
        freq_padding = torch.zeros([*batch_dimensions, channel_dim, num_freq_bins - freq_dim, time_dim]).to(input_tensor.device)
        padded_tensor = torch.cat([input_tensor, freq_padding], -2)

        return padded_tensor

    def calculate_inverse_dimensions(self, input_tensor):
        batch_dimensions = input_tensor.shape[:-3]
        channel_dim, freq_dim, time_dim = input_tensor.shape[-3:]

        num_freq_bins = self.n_fft // 2 + 1

        return batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins

    def prepare_for_istft(self, padded_tensor, batch_dimensions, channel_dim, num_freq_bins, time_dim):
        reshaped_tensor = padded_tensor.reshape([*batch_dimensions, channel_dim // 2, 2, num_freq_bins, time_dim])
        flattened_tensor = reshaped_tensor.reshape([-1, 2, num_freq_bins, time_dim])
        permuted_tensor = flattened_tensor.permute([0, 2, 3, 1])
        complex_tensor = permuted_tensor[..., 0] + permuted_tensor[..., 1] * 1.0j

        return complex_tensor

    def inverse(self, input_tensor):
        is_non_standard_device = not input_tensor.device.type in ["cuda", "cpu"]

        if is_non_standard_device: input_tensor = input_tensor.cpu()

        stft_window = self.hann_window.to(input_tensor.device)

        batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins = self.calculate_inverse_dimensions(input_tensor)

        padded_tensor = self.pad_frequency_dimension(input_tensor, batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins)

        complex_tensor = self.prepare_for_istft(padded_tensor, batch_dimensions, channel_dim, num_freq_bins, time_dim)

        istft_result = torch.istft(complex_tensor, n_fft=self.n_fft, hop_length=self.hop_length, window=stft_window, center=True)

        final_output = istft_result.reshape([*batch_dimensions, 2, -1])

        if is_non_standard_device: final_output = final_output.to(self.device)

        return final_output