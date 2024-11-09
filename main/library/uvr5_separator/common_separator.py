import os
import gc
import torch
import librosa

import numpy as np
import soundfile as sf

from logging import Logger
from pydub import AudioSegment

from . import spec_utils


class CommonSeparator:
    ALL_STEMS = "All Stems"
    VOCAL_STEM = "Vocals"
    INST_STEM = "Instrumental"
    OTHER_STEM = "Other"
    BASS_STEM = "Bass"
    DRUM_STEM = "Drums"
    GUITAR_STEM = "Guitar"
    PIANO_STEM = "Piano"
    SYNTH_STEM = "Synthesizer"
    STRINGS_STEM = "Strings"
    WOODWINDS_STEM = "Woodwinds"
    BRASS_STEM = "Brass"
    WIND_INST_STEM = "Wind Inst"
    NO_OTHER_STEM = "No Other"
    NO_BASS_STEM = "No Bass"
    NO_DRUM_STEM = "No Drums"
    NO_GUITAR_STEM = "No Guitar"
    NO_PIANO_STEM = "No Piano"
    NO_SYNTH_STEM = "No Synthesizer"
    NO_STRINGS_STEM = "No Strings"
    NO_WOODWINDS_STEM = "No Woodwinds"
    NO_WIND_INST_STEM = "No Wind Inst"
    NO_BRASS_STEM = "No Brass"
    PRIMARY_STEM = "Primary Stem"
    SECONDARY_STEM = "Secondary Stem"
    LEAD_VOCAL_STEM = "lead_only"
    BV_VOCAL_STEM = "backing_only"
    LEAD_VOCAL_STEM_I = "with_lead_vocals"
    BV_VOCAL_STEM_I = "with_backing_vocals"
    LEAD_VOCAL_STEM_LABEL = "Lead Vocals"
    BV_VOCAL_STEM_LABEL = "Backing Vocals"
    NO_STEM = "No "

    STEM_PAIR_MAPPER = {VOCAL_STEM: INST_STEM, INST_STEM: VOCAL_STEM, LEAD_VOCAL_STEM: BV_VOCAL_STEM, BV_VOCAL_STEM: LEAD_VOCAL_STEM, PRIMARY_STEM: SECONDARY_STEM}

    NON_ACCOM_STEMS = (VOCAL_STEM, OTHER_STEM, BASS_STEM, DRUM_STEM, GUITAR_STEM, PIANO_STEM, SYNTH_STEM, STRINGS_STEM, WOODWINDS_STEM, BRASS_STEM, WIND_INST_STEM)


    def __init__(self, config):
        self.logger: Logger = config.get("logger")
        self.log_level: int = config.get("log_level")
        self.torch_device = config.get("torch_device")
        self.torch_device_cpu = config.get("torch_device_cpu")
        self.torch_device_mps = config.get("torch_device_mps")
        self.onnx_execution_provider = config.get("onnx_execution_provider")
        self.model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.model_data = config.get("model_data")
        self.output_dir = config.get("output_dir")
        self.output_format = config.get("output_format")
        self.output_bitrate = config.get("output_bitrate")
        self.normalization_threshold = config.get("normalization_threshold")
        self.enable_denoise = config.get("enable_denoise")
        self.output_single_stem = config.get("output_single_stem")
        self.invert_using_spec = config.get("invert_using_spec")
        self.sample_rate = config.get("sample_rate")

        self.primary_stem_name = None
        self.secondary_stem_name = None

        if "training" in self.model_data and "instruments" in self.model_data["training"]:
            instruments = self.model_data["training"]["instruments"]

            if instruments:
                self.primary_stem_name = instruments[0]
                self.secondary_stem_name = instruments[1] if len(instruments) > 1 else self.secondary_stem(self.primary_stem_name)

        if self.primary_stem_name is None:
            self.primary_stem_name = self.model_data.get("primary_stem", "Vocals")
            self.secondary_stem_name = self.secondary_stem(self.primary_stem_name)

        self.is_karaoke = self.model_data.get("is_karaoke", False)
        self.is_bv_model = self.model_data.get("is_bv_model", False)
        self.bv_model_rebalance = self.model_data.get("is_bv_model_rebalanced", 0)

        self.logger.debug(f"Các thông số phổ biến: Tên mô hình = {self.model_name}, Đường dẫn mô hình = {self.model_path}")
        self.logger.debug(f"Các thông số phổ biến: Đường dẫn đầu ra ={self.output_dir}, Định dạng đầu ra = {self.output_format}")
        self.logger.debug(f"Các thông số phổ biến: ngưỡng chuẩn hóa = {self.normalization_threshold}")
        self.logger.debug(f"Các thông số phổ biến: Kích hoạt khữ nhiễu = {self.enable_denoise}, Đầu ra một phần = {self.output_single_stem}")
        self.logger.debug(f"Các thông số phổ biến: Đảo ngược bằng cách sử dụng thông số kỹ thuật = {self.invert_using_spec}, sample_rate = {self.sample_rate}")
        self.logger.debug(f"Các thông số phổ biến: Tên phần gốc chính = {self.primary_stem_name}, Tên phần gốc phụ = {self.secondary_stem_name}")
        self.logger.debug(f"Các thông số phổ biến: Là Karaoke = {self.is_karaoke}, là mô hình bv = {self.is_bv_model}, tái cân bằng mô hình bv = {self.bv_model_rebalance}")

        self.audio_file_path = None
        self.audio_file_base = None

        self.primary_source = None
        self.secondary_source = None

        self.primary_stem_output_path = None
        self.secondary_stem_output_path = None

        self.cached_sources_map = {}

    def secondary_stem(self, primary_stem: str):
        primary_stem = primary_stem if primary_stem else self.NO_STEM

        return self.STEM_PAIR_MAPPER[primary_stem] if primary_stem in self.STEM_PAIR_MAPPER else primary_stem.replace(self.NO_STEM, "") if self.NO_STEM in primary_stem else f"{self.NO_STEM}{primary_stem}"

    def separate(self, audio_file_path):
        raise NotImplementedError("Phương thức sẽ bị ghi đè bởi các lớp con.")

    def final_process(self, stem_path, source, stem_name):
        self.logger.debug(f"Đang hoàn tất quá trình xử lý phần gốc {stem_name} và ghi âm thanh...")
        self.write_audio(stem_path, source)

        return {stem_name: source}

    def cached_sources_clear(self):
        self.cached_sources_map = {}

    def cached_source_callback(self, model_architecture, model_name=None):
        model, sources = None, None

        mapper = self.cached_sources_map[model_architecture]

        for key, value in mapper.items():
            if model_name in key:
                model = key
                sources = value

        return model, sources

    def cached_model_source_holder(self, model_architecture, sources, model_name=None):
        self.cached_sources_map[model_architecture] = {**self.cached_sources_map.get(model_architecture, {}), **{model_name: sources}}

    def prepare_mix(self, mix):
        audio_path = mix

        if not isinstance(mix, np.ndarray):
            self.logger.debug(f"Đang tải âm thanh từ tập tin: {mix}")
            mix, sr = librosa.load(mix, mono=False, sr=self.sample_rate)
            self.logger.debug(f"Đã tải âm thanh. Tốc độ mẫu: {sr}, Hình dạng âm thanh: {mix.shape}")
        else:
            self.logger.debug("Chuyển đổi mảng hỗn hợp được cung cấp.")
            mix = mix.T
            self.logger.debug(f"Hình dạng hỗn hợp chuyển đổi: {mix.shape}")

        if isinstance(audio_path, str):
            if not np.any(mix):
                error_msg = f"Tệp âm thanh {audio_path} trống hoặc không hợp lệ"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else: self.logger.debug("Tệp âm thanh hợp lệ và chứa dữ liệu.")


        if mix.ndim == 1:
            self.logger.debug("Hỗn hợp là đơn sắc. Chuyển đổi sang âm thanh nổi.")
            mix = np.asfortranarray([mix, mix])
            self.logger.debug("Đã chuyển đổi thành bản trộn âm thanh nổi.")

        self.logger.debug("Công tác chuẩn bị hỗn hợp đã hoàn tất.")
        return mix

    def write_audio(self, stem_path: str, stem_source):
        duration_seconds = librosa.get_duration(filename=self.audio_file_path)
        duration_hours = duration_seconds / 3600
        self.logger.info(f"Thời lượng âm thanh là {duration_hours:.2f} giờ ({duration_seconds:.2f} giây).")

        if duration_hours >= 1:
            self.logger.warning(f"Sử dụng soundfile để viết.")
            self.write_audio_soundfile(stem_path, stem_source)
        else:
            self.logger.info(f"Sử dụng pydub để viết.")
            self.write_audio_pydub(stem_path, stem_source)

    def write_audio_pydub(self, stem_path: str, stem_source):
        self.logger.debug(f"Đang nhập write_audio_pydub bằng đường dẫn gốc: {stem_path}")

        stem_source = spec_utils.normalize(wave=stem_source, max_peak=self.normalization_threshold)

        if np.max(np.abs(stem_source)) < 1e-6:
            self.logger.warning("Cảnh báo: mảng nguồn gốc gần như im lặng hoặc trống.")
            return

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            stem_path = os.path.join(self.output_dir, stem_path)

        self.logger.debug(f"Hình dạng dữ liệu âm thanh trước khi xử lý: {stem_source.shape}")
        self.logger.debug(f"Kiểu dữ liệu trước khi chuyển đổi: {stem_source.dtype}")

        if stem_source.dtype != np.int16:
            stem_source = (stem_source * 32767).astype(np.int16)
            self.logger.debug("Đã chuyển đổi gốc_source thành int16.")

        stem_source_interleaved = np.empty((2 * stem_source.shape[0],), dtype=np.int16)
        stem_source_interleaved[0::2] = stem_source[:, 0] 
        stem_source_interleaved[1::2] = stem_source[:, 1]

        self.logger.debug(f"Hình dạng dữ liệu âm thanh xen kẽ: {stem_source_interleaved.shape}")

        try:
            audio_segment = AudioSegment(stem_source_interleaved.tobytes(), frame_rate=self.sample_rate, sample_width=stem_source.dtype.itemsize, channels=2)
            self.logger.debug("Đã tạo AudioSegment thành công.")
        except (IOError, ValueError) as e:
            self.logger.error(f"Lỗi cụ thể khi tạo AudioSegment: {e}")
            return

        file_format = stem_path.lower().split(".")[-1]

        if file_format == "m4a": file_format = "mp4"
        elif file_format == "mka": file_format = "matroska"

        bitrate = "320k" if file_format == "mp3" and self.output_bitrate is None else self.output_bitrate

        try:
            audio_segment.export(stem_path, format=file_format, bitrate=bitrate)
            self.logger.debug(f"Đã xuất thành công tệp âm thanh sang {stem_path}")
        except (IOError, ValueError) as e:
            self.logger.error(f"Lỗi xuất file âm thanh: {e}")

    def write_audio_soundfile(self, stem_path: str, stem_source):
        self.logger.debug(f"Nhập write_audio_soundfile bằng đường dẫn phần gốc: {stem_path}")

        if stem_source.shape[1] == 2:
            if stem_source.flags["F_CONTIGUOUS"]: stem_source = np.ascontiguousarray(stem_source)
            else:
                stereo_interleaved = np.empty((2 * stem_source.shape[0],), dtype=np.int16)

                stereo_interleaved[0::2] = stem_source[:, 0]

                stereo_interleaved[1::2] = stem_source[:, 1]
                stem_source = stereo_interleaved

        self.logger.debug(f"Hình dạng dữ liệu âm thanh xen kẽ: {stem_source.shape}")

        try:
            sf.write(stem_path, stem_source, self.sample_rate)
            self.logger.debug(f"Đã xuất thành công tệp âm thanh sang {stem_path}")
        except Exception as e:
            self.logger.error(f"Lỗi xuất file âm thanh: {e}")

    def clear_gpu_cache(self):
        self.logger.debug("Chạy thu gom rác...")
        gc.collect()

        if self.torch_device == torch.device("mps"):
            self.logger.debug("Xóa bộ nhớ đệm MPS...")
            torch.mps.empty_cache()

        if self.torch_device == torch.device("cuda"):
            self.logger.debug("Xóa bộ đệm CUDA...")
            torch.cuda.empty_cache()

    def clear_file_specific_paths(self):
        self.logger.info("Xóa đường dẫn, nguồn và gốc của tệp âm thanh đầu vào...")
        self.audio_file_path = None
        self.audio_file_base = None
        
        self.primary_source = None
        self.secondary_source = None

        self.primary_stem_output_path = None
        self.secondary_stem_output_path = None