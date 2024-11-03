import os
import sys
import time
import json
import yaml
import torch
import codecs
import hashlib
import logging
import platform
import warnings
import requests
import subprocess

import onnxruntime as ort

from tqdm import tqdm
from importlib import metadata, import_module

now_dir = os.getcwd()
sys.path.append(now_dir)

class Separator:
    def __init__(self, log_level=logging.INFO, log_formatter=None, model_file_dir="assets/model/uvr5", output_dir=None, output_format="wav", output_bitrate=None, normalization_threshold=0.9, output_single_stem=None, invert_using_spec=False, sample_rate=44100, mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": False}, demucs_params={"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True}):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.log_level = log_level
        self.log_formatter = log_formatter

        self.log_handler = logging.StreamHandler()

        if self.log_formatter is None: self.log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

        self.log_handler.setFormatter(self.log_formatter)

        if not self.logger.hasHandlers(): self.logger.addHandler(self.log_handler)
        if log_level > logging.DEBUG: warnings.filterwarnings("ignore")

        self.logger.info(f"Đang khởi tạo với đường dẫn đầu ra: {output_dir}, định dạng đầu ra: {output_format}")

        self.model_file_dir = model_file_dir

        if output_dir is None:
            output_dir = os.getcwd()
            self.logger.info("Thư mục đầu ra không được chỉ định. Sử dụng thư mục làm việc hiện tại.")

        self.output_dir = output_dir

        os.makedirs(self.model_file_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.output_format = output_format
        self.output_bitrate = output_bitrate

        if self.output_format is None: self.output_format = "wav"

        self.normalization_threshold = normalization_threshold

        if normalization_threshold <= 0 or normalization_threshold > 1: raise ValueError("Ngưỡng chuẩn hóa phải lớn hơn 0 và nhỏ hơn hoặc bằng 1.")

        self.output_single_stem = output_single_stem

        if output_single_stem is not None: self.logger.debug(f"Đã yêu cầu đầu ra một gốc nên chỉ có một tệp đầu ra ({output_single_stem}) sẽ được ghi")

        self.invert_using_spec = invert_using_spec
        if self.invert_using_spec: self.logger.debug(f"Bước thứ hai sẽ được đảo ngược bằng cách sử dụng quang phổ thay vì dạng sóng. Điều này có thể cải thiện chất lượng nhưng chậm hơn một chút.")

        try:
            self.sample_rate = int(sample_rate)
            
            if self.sample_rate <= 0: raise ValueError(f"Cài đặt tốc độ mẫu là {self.sample_rate} nhưng phải là số nguyên khác 0.")
            if self.sample_rate > 12800000: raise ValueError(f"Cài đặt tốc độ mẫu là {self.sample_rate}. Nhập một cái gì đó ít tham vọng hơn.")
        except ValueError: 
            raise ValueError("Tốc độ mẫu phải là số nguyên khác 0. Vui lòng cung cấp số nguyên hợp lệ.")

        self.arch_specific_params = {"MDX": mdx_params, "Demucs": demucs_params}
        self.torch_device = None
        self.torch_device_cpu = None
        self.torch_device_mps = None
        self.onnx_execution_provider = None
        self.model_instance = None
        self.model_is_uvr_vip = False
        self.model_friendly_name = None

        self.setup_accelerated_inferencing_device()

    def setup_accelerated_inferencing_device(self):
        system_info = self.get_system_info()
        self.check_ffmpeg_installed()
        self.log_onnxruntime_packages()
        self.setup_torch_device(system_info)

    def get_system_info(self):
        os_name = platform.system()
        os_version = platform.version()

        self.logger.info(f"Hệ điều hành: {os_name} {os_version}")

        system_info = platform.uname()
        self.logger.info(f"Hệ thống: {system_info.system} Tên: {system_info.node} Phát hành: {system_info.release} Máy: {system_info.machine} Vi xử lý: {system_info.processor}")

        python_version = platform.python_version()
        self.logger.info(f"Phiên bản python: {python_version}")

        pytorch_version = torch.__version__
        self.logger.info(f"Phiên bản pytorch: {pytorch_version}")

        return system_info

    def check_ffmpeg_installed(self):
        try:
            ffmpeg_version_output = subprocess.check_output(["ffmpeg", "-version"], text=True)
            first_line = ffmpeg_version_output.splitlines()[0]
            self.logger.info(f"Đã cài đặt FFmpeg: {first_line}")
        except FileNotFoundError:
            self.logger.error("FFmpeg chưa được cài đặt. Vui lòng cài đặt FFmpeg để sử dụng gói này.")
            if "PYTEST_CURRENT_TEST" not in os.environ: raise

    def log_onnxruntime_packages(self):
        onnxruntime_gpu_package = self.get_package_distribution("onnxruntime-gpu")
        onnxruntime_cpu_package = self.get_package_distribution("onnxruntime")

        if onnxruntime_gpu_package is not None: self.logger.info(f"Gói GPU ONNX Runtime được cài đặt cùng với phiên bản: {onnxruntime_gpu_package.version}")
        if onnxruntime_cpu_package is not None: self.logger.info(f"Gói CPU ONNX Runtime được cài đặt cùng với phiên bản: {onnxruntime_cpu_package.version}")

    def setup_torch_device(self, system_info):
        hardware_acceleration_enabled = False
        ort_providers = ort.get_available_providers()

        self.torch_device_cpu = torch.device("cpu")

        if torch.cuda.is_available():
            self.configure_cuda(ort_providers)
            hardware_acceleration_enabled = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and system_info.processor == "arm":
            self.configure_mps(ort_providers)
            hardware_acceleration_enabled = True

        if not hardware_acceleration_enabled:
            self.logger.info("Không thể cấu hình khả năng tăng tốc phần cứng, chạy ở chế độ CPU")
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CPUExecutionProvider"]

    def configure_cuda(self, ort_providers):
        self.logger.info("CUDA có sẵn trong Torch, cài đặt thiết bị Torch thành CUDA")
        self.torch_device = torch.device("cuda")

        if "CUDAExecutionProvider" in ort_providers:
            self.logger.info("ONNXruntime có sẵn CUDAExecutionProvider, cho phép tăng tốc")
            self.onnx_execution_provider = ["CUDAExecutionProvider"]
        else: self.logger.warning("CUDAExecutionProvider không có sẵn trong ONNXruntime, do đó khả năng tăng tốc sẽ KHÔNG được bật")

    def configure_mps(self, ort_providers):
        self.logger.info("Cài đặt thiết bị Torch thành MPS")
        self.torch_device_mps = torch.device("mps")
        self.torch_device = self.torch_device_mps

        if "CoreMLExecutionProvider" in ort_providers:
            self.logger.info("ONNXruntime có sẵn CoreMLExecutionProvider, cho phép tăng tốc")
            self.onnx_execution_provider = ["CoreMLExecutionProvider"]
        else: self.logger.warning("CoreMLExecutionProvider không có sẵn trong ONNXruntime, do đó khả năng tăng tốc sẽ KHÔNG được bật")

    def get_package_distribution(self, package_name):
        try:
            return metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            self.logger.debug(f"Gói Python: {package_name} chưa được cài đặt")
            return None

    def get_model_hash(self, model_path):
        self.logger.debug(f"Tính hash của tệp mô hình {model_path}")

        try:
            with open(model_path, "rb") as f:
                f.seek(-10000 * 1024, 2)
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            self.logger.error(f"IOError đang tìm kiếm -10 MB hoặc đọc tệp mô hình để tính toán hàm băm: {e}")

            return hashlib.md5(open(model_path, "rb").read()).hexdigest()

    def download_file_if_not_exists(self, url, output_path):
        if os.path.isfile(output_path):
            self.logger.debug(f"Tệp đã tồn tại tại {output_path}, bỏ qua quá trình tải xuống")
            return

        self.logger.debug(f"Đang tải tệp từ {url} xuống {output_path} với thời gian chờ 300 giây")
        response = requests.get(url, stream=True, timeout=300)

        if response.status_code == 200:
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))
                    f.write(chunk)

            progress_bar.close()
        else: raise RuntimeError(f"Không tải được tệp xuống từ {url}, mã phản hồi: {response.status_code}")

    def print_uvr_vip_message(self):
        if self.model_is_uvr_vip:
            self.logger.warning(f"Mô hình: '{self.model_friend_name}' là mô hình cao cấp, được Anjok07 dự định chỉ dành cho những người đăng ký trả phí truy cập.")
            self.logger.warning("Này bạn, nếu bạn chưa đăng ký, vui lòng cân nhắc việc hỗ trợ cho nhà phát triển của UVR, Anjok07 bằng cách đăng ký tại đây: https://patreon.com/uvr")

    def list_supported_model_files(self):
        download_checks_path = os.path.join(self.model_file_dir, "download_checks.json")

        model_downloads_list = json.load(open(download_checks_path, encoding="utf-8"))
        self.logger.debug(f"Đã tải danh sách tải xuống mô hình")

        filtered_demucs_v4 = {key: value for key, value in model_downloads_list["demucs_download_list"].items() if key.startswith("Demucs v4")}

        model_files_grouped_by_type = {"MDX": {**model_downloads_list["mdx_download_list"], **model_downloads_list["mdx_download_vip_list"]}, "Demucs": filtered_demucs_v4}
        return model_files_grouped_by_type
    
    def download_model_files(self, model_filename):
        model_path = os.path.join(self.model_file_dir, f"{model_filename}")

        supported_model_files_grouped = self.list_supported_model_files()
        public_model_repo_url_prefix = codecs.decode("uggcf://tvguho.pbz/GEiyie/zbqry_ercb/eryrnfrf/qbjaybnq/nyy_choyvp_hie_zbqryf", "rot13")
        vip_model_repo_url_prefix = codecs.decode("uggcf://tvguho.pbz/Nawbx0109/nv_zntvp/eryrnfrf/qbjaybnq/i5", "rot13")

        audio_separator_models_repo_url_prefix = codecs.decode("uggcf://tvguho.pbz/abznqxnenbxr/clguba-nhqvb-frcnengbe/eryrnfrf/qbjaybnq/zbqry-pbasvtf", "rot13")

        yaml_config_filename = None

        self.logger.debug(f"Đang tìm kiếm mô hình {model_filename} trong tập tin các mô hình được hỗ trợ trong nhóm")

        for model_type, model_list in supported_model_files_grouped.items():
            for model_friendly_name, model_download_list in model_list.items():
                self.model_is_uvr_vip = "VIP" in model_friendly_name
                model_repo_url_prefix = vip_model_repo_url_prefix if self.model_is_uvr_vip else public_model_repo_url_prefix

                if isinstance(model_download_list, str) and model_download_list == model_filename:
                    self.logger.debug(f"Đã xác định được tệp mô hình đơn: {model_friendly_name}")
                    self.model_friendly_name = model_friendly_name

                    try:
                        self.download_file_if_not_exists(f"{model_repo_url_prefix}/{model_filename}", model_path)
                    except RuntimeError:
                        self.logger.debug("Không tìm thấy mô hình trong kho lưu trữ UVR, đang cố tải xuống từ kho lưu trữ mô hình phân tách âm thanh...")
                        self.download_file_if_not_exists(f"{audio_separator_models_repo_url_prefix}/{model_filename}", model_path)

                    self.print_uvr_vip_message()

                    self.logger.debug(f"Đường dẫn trả về cho tệp mô hình đơn: {model_path}")

                    return model_filename, model_type, model_friendly_name, model_path, yaml_config_filename
                elif isinstance(model_download_list, dict):
                    this_model_matches_input_filename = False

                    for file_name, file_url in model_download_list.items():
                        if file_name == model_filename or file_url == model_filename:
                            self.logger.debug(f"Đã tìm thấy tên tệp đầu vào {model_filename} trong mô hình nhiều tệp: {model_friendly_name}")
                            this_model_matches_input_filename = True

                    if this_model_matches_input_filename:
                        self.logger.debug(f"Đã xác định mô hình nhiều tệp: {model_friendly_name}, lặp qua các tệp để tải xuống")
                        self.model_friendly_name = model_friendly_name
                        self.print_uvr_vip_message()

                        for config_key, config_value in model_download_list.items():
                            self.logger.debug(f"Đang cố gắng xác định URL tải xuống cho cặp cấu hình: {config_key} -> {config_value}")

                            if config_value.startswith("http"): self.download_file_if_not_exists(config_value, os.path.join(self.model_file_dir, config_key))
                            elif config_key.endswith(".ckpt"):
                                try:
                                    download_url = f"{model_repo_url_prefix}/{config_key}"
                                    self.download_file_if_not_exists(download_url, os.path.join(self.model_file_dir, config_key))
                                except RuntimeError:
                                    self.logger.debug("Không tìm thấy mô hình trong kho lưu trữ UVR, đang cố tải xuống từ kho lưu trữ mô hình phân tách âm thanh...")
                                    download_url = f"{audio_separator_models_repo_url_prefix}/{config_key}"
                                    self.download_file_if_not_exists(download_url, os.path.join(self.model_file_dir, config_key))

                                if model_filename.endswith(".yaml"):
                                    self.logger.warning(f"Tên mô hình bạn đã chỉ định, {model_filename} thực sự là tệp cấu hình mô hình chứ không phải tệp mô hình.")
                                    self.logger.warning(f"Chúng tôi đã tìm thấy một mô hình khớp với tệp cấu hình này: {config_key} nên chúng tôi sẽ sử dụng tệp mô hình đó cho lần chạy này.")
                                    self.logger.warning("Để tránh hành vi gây nhầm lẫn/không nhất quán trong tương lai, thay vào đó hãy chỉ định tên tệp mô hình thực tế.")

                                    model_filename = config_key
                                    model_path = os.path.join(self.model_file_dir, f"{model_filename}")

                                yaml_config_filename = config_value
                                yaml_config_filepath = os.path.join(self.model_file_dir, yaml_config_filename)

                                try:
                                    url = codecs.decode("uggcf://enj.tvguhohfrepbagrag.pbz/GEiyie/nccyvpngvba_qngn/znva/zqk_zbqry_qngn/zqk_p_pbasvtf", "rot13")
                                    yaml_config_url = f"{url}/{yaml_config_filename}"
                                    self.download_file_if_not_exists(f"{yaml_config_url}", yaml_config_filepath)
                                except RuntimeError:
                                    self.logger.debug("Không tìm thấy tệp cấu hình mô hình YAML trong kho lưu trữ UVR, đang cố tải xuống từ kho lưu trữ mô hình phân tách âm thanh...")
                                    yaml_config_url = f"{audio_separator_models_repo_url_prefix}/{yaml_config_filename}"
                                    self.download_file_if_not_exists(f"{yaml_config_url}", yaml_config_filepath)
                            else:
                                download_url = f"{model_repo_url_prefix}/{config_value}"
                                self.download_file_if_not_exists(download_url, os.path.join(self.model_file_dir, config_value))

                        self.logger.debug(f"Tất cả các tệp đã tải xuống cho mô hình {model_friendly_name}, trả về đường dẫn ban đầu {model_path}")

                        return model_filename, model_type, model_friendly_name, model_path, yaml_config_filename

        raise ValueError(f"Không tìm thấy tệp mô hình {model_filename} trong các tệp mô hình được hỗ trợ")

    def load_model_data_from_yaml(self, yaml_config_filename):
        model_data_yaml_filepath = os.path.join(self.model_file_dir, yaml_config_filename) if not os.path.exists(yaml_config_filename) else yaml_config_filename

        self.logger.debug(f"Đang tải dữ liệu mô hình từ YAML tại đường dẫn {model_data_yaml_filepath}")

        model_data = yaml.load(open(model_data_yaml_filepath, encoding="utf-8"), Loader=yaml.FullLoader)
        self.logger.debug(f"Dữ liệu mô hình được tải từ tệp YAML: {model_data}")

        if "roformer" in model_data_yaml_filepath: model_data["is_roformer"] = True

        return model_data

    def load_model_data_using_hash(self, model_path):
        mdx_model_data_url = codecs.decode("uggcf://enj.tvguhohfrepbagrag.pbz/GEiyie/nccyvpngvba_qngn/znva/zqk_zbqry_qngn/zbqry_qngn_arj.wfba", "rot13")

        self.logger.debug("Tính hash MD5 cho tệp mô hình để xác định các tham số mô hình từ dữ liệu UVR...")
        model_hash = self.get_model_hash(model_path)
        self.logger.debug(f"Mô hình {model_path} có hash {model_hash}")

        mdx_model_data_path = os.path.join(self.model_file_dir, "mdx_model_data.json")
        self.logger.debug(f"Đường dẫn dữ liệu mô hình MDX được đặt thành {mdx_model_data_path}")
        self.download_file_if_not_exists(mdx_model_data_url, mdx_model_data_path)

        self.logger.debug("Đang tải các tham số mô hình MDX và VR từ tệp dữ liệu mô hình UVR...")
        mdx_model_data_object = json.load(open(mdx_model_data_path, encoding="utf-8"))

        if model_hash in mdx_model_data_object: model_data = mdx_model_data_object[model_hash]
        else: raise ValueError(f"Tệp mô hình không được hỗ trợ: không thể tìm thấy tham số cho hash MD5 {model_hash} trong tệp dữ liệu mô hình UVR cho vòm MDX hoặc VR.")

        self.logger.debug(f"Dữ liệu mô hình được tải từ UVR JSON bằng hàm băm {model_hash}: {model_data}")

        return model_data

    def load_model(self, model_filename):
        self.logger.info(f"Đang tải mô hình {model_filename}...")

        load_model_start_time = time.perf_counter()

        model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = self.download_model_files(model_filename)
        model_name = model_filename.split(".")[0]
        self.logger.debug(f"Đã tải xuống mô hình, tên thân thiện: {model_friendly_name}, model_path: {model_path}")

        if model_path.lower().endswith(".yaml"): yaml_config_filename = model_path

        model_data = self.load_model_data_from_yaml(yaml_config_filename) if yaml_config_filename is not None else self.load_model_data_using_hash(model_path)

        common_params = {
            "logger": self.logger,
            "log_level": self.log_level,
            "torch_device": self.torch_device,
            "torch_device_cpu": self.torch_device_cpu,
            "torch_device_mps": self.torch_device_mps,
            "onnx_execution_provider": self.onnx_execution_provider,
            "model_name": model_name,
            "model_path": model_path,
            "model_data": model_data,
            "output_format": self.output_format,
            "output_bitrate": self.output_bitrate,
            "output_dir": self.output_dir,
            "normalization_threshold": self.normalization_threshold,
            "output_single_stem": self.output_single_stem,
            "invert_using_spec": self.invert_using_spec,
            "sample_rate": self.sample_rate,
        }

        separator_classes = {"MDX": "mdx_separator.MDXSeparator", "Demucs": "demucs_separator.DemucsSeparator"}

        if model_type not in self.arch_specific_params or model_type not in separator_classes: raise ValueError(f"Loại mô hình không được hỗ trợ: {model_type}")
        if model_type == "Demucs" and sys.version_info < (3, 10): raise Exception("Các mô hình Demucs yêu cầu phiên bản Python 3.10 trở lên.")

        self.logger.debug(f"Nhập mô-đun cho loại mô hình {model_type}: {separator_classes[model_type]}")

        module_name, class_name = separator_classes[model_type].split(".")
        module = import_module(f"main.library.architectures.{module_name}")
        separator_class = getattr(module, class_name)

        self.logger.debug(f"Khởi tạo lớp phân cách cho loại mô hình {model_type}: {separator_class}")
        self.model_instance = separator_class(common_config=common_params, arch_config=self.arch_specific_params[model_type])

        self.logger.debug("Đang tải mô hình hoàn tất.")
        self.logger.info(f'Tải thời lượng mô hình: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - load_model_start_time)))}')

    def separate(self, audio_file_path):
        self.logger.info(f"Bắt đầu quá trình tách cho đường dẫn tập tin âm thanh: {audio_file_path}")
        separate_start_time = time.perf_counter()

        self.logger.debug(f"Ngưỡng chuẩn hóa được đặt thành {self.normalization_threshold}, dạng sóng sẽ hạ xuống biên độ tối đa này để tránh bị cắt.")

        output_files = self.model_instance.separate(audio_file_path)

        self.model_instance.clear_gpu_cache()

        self.model_instance.clear_file_specific_paths()

        self.print_uvr_vip_message()

        self.logger.debug("Quá trình tách hoàn tất.")
        self.logger.info(f'Thời gian tách: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - separate_start_time)))}')

        return output_files

    def download_model_and_data(self, model_filename):
        self.logger.info(f"Đang tải xuống mô hình {model_filename}...")

        model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = self.download_model_files(model_filename)

        if model_path.lower().endswith(".yaml"): yaml_config_filename = model_path

        model_data = self.load_model_data_from_yaml(yaml_config_filename) if yaml_config_filename is not None else self.load_model_data_using_hash(model_path)

        model_data_dict_size = len(model_data)

        self.logger.info(f"Đã tải xuống mô hình, loại: {model_type}, tên thân thiện: {model_friendly_name}, đường dẫn mô hình: {model_path}, dữ liệu mô hình: {model_data_dict_size} mục")