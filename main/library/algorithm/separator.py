import os
import sys
import time
import yaml
import torch
import codecs
import hashlib
import logging
import platform
import warnings
import requests
import onnxruntime

from importlib import metadata, import_module

now_dir = os.getcwd()
sys.path.append(now_dir)

from main.configs.config import Config
translations = Config().translations

class Separator:
    def __init__(self, logger=logging.getLogger(__name__), log_level=logging.INFO, log_formatter=None, model_file_dir="assets/models/uvr5", output_dir=None, output_format="wav", output_bitrate=None, normalization_threshold=0.9, output_single_stem=None, invert_using_spec=False, sample_rate=44100, mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": False}, demucs_params={"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True}):
        self.logger = logger
        self.log_level = log_level
        self.log_formatter = log_formatter
        self.log_handler = logging.StreamHandler()

        if self.log_formatter is None: self.log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.log_handler.setFormatter(self.log_formatter)
        if not self.logger.hasHandlers(): self.logger.addHandler(self.log_handler)
        if log_level > logging.DEBUG: warnings.filterwarnings("ignore")

        self.logger.info(translations["separator_info"].format(output_dir=output_dir, output_format=output_format))
        self.model_file_dir = model_file_dir

        if output_dir is None:
            output_dir = now_dir
            self.logger.info(translations["output_dir_is_none"])

        self.output_dir = output_dir

        os.makedirs(self.model_file_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_format = output_format
        self.output_bitrate = output_bitrate

        if self.output_format is None: self.output_format = "wav"
        self.normalization_threshold = normalization_threshold
        if normalization_threshold <= 0 or normalization_threshold > 1: raise ValueError(translations[">0or=1"])

        self.output_single_stem = output_single_stem
        if output_single_stem is not None: self.logger.debug(translations["output_single"].format(output_single_stem=output_single_stem))
        self.invert_using_spec = invert_using_spec
        if self.invert_using_spec: self.logger.debug(translations["step2"])

        self.sample_rate = int(sample_rate)
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
        self.log_onnxruntime_packages()
        self.setup_torch_device(system_info)

    def get_system_info(self):
        os_name = platform.system()
        os_version = platform.version()
        self.logger.info(f"{translations['os']}: {os_name} {os_version}")
        system_info = platform.uname()
        self.logger.info(translations["platform_info"].format(system_info=system_info, node=system_info.node, release=system_info.release, machine=system_info.machine, processor=system_info.processor))
        python_version = platform.python_version()
        self.logger.info(f"{translations['name_ver'].format(name='python')}: {python_version}")
        pytorch_version = torch.__version__
        self.logger.info(f"{translations['name_ver'].format(name='pytorch')}: {pytorch_version}")

        return system_info

    def log_onnxruntime_packages(self):
        onnxruntime_gpu_package = self.get_package_distribution("onnxruntime-gpu")
        onnxruntime_cpu_package = self.get_package_distribution("onnxruntime")

        if onnxruntime_gpu_package is not None: self.logger.info(f"{translations['install_onnx'].format(pu='GPU')}: {onnxruntime_gpu_package.version}")
        if onnxruntime_cpu_package is not None: self.logger.info(f"{translations['install_onnx'].format(pu='CPU')}: {onnxruntime_cpu_package.version}")

    def setup_torch_device(self, system_info):
        hardware_acceleration_enabled = False
        ort_providers = onnxruntime.get_available_providers()
        self.torch_device_cpu = torch.device("cpu")

        if torch.cuda.is_available():
            self.configure_cuda(ort_providers)
            hardware_acceleration_enabled = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and system_info.processor == "arm":
            self.configure_mps(ort_providers)
            hardware_acceleration_enabled = True

        if not hardware_acceleration_enabled:
            self.logger.info(translations["running_in_cpu"])
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CPUExecutionProvider"]

    def configure_cuda(self, ort_providers):
        self.logger.info(translations["running_in_cuda"])
        self.torch_device = torch.device("cuda")

        if "CUDAExecutionProvider" in ort_providers:
            self.logger.info(translations["onnx_have"].format(have='CUDAExecutionProvider'))
            self.onnx_execution_provider = ["CUDAExecutionProvider"]
        else: self.logger.warning(translations["onnx_not_have"].format(have='CUDAExecutionProvider'))

    def configure_mps(self, ort_providers):
        self.logger.info(translations["set_torch_mps"])
        self.torch_device_mps = torch.device("mps")
        self.torch_device = self.torch_device_mps

        if "CoreMLExecutionProvider" in ort_providers:
            self.logger.info(translations["onnx_have"].format(have='CoreMLExecutionProvider'))
            self.onnx_execution_provider = ["CoreMLExecutionProvider"]
        else: self.logger.warning(translations["onnx_not_have"].format(have='CoreMLExecutionProvider'))

    def get_package_distribution(self, package_name):
        try:
            return metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            self.logger.debug(translations["python_not_install"].format(package_name=package_name))
            return None

    def get_model_hash(self, model_path):
        self.logger.debug(translations["hash"].format(model_path=model_path))

        try:
            with open(model_path, "rb") as f:
                f.seek(-10000 * 1024, 2)
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            self.logger.error(translations["ioerror"].format(e=e))
            return hashlib.md5(open(model_path, "rb").read()).hexdigest()

    def download_file_if_not_exists(self, url, output_path):
        if os.path.isfile(output_path):
            self.logger.debug(translations["cancel_download"].format(output_path=output_path))
            return

        self.logger.debug(translations["download_model"].format(url=url, output_path=output_path))
        response = requests.get(url, stream=True, timeout=300)

        if response.status_code == 200:
            from tqdm import tqdm

            progress_bar = tqdm(total=int(response.headers.get("content-length", 0)), ncols=100, unit="byte")

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))
                    f.write(chunk)

            progress_bar.close()
        else: raise RuntimeError(translations["download_error"].format(url=url, status_code=response.status_code))

    def print_uvr_vip_message(self):
        if self.model_is_uvr_vip:
            self.logger.warning(translations["vip_model"].format(model_friendly_name=self.model_friendly_name))
            self.logger.warning(translations["vip_print"])

    def list_supported_model_files(self):
        response = requests.get(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/enj/znva/wfba/hie_zbqryf.wfba", "rot13"))
        response.raise_for_status()
        model_downloads_list = response.json()
        self.logger.debug(translations["load_download_json"])

        return {"MDX": {**model_downloads_list["mdx_download_list"], **model_downloads_list["mdx_download_vip_list"]}, "Demucs": {key: value for key, value in model_downloads_list["demucs_download_list"].items() if key.startswith("Demucs v4")}}
    
    def download_model_files(self, model_filename):
        model_path = os.path.join(self.model_file_dir, model_filename)
        supported_model_files_grouped = self.list_supported_model_files()

        yaml_config_filename = None
        self.logger.debug(translations["search_model"].format(model_filename=model_filename))

        for model_type, model_list in supported_model_files_grouped.items():
            for model_friendly_name, model_download_list in model_list.items():
                self.model_is_uvr_vip = "VIP" in model_friendly_name
                model_repo_url_prefix = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/hie5_zbqryf", "rot13")

                if isinstance(model_download_list, str) and model_download_list == model_filename:
                    self.logger.debug(translations["single_model"].format(model_friendly_name=model_friendly_name))
                    self.model_friendly_name = model_friendly_name

                    try:
                        self.download_file_if_not_exists(f"{model_repo_url_prefix}/MDX/{model_filename}", model_path)
                    except RuntimeError:
                        self.logger.warning(translations["not_found_model"])
                        self.download_file_if_not_exists(f"{model_repo_url_prefix}/Demucs/{model_filename}", model_path)

                    self.print_uvr_vip_message()
                    self.logger.debug(translations["single_model_path"].format(model_path=model_path))

                    return model_filename, model_type, model_friendly_name, model_path, yaml_config_filename
                elif isinstance(model_download_list, dict):
                    this_model_matches_input_filename = False

                    for file_name, file_url in model_download_list.items():
                        if file_name == model_filename or file_url == model_filename:
                            self.logger.debug(translations["find_model"].format(model_filename=model_filename, model_friendly_name=model_friendly_name))
                            this_model_matches_input_filename = True

                    if this_model_matches_input_filename:
                        self.logger.debug(translations["find_models"].format(model_friendly_name=model_friendly_name))
                        self.model_friendly_name = model_friendly_name
                        self.print_uvr_vip_message()

                        for config_key, config_value in model_download_list.items():
                            self.logger.debug(f"{translations['find_path']}: {config_key} -> {config_value}")

                            if config_value.startswith("http"): self.download_file_if_not_exists(config_value, os.path.join(self.model_file_dir, config_key))
                            elif config_key.endswith(".ckpt"):
                                try:
                                    self.download_file_if_not_exists(f"{model_repo_url_prefix}/Demucs/{config_key}", os.path.join(self.model_file_dir, config_key))
                                except RuntimeError:
                                    self.logger.warning(translations["not_found_model_warehouse"])

                                if model_filename.endswith(".yaml"):
                                    self.logger.warning(translations["yaml_warning"].format(model_filename=model_filename))
                                    self.logger.warning(translations["yaml_warning_2"].format(config_key=config_key))
                                    self.logger.warning(translations["yaml_warning_3"])

                                    model_filename = config_key
                                    model_path = os.path.join(self.model_file_dir, f"{model_filename}")

                                yaml_config_filename = config_value
                                yaml_config_filepath = os.path.join(self.model_file_dir, yaml_config_filename)

                                try:
                                    self.download_file_if_not_exists(f"{model_repo_url_prefix}/mdx_c_configs/{yaml_config_filename}", yaml_config_filepath)
                                except RuntimeError:
                                    self.logger.debug(translations["yaml_debug"])
                            else: self.download_file_if_not_exists(f"{model_repo_url_prefix}/Demucs/{config_value}", os.path.join(self.model_file_dir, config_value))

                        self.logger.debug(translations["download_model_friendly"].format(model_friendly_name=model_friendly_name, model_path=model_path))
                        return model_filename, model_type, model_friendly_name, model_path, yaml_config_filename

        raise ValueError(translations["not_found_model_2"].format(model_filename=model_filename))

    def load_model_data_from_yaml(self, yaml_config_filename):
        model_data_yaml_filepath = os.path.join(self.model_file_dir, yaml_config_filename) if not os.path.exists(yaml_config_filename) else yaml_config_filename
        self.logger.debug(translations["load_yaml"].format(model_data_yaml_filepath=model_data_yaml_filepath))
        model_data = yaml.load(open(model_data_yaml_filepath, encoding="utf-8"), Loader=yaml.FullLoader)
        self.logger.debug(translations["load_yaml_2"].format(model_data=model_data))

        if "roformer" in model_data_yaml_filepath: model_data["is_roformer"] = True
        return model_data

    def load_model_data_using_hash(self, model_path):
        self.logger.debug(translations["hash_md5"])
        model_hash = self.get_model_hash(model_path)
        self.logger.debug(translations["model_hash"].format(model_path=model_path, model_hash=model_hash))
        mdx_model_data_path = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/enj/znva/wfba/zbqry_qngn.wfba", "rot13")
        self.logger.debug(translations["mdx_data"].format(mdx_model_data_path=mdx_model_data_path))

        response = requests.get(mdx_model_data_path)
        response.raise_for_status()
        mdx_model_data_object = response.json()
        self.logger.debug(translations["load_mdx"])

        if model_hash in mdx_model_data_object: model_data = mdx_model_data_object[model_hash]
        else: raise ValueError(translations["model_not_support"].format(model_hash=model_hash))

        self.logger.debug(translations["uvr_json"].format(model_hash=model_hash, model_data=model_data))
        return model_data

    def load_model(self, model_filename):
        self.logger.info(translations["loading_model"].format(model_filename=model_filename))
        load_model_start_time = time.perf_counter()
        model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = self.download_model_files(model_filename)
        self.logger.debug(translations["download_model_friendly_2"].format(model_friendly_name=model_friendly_name, model_path=model_path))

        if model_path.lower().endswith(".yaml"): yaml_config_filename = model_path

        common_params = {
            "logger": self.logger,
            "log_level": self.log_level,
            "torch_device": self.torch_device,
            "torch_device_cpu": self.torch_device_cpu,
            "torch_device_mps": self.torch_device_mps,
            "onnx_execution_provider": self.onnx_execution_provider,
            "model_name": model_filename.split(".")[0],
            "model_path": model_path,
            "model_data": self.load_model_data_from_yaml(yaml_config_filename) if yaml_config_filename is not None else self.load_model_data_using_hash(model_path),
            "output_format": self.output_format,
            "output_bitrate": self.output_bitrate,
            "output_dir": self.output_dir,
            "normalization_threshold": self.normalization_threshold,
            "output_single_stem": self.output_single_stem,
            "invert_using_spec": self.invert_using_spec,
            "sample_rate": self.sample_rate,
        }

        separator_classes = {"MDX": "mdx_separator.MDXSeparator", "Demucs": "demucs_separator.DemucsSeparator"}

        if model_type not in self.arch_specific_params or model_type not in separator_classes: raise ValueError(translations["model_type_not_support"].format(model_type=model_type))
        if model_type == "Demucs" and sys.version_info < (3, 10): raise Exception(translations["demucs_not_support_python<3.10"])

        self.logger.debug(f"{translations['import_module']} {model_type}: {separator_classes[model_type]}")
        module_name, class_name = separator_classes[model_type].split(".")
        separator_class = getattr(import_module(f"main.library.architectures.{module_name}"), class_name)

        self.logger.debug(f"{translations['initialization']} {model_type}: {separator_class}")
        self.model_instance = separator_class(common_config=common_params, arch_config=self.arch_specific_params[model_type])
        self.logger.debug(translations["loading_model_success"])
        self.logger.info(f"{translations['loading_model_duration']}: {time.strftime('%H:%M:%S', time.gmtime(int(time.perf_counter() - load_model_start_time)))}")

    def separate(self, audio_file_path):
        self.logger.info(f"{translations['starting_separator']}: {audio_file_path}")
        separate_start_time = time.perf_counter()

        self.logger.debug(translations["normalization"].format(normalization_threshold=self.normalization_threshold))
        output_files = self.model_instance.separate(audio_file_path)

        self.model_instance.clear_gpu_cache()
        self.model_instance.clear_file_specific_paths()

        self.print_uvr_vip_message()

        self.logger.debug(translations["separator_success_3"])
        self.logger.info(f"{translations['separator_duration']}: {time.strftime('%H:%M:%S', time.gmtime(int(time.perf_counter() - separate_start_time)))}")
        return output_files

    def download_model_and_data(self, model_filename):
        self.logger.info(translations["loading_separator_model"].format(model_filename=model_filename))
        model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = self.download_model_files(model_filename)
        if model_path.lower().endswith(".yaml"): yaml_config_filename = model_path
        self.logger.info(translations["downloading_model"].format(model_type=model_type, model_friendly_name=model_friendly_name, model_path=model_path, model_data_dict_size=len(self.load_model_data_from_yaml(yaml_config_filename) if yaml_config_filename is not None else self.load_model_data_using_hash(model_path))))