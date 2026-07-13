import os
import sys
import time
import torch
import codecs
import hashlib
import requests
import warnings

from importlib import import_module

now_dir = os.getcwd()
sys.path.append(now_dir)

from main.library.utils import clear_gpu_cache
from main.tools.huggingface import HF_download_file
from main.app.variables import config, translations, configs

if not config.debug_mode: warnings.filterwarnings("ignore")

class Separator: 
    """
    The orchestrator class for Ultimate Vocal Remover v5 (UVR5) inference pipelines.
    Handles dynamic model downloads, architecture switching (MDX, Demucs, VR), 
    weight checksum validation, and file processing orchestration.
    """

    def __init__(
        self, 
        logger, 
        model_file_dir=config.configs["uvr5_path"], 
        output_dir=None, 
        output_format="wav", 
        normalization_threshold=0.9, 
        sample_rate=44100, 
        mdx_params={
            "hop_length": 1024, 
            "segment_size": 256, 
            "overlap": 0.25, 
            "batch_size": 1, 
            "enable_denoise": False
        }, 
        demucs_params={
            "segment_size": "Default", 
            "shifts": 2, 
            "overlap": 0.25, 
            "segments_enabled": True
        }, 
        vr_params={
            "batch_size": 1, 
            "window_size": 512, 
            "aggression": 5, 
            "enable_tta": False, 
            "enable_post_process": False, 
            "post_process_threshold": 0.2, 
            "high_end_process": False
        }
    ):
        """
        Initializes runtime parameters, cross-validates thresholds, and maps model-specific hyperparameters.

        Args:
            logger: System logger instance for tracking process messages.
            model_file_dir (str): Folder where model weights are stored or downloaded.
            output_dir (str, optional): Target directory for saving split tracks. Defaults to current directory.
            output_format (str): Audio container format extension (e.g., 'wav', 'mp3'). Defaults to "wav".
            normalization_threshold (float): Amplitude limit peak scaling factor. Defaults to 0.9.
            sample_rate (int): Target sampling frequency rate. Defaults to 44100.
            mdx_params (dict): Processing hyperparameters for MDX models.
            demucs_params (dict): Processing hyperparameters for Demucs architectures.
            vr_params (dict): Processing hyperparameters for VR (Vocal Remover) models.

        Raises:
            ValueError: If `normalization_threshold` falls outside the allowed range (0, 1].
        """

        self.logger = logger
        self.logger.info(translations["separator_info"].format(output_dir=output_dir, output_format=output_format))
        # Establish and verify filesystem storage spaces
        self.output_dir = output_dir if output_dir is not None else now_dir
        self.model_file_dir = model_file_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_file_dir, exist_ok=True)

        self.output_format = output_format if output_format is not None else "wav"
        self.normalization_threshold = normalization_threshold
        self.sample_rate = int(sample_rate)
        # Enforce valid range constraints on audio peak limits
        if normalization_threshold <= 0 or normalization_threshold > 1: raise ValueError("Normalization threshold must be between 0 and 1 (exclusive of 0).")

        # Map execution params to their respective structural categories
        self.arch_specific_params = {"MDX": mdx_params, "Demucs": demucs_params, "VR": vr_params}
        self.torch_device = torch.device(config.device)
        self.onnx_execution_provider = config.providers
        self.model_instance = None

    def download_file_if_not_exists(self, url, output_path):
        """
        Downloads files via the HuggingFace bridge API if not present in the local file system.

        Args:
            url (str): Remote file URL.
            output_path (str): Target filesystem download path.
        """

        if os.path.isfile(output_path): return
        HF_download_file(url, output_path)

    def list_supported_model_files(self):
        """
        Fetches the remote model repository inventory map. Obfuscated URLs are decoded via ROT13.

        Returns:
            dict: Structured directory categorization map containing model filenames.
        """

        # Remote repository JSON schema index file (Decoded using ROT13 cipher swap)
        response = requests.get(
            codecs.decode(
                "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/enj/znva/wfba/hie_zbqryf.wfba", 
                "rot13"
            )
        )

        response.raise_for_status()
        model_downloads_list = response.json()

        return {
            "MDX": {
                **model_downloads_list["mdx_download_list"], 
                **model_downloads_list["mdx_download_vip_list"]
            }, 
            "Demucs": {
                **model_downloads_list["demucs_download_list"]
            },
            "VR": {
                **model_downloads_list["vr_download_list"]
            }
        }
    
    def download_model_files(self, model_filename):
        """
        Identifies, down-links, and verifies target model weights across available remote categories.

        Args:
            model_filename (str): Base filename of the target weights checkpoint.

        Returns:
            tuple: (model_type, model_path)
                - model_type (str): Identified architecture family ('MDX', 'Demucs', or 'VR').
                - model_path (str): Absolute local filesystem path to the downloaded weights file.

        Raises:
            ValueError: If the model file name cannot be matched against remote inventory catalogs.
        """

        model_repo = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/hie5_zbqryf", "rot13")
        model_path = os.path.join(self.model_file_dir, model_filename)

        # Traverse model indexes to pinpoint the matching download endpoint
        for model_type, model_list in self.list_supported_model_files().items():
            for _, files in model_list.items():
                # Handling standalone model file items
                if isinstance(files, str) and files == model_filename:
                    # Fallback trial sequence to catch models across varying repository endpoints
                    try:
                        self.download_file_if_not_exists(f"{model_repo}/MDX/{model_filename}", model_path)
                    except:
                        try:
                            self.download_file_if_not_exists(f"{model_repo}/VR/{model_filename}",  model_path)
                        except:
                            self.download_file_if_not_exists(f"{model_repo}/Demucs/{model_filename}", model_path)

                    return model_type, model_path
                elif isinstance(files, dict) and any(model_filename in (k, v) for k, v in files.items()): # Handling multi-file compound structures (e.g., Demucs collections)
                    for _, file_val in files.items():
                        self.download_file_if_not_exists(f"{model_repo}/Demucs/{file_val}", os.path.join(self.model_file_dir, file_val))

                    return model_type, model_path

        raise ValueError(f"Model file '{model_filename}' not found in remote database index records.")

    def load_model_data(self, model_path = None, model_type = None):
        """
        Computes partial/full file checksum digests to fetch structural metadata definitions from the server.

        Args:
            model_path (str, optional): Target model weights path.
            model_type (str, optional): Architecture type string.

        Returns:
            dict: Model parsing parameters extracted from the server index database.

        Raises:
            ValueError: If hash mappings are missing or configuration parameters are invalid.
        """

        # Demucs handles configurations natively via package attributes
        if model_type == "Demucs": model_data = {}
        elif model_path is not None:
            try:
                with open(model_path, "rb") as f:
                    # Quick Hash Optimization: Read and parse only the trailing block segment
                    f.seek(-10000 * 1024, 2)
                    model_hash = hashlib.md5(f.read()).hexdigest()
            except IOError:
                # Fallback: Parse the full file stream if segment-seeking fails
                model_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()

            # Retrieve architectural details map from database server indices
            response = requests.get(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/enj/znva/wfba/zbqry_qngn.wfba", "rot13"))
            response.raise_for_status()
            model_data_object = response.json()

            if model_hash in model_data_object: model_data = model_data_object[model_hash]
            else: raise ValueError("Model signature verification failed. Weights are unknown or corrupted.")
        else: raise ValueError("Invalid parameters provided for metadata loading configurations.")

        return model_data

    def load_model(self, model_filename):
        """
        Downloads structural assets, extracts structural configuration logs, and dynamically 
        instantiates the proper processing class subclass.

        Args:
            model_filename (str): Name of the model file to activate.

        Raises:
            ValueError: If the detected model type configuration lacks internal runner code support.
        """

        self.logger.info(translations["loading_model"].format(model_filename=model_filename))
        model_type, model_path = self.download_model_files(model_filename)

        # Mapping names to their respective module and class identifiers
        separator_classes = {"MDX": "mdx_separator.MDXSeparator", "Demucs": "demucs_separator.DemucsSeparator", "VR": "vr_separator.VRSeparator"}
        # Consolidate standard runtime configurations
        common_params = {
            "logger": self.logger, 
            "torch_device": self.torch_device, 
            "onnx_execution_provider": self.onnx_execution_provider, 
            "model_name": model_filename.split(".")[0], 
            "model_path": model_path, 
            "model_data": self.load_model_data(model_path, model_type), 
            "output_format": self.output_format, 
            "output_dir": self.output_dir, 
            "normalization_threshold": self.normalization_threshold, 
            "invert_using_spec": configs.get("invert_using_spec", False), 
            "sample_rate": self.sample_rate
        }

        if model_type not in self.arch_specific_params or model_type not in separator_classes: 
            raise ValueError(translations["model_type_not_support"].format(model_type=model_type))

        # Dynamically import module and initialize class instantiations at runtime
        module_name, class_name = separator_classes[model_type].split(".")
        separator_class = getattr(import_module(f"main.library.uvr5_lib.{module_name}"), class_name)
        # Instantiate active runtime context interface class
        self.model_instance = separator_class(common_config=common_params, arch_config=self.arch_specific_params[model_type])

    def separate(self, audio_file_path):
        """
        Triggers the underlying model extraction pipeline on the target audio track.

        Args:
            audio_file_path (str): Target mixed track file path destination.

        Returns:
            list: Generated target stems output path destinations.
        """

        self.logger.info(f"{translations['starting_separator']}: {audio_file_path}")
        # Execute processing pipeline within the allocated instance class
        output_files = self.model_instance.separate(audio_file_path)

        # Clear specific variables and flush memory pipelines to prepare for subsequent batches
        self.model_instance.clear_file_specific_paths()
        clear_gpu_cache()

        self.logger.info(translations["separator_success_3"])
        return output_files