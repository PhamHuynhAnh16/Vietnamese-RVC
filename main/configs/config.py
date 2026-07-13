import os
import sys
import json
import torch
import onnxruntime

sys.path.append(os.getcwd())

from main.library.backends import directml, opencl, xpu, zluda

def singleton(cls):
    """
    A decorator to implement the Singleton pattern for a class.
    Ensures only one instance of the class is created across the application.
    """

    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances: instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Config:
    """
    Configuration manager class that loads, parses, and validates settings 
    from a JSON file, and automatically configures hardware backends 
    (CUDA, XPU, DirectML, OpenCL, MPS, CPU) for PyTorch and ONNX Runtime.
    """

    def __init__(self):
        """Initializes configuration paths, hardware status, and optimization flags."""

        # Define paths and load JSON configuration file
        self.configs_path = os.path.join("main", "configs", "config.json")
        self.configs = json.load(open(self.configs_path, "r"))
        # Base execution flags
        self.cpu_mode = self.configs.get("cpu_mode", False)
        self.cuda_available = torch.cuda.is_available()
        # Check for AMD ZLUDA environment mimicking CUDA
        self.is_zluda = self.cuda_available and torch.cuda.get_device_name().endswith("[ZLUDA]")
        self.hip_version = torch.version.hip if hasattr(torch.version, "hip") else None
        if self.is_zluda: zluda.init_zluda()
        # Debug and localization setups
        self.debug_mode = self.configs.get("debug_mode", False)
        self.translations = self.multi_language()
        # Device capability trackers
        self.invalid_gpu = False
        self.gpu_mem = None
        self.major = None # Compute capability major version
        # Map target device indexes for PyTorch and ONNX
        self.pytorch_device_idx = self.device_idx(self.configs.get("gpu_idx", 0))
        self.onnx_device_idx = {"device_id": self.pytorch_device_idx}
        # Default scaling and device auto-detection
        self.per_preprocess = 3.7
        self.device = self.get_default_device()
        self.tensorrt = self.configs.get("tensorrt", False)
        self.providers = self.get_providers()
        # Check precision acceleration capabilities (TF32 & BF16)
        self.cuda_tf32 = self.cuda_available and hasattr(torch.cuda, "is_tf32_supported") and self.major is not None and self.major >= 8 and torch.cuda.is_tf32_supported()
        self.cuda_bf16 = self.cuda_available and hasattr(torch.cuda, "is_bf16_supported") and self.major is not None and self.major >= 8 and torch.cuda.is_bf16_supported()
        self.xpu_fp16 = hasattr(torch, "xpu") and torch.xpu.is_available() and hasattr(torch.xpu, "is_bf16_supported") and torch.xpu.is_bf16_supported()
        # Determine structural support based on current active device
        self.tf32_support = self.device.startswith("cuda") and self.cuda_tf32
        self.bf16_support = self.device.startswith(("cuda", "xpu")) and (self.cuda_bf16 or self.xpu_fp16)
        # Apply optimal math precision modes (Bfloat16 / TensorFloat32)
        self.brain = self.configs.get("brain", False) and self.bf16_support and not self.cpu_mode
        self.tf32 = self.configs.get("tf32", False) and self.tf32_support and not self.is_zluda and not self.cpu_mode
        # Configure model quantization and compiler variables
        self.is_half = self.is_fp16()
        self.int8 = self.configs.get("int8", False)
        self.compile_all = self.device.startswith("cuda") and self.configs.get("compile_all", False)
        self.compile_mode = self.configs.get("compile_mode", None)
        self.allow_is_half = not self.device.startswith(("cpu", "mps", "ocl", "privateuseone")) or self.configs.get("allow_fp16_all_backend", False)
        # Pull pad/query/center/max threshold properties based on the device configurations
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()
        if self.compile_all: self.setup_compile()

    def device_idx(self, device_id):
        """
        Validates whether the provided device ID falls within the available range of hardware backends.

        Args:
            device_id (int): Target hardware device index.

        Returns:
            int: Validated device index (defaults to 0 if out of range).
        """

        total_device = torch.cuda.device_count() + directml.device_count() + opencl.device_count()
        if hasattr(torch, "xpu"): total_device += torch.xpu.device_count()

        if device_id >= 0 and device_id < total_device: return device_id
        else: 
            self.invalid_gpu = True
            return 0
    
    def multi_language(self):
        """
        Loads localization strings from JSON translation files based on the config settings.

        Returns:
            dict: Translated key-value pairs or None if an error occurs.
        """

        translations = None

        try:
            lang = self.configs.get("language", "vi-VN")
            # Check if directory contains any valid localization files
            if len([l for l in os.listdir(self.configs["language_path"]) if l.endswith(".json")]) < 1: 
                raise FileNotFoundError("Không tìm thấy bất cứ gói ngôn ngữ nào(No package languages found)")

            if not lang: lang = "vi-VN"
            # Validate language compatibility
            if lang not in self.configs["support_language"]: 
                raise ValueError("Ngôn ngữ không được hỗ trợ (Language not supported)")

            # Resolve paths, fall back to Vietnamese if target language file is missing
            lang_path = os.path.join(self.configs["language_path"], f"{lang}.json")
            if not os.path.exists(lang_path): 
                lang_path = os.path.join(self.configs["language_path"], "vi-VN.json")

            with open(lang_path, encoding="utf-8") as f:
                translations = json.load(f)
        except json.JSONDecodeError:
            # Output error using pre-existing translations structure if possible
            print(self.translations["empty_json"].format(file=lang))
            pass

        return translations
    
    def is_fp16(self):
        """
        Determines if FP16 precision is enabled and turns it off if the current hardware doesn't natively support it.

        Returns:
            bool: FP16 activation state.
        """

        fp16 = self.configs.get("fp16", False)
        # Force disable FP16 for basic or unoptimized backends to avoid runtime arithmetic crashes
        if self.device in ["cpu", "mps", "ocl", "privateuseone"] and fp16:
            self.configs["fp16"] = False
            fp16 = False

            # Update the configuration file with modified parameters
            with open(self.configs_path, "w") as f:
                json.dump(self.configs, f, indent=4)
        
        if not fp16: self.per_preprocess = 3.0
        return fp16
    
    def setup_compile(self):
        """
        Validates requirements and prepares environment configurations for PyTorch 2.0+ graph compilation (torch.compile).
        """

        import importlib

        # Check for Triton availability required by TorchInductor
        try:
            importlib.import_module("triton")
        except ModuleNotFoundError:
            self.compile_all = False
            self.configs["compile_all"] = False

            with open(self.configs_path, "w") as f:
                json.dump(self.configs, f, indent=4)

        if self.compile_all:
            dirs = self.configs.get("compile_cache_dir", "none")
            # Setup persistent compilation cache paths
            if dirs != "none":
                if not os.path.exists(dirs): os.makedirs(dirs)
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = dirs
            
            import torch._inductor.config as config
            # Optimize graph captures and model freezing behaviors
            torch._dynamo.config.capture_scalar_outputs = True
            config.freezing = True

    def device_config(self):
        """
        Retrieves baseline model matrix padding and processing thresholds based on target device VRAM or precision.

        Returns:
            tuple: Contains (x_pad, x_query, x_center, x_max) parameter thresholds.
        """

        # Lower processing parameters for low-end VRAM profiles (<= 4GB)
        if self.gpu_mem is not None and self.gpu_mem <= 4: 
            self.per_preprocess = 3.0
            return 1, 5, 30, 32
        
        # High parameters for standard FP16 execution, versus normal FP32 processing
        return (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
    
    def setup_cpu_mode(self):
        """
        Configures fallback CPU multithreading environment, turns off gradients, 
        and stubs GPU availabilities to optimize pure CPU tracking pipelines.
        """

        # Allocate logical processor thread constraints
        cores = max(os.cpu_count() - 2, 2)
        torch.set_num_threads(cores)
        torch.set_num_interop_threads(cores)

        # General inference speed enhancements for CPU
        torch.set_grad_enabled(False)
        torch.backends.mkldnn.enabled = True
        # Monkey-patch availability hooks to force runtime modules onto the CPU
        torch.cuda.is_available = lambda : False
        directml.is_available = lambda : False
        opencl.is_available = lambda : False
        torch.backends.mps.is_available = lambda : False
        if hasattr(torch, "xpu"): torch.xpu.is_available = lambda: False

    def get_default_device(self):
        """
        Scans all available hardware backends across the operating system 
        and extracts total memory details to determine the default target device string.

        Returns:
            str: Selected Torch device identifier string.
        """

        if not self.cpu_mode:
            if torch.cuda.is_available(): # 1. NVIDIA CUDA Execution Track
                device = f"cuda:{self.pytorch_device_idx}"
                device_properties = torch.cuda.get_device_properties(self.pytorch_device_idx)

                self.major = device_properties.major
                self.gpu_mem = device_properties.total_memory // (1024**3)
            elif hasattr(torch, "xpu") and torch.xpu.is_available(): # 2. Intel XPU Execution Track
                device = f"xpu:{self.pytorch_device_idx}"
                device_properties = torch.xpu.get_device_properties(self.pytorch_device_idx)

                self.gpu_mem = device_properties.total_memory // (1024**3)
                self.onnx_device_idx["device_type"] = "GPU"

                xpu.setup_onnxruntime_xpu()
                xpu.setup_gradscaler()
            elif directml.is_available(): # 3. Windows DirectML Execution Track
                device = f"privateuseone:{self.pytorch_device_idx}"
                device_properties = directml.get_device_properties(self.pytorch_device_idx)

                self.gpu_mem = device_properties.total_memory // (1024**3)
                self.onnx_device_idx = {"device_id": directml.mapping_directml(self.pytorch_device_idx)}
            elif opencl.is_available(): # 4. Cross-platform OpenCL Track
                device = f"ocl:{self.pytorch_device_idx}"
                device_properties = opencl.get_device_properties(self.pytorch_device_idx)

                self.gpu_mem = device_properties.total_memory // (1024**3)
            elif torch.backends.mps.is_available(): # 5. Apple Silicon Metal Performance Shaders (MPS) Track
                # Not entirely certain; it's just inherited from the original RVC.
                device = "mps"
            else: # 6. Fallback path if no accelerators are configured or detected
                device = "cpu"
                self.setup_cpu_mode()
        else:
            device = "cpu"
            self.setup_cpu_mode()

        return device 

    def get_providers(self):
        """
        Generates an ordered priority list of execution provider mappings for ONNX Runtime sessions.

        Returns:
            list: List of execution provider tuples containing backend names and configuration dicts.
        """

        ort_providers = onnxruntime.get_available_providers() if self.cpu_mode is False else []
        providers = []

        if not self.cpu_mode and not self.device.startswith("cpu"):
            # Check and append Nvidia hardware acceleration stacks
            if "TensorrtExecutionProvider" in ort_providers and self.tensorrt: providers.append(("TensorrtExecutionProvider", self.onnx_device_idx))
            if "CUDAExecutionProvider" in ort_providers: providers.append(("CUDAExecutionProvider", self.onnx_device_idx))
            # Check and append AMD ROCm hardware acceleration stacks
            if "ROCMExecutionProvider" in ort_providers: providers.append(("ROCMExecutionProvider", self.onnx_device_idx))
            if "MIGraphXExecutionProvider" in ort_providers: providers.append(("MIGraphXExecutionProvider", self.onnx_device_idx))
            # Check and append DirectX/DirectML or Intel OpenVINO stacks
            if "DmlExecutionProvider" in ort_providers: providers.append(("DmlExecutionProvider", self.onnx_device_idx))
            if "OpenVINOExecutionProvider" in ort_providers: providers.append(("OpenVINOExecutionProvider", self.onnx_device_idx))
            # Intel OneDNN or Apple CoreML structures
            if "DnnlExecutionProvider" in ort_providers: providers.append(("DnnlExecutionProvider", {}))
            if "CoreMLExecutionProvider" in ort_providers: providers.append(("CoreMLExecutionProvider", {}))

        # Ensure baseline CPU fallback option always sits at the bottom of the stack
        providers.append(("CPUExecutionProvider", {}))
        return providers

    def get_gpu_list(self):
        """
        Retrieves a collection listing names and storage capacities of every usable graphics card 
        matching the current application context.

        Returns:
            tuple: (int: total_gpu_count, list: list_of_gpu_specification_strings)
        """

        if self.device.startswith("cuda"):
            ngpu = torch.cuda.device_count()
            gpu_infos = [
                (
                    f"{i}: {torch.cuda.get_device_name(i)} " + 
                    f"({int(torch.cuda.get_device_properties(i).total_memory / (1024**3) + 0.4)} GB)"
                ) 
                for i in range(ngpu) 
                if self.device.startswith("cuda") or ngpu != 0
            ]
        elif self.device.startswith("xpu"):
            ngpu = torch.xpu.device_count()
            gpu_infos = [
                (
                    f"{i}: {torch.xpu.get_device_name(i)} " + 
                    f"({int(torch.xpu.get_device_properties(i).total_memory / (1024**3) + 0.4)} GB)"
                ) 
                for i in range(ngpu) 
                if self.device.startswith("xpu") or ngpu != 0
            ]
        elif self.device.startswith("privateuseone"):
            ngpu = directml.device_count()

            gpu_infos = [
                (
                    f"{i}: {directml.get_device_name(i)} " + 
                    f"({int(directml.get_device_properties(i).total_memory / (1024**3) + 0.4)} GB)"
                )
                for i in range(ngpu) 
                if self.device.startswith("privateuseone") or ngpu != 0
            ]
        elif self.device.startswith("ocl"):
            ngpu = opencl.device_count()

            gpu_infos = [
                (
                    f"{i}: {opencl.get_device_name(i)} " + 
                    f"({int(opencl.get_device_properties(i).total_memory / (1024**3) + 0.4)} GB)"
                )
                for i in range(ngpu) 
                if self.device.startswith("ocl") or ngpu != 0
            ]
        else:
            ngpu = 0
            gpu_infos = []
        
        return ngpu, gpu_infos