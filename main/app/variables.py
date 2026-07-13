import os
import re
import sys
import csv
import json
import logging
import logging.handlers

import gradio as gr

from packaging import version

sys.path.append(os.getcwd())

from main.configs.config import Config

# Load core system environment settings
config = Config()
python = sys.executable
configs = config.configs
translations = config.translations 
configs_json = config.configs_path

log_file = os.path.join(configs["logs_path"], "app.log")

# Automatically cleans up the log file if it exceeds the limit.
if os.path.exists(log_file):
    file_size = os.path.getsize(log_file)
    try:
        if file_size >= (10 * 1024 * 1024):
            f = open(log_file, "w", encoding="utf-8")
            f.close()
    except:
        pass

# Logger Initialization & Hierarchy Isolation
logger = logging.getLogger(__name__)
logger.propagate = False # Prevent logs from bubbling up to the root logger duplicate entries

# Detect Gradio version to backward-stabilize structural parameter compatibility
gradio_version = version.parse(gr.__version__) <= version.parse("6.0.0")

# Set up operational audio parameters based on the parsed Gradio version spec
audio_params = {
    "show_download_button": True,
    "show_share_button": True
} if gradio_version else {
    "buttons": ["download", "share"]
}

# When the application restarts, the web interface does not reload; this code is used to reload the interface.
reload_js = """
() => {
    function Checking() {
        fetch(window.location.origin)
            .then(response => {
                if (response.ok) {
                    location.reload();
                } else {
                    setTimeout(Checking, 3000);
                }
            })
            .catch(err => {
                setTimeout(Checking, 3000);
            });
    }
    setTimeout(Checking, 10000);
}
"""

# Configure File & Console Logging Handlers
if not logger.hasHandlers():
    # Console stream logging configuration
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG if config.debug_mode else logging.INFO)
    # Rotating file storage logging configuration (Persists debug logs to app.log)
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=0, backupCount=3, encoding='utf-8')
    file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    # Attach pipelines to active system interface logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

# Guard 1: Verify FP16 half-precision computation compatibility on low-tier or non-CUDA compute backends
if config.device.startswith(("cpu", "mps", "ocl", "privateuseone")) and configs.get("fp16", False) and not configs.get("allow_fp16_all_backend", False):
    logger.warning(translations["fp16_not_support"])
    configs["fp16"] = config.is_half = False

# Guard 2: Handle fallback scenarios for invalid/non-standard GPU requests by forcing host defaults
if config.invalid_gpu and not config.device.startswith("cpu"):
    logger.warning(translations["invalid_gpu"])
    config.onnx_device_idx = {"device_id": 0}
    config.pytorch_device_idx = 0
    configs["gpu_idx"] = 0

# Guard 3: Warn against incompatible combinations of INT8 quantization inside TensorRT architectures
if config.int8 and config.providers[0][0].startswith("Tensor"):
    logger.warning(translations["int8_warn"])
    configs["int8"] = config.int8 = False

# Commit synchronized setup back to state JSON configuration files
with open(configs_json, "w") as f:
    json.dump(configs, f, indent=4)

# Optional: Enable Audio Stream Input/Output (ASIO) environment flag flags if explicitly targeted
if configs.get("asio_enabled", False):
    logger.info(translations["enabled_asio"])
    os.environ["SD_ENABLE_ASIO"] = "1"

# Static Data Structures & Choices Mappings
models = {}
model_options = {}

file_types = [
    ".wav", 
    ".mp3", 
    ".flac", 
    ".ogg", 
    ".opus", 
    ".m4a", 
    ".mp4", 
    ".aac", 
    ".alac", 
    ".wma", 
    ".aiff", 
    ".webm", 
    ".ac3"
]

export_format_choices = [
    "wav", 
    "mp3", 
    "flac", 
    "ogg", 
    "opus", 
    "m4a", 
    "mp4", 
    "aac", 
    "alac", 
    "wma", 
    "aiff", 
    "webm", 
    "ac3"
]

# Fundamental base variants for F0 pitch extraction algorithms
method_f0 = [
    "mangio-crepe-full", 
    "crepe-full", 
    "fcpe", 
    "rmvpe", 
    "harvest-stonemask", 
    "pyin", 
    "hybrid"
]
# Comprehensive enumeration list covering all supported F0 model configurations
method_f0_full = [
    "rmvpe-mix", 
    "rmvpe-mix-medfilt", 
    "hpa-rmvpe", 
    "hpa-rmvpe-medfilt", 
    "hpa-rmvpe-previous", 
    "hpa-rmvpe-previous-medfilt", 
    "hpa-rmvpe-v4", 
    "hpa-rmvpe-v4-medfilt",
    "rmvpe", 
    "rmvpe-medfilt",
    "djcm", 
    "djcm-medfilt", 
    "djcm-svs", 
    "djcm-svs-medfilt", 
    "swift", 
    "fcpe", 
    "fcpe-previous", 
    "fcpe-legacy", 
    "mangio-crepe-full", 
    "crepe-full", 
    "mangio-crepe-large", 
    "crepe-large", 
    "mangio-crepe-medium", 
    "crepe-medium", 
    "mangio-crepe-small", 
    "crepe-small", 
    "mangio-crepe-tiny", 
    "crepe-tiny", 
    "harvest-stonemask", 
    "harvest", 
    "pesto", 
    "swipe-stonemask", 
    "swipe", 
    "mangio-penn", 
    "penn", 
    "dio-stonemask", 
    "dio", 
    "pm-ac", 
    "pm-cc", 
    "pm-shs", 
    "pyin", 
    "yin", 
    "piptrack", 
    "hybrid"
]
# Hybrid execution combinations for combining pitch models
hybrid_f0_method = [
    "hybrid[pm+dio]", 
    "hybrid[pm+crepe-tiny]", 
    "hybrid[pm+crepe]", 
    "hybrid[pm+fcpe]", 
    "hybrid[pm+rmvpe]", 
    "hybrid[pm+harvest]", 
    "hybrid[pm+yin]", 
    "hybrid[dio+crepe-tiny]", 
    "hybrid[dio+crepe]", 
    "hybrid[dio+fcpe]", 
    "hybrid[dio+rmvpe]", 
    "hybrid[dio+harvest]", 
    "hybrid[dio+yin]", 
    "hybrid[crepe-tiny+crepe]", 
    "hybrid[crepe-tiny+fcpe]", 
    "hybrid[crepe-tiny+rmvpe]", 
    "hybrid[crepe-tiny+harvest]", 
    "hybrid[crepe+fcpe]", 
    "hybrid[crepe+rmvpe]", 
    "hybrid[crepe+harvest]", 
    "hybrid[crepe+yin]", 
    "hybrid[fcpe+rmvpe]", 
    "hybrid[fcpe+harvest]", 
    "hybrid[fcpe+yin]", 
    "hybrid[rmvpe+harvest]", 
    "hybrid[rmvpe+yin]", 
    "hybrid[harvest+yin]"
]

embedders_mode = [
    "fairseq", 
    "onnx", 
    "transformers"
]

embedders_model = [
    "hubert_base", 
    "contentvec_base", 
    "vietnamese_hubert_base", 
    "japanese_hubert_base", 
    "korean_hubert_base", 
    "chinese_hubert_base", 
    "portuguese_hubert_base", 
    "spin-v1", 
    "spin-v2",
    "custom"
]

whisper_model = [
    "tiny", 
    "tiny.en", 
    "base", 
    "base.en", 
    "small", 
    "small.en", 
    "medium", 
    "medium.en", 
    "large-v1", 
    "large-v2", 
    "large-v3", 
    "large-v3-turbo"
]

whisper_languages = configs.get("whisper_languages", ["vi", "en"])

sample_rate_choice = [
    8000, 
    11025, 
    12000, 
    16000, 
    22050, 
    24000, 
    32000, 
    44100, 
    48000, 
    88200, 
    96000, 
    176400, 
    192000, 
    352800, 
    384000
]

# Discover and sort available input source audio files
paths_for_files = sorted([
    os.path.abspath(os.path.join(root, f)) 
    for root, _, files in os.walk(configs["audios_path"]) 
    for f in files 
    if os.path.splitext(f)[1].lower() in file_types and not f.startswith(("Convert", "output"))
])

# Scan and discover references dataset folders
reference_list = sorted([
    re.sub(r'_v\d+_(?:[A-Za-z0-9_]+?)_(True|False)$', '', name) 
    for name in os.listdir(configs["reference_path"]) 
    if (
        os.path.exists(os.path.join(configs["reference_path"], name)) and 
        os.path.isdir(os.path.join(configs["reference_path"], name))
    )
])

# Scan available deployment weights (.pth / .onnx inference configurations)
model_name = sorted([
    model for model in os.listdir(configs["weights_path"]) 
    if model.endswith((".pth", ".onnx")) and not model.startswith("G_") and not model.startswith("D_")
])

# Map existing feature index alignment vectors (.index clusters)
index_path = sorted([
    os.path.join(root, name) 
    for root, _, files in os.walk(configs["logs_path"], topdown=False) 
    for name in files 
    if name.endswith(".index") and "trained" not in name
])

# Load custom model components from paths
pretraineds = os.listdir(configs["pretrained_custom_path"])

pretrainedD = [
    model for model in pretraineds 
    if model.endswith(".pth") and "D" in model
]

pretrainedG = [
    model for model in pretraineds 
    if model.endswith(".pth") and "G" in model
]

# Parse operational and design presets across UI modules
all_presets = os.listdir(configs["presets_path"])

presets_file = sorted([
    f for f in all_presets 
    if f.endswith(".conversion.json")
])

audio_effect_presets_file = sorted([
    f for f in all_presets 
    if f.endswith(".effect.json")
])

realtime_presets_file = sorted([
    f for f in all_presets 
    if f.endswith(".realtime.json")
])

# Scan pre-extracted pitch cache metrics maps (.txt pitch data files)
f0_file = sorted([
    os.path.abspath(os.path.join(root, f)) 
    for root, _, files in os.walk(configs["f0_path"]) 
    for f in files if f.endswith(".txt")
])

# Localization and aesthetic assets setup
language = configs.get("language", "vi-VN")
theme = configs.get("theme", "NoCrypt/miku")
# TTS configurations structures loading
edgetts = configs.get("edge_tts", ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"])
google_tts_voice = configs.get("google_tts_voice", ["vi", "en"])
# UVR background processing neural architectures dictionary map
vr_models = configs.get("vr_models", "")
demucs_models = configs.get("demucs_models", "")
mdx_models = configs.get("mdx_models", "")
karaoke_models = configs.get("karaoke_models", "")
reverb_models = configs.get("reverb_models", "")
denoise_models = configs.get("denoise_models", "")
uvr_model = list(demucs_models.keys()) + list(vr_models.keys()) + list(mdx_models.keys())

font = configs.get("font", "https://fonts.googleapis.com/css2?family=Saira&display=swap")
csv_path = configs["csv_path"]

# OS Specific Extended Storage Privileges Isolation (Grading full system access paths)
try:
    if "--allow_all_disk" in sys.argv:
        import ctypes

        allow_disk = []

        if sys.platform == "win32": # Retrieve all available logical drive letters on Windows platforms
            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if bitmask & 1: allow_disk.append(f"{letter}:\\")
                bitmask >>= 1
        elif sys.platform == "linux": # Traverse systemic mounts lists via standard library functions on Linux systems
            class Mntent(ctypes.Structure):
                _fields_ = [("mnt_fsname", ctypes.c_char_p), ("mnt_dir", ctypes.c_char_p), ("mnt_type", ctypes.c_char_p), ("mnt_opts", ctypes.c_char_p), ("mnt_freq", ctypes.c_int), ("mnt_passno", ctypes.c_int)]
            
            libc = ctypes.CDLL(None)
            libc.setmntent.restype = ctypes.c_void_p
            libc.getmntent.restype = ctypes.POINTER(Mntent)
            libc.endmntent.restype = ctypes.c_int

            fp = libc.setmntent(b"/proc/mounts", b"r")

            if not fp: allow_disk = []
            else:
                while 1:
                    entry_ptr = libc.getmntent(fp)
                    if not entry_ptr: break
                    allow_disk.append(entry_ptr.contents.mnt_fsname.decode('utf-8', errors='ignore'))
        else: allow_disk = []
    else: allow_disk = []
except:
    allow_disk = []

# Index Hub Content Aggregation (Parsing HuggingFace mirrors links from CSV trackers)
try:
    for row in list(csv.DictReader(open(csv_path, newline='', encoding='utf-8'))):
        filename = row['Filename']
        url = None

        for value in row.values():
            if isinstance(value, str) and "huggingface" in value:
                url = value
                break

        if url: models[filename] = url
except:
    pass