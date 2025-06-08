import os
import sys
import csv
import json
import codecs
import logging
import urllib.request
import logging.handlers

sys.path.append(os.getcwd())

from main.configs.config import Config

logger = logging.getLogger(__name__)
logger.propagate = False

config = Config()
python = sys.executable
translations = config.translations 
configs_json = os.path.join("main", "configs", "config.json")
configs = json.load(open(configs_json, "r"))

if logger.hasHandlers(): logger.handlers.clear()
else:
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG if config.debug_mode else logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(os.path.join(configs["logs_path"], "app.log"), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

if config.device in ["cpu", "mps", "ocl:0"] and configs.get("fp16", False):
    logger.warning(translations["fp16_not_support"])
    configs["fp16"] = config.is_half = False

    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)

models = {}
model_options = {}

method_f0 = ["mangio-crepe-full", "crepe-full", "fcpe", "rmvpe", "harvest", "pyin", "hybrid"]
method_f0_full = ["pm", "dio", "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium", "mangio-crepe-large", "mangio-crepe-full", "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", "crepe-full", "fcpe", "fcpe-legacy", "rmvpe", "rmvpe-legacy", "harvest", "yin", "pyin", "swipe", "hybrid"]

embedders_mode = ["fairseq", "onnx", "transformers", "spin"]
embedders_model = ["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "portuguese_hubert_base", "custom"]

paths_for_files = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(configs["audios_path"]) for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])

model_name = sorted(list(model for model in os.listdir(configs["weights_path"]) if model.endswith((".pth", ".onnx")) and not model.startswith("G_") and not model.startswith("D_"))) 
index_path = sorted([os.path.join(root, name) for root, _, files in os.walk(configs["logs_path"], topdown=False) for name in files if name.endswith(".index") and "trained" not in name])

pretrainedD = [model for model in os.listdir(configs["pretrained_custom_path"]) if model.endswith(".pth") and "D" in model]
pretrainedG = [model for model in os.listdir(configs["pretrained_custom_path"]) if model.endswith(".pth") and "G" in model]

presets_file = sorted(list(f for f in os.listdir(configs["presets_path"]) if f.endswith(".conversion.json")))
audio_effect_presets_file = sorted(list(f for f in os.listdir(configs["presets_path"]) if f.endswith(".effect.json")))
f0_file = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(configs["f0_path"]) for f in files if f.endswith(".txt")])

language = configs.get("language", "vi-VN")
theme = configs.get("theme", "NoCrypt/miku")

edgetts = configs.get("edge_tts", ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"])
google_tts_voice = configs.get("google_tts_voice", ["vi", "en"])

mdx_model = configs.get("mdx_model", "MDXNET_Main")
uvr_model = configs.get("demucs_model", "HD_MMI") + mdx_model

font = configs.get("font", "https://fonts.googleapis.com/css2?family=Courgette&display=swap")
sample_rate_choice = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000]
csv_path = configs["csv_path"]

if "--allow_all_disk" in sys.argv and sys.platform == "win32":
    try:
        import win32api
    except:
        os.system(f"{python} -m pip install pywin32")
        import win32api

    allow_disk = win32api.GetLogicalDriveStrings().split('\x00')[:-1]
else: allow_disk = []

if os.path.exists(csv_path): reader = list(csv.DictReader(open(csv_path, newline='', encoding='utf-8')))
else:
    reader = list(csv.DictReader([line.decode('utf-8') for line in urllib.request.urlopen(codecs.decode("uggcf://qbpf.tbbtyr.pbz/fcernqfurrgf/q/1gNHnDeRULtEfz1Yieaw14USUQjWJy0Oq9k0DrCrjApb/rkcbeg?sbezng=pfi&tvq=1977693859", "rot13")).readlines()]))
    writer = csv.DictWriter(open(csv_path, mode='w', newline='', encoding='utf-8'), fieldnames=reader[0].keys())
    writer.writeheader()
    writer.writerows(reader)

for row in reader:
    filename = row['Filename']
    url = None

    for value in row.values():
        if isinstance(value, str) and "huggingface" in value:
            url = value
            break

    if url: models[filename] = url