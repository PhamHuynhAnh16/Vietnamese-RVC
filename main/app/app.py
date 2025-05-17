import os
import re
import ssl
import sys
import json
import torch
import codecs
import shutil
import asyncio
import librosa
import logging
import datetime
import platform
import requests
import warnings
import threading
import subprocess
import logging.handlers

import numpy as np
import gradio as gr
import pandas as pd
import soundfile as sf

from time import sleep
from multiprocessing import cpu_count

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.configs.config import Config

ssl._create_default_https_context = ssl._create_unverified_context
logger = logging.getLogger(__name__)
logger.propagate = False

if logger.hasHandlers(): logger.handlers.clear()
else:
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(os.path.join("assets", "logs", "app.log"), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

warnings.filterwarnings("ignore")
for l in ["httpx", "gradio", "uvicorn", "httpcore", "urllib3"]:
    logging.getLogger(l).setLevel(logging.ERROR)

config = Config()
python = sys.executable
translations = config.translations 
configs_json = os.path.join("main", "configs", "config.json")
configs = json.load(open(configs_json, "r"))

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

if config.device in ["cpu", "mps"] and configs.get("fp16", False):
    logger.warning(translations["fp16_not_support"])
    configs["fp16"] = config.is_half = False
    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)

models, model_options = {}, {}
method_f0 = ["mangio-crepe-full", "crepe-full", "fcpe", "rmvpe", "harvest", "pyin"]
method_f0_full = ["pm", "dio", "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium", "mangio-crepe-large", "mangio-crepe-full", "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", "crepe-full", "fcpe", "fcpe-legacy", "rmvpe", "rmvpe-legacy", "harvest", "yin", "pyin", "swipe"]
embedders_mode = ["fairseq", "onnx", "transformers", "spin"]
embedders_model = ["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "portuguese_hubert_base", "custom"]
paths_for_files = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk("audios") for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])
model_name, index_path = sorted(list(model for model in os.listdir(os.path.join("assets", "weights")) if model.endswith((".pth", ".onnx")) and not model.startswith("G_") and not model.startswith("D_"))), sorted([os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".index") and "trained" not in name])
pretrainedD, pretrainedG, Allpretrained = ([model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "D" in model], [model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "G" in model], [os.path.join("assets", "models", path, model) for path in ["pretrained_v1", "pretrained_v2", "pretrained_custom"] for model in os.listdir(os.path.join("assets", "models", path)) if model.endswith(".pth") and ("D" in model or "G" in model)])
separate_model = sorted([os.path.join("assets", "models", "uvr5", models) for models in os.listdir(os.path.join("assets", "models", "uvr5")) if models.endswith((".th", ".yaml", ".onnx"))])
presets_file = sorted(list(f for f in os.listdir(os.path.join("assets", "presets")) if f.endswith(".json")))
f0_file = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(os.path.join("assets", "f0")) for f in files if f.endswith(".txt")])
language, theme, edgetts, google_tts_voice, mdx_model, uvr_model, font = configs.get("language", "vi-VN"), configs.get("theme", "NoCrypt/miku"), configs.get("edge_tts", ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]), configs.get("google_tts_voice", ["vi", "en"]), configs.get("mdx_model", "MDXNET_Main"), (configs.get("demucs_model", "HD_MMI") + configs.get("mdx_model", "MDXNET_Main")), configs.get("font", "https://fonts.googleapis.com/css2?family=Courgette&display=swap")
sample_rate_choice = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000]
csv_path = os.path.join("assets", "spreadsheet.csv")
logger.info(config.device)

if "--allow_all_disk" in sys.argv:
    import win32api
    allow_disk = win32api.GetLogicalDriveStrings().split('\x00')[:-1]
else: allow_disk = []

if language == "vi-VN": 
    import gradio.strings
    gradio.strings.en = {"RUNNING_LOCALLY": "* Chạy trên liên kết nội bộ:  {}://{}:{}", "RUNNING_LOCALLY_SSR": "* Chạy trên liên kết nội bộ:  {}://{}:{}, với SSR ⚡ (thử nghiệm, để tắt hãy dùng `ssr=False` trong `launch()`)", "SHARE_LINK_DISPLAY": "* Chạy trên liên kết công khai: {}", "COULD_NOT_GET_SHARE_LINK": "\nKhông thể tạo liên kết công khai. Vui lòng kiểm tra kết nối mạng của bạn hoặc trang trạng thái của chúng tôi: https://status.gradio.app.", "COULD_NOT_GET_SHARE_LINK_MISSING_FILE": "\nKhông thể tạo liên kết công khai. Thiếu tập tin: {}. \n\nVui lòng kiểm tra kết nối internet của bạn. Điều này có thể xảy ra nếu phần mềm chống vi-rút của bạn chặn việc tải xuống tệp này. Bạn có thể cài đặt thủ công bằng cách làm theo các bước sau: \n\n1. Tải xuống tệp này: {}\n2. Đổi tên tệp đã tải xuống thành: {}\n3. Di chuyển tệp đến vị trí này: {}", "COLAB_NO_LOCAL": "Không thể hiển thị giao diện nội bộ trên google colab, liên kết công khai đã được tạo.", "PUBLIC_SHARE_TRUE": "\nĐể tạo một liên kết công khai, hãy đặt `share=True` trong `launch()`.", "MODEL_PUBLICLY_AVAILABLE_URL": "Mô hình được cung cấp công khai tại: {} (có thể mất tới một phút để sử dụng được liên kết)", "GENERATING_PUBLIC_LINK": "Đang tạo liên kết công khai (có thể mất vài giây...):", "BETA_INVITE": "\nCảm ơn bạn đã là người dùng Gradio! Nếu bạn có thắc mắc hoặc phản hồi, vui lòng tham gia máy chủ Discord của chúng tôi và trò chuyện với chúng tôi: https://discord.gg/feTf9x3ZSB", "COLAB_DEBUG_TRUE": "Đã phát hiện thấy sổ tay Colab. Ô này sẽ chạy vô thời hạn để bạn có thể xem lỗi và nhật ký. " "Để tắt, hãy đặt debug=False trong launch().", "COLAB_DEBUG_FALSE": "Đã phát hiện thấy sổ tay Colab. Để hiển thị lỗi trong sổ ghi chép colab, hãy đặt debug=True trong launch()", "COLAB_WARNING": "Lưu ý: việc mở Chrome Inspector có thể làm hỏng bản demo trong sổ tay Colab.", "SHARE_LINK_MESSAGE": "\nLiên kết công khai sẽ hết hạn sau 72 giờ. Để nâng cấp GPU và lưu trữ vĩnh viễn miễn phí, hãy chạy `gradio deploy` từ terminal trong thư mục làm việc để triển khai lên huggingface (https://huggingface.co/spaces)", "INLINE_DISPLAY_BELOW": "Đang tải giao diện bên dưới...", "COULD_NOT_GET_SHARE_LINK_CHECKSUM": "\nKhông thể tạo liên kết công khai. Tổng kiểm tra không khớp cho tập tin: {}."}

if os.path.exists(csv_path): cached_data = pd.read_csv(csv_path) 
else:
    cached_data = pd.read_csv(codecs.decode("uggcf://qbpf.tbbtyr.pbz/fcernqfurrgf/q/1gNHnDeRULtEfz1Yieaw14USUQjWJy0Oq9k0DrCrjApb/rkcbeg?sbezng=pfi&tvq=1977693859", "rot13"))
    cached_data.to_csv(csv_path, index=False)

for _, row in cached_data.iterrows():
    filename = row['Filename']
    url = None

    for value in row.values:
        if isinstance(value, str) and "huggingface" in value:
            url = value
            break

    if url: models[filename] = url

def gr_info(message):
    gr.Info(message, duration=2)
    logger.info(message)

def gr_warning(message):
    gr.Warning(message, duration=2)
    logger.warning(message)

def gr_error(message):
    gr.Error(message=message, duration=6)
    logger.error(message)

def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = [f"{i}: {torch.cuda.get_device_name(i)} ({int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4)} GB)" for i in range(ngpu) if torch.cuda.is_available() or ngpu != 0]
    return "\n".join(gpu_infos) if len(gpu_infos) > 0 else translations["no_support_gpu"]

def change_f0_choices(): 
    f0_file = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(os.path.join("assets", "f0")) for f in files if f.endswith(".txt")])
    return {"value": f0_file[0] if len(f0_file) >= 1 else "", "choices": f0_file, "__type__": "update"}

def change_audios_choices(input_audio): 
    audios = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk("audios") for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])
    return {"value": input_audio if input_audio != "" else (audios[0] if len(audios) >= 1 else ""), "choices": audios, "__type__": "update"}

def change_separate_choices():
    return [{"choices": sorted([os.path.join("assets", "models", "uvr5", models) for models in os.listdir(os.path.join("assets", "models", "uvr5")) if model.endswith((".th", ".yaml", ".onnx"))]), "__type__": "update"}]

def change_models_choices():
    model, index = sorted(list(model for model in os.listdir(os.path.join("assets", "weights")) if model.endswith((".pth", ".onnx")) and not model.startswith("G_") and not model.startswith("D_"))), sorted([os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".index") and "trained" not in name])
    return [{"value": model[0] if len(model) >= 1 else "", "choices": model, "__type__": "update"}, {"value": index[0] if len(index) >= 1 else "", "choices": index, "__type__": "update"}]

def change_allpretrained_choices():
    return [{"choices": sorted([os.path.join("assets", "models", path, model) for path in ["pretrained_v1", "pretrained_v2", "pretrained_custom"] for model in os.listdir(os.path.join("assets", "models", path)) if model.endswith(".pth") and ("D" in model or "G" in model)]), "__type__": "update"}]

def change_pretrained_choices():
    return [{"choices": sorted([model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "D" in model]), "__type__": "update"}, {"choices": sorted([model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "G" in model]), "__type__": "update"}]

def change_choices_del():
    return [{"choices": sorted(list(model for model in os.listdir(os.path.join("assets", "weights")) if model.endswith(".pth") and not model.startswith("G_") and not model.startswith("D_"))), "__type__": "update"}, {"choices": sorted([os.path.join("assets", "logs", f) for f in os.listdir(os.path.join("assets", "logs")) if "mute" not in f and os.path.isdir(os.path.join("assets", "logs", f))]), "__type__": "update"}]

def change_preset_choices():
    return {"value": "", "choices": sorted(list(f for f in os.listdir(os.path.join("assets", "presets")) if f.endswith(".json"))), "__type__": "update"}

def change_tts_voice_choices(google):
    return {"choices": google_tts_voice if google else edgetts, "value": google_tts_voice[0] if google else edgetts[0], "__type__": "update"}

def change_backing_choices(backing, merge):
    if backing or merge: return {"value": False, "interactive": False, "__type__": "update"}
    elif not backing or not merge: return  {"interactive": True, "__type__": "update"}
    else: gr_warning(translations["option_not_valid"])

def change_download_choices(select):
    selects = [False]*10

    if select == translations["download_url"]: selects[0] = selects[1] = selects[2] = True
    elif select == translations["download_from_csv"]:  selects[3] = selects[4] = True
    elif select == translations["search_models"]: selects[5] = selects[6] = True
    elif select == translations["upload"]: selects[9] = True
    else: gr_warning(translations["option_not_valid"])

    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]

def change_download_pretrained_choices(select):
    selects = [False]*8

    if select == translations["download_url"]: selects[0] = selects[1] = selects[2] = True
    elif select == translations["list_model"]: selects[3] = selects[4] = selects[5] = True
    elif select == translations["upload"]: selects[6] = selects[7] = True
    else: gr_warning(translations["option_not_valid"])

    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]

def get_index(model):
    model = os.path.basename(model).split("_")[0]
    return {"value": next((f for f in [os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".index") and "trained" not in name] if model.split(".")[0] in f), ""), "__type__": "update"} if model else None

def index_strength_show(index):
    return {"visible": index != "" and os.path.exists(index), "value": 0.5, "__type__": "update"}

def hoplength_show(method, hybrid_method=None):
    show_hop_length_method = ["mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium", "mangio-crepe-large", "mangio-crepe-full", "fcpe", "fcpe-legacy", "yin", "pyin"]

    if method in show_hop_length_method: visible = True
    elif method == "hybrid":
        methods_str = re.search("hybrid\[(.+)\]", hybrid_method)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        for i in methods:
            visible = i in show_hop_length_method
            if visible: break
    else: visible = False
    
    return {"visible": visible, "__type__": "update"}

def visible(value):
    return {"visible": value, "__type__": "update"}

def valueFalse_interactive(inp): 
    return {"value": False, "interactive": inp, "__type__": "update"}

def valueEmpty_visible1(inp1): 
    return {"value": "", "visible": inp1, "__type__": "update"}

def process_input(file_path):
    file_contents = ""

    if not file_path.endswith(".srt"):
        with open(file_path, "r", encoding="utf-8") as file:
            file_contents = file.read()

    gr_info(translations["upload_success"].format(name=translations["text"]))
    return file_contents

def fetch_pretrained_data():
    response = requests.get(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/wfba/phfgbz_cergenvarq.wfba", "rot13"))
    response.raise_for_status()

    return response.json()

def update_sample_rate_dropdown(model):
    data = fetch_pretrained_data()
    if model != translations["success"]: return {"choices": list(data[model].keys()), "value": list(data[model].keys())[0], "__type__": "update"}

def if_done(done, p):
    while 1:
        if p.poll() is None: sleep(0.5)
        else: break

    done[0] = True

def restart_app():
    global app

    gr_info(translations["15s"])
    os.system("cls" if platform.system() == "Windows" else "clear")
    
    app.close()
    subprocess.run([python, os.path.join("main", "app", "app.py")] + sys.argv[1:])

def change_language(lang):
    configs = json.load(open(configs_json, "r"))
    configs["language"] = lang

    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)

    restart_app()

def change_theme(theme):
    with open(configs_json, "r") as f:
        configs = json.load(f)

    configs["theme"] = theme
    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)

    restart_app()

def change_font(font):
    with open(configs_json, "r") as f:
        configs = json.load(f)

    configs["font"] = font
    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)

    restart_app()

def zip_file(name, pth, index):
    pth_path = os.path.join("assets", "weights", pth)
    if not pth or not os.path.exists(pth_path) or not pth.endswith((".pth", ".onnx")): return gr_warning(translations["provide_file"].format(filename=translations["model"]))

    zip_file_path = os.path.join("assets", "logs", name, name + ".zip")
    gr_info(translations["start"].format(start=translations["zip"]))

    import zipfile
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(pth_path, os.path.basename(pth_path))
        if index: zipf.write(index, os.path.basename(index))

    gr_info(translations["success"])
    return {"visible": True, "value": zip_file_path, "__type__": "update"}

def fetch_models_data(search):
    all_table_data = [] 
    page = 1 

    while 1:
        try:
            response = requests.post(url=codecs.decode("uggcf://ibvpr-zbqryf.pbz/srgpu_qngn.cuc", "rot13"), data={"page": page, "search": search})

            if response.status_code == 200:
                table_data = response.json().get("table", "")
                if not table_data.strip(): break  
                all_table_data.append(table_data)
                page += 1
            else:
                logger.debug(f"{translations['code_error']} {response.status_code}")
                break  
        except json.JSONDecodeError:
            logger.debug(translations["json_error"])
            break
        except requests.RequestException as e:
            logger.debug(translations["requests_error"].format(e=e))
            break
    return all_table_data

def search_models(name):
    gr_info(translations["start"].format(start=translations["search"]))
    tables = fetch_models_data(name)

    if len(tables) == 0:
        gr_info(translations["not_found"].format(name=name))
        return [None]*2
    else:
        model_options.clear()
        
        from bs4 import BeautifulSoup

        for table in tables:
            for row in BeautifulSoup(table, "html.parser").select("tr"):
                name_tag, url_tag = row.find("a", {"class": "fs-5"}), row.find("a", {"class": "btn btn-sm fw-bold btn-light ms-0 p-1 ps-2 pe-2"})
                url = url_tag["href"].replace("https://easyaivoice.com/run?url=", "")
                if "huggingface" in url:
                    if name_tag and url_tag: model_options[name_tag.text.replace(".onnx", "").replace(".pth", "").replace(".index", "").replace(".zip", "").replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip()] = url

        gr_info(translations["found"].format(results=len(model_options)))
        return [{"value": "", "choices": model_options, "interactive": True, "visible": True, "__type__": "update"}, {"value": translations["downloads"], "visible": True, "__type__": "update"}]

def move_files_from_directory(src_dir, dest_weights, dest_logs, model_name):
    for root, _, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".index"):
                model_log_dir = os.path.join(dest_logs, model_name)
                os.makedirs(model_log_dir, exist_ok=True)

                filepath = os.path.join(model_log_dir, file.replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip())
                if os.path.exists(filepath): os.remove(filepath)

                shutil.move(file_path, filepath)
            elif file.endswith(".pth") and not file.startswith("D_") and not file.startswith("G_"):
                pth_path = os.path.join(dest_weights, model_name + ".pth")
                if os.path.exists(pth_path): os.remove(pth_path)

                shutil.move(file_path, pth_path)
            elif file.endswith(".onnx") and not file.startswith("D_") and not file.startswith("G_"):
                pth_path = os.path.join(dest_weights, model_name + ".onnx")
                if os.path.exists(pth_path): os.remove(pth_path)

                shutil.move(file_path, pth_path)

def download_url(url):
    import yt_dlp

    if not url: return gr_warning(translations["provide_url"])
    if not os.path.exists("audios"): os.makedirs("audios", exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ydl_opts = {"format": "bestaudio/best", "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}], "quiet": True, "no_warnings": True, "noplaylist": True, "verbose": False}

        gr_info(translations["start"].format(start=translations["download_music"]))

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            audio_output = os.path.join("audios", re.sub(r'\s+', '-', re.sub(r'[^\w\s\u4e00-\u9fff\uac00-\ud7af\u0400-\u04FF\u1100-\u11FF]', '', ydl.extract_info(url, download=False).get('title', 'video')).strip()))
            if os.path.exists(audio_output): shutil.rmtree(audio_output, ignore_errors=True)

            ydl_opts['outtmpl'] = audio_output
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
            audio_output = audio_output + ".wav"
            if os.path.exists(audio_output): os.remove(audio_output)
            
            ydl.download([url])

        gr_info(translations["success"])
        return [audio_output, audio_output, translations["success"]]

def download_model(url=None, model=None):
    if not url: return gr_warning(translations["provide_url"])
    if not model: return gr_warning(translations["provide_name_is_save"])

    model = model.replace(".onnx", "").replace(".pth", "").replace(".index", "").replace(".zip", "").replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip()
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()

    download_dir = os.path.join("download_model")
    weights_dir = os.path.join("assets", "weights")
    logs_dir = os.path.join("assets", "logs")

    if not os.path.exists(download_dir): os.makedirs(download_dir, exist_ok=True)
    if not os.path.exists(weights_dir): os.makedirs(weights_dir, exist_ok=True)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir, exist_ok=True)
    
    try:
        gr_info(translations["start"].format(start=translations["download"]))

        if url.endswith(".pth"): huggingface.HF_download_file(url, os.path.join(weights_dir, f"{model}.pth"))
        elif url.endswith(".onnx"): huggingface.HF_download_file(url, os.path.join(weights_dir, f"{model}.onnx"))
        elif url.endswith(".index"):
            model_log_dir = os.path.join(logs_dir, model)
            os.makedirs(model_log_dir, exist_ok=True)

            huggingface.HF_download_file(url, os.path.join(model_log_dir, f"{model}.index"))
        elif url.endswith(".zip"):
            output_path = huggingface.HF_download_file(url, os.path.join(download_dir, model + ".zip"))
            shutil.unpack_archive(output_path, download_dir)

            move_files_from_directory(download_dir, weights_dir, logs_dir, model)
        else:
            if "drive.google.com" in url or "drive.usercontent.google.com" in url:
                file_id = None

                from main.tools import gdown

                if "/file/d/" in url: file_id = url.split("/d/")[1].split("/")[0]
                elif "open?id=" in url: file_id = url.split("open?id=")[1].split("/")[0]
                elif "/download?id=" in url: file_id = url.split("/download?id=")[1].split("&")[0]
                
                if file_id:
                    file = gdown.gdown_download(id=file_id, output=download_dir)
                    if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)

                    move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif "mega.nz" in url:
                from main.tools import meganz
                
                meganz.mega_download_url(url, download_dir)

                file_download = next((f for f in os.listdir(download_dir)), None)
                if file_download.endswith(".zip"): shutil.unpack_archive(os.path.join(download_dir, file_download), download_dir)

                move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif "mediafire.com" in url:
                from main.tools import mediafire

                file = mediafire.Mediafire_Download(url, download_dir)
                if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)

                move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif "pixeldrain.com" in url:
                from main.tools import pixeldrain

                file = pixeldrain.pixeldrain(url, download_dir)
                if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)

                move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            else:
                gr_warning(translations["not_support_url"])
                return translations["not_support_url"]
        
        gr_info(translations["success"])
        return translations["success"]
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))
        logger.debug(e)
        return translations["error_occurred"].format(e=e)
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)

def save_drop_model(dropbox):
    weight_folder = os.path.join("assets", "weights")
    logs_folder = os.path.join("assets", "logs")
    save_model_temp = os.path.join("save_model_temp")

    if not os.path.exists(weight_folder): os.makedirs(weight_folder, exist_ok=True)
    if not os.path.exists(logs_folder): os.makedirs(logs_folder, exist_ok=True)
    if not os.path.exists(save_model_temp): os.makedirs(save_model_temp, exist_ok=True)

    shutil.move(dropbox, save_model_temp)

    try:
        file_name = os.path.basename(dropbox)

        if file_name.endswith(".pth") and file_name.endswith(".onnx") and file_name.endswith(".index"): gr_warning(translations["not_model"])
        else:    
            if file_name.endswith(".zip"):
                shutil.unpack_archive(os.path.join(save_model_temp, file_name), save_model_temp)
                move_files_from_directory(save_model_temp, weight_folder, logs_folder, file_name.replace(".zip", ""))
            elif file_name.endswith((".pth", ".onnx")): 
                output_file = os.path.join(weight_folder, file_name)
                if os.path.exists(output_file): os.remove(output_file)
                
                shutil.move(os.path.join(save_model_temp, file_name), output_file)
            elif file_name.endswith(".index"):
                def extract_name_model(filename):
                    match = re.search(r"([A-Za-z]+)(?=_v|\.|$)", filename)
                    return match.group(1) if match else None
                
                model_logs = os.path.join(logs_folder, extract_name_model(file_name))
                if not os.path.exists(model_logs): os.makedirs(model_logs, exist_ok=True)
                shutil.move(os.path.join(save_model_temp, file_name), model_logs)
            else: 
                gr_warning(translations["unable_analyze_model"])
                return None
        
        gr_info(translations["upload_success"].format(name=translations["model"]))
        return None
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))
        logger.debug(e)
        return None
    finally:
        shutil.rmtree(save_model_temp, ignore_errors=True)

def download_pretrained_model(choices, model, sample_rate):
    pretraineds_custom_path = os.path.join("assets", "models", "pretrained_custom")
    if choices == translations["list_model"]:
        paths = fetch_pretrained_data()[model][sample_rate]

        if not os.path.exists(pretraineds_custom_path): os.makedirs(pretraineds_custom_path, exist_ok=True)
        url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cergenvarq_phfgbz/", "rot13") + paths

        gr_info(translations["download_pretrain"])
        file = huggingface.HF_download_file(url.replace("/blob/", "/resolve/").replace("?download=true", "").strip(), os.path.join(pretraineds_custom_path, paths))

        if file.endswith(".zip"): 
            shutil.unpack_archive(file, pretraineds_custom_path)
            os.remove(file)

        gr_info(translations["success"])
        return translations["success"]
    elif choices == translations["download_url"]:
        if not model: return gr_warning(translations["provide_pretrain"].format(dg="D"))
        if not sample_rate: return gr_warning(translations["provide_pretrain"].format(dg="G"))

        gr_info(translations["download_pretrain"])

        huggingface.HF_download_file(model.replace("/blob/", "/resolve/").replace("?download=true", "").strip(), pretraineds_custom_path)
        huggingface.HF_download_file(sample_rate.replace("/blob/", "/resolve/").replace("?download=true", "").strip(), pretraineds_custom_path)

        gr_info(translations["success"])
        return translations["success"]

def fushion_model_pth(name, pth_1, pth_2, ratio):
    if not name.endswith(".pth"): name = name + ".pth"

    if not pth_1 or not os.path.exists(pth_1) or not pth_1.endswith(".pth"):
        gr_warning(translations["provide_file"].format(filename=translations["model"] + " 1"))
        return [translations["provide_file"].format(filename=translations["model"] + " 1"), None]
    
    if not pth_2 or not os.path.exists(pth_2) or not pth_2.endswith(".pth"):
        gr_warning(translations["provide_file"].format(filename=translations["model"] + " 2"))
        return [translations["provide_file"].format(filename=translations["model"] + " 2"), None]
    
    from collections import OrderedDict

    def extract(ckpt):
        a = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}

        for key in a.keys():
            if "enc_q" in key: continue

            opt["weight"][key] = a[key]

        return opt
    
    try:
        ckpt1 = torch.load(pth_1, map_location="cpu")
        ckpt2 = torch.load(pth_2, map_location="cpu")

        if ckpt1["sr"] != ckpt2["sr"]: 
            gr_warning(translations["sr_not_same"])
            return [translations["sr_not_same"], None]

        cfg = ckpt1["config"]
        cfg_f0 = ckpt1["f0"]
        cfg_version = ckpt1["version"]
        cfg_sr = ckpt1["sr"]

        vocoder = ckpt1.get("vocoder", "Default")

        ckpt1 = extract(ckpt1) if "model" in ckpt1 else ckpt1["weight"]
        ckpt2 = extract(ckpt2) if "model" in ckpt2 else ckpt2["weight"]

        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())): 
            gr_warning(translations["architectures_not_same"])
            return [translations["architectures_not_same"], None]
         
        gr_info(translations["start"].format(start=translations["fushion_model"]))

        opt = OrderedDict()
        opt["weight"] = {}

        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (ratio * (ckpt1[key][:min_shape0].float()) + (1 - ratio) * (ckpt2[key][:min_shape0].float())).half()
            else: opt["weight"][key] = (ratio * (ckpt1[key].float()) + (1 - ratio) * (ckpt2[key].float())).half()

        opt["config"] = cfg
        opt["sr"] = cfg_sr
        opt["f0"] = cfg_f0
        opt["version"] = cfg_version
        opt["infos"] = translations["model_fushion_info"].format(name=name, pth_1=pth_1, pth_2=pth_2, ratio=ratio)
        opt["vocoder"] = vocoder

        output_model = os.path.join("assets", "weights")
        if not os.path.exists(output_model): os.makedirs(output_model, exist_ok=True)

        torch.save(opt, os.path.join(output_model, name))

        gr_info(translations["success"])
        return [translations["success"], os.path.join(output_model, name)]
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))
        logger.debug(e)
        return [e, None]

def fushion_model(name, path_1, path_2, ratio):
    if not name:
        gr_warning(translations["provide_name_is_save"]) 
        return [translations["provide_name_is_save"], None]

    if path_1.endswith(".pth") and path_2.endswith(".pth"): return fushion_model_pth(name.replace(".onnx", ".pth"), path_1, path_2, ratio)
    else:
        gr_warning(translations["format_not_valid"])
        return [None, None]
    
def onnx_export(model_path):
    from main.library.algorithm.onnx_export import onnx_exporter
    
    if not model_path.endswith(".pth"): model_path + ".pth"
    if not model_path or not os.path.exists(model_path) or not model_path.endswith(".pth"):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return [None, translations["provide_file"].format(filename=translations["model"])]
    
    try:
        gr_info(translations["start_onnx_export"])
        output = onnx_exporter(model_path, model_path.replace(".pth", ".onnx"), is_half=config.is_half, device=config.device)

        gr_info(translations["success"])
        return [output, translations["success"]]
    except Exception as e:
        return [None, e]
    
def model_info(path):
    if not path or not os.path.exists(path) or os.path.isdir(path) or not path.endswith((".pth", ".onnx")): return gr_warning(translations["provide_file"].format(filename=translations["model"]))
    
    def prettify_date(date_str):
        if date_str == translations["not_found_create_time"]: return None

        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            logger.debug(e)
            return translations["format_not_valid"]
    
    if path.endswith(".pth"): model_data = torch.load(path, map_location=torch.device("cpu"))
    else:
        import onnx
        
        model = onnx.load(path)
        model_data = None

        for prop in model.metadata_props:
            if prop.key == "model_info":
                model_data = json.loads(prop.value)
                break

    gr_info(translations["read_info"])

    epochs = model_data.get("epoch", None)
    if epochs is None: 
        epochs = model_data.get("info", None)
        try:
            epoch = epochs.replace("epoch", "").replace("e", "").isdigit()
            if epoch and epochs is None: epochs = translations["not_found"].format(name=translations["epoch"])
        except: 
            pass

    steps = model_data.get("step", translations["not_found"].format(name=translations["step"]))
    sr = model_data.get("sr", translations["not_found"].format(name=translations["sr"]))
    f0 = model_data.get("f0", translations["not_found"].format(name=translations["f0"]))
    version = model_data.get("version", translations["not_found"].format(name=translations["version"]))
    creation_date = model_data.get("creation_date", translations["not_found_create_time"])
    model_hash = model_data.get("model_hash", translations["not_found"].format(name="model_hash"))
    pitch_guidance = translations["trained_f0"] if f0 else translations["not_f0"]
    creation_date_str = prettify_date(creation_date) if creation_date else translations["not_found_create_time"]
    model_name = model_data.get("model_name", translations["unregistered"])
    model_author = model_data.get("author", translations["not_author"])
    vocoder = model_data.get("vocoder", "Default")

    gr_info(translations["success"])
    return translations["model_info"].format(model_name=model_name, model_author=model_author, epochs=epochs, steps=steps, version=version, sr=sr, pitch_guidance=pitch_guidance, model_hash=model_hash, creation_date_str=creation_date_str, vocoder=vocoder)

def audio_effects(input_path, output_path, resample, resample_sr, chorus_depth, chorus_rate, chorus_mix, chorus_delay, chorus_feedback, distortion_drive, reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level, reverb_width, reverb_freeze_mode, pitch_shift, delay_seconds, delay_feedback, delay_mix, compressor_threshold, compressor_ratio, compressor_attack_ms, compressor_release_ms, limiter_threshold, limiter_release, gain_db, bitcrush_bit_depth, clipping_threshold, phaser_rate_hz, phaser_depth, phaser_centre_frequency_hz, phaser_feedback, phaser_mix, bass_boost_db, bass_boost_frequency, treble_boost_db, treble_boost_frequency, fade_in_duration, fade_out_duration, export_format, chorus, distortion, reverb, delay, compressor, limiter, gain, bitcrush, clipping, phaser, treble_bass_boost, fade_in_out, audio_combination, audio_combination_input, main_vol, combine_vol):
    if not input_path or not os.path.exists(input_path) or os.path.isdir(input_path): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_path:
        gr_warning(translations["output_not_valid"])
        return None
    
    if os.path.isdir(output_path): output_path = os.path.join(output_path, f"audio_effects.{export_format}")
    output_dir = os.path.dirname(output_path) or output_path

    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path): os.remove(output_path)
    
    gr_info(translations["start"].format(start=translations["apply_effect"]))
    subprocess.run([python, "main/inference/audio_effects.py", "--input_path", input_path, "--output_path", output_path, "--resample", str(resample), "--resample_sr", str(resample_sr), "--chorus_depth", str(chorus_depth), "--chorus_rate", str(chorus_rate), "--chorus_mix", str(chorus_mix), "--chorus_delay", str(chorus_delay), "--chorus_feedback", str(chorus_feedback), "--drive_db", str(distortion_drive), "--reverb_room_size", str(reverb_room_size), "--reverb_damping", str(reverb_damping), "--reverb_wet_level", str(reverb_wet_level), "--reverb_dry_level", str(reverb_dry_level), "--reverb_width", str(reverb_width), "--reverb_freeze_mode", str(reverb_freeze_mode), "--pitch_shift", str(pitch_shift), "--delay_seconds", str(delay_seconds), "--delay_feedback", str(delay_feedback), "--delay_mix", str(delay_mix), "--compressor_threshold", str(compressor_threshold), "--compressor_ratio", str(compressor_ratio), "--compressor_attack_ms", str(compressor_attack_ms), "--compressor_release_ms", str(compressor_release_ms), "--limiter_threshold", str(limiter_threshold), "--limiter_release", str(limiter_release), "--gain_db", str(gain_db), "--bitcrush_bit_depth", str(bitcrush_bit_depth), "--clipping_threshold", str(clipping_threshold), "--phaser_rate_hz", str(phaser_rate_hz), "--phaser_depth", str(phaser_depth), "--phaser_centre_frequency_hz", str(phaser_centre_frequency_hz), "--phaser_feedback", str(phaser_feedback), "--phaser_mix", str(phaser_mix), "--bass_boost_db", str(bass_boost_db), "--bass_boost_frequency", str(bass_boost_frequency), "--treble_boost_db", str(treble_boost_db), "--treble_boost_frequency", str(treble_boost_frequency), "--fade_in_duration", str(fade_in_duration), "--fade_out_duration", str(fade_out_duration), "--export_format", export_format, "--chorus", str(chorus), "--distortion", str(distortion), "--reverb", str(reverb), "--pitchshift", str(pitch_shift != 0), "--delay", str(delay), "--compressor", str(compressor), "--limiter", str(limiter), "--gain", str(gain), "--bitcrush", str(bitcrush), "--clipping", str(clipping), "--phaser", str(phaser), "--treble_bass_boost", str(treble_bass_boost), "--fade_in_out", str(fade_in_out), "--audio_combination", str(audio_combination), "--audio_combination_input", audio_combination_input, "--main_volume", str(main_vol), "--combination_volume", str(combine_vol)])

    gr_info(translations["success"])
    return output_path.replace("wav", export_format)

def synthesize_tts(prompt, voice, speed, output, pitch, google):
    if not google: 
        from edge_tts import Communicate

        asyncio.run(Communicate(text=prompt, voice=voice, rate=f"+{speed}%" if speed >= 0 else f"{speed}%", pitch=f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz").save(output))
    else: 
        response = requests.get(codecs.decode("uggcf://genafyngr.tbbtyr.pbz/genafyngr_ggf", "rot13"), params={"ie": "UTF-8", "q": prompt, "tl": voice, "ttsspeed": speed, "client": "tw-ob"}, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"})

        if response.status_code == 200:
            with open(output, "wb") as f:
                f.write(response.content)

            if pitch != 0 or speed != 0:
                y, sr = librosa.load(output, sr=None)

                if pitch != 0: y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
                if speed != 0: y = librosa.effects.time_stretch(y, rate=speed)

                sf.write(file=output, data=y, samplerate=sr, format=os.path.splitext(os.path.basename(output))[-1].lower().replace('.', ''))
        else: gr_error(f"{response.status_code}, {response.text}")

def time_stretch(y, sr, target_duration):
    rate = (len(y) / sr) / target_duration
    if rate != 1.0: y = librosa.effects.time_stretch(y=y.astype(np.float32), rate=rate)

    n_target = int(round(target_duration * sr))
    return np.pad(y, (0, n_target - len(y))) if len(y) < n_target else y[:n_target]

def pysrttime_to_seconds(t):
    return (t.hours * 60 + t.minutes) * 60 + t.seconds + t.milliseconds / 1000

def srt_tts(srt_file, out_file, voice, rate = 0, sr = 24000, google = False):
    import pysrt
    import tempfile

    subs = pysrt.open(srt_file)
    if not subs: raise ValueError(translations["srt"])

    final_audio = np.zeros(int(round(pysrttime_to_seconds(subs[-1].end) * sr)), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tempdir:
        for idx, seg in enumerate(subs):
            wav_path = os.path.join(tempdir, f"seg_{idx}.wav")
            synthesize_tts(" ".join(seg.text.splitlines()), voice, 0, wav_path, rate, google)

            audio, file_sr = sf.read(wav_path, dtype=np.float32)
            if file_sr != sr: audio = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * sr / file_sr)), np.arange(len(audio)), audio)
            adjusted = time_stretch(audio, sr, pysrttime_to_seconds(seg.duration))

            start_sample = int(round(pysrttime_to_seconds(seg.start) * sr))
            end_sample = start_sample + adjusted.shape[0]

            if end_sample > final_audio.shape[0]:
                adjusted = adjusted[: final_audio.shape[0] - start_sample]
                end_sample = final_audio.shape[0]

            final_audio[start_sample:end_sample] += adjusted

    sf.write(out_file, final_audio, sr)

def TTS(prompt, voice, speed, output, pitch, google, srt_input):
    if not srt_input: srt_input = ""

    if not prompt and not srt_input.endswith(".srt"):
        gr_warning(translations["enter_the_text"])
        return None
    
    if not voice:
        gr_warning(translations["choose_voice"])
        return None
    
    if not output: 
        gr_warning(translations["output_not_valid"])
        return None
    
    if os.path.isdir(output): output = os.path.join(output, f"tts.wav")
    gr_info(translations["convert"].format(name=translations["text"]))

    output_dir = os.path.dirname(output) or output
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    if srt_input.endswith(".srt"): srt_tts(srt_input, output, voice, 0, 24000, google)
    else: synthesize_tts(prompt, voice, speed, output, pitch, google)

    gr_info(translations["success"])
    return output

def separator_music(input, output_audio, format, shifts, segments_size, overlap, clean_audio, clean_strength, denoise, separator_model, kara_model, backing, reverb, backing_reverb, hop_length, batch_size, sample_rate):
    output = os.path.dirname(output_audio) or output_audio

    if not input or not os.path.exists(input) or os.path.isdir(input): 
        gr_warning(translations["input_not_valid"])
        return [None]*4
    
    if not os.path.exists(output): 
        gr_warning(translations["output_not_valid"])
        return [None]*4

    if not os.path.exists(output): os.makedirs(output)
    gr_info(translations["start"].format(start=translations["separator_music"]))

    subprocess.run([python, "main/inference/separator_music.py", "--input_path", input, "--output_path", output, "--format", format, "--shifts", str(shifts), "--segments_size", str(segments_size), "--overlap", str(overlap), "--mdx_hop_length", str(hop_length), "--mdx_batch_size", str(batch_size), "--clean_audio", str(clean_audio), "--clean_strength", str(clean_strength), "--kara_model", kara_model, "--backing", str(backing), "--mdx_denoise", str(denoise), "--reverb", str(reverb), "--backing_reverb", str(backing_reverb), "--model_name", separator_model, "--sample_rate", str(sample_rate)])
    gr_info(translations["success"])

    filename, _ = os.path.splitext(os.path.basename(input))
    output = os.path.join(output, filename)

    return [os.path.join(output, f"Original_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Original_Vocals.{format}"), os.path.join(output, f"Instruments.{format}"), (os.path.join(output, f"Main_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Main_Vocals.{format}") if backing else None), (os.path.join(output, f"Backing_Vocals_No_Reverb.{format}") if backing_reverb else os.path.join(output, f"Backing_Vocals.{format}") if backing else None)] if os.path.isfile(input) else [None]*4

def convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, clean_audio, clean_strength, export_format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file):    
    subprocess.run([python, "main/inference/convert.py", "--pitch", str(pitch), "--filter_radius", str(filter_radius), "--index_rate", str(index_rate), "--volume_envelope", str(volume_envelope), "--protect", str(protect), "--hop_length", str(hop_length), "--f0_method", f0_method, "--input_path", input_path, "--output_path", output_path, "--pth_path", pth_path, "--index_path", index_path if index_path else "", "--f0_autotune", str(f0_autotune), "--clean_audio", str(clean_audio), "--clean_strength", str(clean_strength), "--export_format", export_format, "--embedder_model", embedder_model, "--resample_sr", str(resample_sr), "--split_audio", str(split_audio), "--f0_autotune_strength", str(f0_autotune_strength), "--checkpointing", str(checkpointing), "--f0_onnx", str(onnx_f0_mode), "--embedders_mode", embedders_mode, "--formant_shifting", str(formant_shifting), "--formant_qfrency", str(formant_qfrency), "--formant_timbre", str(formant_timbre), "--f0_file", f0_file])

def convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, input_audio_name, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode):
    model_path = os.path.join("assets", "weights", model)

    return_none = [None]*6
    return_none[5] = {"visible": True, "__type__": "update"}

    if not use_audio:
        if merge_instrument or not_merge_backing or convert_backing or use_original:
            gr_warning(translations["turn_on_use_audio"])
            return return_none

    if use_original:
        if convert_backing:
            gr_warning(translations["turn_off_convert_backup"])
            return return_none
        elif not_merge_backing:
            gr_warning(translations["turn_off_merge_backup"])
            return return_none

    if not model or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith((".pth", ".onnx")):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return return_none

    f0method, embedder_model = (method if method != "hybrid" else hybrid_method), (embedders if embedders != "custom" else custom_embedders)

    if use_audio:
        output_audio = os.path.join("audios", input_audio_name)

        from main.library.utils import pydub_load
        
        def get_audio_file(label):
            matching_files = [f for f in os.listdir(output_audio) if label in f]

            if not matching_files: return translations["notfound"]   
            return os.path.join(output_audio, matching_files[0])

        output_path = os.path.join(output_audio, f"Convert_Vocals.{format}")
        output_backing = os.path.join(output_audio, f"Convert_Backing.{format}")
        output_merge_backup = os.path.join(output_audio, f"Vocals+Backing.{format}")
        output_merge_instrument = os.path.join(output_audio, f"Vocals+Instruments.{format}")

        if os.path.exists(output_audio): os.makedirs(output_audio, exist_ok=True)
        if os.path.exists(output_path): os.remove(output_path)

        if use_original:
            original_vocal = get_audio_file('Original_Vocals_No_Reverb.')

            if original_vocal == translations["notfound"]: original_vocal = get_audio_file('Original_Vocals.')

            if original_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_original_vocal"])
                return return_none
            
            input_path = original_vocal
        else:
            main_vocal = get_audio_file('Main_Vocals_No_Reverb.')
            backing_vocal = get_audio_file('Backing_Vocals_No_Reverb.')

            if main_vocal == translations["notfound"]: main_vocal = get_audio_file('Main_Vocals.')
            if not not_merge_backing and backing_vocal == translations["notfound"]: backing_vocal = get_audio_file('Backing_Vocals.')

            if main_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_main_vocal"])
                return return_none
            
            if not not_merge_backing and backing_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_backing_vocal"])
                return return_none
            
            input_path = main_vocal
            backing_path = backing_vocal

        gr_info(translations["convert_vocal"])

        convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input_path, output_path, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file)

        gr_info(translations["convert_success"])

        if convert_backing:
            if os.path.exists(output_backing): os.remove(output_backing)

            gr_info(translations["convert_backup"])

            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, backing_path, output_backing, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file)

            gr_info(translations["convert_backup_success"])

        try:
            if not not_merge_backing and not use_original:
                backing_source = output_backing if convert_backing else backing_vocal

                if os.path.exists(output_merge_backup): os.remove(output_merge_backup)

                gr_info(translations["merge_backup"])

                pydub_load(output_path, volume=-4).overlay(pydub_load(backing_source, volume=-6)).export(output_merge_backup, format=format)

                gr_info(translations["merge_success"])

            if merge_instrument:    
                vocals = output_merge_backup if not not_merge_backing and not use_original else output_path

                if os.path.exists(output_merge_instrument): os.remove(output_merge_instrument)

                gr_info(translations["merge_instruments_process"])

                instruments = get_audio_file('Instruments.')
                
                if instruments == translations["notfound"]: 
                    gr_warning(translations["not_found_instruments"])
                    output_merge_instrument = None
                else: pydub_load(instruments, volume=-7).overlay(pydub_load(vocals, volume=-4 if use_original else None)).export(output_merge_instrument, format=format)
                
                gr_info(translations["merge_success"])
        except:
            return return_none

        return [(None if use_original else output_path), output_backing, (None if not_merge_backing and use_original else output_merge_backup), (output_path if use_original else None), (output_merge_instrument if merge_instrument else None), {"visible": True, "__type__": "update"}]
    else:
        if not input or not os.path.exists(input) or os.path.isdir(input): 
            gr_warning(translations["input_not_valid"])
            return return_none
        
        if not output:
            gr_warning(translations["output_not_valid"])
            return return_none
        
        output = output.replace("wav", format)

        if os.path.isdir(input):
            gr_info(translations["is_folder"])

            if not [f for f in os.listdir(input) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]:
                gr_warning(translations["not_found_in_folder"])
                return return_none
            
            gr_info(translations["batch_convert"])

            output_dir = os.path.dirname(output) or output
            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output_dir, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file)

            gr_info(translations["batch_convert_success"])

            return return_none
        else:
            output_dir = os.path.dirname(output) or output

            if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(output): os.remove(output)

            gr_info(translations["convert_vocal"])

            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file)

            gr_info(translations["convert_success"])

            return_none[0] = output
            return return_none

def convert_selection(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode):
    if use_audio:
        gr_info(translations["search_separate"])

        choice = [f for f in os.listdir("audios") if os.path.isdir(os.path.join("audios", f))]

        gr_info(translations["found_choice"].format(choice=len(choice)))

        if len(choice) == 0: 
            gr_warning(translations["separator==0"])

            return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, None, None, None, None, None, {"visible": True, "__type__": "update"}]
        elif len(choice) == 1:
            convert_output = convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, None, None, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, choice[0], checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode)

            return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, convert_output[0], convert_output[1], convert_output[2], convert_output[3], convert_output[4], {"visible": True, "__type__": "update"}]
        else: return [{"choices": choice, "value": "", "interactive": True, "visible": True, "__type__": "update"}, None, None, None, None, None, {"visible": False, "__type__": "update"}]
    else:
        main_convert = convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, None, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode)

        return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, main_convert[0], None, None, None, None, {"visible": True, "__type__": "update"}]
    
def convert_with_whisper(num_spk, model_size, cleaner, clean_strength, autotune, f0_autotune_strength, checkpointing, model_1, model_2, model_index_1, model_index_2, pitch_1, pitch_2, index_strength_1, index_strength_2, export_format, input_audio, output_audio, onnx_f0_mode, method, hybrid_method, hop_length, embed_mode, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, formant_shifting, formant_qfrency_1, formant_timbre_1, formant_qfrency_2, formant_timbre_2):
    from pydub import AudioSegment
    from sklearn.cluster import AgglomerativeClustering
    
    from main.library.speaker_diarization.audio import Audio
    from main.library.speaker_diarization.segment import Segment
    from main.library.speaker_diarization.whisper import load_model
    from main.library.utils import check_spk_diarization, pydub_load
    from main.library.speaker_diarization.embedding import SpeechBrainPretrainedSpeakerEmbedding
    
    check_spk_diarization(model_size)
    model_pth_1, model_pth_2 = os.path.join("assets", "weights", model_1), os.path.join("assets", "weights", model_2)

    if (not model_1 or not os.path.exists(model_pth_1) or os.path.isdir(model_pth_1) or not model_pth_1.endswith((".pth", ".onnx"))) and (not model_2 or not os.path.exists(model_pth_2) or os.path.isdir(model_pth_2) or not model_pth_2.endswith((".pth", ".onnx"))):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return None
    
    if not model_1: model_pth_1 = model_pth_2
    if not model_2: model_pth_2 = model_pth_1

    if not input_audio or not os.path.exists(input_audio) or os.path.isdir(input_audio): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_audio:
        gr_warning(translations["output_not_valid"])
        return None
    
    if os.path.exists(output_audio): os.remove(output_audio)
    gr_info(translations["start_whisper"])
    
    try:
        audio = Audio()

        embedding_model = SpeechBrainPretrainedSpeakerEmbedding(device=config.device)
        segments = load_model(model_size, device=config.device).transcribe(input_audio, fp16=configs.get("fp16", False), word_timestamps=True)["segments"]

        y, sr = librosa.load(input_audio, sr=None)  
        duration = len(y) / sr
            
        def segment_embedding(segment):
            waveform, _ = audio.crop(input_audio, Segment(segment["start"], min(duration, segment["end"])))
            return embedding_model(waveform.mean(dim=0, keepdim=True)[None] if waveform.shape[0] == 2 else waveform[None])  
        
        def time(secs):
            return datetime.timedelta(seconds=round(secs))
        
        def merge_audio(files_list, time_stamps, original_file_path, output_path, format):
            def extract_number(filename):
                match = re.search(r'_(\d+)', filename)
                return int(match.group(1)) if match else 0

            total_duration = len(pydub_load(original_file_path))
            combined = AudioSegment.empty() 
            current_position = 0 

            for file, (start_i, end_i) in zip(sorted(files_list, key=extract_number), time_stamps):
                if start_i > current_position: combined += AudioSegment.silent(duration=start_i - current_position)  
                
                combined += pydub_load(file)  
                current_position = end_i

            if current_position < total_duration: combined += AudioSegment.silent(duration=total_duration - current_position)
            combined.export(output_path, format=format)

            return output_path

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)

        labels = AgglomerativeClustering(num_spk).fit(np.nan_to_num(embeddings)).labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        merged_segments, current_text = [], []
        current_speaker, current_start = None, None

        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            start_time = segment["start"]
            text = segment["text"][1:]  

            if speaker == current_speaker:
                current_text.append(text)
                end_time = segment["end"]
            else:
                if current_speaker is not None: merged_segments.append({"speaker": current_speaker, "start": current_start, "end": end_time, "text": " ".join(current_text)})
                
                current_speaker = speaker
                current_start = start_time
                current_text = [text]
                end_time = segment["end"]

        if current_speaker is not None: merged_segments.append({"speaker": current_speaker, "start": current_start, "end": end_time, "text": " ".join(current_text)})

        gr_info(translations["whisper_done"])

        x = ""
        for segment in merged_segments:
            x += f"\n{segment['speaker']} {str(time(segment['start']))} - {str(time(segment['end']))}\n"
            x += segment["text"] + "\n"

        logger.info(x)

        gr_info(translations["process_audio"])

        audio = pydub_load(input_audio)
        output_folder = "audios_temp"

        if os.path.exists(output_folder): shutil.rmtree(output_folder, ignore_errors=True)
        for f in [output_folder, os.path.join(output_folder, "1"), os.path.join(output_folder, "2")]:
            os.makedirs(f, exist_ok=True)

        time_stamps, processed_segments = [], []
        for i, segment in enumerate(merged_segments):
            start_ms = int(segment["start"] * 1000) 
            end_ms = int(segment["end"] * 1000)

            index = i + 1

            segment_filename = os.path.join(output_folder, "1" if i % 2 == 1 else "2", f"segment_{index}.wav")
            audio[start_ms:end_ms].export(segment_filename, format="wav")

            processed_segments.append(os.path.join(output_folder, "1" if i % 2 == 1 else "2", f"segment_{index}_output.wav"))
            time_stamps.append((start_ms, end_ms))

        f0method, embedder_model = (method if method != "hybrid" else hybrid_method), (embedders if embedders != "custom" else custom_embedders)

        gr_info(translations["process_done_start_convert"])

        convert(pitch_1, filter_radius, index_strength_1, volume_envelope, protect, hop_length, f0method, os.path.join(output_folder, "1"), output_folder, model_pth_1, model_index_1, autotune, cleaner, clean_strength, "wav", embedder_model, resample_sr, False, f0_autotune_strength, checkpointing, onnx_f0_mode, embed_mode, formant_shifting, formant_qfrency_1, formant_timbre_1, "")
        convert(pitch_2, filter_radius, index_strength_2, volume_envelope, protect, hop_length, f0method, os.path.join(output_folder, "2"), output_folder, model_pth_2, model_index_2, autotune, cleaner, clean_strength, "wav", embedder_model, resample_sr, False, f0_autotune_strength, checkpointing, onnx_f0_mode, embed_mode, formant_shifting, formant_qfrency_2, formant_timbre_2, "")

        gr_info(translations["convert_success"])
        return merge_audio(processed_segments, time_stamps, input_audio, output_audio.replace("wav", export_format), export_format)
    except Exception as e:
        gr_error(translations["error_occurred"].format(e=e))
        import traceback
        logger.debug(traceback.format_exc())
        return None
    finally:
        if os.path.exists("audios_temp"): shutil.rmtree("audios_temp", ignore_errors=True)

def convert_tts(clean, autotune, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode):
    model_path = os.path.join("assets", "weights", model)

    if not model_path or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith((".pth", ".onnx")):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return None

    if not input or not os.path.exists(input): 
        gr_warning(translations["input_not_valid"])
        return None
    
    if os.path.isdir(input): 
        input_audio = [f for f in os.listdir(input) if "tts" in f and f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]
        
        if not input_audio:
            gr_warning(translations["not_found_in_folder"])
            return None
        
        input = os.path.join(input, input_audio[0])
    
    if not output:
        gr_warning(translations["output_not_valid"])
        return None
    
    output = output.replace("wav", format)
    if os.path.isdir(output): output = os.path.join(output, f"tts.{format}")

    output_dir = os.path.dirname(output)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output): os.remove(output)

    f0method = method if method != "hybrid" else hybrid_method
    embedder_model = embedders if embedders != "custom" else custom_embedders

    gr_info(translations["convert_vocal"])

    convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file)

    gr_info(translations["convert_success"])
    return output

def log_read(log_file, done):
    f = open(log_file, "w", encoding="utf-8")
    f.close()

    while 1:
        with open(log_file, "r", encoding="utf-8") as f:
            yield "".join(line for line in f.readlines() if "DEBUG" not in line and line.strip() != "")

        sleep(1)
        if done[0]: break

    with open(log_file, "r", encoding="utf-8") as f:
        log = "".join(line for line in f.readlines() if "DEBUG" not in line and line.strip() != "")

    yield log

def create_dataset(input_audio, output_dataset, clean_dataset, clean_strength, separator_reverb, kim_vocals_version, overlap, segments_size, denoise_mdx, skip, skip_start, skip_end, hop_length, batch_size, sample_rate):
    version = 1 if kim_vocals_version == "Version-1" else 2

    gr_info(translations["start"].format(start=translations["create"]))

    p = subprocess.Popen(f'{python} main/inference/create_dataset.py --input_audio "{input_audio}" --output_dataset "{output_dataset}" --clean_dataset {clean_dataset} --clean_strength {clean_strength} --separator_reverb {separator_reverb} --kim_vocal_version {version} --overlap {overlap} --segments_size {segments_size} --mdx_hop_length {hop_length} --mdx_batch_size {batch_size} --denoise_mdx {denoise_mdx} --skip {skip} --skip_start_audios "{skip_start}" --skip_end_audios "{skip_end}" --sample_rate {sample_rate}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(os.path.join("assets", "logs", "create_dataset.log"), done):
        yield log

def preprocess(model_name, sample_rate, cpu_core, cut_preprocess, process_effects, dataset, clean_dataset, clean_strength):
    sr = int(float(sample_rate.rstrip("k")) * 1000)

    if not model_name: return gr_warning(translations["provide_name"])
    if not os.path.exists(dataset) or not any(f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3")) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))): return gr_warning(translations["not_found_data"])
    
    model_dir = os.path.join("assets", "logs", model_name)
    if os.path.exists(model_dir): shutil.rmtree(model_dir, ignore_errors=True)

    p = subprocess.Popen(f'{python} main/inference/preprocess.py --model_name "{model_name}" --dataset_path "{dataset}" --sample_rate {sr} --cpu_cores {cpu_core} --cut_preprocess {cut_preprocess} --process_effects {process_effects} --clean_dataset {clean_dataset} --clean_strength {clean_strength}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(os.path.join(model_dir, "preprocess.log"), done):
        yield log

def extract(model_name, version, method, pitch_guidance, hop_length, cpu_cores, gpu, sample_rate, embedders, custom_embedders, onnx_f0_mode, embedders_mode):
    embedder_model = embedders if embedders != "custom" else custom_embedders
    sr = int(float(sample_rate.rstrip("k")) * 1000)

    if not model_name: return gr_warning(translations["provide_name"])
    model_dir = os.path.join("assets", "logs", model_name)

    try:
        if not any(os.path.isfile(os.path.join(model_dir, "sliced_audios", f)) for f in os.listdir(os.path.join(model_dir, "sliced_audios"))) or not any(os.path.isfile(os.path.join(model_dir, "sliced_audios_16k", f)) for f in os.listdir(os.path.join(model_dir, "sliced_audios_16k"))): return gr_warning(translations["not_found_data_preprocess"])
    except:
        return gr_warning(translations["not_found_data_preprocess"])
    
    p = subprocess.Popen(f'{python} main/inference/extract.py --model_name "{model_name}" --rvc_version {version} --f0_method {method} --pitch_guidance {pitch_guidance} --hop_length {hop_length} --cpu_cores {cpu_cores} --gpu {gpu} --sample_rate {sr} --embedder_model {embedder_model} --f0_onnx {onnx_f0_mode} --embedders_mode {embedders_mode}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(os.path.join(model_dir, "extract.log"), done):
        yield log

def create_index(model_name, rvc_version, index_algorithm):
    if not model_name: return gr_warning(translations["provide_name"])
    model_dir = os.path.join("assets", "logs", model_name)
    
    try:
        if not any(os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))): return gr_warning(translations["not_found_data_extract"])
    except:
        return gr_warning(translations["not_found_data_extract"])
    
    p = subprocess.Popen(f'{python} main/inference/create_index.py --model_name "{model_name}" --rvc_version {rvc_version} --index_algorithm {index_algorithm}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(os.path.join(model_dir, "create_index.log"), done):
        yield log

def training(model_name, rvc_version, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sample_rate, batch_size, gpu, pitch_guidance, not_pretrain, custom_pretrained, pretrain_g, pretrain_d, detector, threshold, clean_up, cache, model_author, vocoder, checkpointing, deterministic, benchmark, optimizer):
    sr = int(float(sample_rate.rstrip("k")) * 1000)
    if not model_name: return gr_warning(translations["provide_name"])

    model_dir = os.path.join("assets", "logs", model_name)
    if os.path.exists(os.path.join(model_dir, "train_pid.txt")): os.remove(os.path.join(model_dir, "train_pid.txt"))
    
    try:
        if not any(os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))): return gr_warning(translations["not_found_data_extract"])
    except:
        return gr_warning(translations["not_found_data_extract"])
    
    if not not_pretrain:
        if not custom_pretrained: 
            pretrained_selector = {True: {32000: ("f0G32k.pth", "f0D32k.pth"), 40000: ("f0G40k.pth", "f0D40k.pth"), 48000: ("f0G48k.pth", "f0D48k.pth")}, False: {32000: ("G32k.pth", "D32k.pth"), 40000: ("G40k.pth", "D40k.pth"), 48000: ("G48k.pth", "D48k.pth")}}

            pg, pd = pretrained_selector[pitch_guidance][sr]
        else:
            if not pretrain_g: return gr_warning(translations["provide_pretrained"].format(dg="G"))
            if not pretrain_d: return gr_warning(translations["provide_pretrained"].format(dg="D"))
            
            pg, pd = pretrain_g, pretrain_d

        pretrained_G, pretrained_D = (os.path.join("assets", "models", f"pretrained_{rvc_version}", f"{vocoder}_{pg}" if vocoder != 'Default' else pg), os.path.join("assets", "models", f"pretrained_{rvc_version}", f"{vocoder}_{pd}" if vocoder != 'Default' else pd)) if not custom_pretrained else (os.path.join("assets", "models", f"pretrained_custom", pg), os.path.join("assets", "models", f"pretrained_custom", pd))
        download_version = codecs.decode(f"uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cergenvarq_i{'2' if rvc_version == 'v2' else '1'}/", "rot13")
        
        if not custom_pretrained:
            try:
                if not os.path.exists(pretrained_G):
                    gr_info(translations["download_pretrained"].format(dg="G", rvc_version=rvc_version))
                    huggingface.HF_download_file("".join([download_version, vocoder, "_", pg]) if vocoder != 'Default' else (download_version + pg), os.path.join("assets", "models", f"pretrained_{rvc_version}", f"{vocoder}_{pg}" if vocoder != 'Default' else pg))
                        
                if not os.path.exists(pretrained_D):
                    gr_info(translations["download_pretrained"].format(dg="D", rvc_version=rvc_version))
                    huggingface.HF_download_file("".join([download_version, vocoder, "_", pd]) if vocoder != 'Default' else (download_version + pd), os.path.join("assets", "models", f"pretrained_{rvc_version}", f"{vocoder}_{pd}" if vocoder != 'Default' else pd))
            except:
                gr_warning(translations["not_use_pretrain_error_download"])
                pretrained_G = pretrained_D = None
        else:
            if not os.path.exists(pretrained_G): return gr_warning(translations["not_found_pretrain"].format(dg="G"))
            if not os.path.exists(pretrained_D): return gr_warning(translations["not_found_pretrain"].format(dg="D"))
    else: 
        pretrained_G = pretrained_D = None
        gr_warning(translations["not_use_pretrain"])

    gr_info(translations["start"].format(start=translations["training"]))

    p = subprocess.Popen(f'{python} main/inference/train.py --model_name "{model_name}" --rvc_version {rvc_version} --save_every_epoch {save_every_epoch} --save_only_latest {save_only_latest} --save_every_weights {save_every_weights} --total_epoch {total_epoch} --sample_rate {sr} --batch_size {batch_size} --gpu {gpu} --pitch_guidance {pitch_guidance} --overtraining_detector {detector} --overtraining_threshold {threshold} --cleanup {clean_up} --cache_data_in_gpu {cache} --g_pretrained_path "{pretrained_G}" --d_pretrained_path "{pretrained_D}" --model_author "{model_author}" --vocoder "{vocoder}" --checkpointing {checkpointing} --deterministic {deterministic} --benchmark {benchmark} --optimizer {optimizer}', shell=True)
    done = [False]

    with open(os.path.join(model_dir, "train_pid.txt"), "w") as pid_file:
        pid_file.write(str(p.pid))

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(os.path.join(model_dir, "train.log"), done):
        lines = log.splitlines()
        if len(lines) > 100: log = "\n".join(lines[-100:])
        yield log

def stop_pid(pid_file, model_name=None, train=False):
    try:
        pid_file_path = os.path.join("assets", f"{pid_file}.txt") if model_name is None else os.path.join("assets", "logs", model_name, f"{pid_file}.txt")

        if not os.path.exists(pid_file_path): return gr_warning(translations["not_found_pid"])
        else:
            with open(pid_file_path, "r") as pid_file:
                pids = [int(pid) for pid in pid_file.readlines()]

            for pid in pids:
                os.kill(pid, 9)

            if os.path.exists(pid_file_path): os.remove(pid_file_path)

        pid_file_path = os.path.join("assets", "logs", model_name, "config.json")

        if train and os.path.exists(pid_file_path):
            with open(pid_file_path, "r") as pid_file:
                pid_data = json.load(pid_file)
                pids = pid_data.get("process_pids", [])

            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)

                json.dump(pid_data, pid_file, indent=4)

            for pid in pids:
                os.kill(pid, 9)

            gr_info(translations["end_pid"])
    except:
        pass

def load_presets(presets, cleaner, autotune, pitch, clean_strength, index_strength, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, formant_shifting, formant_qfrency, formant_timbre):
    if not presets: return gr_warning(translations["provide_file_settings"])

    with open(os.path.join("assets", "presets", presets)) as f:
        file = json.load(f)

    gr_info(translations["load_presets"].format(presets=presets))
    return file.get("cleaner", cleaner), file.get("autotune", autotune), file.get("pitch", pitch), file.get("clean_strength", clean_strength), file.get("index_strength", index_strength), file.get("resample_sr", resample_sr), file.get("filter_radius", filter_radius), file.get("volume_envelope", volume_envelope), file.get("protect", protect), file.get("split_audio", split_audio), file.get("f0_autotune_strength", f0_autotune_strength), file.get("formant_shifting", formant_shifting), file.get("formant_qfrency", formant_qfrency), file.get("formant_timbre", formant_timbre)

def save_presets(name, cleaner, autotune, pitch, clean_strength, index_strength, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, cleaner_chbox, autotune_chbox, pitch_chbox, index_strength_chbox, resample_sr_chbox, filter_radius_chbox, volume_envelope_chbox, protect_chbox, split_audio_chbox, formant_shifting_chbox, formant_shifting, formant_qfrency, formant_timbre):  
    if not name: return gr_warning(translations["provide_filename_settings"])
    if not any([cleaner_chbox, autotune_chbox, pitch_chbox, index_strength_chbox, resample_sr_chbox, filter_radius_chbox, volume_envelope_chbox, protect_chbox, split_audio_chbox, formant_shifting_chbox]): return gr_warning(translations["choose1"])

    settings = {}

    for checkbox, data in [(cleaner_chbox, {"cleaner": cleaner, "clean_strength": clean_strength}), (autotune_chbox, {"autotune": autotune, "f0_autotune_strength": f0_autotune_strength}), (pitch_chbox, {"pitch": pitch}), (index_strength_chbox, {"index_strength": index_strength}), (resample_sr_chbox, {"resample_sr": resample_sr}), (filter_radius_chbox, {"filter_radius": filter_radius}), (volume_envelope_chbox, {"volume_envelope": volume_envelope}), (protect_chbox, {"protect": protect}), (split_audio_chbox, {"split_audio": split_audio}), (formant_shifting_chbox, {"formant_shifting": formant_shifting, "formant_qfrency": formant_qfrency, "formant_timbre": formant_timbre})]:
        if checkbox: settings.update(data)

    with open(os.path.join("assets", "presets", name + ".json"), "w") as f:
        json.dump(settings, f, indent=4)

    gr_info(translations["export_settings"])
    return change_preset_choices()

def report_bug(error_info, provide):
    report_path = os.path.join("assets", "logs", "report_bugs.log")
    if os.path.exists(report_path): os.remove(report_path)

    report_url = codecs.decode(requests.get(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/jroubbx.gkg", "rot13")).text, "rot13")
    if not error_info: error_info = "Không Có"

    gr_info(translations["thank"])

    if provide:
        try:
            for log in [os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".log")]:
                with open(log, "r", encoding="utf-8") as r:
                    with open(report_path, "a", encoding="utf-8") as w:
                        w.write(str(r.read()))
                        w.write("\n")
        except Exception as e:
            gr_error(translations["error_read_log"])
            logger.debug(e)

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            requests.post(report_url, json={"embeds": [{"title": "Báo Cáo Lỗi", "description": f"Mô tả lỗi: {error_info}", "color": 15158332, "author": {"name": "Vietnamese_RVC", "icon_url": codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/vpb.cat", "rot13"), "url": codecs.decode("uggcf://tvguho.pbz/CunzUhlauNau16/Ivrganzrfr-EIP/gerr/znva","rot13")}, "thumbnail": {"url": codecs.decode("uggcf://p.grabe.pbz/7dADJbv-36fNNNNq/grabe.tvs", "rot13")}, "fields": [{"name": "Số Lượng Gỡ Lỗi", "value": content.count("DEBUG")}, {"name": "Số Lượng Thông Tin", "value": content.count("INFO")}, {"name": "Số Lượng Cảnh Báo", "value": content.count("WARNING")}, {"name": "Số Lượng Lỗi", "value": content.count("ERROR")}], "footer": {"text": f"Tên Máy: {platform.uname().node} - Hệ Điều Hành: {platform.system()}-{platform.version()}\nThời Gian Báo Cáo Lỗi: {datetime.datetime.now()}."}}]})

            with open(report_path, "rb") as f:
                requests.post(report_url, files={"file": f})
        except Exception as e:
            gr_error(translations["error_send"])
            logger.debug(e)
        finally:
            if os.path.exists(report_path): os.remove(report_path)
    else: requests.post(report_url, json={"embeds": [{"title": "Báo Cáo Lỗi", "description": error_info}]})

def f0_extract(audio, f0_method, f0_onnx):
    if not audio or not os.path.exists(audio) or os.path.isdir(audio): 
        gr_warning(translations["input_not_valid"])
        return [None]*2

    from matplotlib import pyplot as plt
    from main.library.utils import check_predictors
    from main.inference.extract import FeatureInput

    check_predictors(f0_method, f0_onnx)

    f0_path = os.path.join("assets", "f0", os.path.splitext(os.path.basename(audio))[0])
    image_path = os.path.join(f0_path, "f0.png")
    txt_path = os.path.join(f0_path, "f0.txt")

    gr_info(translations["start_extract"])

    if not os.path.exists(f0_path): os.makedirs(f0_path, exist_ok=True)

    y, sr = librosa.load(audio, sr=None)

    feats = FeatureInput(sample_rate=sr, is_half=config.is_half, device=config.device)
    feats.f0_max = 1600.0

    F_temp = np.array(feats.compute_f0(y.flatten(), f0_method, 160, f0_onnx), dtype=np.float32)
    F_temp[F_temp == 0] = np.nan

    f0 = 1200 * np.log2(F_temp / librosa.midi_to_hz(0))

    plt.figure(figsize=(10, 4))
    plt.plot(f0)
    plt.title(f0_method)
    plt.xlabel(translations["time_frames"])
    plt.ylabel(translations["Frequency"])
    plt.savefig(image_path)
    plt.close()

    with open(txt_path, "w") as f:
        for i, f0_value in enumerate(f0):
            f.write(f"{i * sr / 160},{f0_value}\n")

    gr_info(translations["extract_done"])

    return [txt_path, image_path]

def pitch_guidance_lock(vocoders):
    return {"value": True, "interactive": vocoders == "Default", "__type__": "update"}

def vocoders_lock(pitch, vocoders):
    return {"value": vocoders if pitch else "Default", "interactive": pitch, "__type__": "update"}

def run_audioldm2(input_path, output_path, export_format, sample_rate, audioldm_model, source_prompt, target_prompt, steps, cfg_scale_src, cfg_scale_tar, t_start, save_compute):
    if not input_path or not os.path.exists(input_path) or os.path.isdir(input_path): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_path:
        gr_warning(translations["output_not_valid"])
        return None
    
    output_path = output_path.replace("wav", export_format)

    if os.path.exists(output_path): os.remove(output_path)

    gr_info(translations["start_edit"].format(input_path=input_path))
    subprocess.run([python, "main/inference/audioldm2.py", "--input_path", input_path, "--output_path", output_path, "--export_format", str(export_format), "--sample_rate", str(sample_rate), "--audioldm_model", audioldm_model, "--source_prompt", source_prompt, "--target_prompt", target_prompt, "--steps", str(steps), "--cfg_scale_src", str(cfg_scale_src), "--cfg_scale_tar", str(cfg_scale_tar), "--t_start", str(t_start), "--save_compute", str(save_compute)])
    
    gr_info(translations["success"])
    return output_path

def change_fp(fp):
    fp16 = fp == "fp16"

    if fp16 and config.device == "cpu": 
        gr_warning(translations["fp16_not_support"])
        return "fp32"
    else:
        gr_info(translations["start_update_precision"])

        configs = json.load(open(configs_json, "r"))
        configs["fp16"] = config.is_half = fp16

        with open(configs_json, "w") as f:
            json.dump(configs, f, indent=4)

        gr_info(translations["success"])
        return "fp16" if fp16 else "fp32"

def unlock_f0(value):
    return {"choices": method_f0_full if value else method_f0, "value": "rmvpe", "__type__": "update"} 

def unlock_vocoder(value, vocoder):
    return {"value": vocoder if value == "v2" else "Default", "interactive": value == "v2", "__type__": "update"} 

def unlock_ver(value, vocoder):
    return {"value": "v2" if vocoder == "Default" else value, "interactive": vocoder == "Default", "__type__": "update"}

def visible_embedders(value):
    return {"visible": value != "spin", "__type__": "update"}

with gr.Blocks(title="📱 Vietnamese-RVC GUI BY ANH", theme=theme, css="<style> @import url('{fonts}'); * {{font-family: 'Courgette', cursive !important;}} body, html {{font-family: 'Courgette', cursive !important;}} h1, h2, h3, h4, h5, h6, p, button, input, textarea, label, span, div, select {{font-family: 'Courgette', cursive !important;}} </style>".format(fonts=font or "https://fonts.googleapis.com/css2?family=Courgette&display=swap")) as app:
    gr.HTML("<h1 style='text-align: center;'>🎵VIETNAMESE RVC BY ANH🎵</h1>")
    gr.HTML(f"<h3 style='text-align: center;'>{translations['title']}</h3>")

    with gr.Tabs():      
        with gr.TabItem(translations["separator_tab"], visible=configs.get("separator_tab", True)):
            gr.Markdown(f"## {translations['separator_tab']}")
            with gr.Row(): 
                gr.Markdown(translations["4_part"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():       
                            cleaner = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True, min_width=140)       
                            backing = gr.Checkbox(label=translations["separator_backing"], value=False, interactive=True, min_width=140)
                            reverb = gr.Checkbox(label=translations["dereveb_audio"], value=False, interactive=True, min_width=140)
                            backing_reverb = gr.Checkbox(label=translations["dereveb_backing"], value=False, interactive=False, min_width=140)               
                            denoise = gr.Checkbox(label=translations["denoise_mdx"], value=False, interactive=False, min_width=140)     
                        with gr.Row():
                            separator_model = gr.Dropdown(label=translations["separator_model"], value=uvr_model[0], choices=uvr_model, interactive=True)
                            separator_backing_model = gr.Dropdown(label=translations["separator_backing_model"], value="Version-1", choices=["Version-1", "Version-2"], interactive=True, visible=backing.value)
            with gr.Row():
                with gr.Column():
                    separator_button = gr.Button(translations["separator_tab"], variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            shifts = gr.Slider(label=translations["shift"], info=translations["shift_info"], minimum=1, maximum=20, value=2, step=1, interactive=True)
                            segment_size = gr.Slider(label=translations["segments_size"], info=translations["segments_size_info"], minimum=32, maximum=3072, value=256, step=32, interactive=True)
                        with gr.Row():
                            mdx_batch_size = gr.Slider(label=translations["batch_size"], info=translations["mdx_batch_size_info"], minimum=1, maximum=64, value=1, step=1, interactive=True, visible=backing.value or reverb.value or separator_model.value in mdx_model)
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            overlap = gr.Radio(label=translations["overlap"], info=translations["overlap_info"], choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True)
                        with gr.Row():
                            mdx_hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=8192, value=1024, step=1, interactive=True, visible=backing.value or reverb.value or separator_model.value in mdx_model)
            with gr.Row():
                with gr.Column():
                    input = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])    
                    with gr.Accordion(translations["use_url"], open=False):
                        url = gr.Textbox(label=translations["url_audio"], value="", placeholder="https://www.youtube.com/...", scale=6)
                        download_button = gr.Button(translations["downloads"])
                with gr.Column():
                    with gr.Row():
                        clean_strength = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner.value)
                        sample_rate1 = gr.Radio(choices=sample_rate_choice, value=44100, label=translations["sr"], info=translations["sr_info"], interactive=True)
                    with gr.Accordion(translations["input_output"], open=False):
                        format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                        input_audio = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, allow_custom_value=True, interactive=True)
                        refesh_separator = gr.Button(translations["refesh"])
                        output_separator = gr.Textbox(label=translations["output_folder"], value="audios", placeholder="audios", info=translations["output_folder_info"], interactive=True)
                    audio_input = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
            with gr.Row():
                gr.Markdown(translations["output_separator"])
            with gr.Row():
                instruments_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["instruments"])
                original_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["original_vocal"])
                main_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["main_vocal"], visible=backing.value)
                backing_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["backing_vocal"], visible=backing.value)
            with gr.Row():
                separator_model.change(fn=lambda a, b, c: [visible(a or b or c in mdx_model), visible(a or b or c in mdx_model), valueFalse_interactive(a or b or c in mdx_model), visible(c not in mdx_model)], inputs=[backing, reverb, separator_model], outputs=[mdx_batch_size, mdx_hop_length, denoise, shifts])
                backing.change(fn=lambda a, b, c: [visible(a or b or c in mdx_model), visible(a or b or c in mdx_model), valueFalse_interactive(a or b or c in mdx_model), visible(a), visible(a), visible(a), valueFalse_interactive(a and b)], inputs=[backing, reverb, separator_model], outputs=[mdx_batch_size, mdx_hop_length, denoise, separator_backing_model, main_vocals, backing_vocals, backing_reverb])
                reverb.change(fn=lambda a, b, c: [visible(a or b or c in mdx_model), visible(a or b or c in mdx_model), valueFalse_interactive(a or b or c in mdx_model), valueFalse_interactive(a and b)], inputs=[backing, reverb, separator_model], outputs=[mdx_batch_size, mdx_hop_length, denoise, backing_reverb])
            with gr.Row():
                input_audio.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio], outputs=[audio_input])
                cleaner.change(fn=visible, inputs=[cleaner], outputs=[clean_strength])
            with gr.Row():
                input.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[input], outputs=[input_audio])
                refesh_separator.click(fn=change_audios_choices, inputs=[input_audio], outputs=[input_audio])
            with gr.Row():
                download_button.click(
                    fn=download_url, 
                    inputs=[url], 
                    outputs=[input_audio, audio_input, url],
                    api_name='download_url'
                )
                separator_button.click(
                    fn=separator_music, 
                    inputs=[
                        input_audio, 
                        output_separator,
                        format, 
                        shifts, 
                        segment_size, 
                        overlap, 
                        cleaner, 
                        clean_strength, 
                        denoise, 
                        separator_model, 
                        separator_backing_model, 
                        backing,
                        reverb, 
                        backing_reverb,
                        mdx_hop_length,
                        mdx_batch_size,
                        sample_rate1
                    ],
                    outputs=[original_vocals, instruments_audio, main_vocals, backing_vocals],
                    api_name='separator_music'
                )

        with gr.TabItem(translations["convert_audio"], visible=configs.get("convert_tab", True)):
            gr.Markdown(f"## {translations['convert_audio']}")
            with gr.Row():
                gr.Markdown(translations["convert_info"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            cleaner0 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                            autotune = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
                            use_audio = gr.Checkbox(label=translations["use_audio"], value=False, interactive=True)
                            checkpointing = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)
                        with gr.Row():
                            use_original = gr.Checkbox(label=translations["convert_original"], value=False, interactive=True, visible=use_audio.value) 
                            convert_backing = gr.Checkbox(label=translations["convert_backing"], value=False, interactive=True, visible=use_audio.value)   
                            not_merge_backing = gr.Checkbox(label=translations["not_merge_backing"], value=False, interactive=True, visible=use_audio.value)
                            merge_instrument = gr.Checkbox(label=translations["merge_instruments"], value=False, interactive=True, visible=use_audio.value) 
                    with gr.Row():
                        pitch = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                        clean_strength0 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner0.value)
                    with gr.Row(): 
                        with gr.Column():
                            audio_select = gr.Dropdown(label=translations["select_separate"], choices=[], value="", interactive=True, allow_custom_value=True, visible=False)
                            convert_button_2 = gr.Button(translations["convert_audio"], visible=False)
            with gr.Row():
                with gr.Column():
                    convert_button = gr.Button(translations["convert_audio"], variant="primary")
            with gr.Row():
                with gr.Column():
                    input0 = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])  
                    play_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
                with gr.Column():
                    with gr.Accordion(translations["model_accordion"], open=True):
                        with gr.Row():
                            model_pth = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                            model_index = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refesh = gr.Button(translations["refesh"])
                        with gr.Row():
                            index_strength = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index.value != "")
                    with gr.Accordion(translations["input_output"], open=False):
                        with gr.Column():
                            export_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                            input_audio0 = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                            output_audio = gr.Textbox(label=translations["output_path"], value="audios/output.wav", placeholder="audios/output.wav", info=translations["output_path_info"], interactive=True)
                        with gr.Column():
                            refesh0 = gr.Button(translations["refesh"])
                    with gr.Accordion(translations["setting"], open=False):
                        with gr.Accordion(translations["f0_method"], open=False):
                            with gr.Group():
                                with gr.Row():
                                    onnx_f0_mode = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                                    unlock_full_method = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                                method = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0+["hybrid"], value="rmvpe", interactive=True)
                                hybrid_method = gr.Dropdown(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[pm+yin]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[dio+yin]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]", "hybrid[rmvpe+yin]", "hybrid[harvest+yin]"], value="hybrid[pm+dio]", interactive=True, allow_custom_value=True, visible=method.value == "hybrid")
                            hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion(translations["f0_file"], open=False):
                            upload_f0_file = gr.File(label=translations["upload_f0"], file_types=[".txt"])  
                            f0_file_dropdown = gr.Dropdown(label=translations["f0_file_2"], value="", choices=f0_file, allow_custom_value=True, interactive=True)
                            refesh_f0_file = gr.Button(translations["refesh"])
                        with gr.Accordion(translations["hubert_model"], open=False):
                            embed_mode = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                            embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                            custom_embedders = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=embedders.value == "custom")    
                        with gr.Accordion(translations["use_presets"], open=False):
                            with gr.Row():
                                presets_name = gr.Dropdown(label=translations["file_preset"], choices=presets_file, value=presets_file[0] if len(presets_file) > 0 else '', interactive=True, allow_custom_value=True)
                            with gr.Row():
                                load_click = gr.Button(translations["load_file"], variant="primary")
                                refesh_click = gr.Button(translations["refesh"])
                            with gr.Accordion(translations["export_file"], open=False):
                                with gr.Row():
                                    with gr.Column():
                                        with gr.Group():
                                            with gr.Row():
                                                cleaner_chbox = gr.Checkbox(label=translations["save_clean"], value=True, interactive=True)
                                                autotune_chbox = gr.Checkbox(label=translations["save_autotune"], value=True, interactive=True)
                                                pitch_chbox = gr.Checkbox(label=translations["save_pitch"], value=True, interactive=True)
                                                index_strength_chbox = gr.Checkbox(label=translations["save_index_2"], value=True, interactive=True)
                                                resample_sr_chbox = gr.Checkbox(label=translations["save_resample"], value=True, interactive=True)
                                                filter_radius_chbox = gr.Checkbox(label=translations["save_filter"], value=True, interactive=True)
                                                volume_envelope_chbox = gr.Checkbox(label=translations["save_envelope"], value=True, interactive=True)
                                                protect_chbox = gr.Checkbox(label=translations["save_protect"], value=True, interactive=True)
                                                split_audio_chbox = gr.Checkbox(label=translations["save_split"], value=True, interactive=True)
                                                formant_shifting_chbox = gr.Checkbox(label=translations["formantshift"], value=True, interactive=True)
                                with gr.Row():
                                    with gr.Column():
                                        name_to_save_file = gr.Textbox(label=translations["filename_to_save"])
                                        save_file_button = gr.Button(translations["export_file"])
                            with gr.Row():
                                upload_presets = gr.File(label=translations["upload_presets"], file_types=[".json"])  
                        with gr.Column():
                            with gr.Row():
                                split_audio = gr.Checkbox(label=translations["split_audio"], value=False, interactive=True)
                                formant_shifting = gr.Checkbox(label=translations["formantshift"], value=False, interactive=True)
                            f0_autotune_strength = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=autotune.value)
                            resample_sr = gr.Radio(choices=[0]+sample_rate_choice, label=translations["resample"], info=translations["resample_info"], value=0, interactive=True)
                            filter_radius = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                            volume_envelope = gr.Slider(minimum=0, maximum=1, label=translations["volume_envelope"], info=translations["volume_envelope_info"], value=1, step=0.1, interactive=True)
                            protect = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.5, step=0.01, interactive=True)
                        with gr.Row():
                            formant_qfrency = gr.Slider(value=1.0, label=translations["formant_qfrency"], info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                            formant_timbre = gr.Slider(value=1.0, label=translations["formant_timbre"], info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
            with gr.Row():
                gr.Markdown(translations["output_convert"])
            with gr.Row():
                main_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["main_convert"])
                backing_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["convert_backing"], visible=convert_backing.value)
                main_backing = gr.Audio(show_download_button=True, interactive=False, label=translations["main_or_backing"], visible=convert_backing.value)  
            with gr.Row():
                original_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["convert_original"], visible=use_original.value)
                vocal_instrument = gr.Audio(show_download_button=True, interactive=False, label=translations["voice_or_instruments"], visible=merge_instrument.value)  
            with gr.Row():
                upload_f0_file.upload(fn=lambda inp: shutil.move(inp.name, os.path.join("assets", "f0")), inputs=[upload_f0_file], outputs=[f0_file_dropdown])
                refesh_f0_file.click(fn=change_f0_choices, inputs=[], outputs=[f0_file_dropdown])
                unlock_full_method.change(fn=unlock_f0, inputs=[unlock_full_method], outputs=[method])
            with gr.Row():
                load_click.click(
                    fn=load_presets, 
                    inputs=[
                        presets_name, 
                        cleaner0, 
                        autotune, 
                        pitch, 
                        clean_strength0, 
                        index_strength, 
                        resample_sr, 
                        filter_radius, 
                        volume_envelope, 
                        protect, 
                        split_audio, 
                        f0_autotune_strength, 
                        formant_qfrency, 
                        formant_timbre
                    ], 
                    outputs=[
                        cleaner0, 
                        autotune, 
                        pitch, 
                        clean_strength0, 
                        index_strength, 
                        resample_sr, 
                        filter_radius, 
                        volume_envelope, 
                        protect, 
                        split_audio, 
                        f0_autotune_strength, 
                        formant_shifting, 
                        formant_qfrency, 
                        formant_timbre
                    ]
                )
                refesh_click.click(fn=change_preset_choices, inputs=[], outputs=[presets_name])
                save_file_button.click(
                    fn=save_presets, 
                    inputs=[
                        name_to_save_file, 
                        cleaner0, 
                        autotune, 
                        pitch, 
                        clean_strength0, 
                        index_strength, 
                        resample_sr, 
                        filter_radius, 
                        volume_envelope, 
                        protect, 
                        split_audio, 
                        f0_autotune_strength, 
                        cleaner_chbox, 
                        autotune_chbox, 
                        pitch_chbox, 
                        index_strength_chbox, 
                        resample_sr_chbox, 
                        filter_radius_chbox, 
                        volume_envelope_chbox, 
                        protect_chbox, 
                        split_audio_chbox, 
                        formant_shifting_chbox, 
                        formant_shifting, 
                        formant_qfrency, 
                        formant_timbre
                    ], 
                    outputs=[presets_name]
                )
            with gr.Row():
                upload_presets.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("assets", "presets")), inputs=[upload_presets], outputs=[presets_name])
                autotune.change(fn=visible, inputs=[autotune], outputs=[f0_autotune_strength])
                use_audio.change(fn=lambda a: [visible(a), visible(a), visible(a), visible(a), visible(a), valueFalse_interactive(a), valueFalse_interactive(a), valueFalse_interactive(a), valueFalse_interactive(a), visible(not a), visible(not a), visible(not a), visible(not a)], inputs=[use_audio], outputs=[main_backing, use_original, convert_backing, not_merge_backing, merge_instrument, use_original, convert_backing, not_merge_backing, merge_instrument, input_audio0, output_audio, input0, play_audio])
            with gr.Row():
                convert_backing.change(fn=lambda a,b: [change_backing_choices(a, b), visible(a)], inputs=[convert_backing, not_merge_backing], outputs=[use_original, backing_convert])
                use_original.change(fn=lambda audio, original: [visible(original), visible(not original), visible(audio and not original), valueFalse_interactive(not original), valueFalse_interactive(not original)], inputs=[use_audio, use_original], outputs=[original_convert, main_convert, main_backing, convert_backing, not_merge_backing])
                cleaner0.change(fn=visible, inputs=[cleaner0], outputs=[clean_strength0])
            with gr.Row():
                merge_instrument.change(fn=visible, inputs=[merge_instrument], outputs=[vocal_instrument])
                not_merge_backing.change(fn=lambda audio, merge, cvb: [visible(audio and not merge), change_backing_choices(cvb, merge)], inputs=[use_audio, not_merge_backing, convert_backing], outputs=[main_backing, use_original])
                method.change(fn=lambda method, hybrid: [visible(method == "hybrid"), hoplength_show(method, hybrid)], inputs=[method, hybrid_method], outputs=[hybrid_method, hop_length])
            with gr.Row():
                hybrid_method.change(fn=hoplength_show, inputs=[method, hybrid_method], outputs=[hop_length])
                refesh.click(fn=change_models_choices, inputs=[], outputs=[model_pth, model_index])
                model_pth.change(fn=get_index, inputs=[model_pth], outputs=[model_index])
            with gr.Row():
                input0.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[input0], outputs=[input_audio0])
                input_audio0.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio0], outputs=[play_audio])
                formant_shifting.change(fn=lambda a: [visible(a)]*2, inputs=[formant_shifting], outputs=[formant_qfrency, formant_timbre])
            with gr.Row():
                embedders.change(fn=lambda embedders: visible(embedders == "custom"), inputs=[embedders], outputs=[custom_embedders])
                refesh0.click(fn=change_audios_choices, inputs=[input_audio0], outputs=[input_audio0])
                model_index.change(fn=index_strength_show, inputs=[model_index], outputs=[index_strength])
            with gr.Row():
                audio_select.change(fn=lambda: visible(True), inputs=[], outputs=[convert_button_2])
                convert_button.click(fn=lambda: visible(False), inputs=[], outputs=[convert_button])
                convert_button_2.click(fn=lambda: [visible(False), visible(False)], inputs=[], outputs=[audio_select, convert_button_2])
            with gr.Row():
                convert_button.click(
                    fn=convert_selection,
                    inputs=[
                        cleaner0,
                        autotune,
                        use_audio,
                        use_original,
                        convert_backing,
                        not_merge_backing,
                        merge_instrument,
                        pitch,
                        clean_strength0,
                        model_pth,
                        model_index,
                        index_strength,
                        input_audio0,
                        output_audio,
                        export_format,
                        method,
                        hybrid_method,
                        hop_length,
                        embedders,
                        custom_embedders,
                        resample_sr,
                        filter_radius,
                        volume_envelope,
                        protect,
                        split_audio,
                        f0_autotune_strength,
                        checkpointing,
                        onnx_f0_mode,
                        formant_shifting, 
                        formant_qfrency, 
                        formant_timbre,
                        f0_file_dropdown,
                        embed_mode
                    ],
                    outputs=[audio_select, main_convert, backing_convert, main_backing, original_convert, vocal_instrument, convert_button],
                    api_name="convert_selection"
                )
                embed_mode.change(fn=visible_embedders, inputs=[embed_mode], outputs=[embedders])
                convert_button_2.click(
                    fn=convert_audio,
                    inputs=[
                        cleaner0,
                        autotune,
                        use_audio,
                        use_original,
                        convert_backing,
                        not_merge_backing,
                        merge_instrument,
                        pitch,
                        clean_strength0,
                        model_pth,
                        model_index,
                        index_strength,
                        input_audio0,
                        output_audio,
                        export_format,
                        method,
                        hybrid_method,
                        hop_length,
                        embedders,
                        custom_embedders,
                        resample_sr,
                        filter_radius,
                        volume_envelope,
                        protect,
                        split_audio,
                        f0_autotune_strength,
                        audio_select,
                        checkpointing,
                        onnx_f0_mode,
                        formant_shifting, 
                        formant_qfrency, 
                        formant_timbre,
                        f0_file_dropdown,
                        embed_mode
                    ],
                    outputs=[main_convert, backing_convert, main_backing, original_convert, vocal_instrument, convert_button],
                    api_name="convert_audio"
                )

        with gr.TabItem(translations["convert_with_whisper"], visible=configs.get("convert_with_whisper", True)):
            gr.Markdown(f"## {translations['convert_with_whisper']}")
            with gr.Row():
                gr.Markdown(translations["convert_with_whisper_info"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            cleaner2 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                            autotune2 = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
                            checkpointing2 = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)
                            formant_shifting2 = gr.Checkbox(label=translations["formantshift"], value=False, interactive=True)
                        with gr.Row():
                            num_spk = gr.Slider(minimum=2, maximum=8, step=1, info=translations["num_spk_info"], label=translations["num_spk"], value=2, interactive=True)
            with gr.Row():
                with gr.Column():
                    convert_button3 = gr.Button(translations["convert_audio"], variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(translations["model_accordion"] + " 1", open=True):
                        with gr.Row():
                            model_pth2 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                            model_index2 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refesh2 = gr.Button(translations["refesh"])
                        with gr.Row():
                            pitch3 = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                            index_strength2 = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index2.value != "")
                    with gr.Accordion(translations["input_output"], open=False):
                        with gr.Column():
                            export_format2 = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                            input_audio1 = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                            output_audio2 = gr.Textbox(label=translations["output_path"], value="audios/output.wav", placeholder="audios/output.wav", info=translations["output_path_info"], interactive=True)
                        with gr.Column():
                            refesh4 = gr.Button(translations["refesh"])
                        with gr.Row():
                            input2 = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])
                with gr.Column():
                    with gr.Accordion(translations["model_accordion"] + " 2", open=True):
                        with gr.Row():
                            model_pth3 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                            model_index3 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refesh3 = gr.Button(translations["refesh"])
                        with gr.Row():
                            pitch4 = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                            index_strength3 = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index3.value != "")
                    with gr.Accordion(translations["setting"], open=False):
                        with gr.Row():
                            model_size = gr.Radio(label=translations["model_size"], info=translations["model_size_info"], choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"], value="medium", interactive=True)
                        with gr.Accordion(translations["f0_method"], open=False):
                            with gr.Group():
                                with gr.Row():
                                    onnx_f0_mode4 = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                                    unlock_full_method2 = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                                method3 = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0+["hybrid"], value="rmvpe", interactive=True)
                                hybrid_method3 = gr.Dropdown(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[pm+yin]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[dio+yin]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]", "hybrid[rmvpe+yin]", "hybrid[harvest+yin]"], value="hybrid[pm+dio]", interactive=True, allow_custom_value=True, visible=method3.value == "hybrid")
                            hop_length3 = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion(translations["hubert_model"], open=False):
                            embed_mode3 = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                            embedders3 = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                            custom_embedders3 = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=embedders3.value == "custom")
                        with gr.Column():      
                            clean_strength3 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner2.value)
                            f0_autotune_strength3 = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=autotune.value)
                            resample_sr3 = gr.Radio(choices=[0]+sample_rate_choice, label=translations["resample"], info=translations["resample_info"], value=0, interactive=True)
                            filter_radius3 = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                            volume_envelope3 = gr.Slider(minimum=0, maximum=1, label=translations["volume_envelope"], info=translations["volume_envelope_info"], value=1, step=0.1, interactive=True)
                            protect3 = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.5, step=0.01, interactive=True)
                        with gr.Row():
                            formant_qfrency3 = gr.Slider(value=1.0, label=translations["formant_qfrency"] + " 1", info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                            formant_timbre3 = gr.Slider(value=1.0, label=translations["formant_timbre"] + " 1", info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                        with gr.Row():
                            formant_qfrency4 = gr.Slider(value=1.0, label=translations["formant_qfrency"] + " 2", info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                            formant_timbre4 = gr.Slider(value=1.0, label=translations["formant_timbre"] + " 2", info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
            with gr.Row():
                gr.Markdown(translations["input_output"])
            with gr.Row():
                play_audio2 = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
                play_audio3 = gr.Audio(show_download_button=True, interactive=False, label=translations["output_file_tts_convert"])
            with gr.Row():
                autotune2.change(fn=visible, inputs=[autotune2], outputs=[f0_autotune_strength3])
                cleaner2.change(fn=visible, inputs=[cleaner2], outputs=[clean_strength3])
                method3.change(fn=lambda method, hybrid: [visible(method == "hybrid"), hoplength_show(method, hybrid)], inputs=[method3, hybrid_method3], outputs=[hybrid_method3, hop_length3])
            with gr.Row():
                hybrid_method3.change(fn=hoplength_show, inputs=[method3, hybrid_method3], outputs=[hop_length3])
                refesh2.click(fn=change_models_choices, inputs=[], outputs=[model_pth2, model_index2])
                model_pth2.change(fn=get_index, inputs=[model_pth2], outputs=[model_index2])
            with gr.Row():
                refesh3.click(fn=change_models_choices, inputs=[], outputs=[model_pth3, model_index3])
                model_pth3.change(fn=get_index, inputs=[model_pth3], outputs=[model_index3])
                input2.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[input2], outputs=[input_audio1])
            with gr.Row():
                input_audio1.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio1], outputs=[play_audio2])
                formant_shifting2.change(fn=lambda a: [visible(a)]*4, inputs=[formant_shifting2], outputs=[formant_qfrency3, formant_timbre3, formant_qfrency4, formant_timbre4])
                embedders3.change(fn=lambda embedders: visible(embedders == "custom"), inputs=[embedders3], outputs=[custom_embedders3])
            with gr.Row():
                refesh4.click(fn=change_audios_choices, inputs=[input_audio1], outputs=[input_audio1])
                model_index2.change(fn=index_strength_show, inputs=[model_index2], outputs=[index_strength2])
                model_index3.change(fn=index_strength_show, inputs=[model_index3], outputs=[index_strength3])
            with gr.Row():
                unlock_full_method2.change(fn=unlock_f0, inputs=[unlock_full_method2], outputs=[method3])
                embed_mode3.change(fn=visible_embedders, inputs=[embed_mode3], outputs=[embedders3])
                convert_button3.click(
                    fn=convert_with_whisper,
                    inputs=[
                        num_spk,
                        model_size,
                        cleaner2,
                        clean_strength3,
                        autotune2,
                        f0_autotune_strength3,
                        checkpointing2,
                        model_pth2,
                        model_pth3,
                        model_index2,
                        model_index3,
                        pitch3,
                        pitch4,
                        index_strength2,
                        index_strength3,
                        export_format2,
                        input_audio1,
                        output_audio2,
                        onnx_f0_mode4,
                        method3,
                        hybrid_method3,
                        hop_length3,
                        embed_mode3,
                        embedders3,
                        custom_embedders3,
                        resample_sr3,
                        filter_radius3,
                        volume_envelope3,
                        protect3,
                        formant_shifting2,
                        formant_qfrency3,
                        formant_timbre3,
                        formant_qfrency4,
                        formant_timbre4,
                    ],
                    outputs=[play_audio3],
                    api_name="convert_with_whisper"
                )

        with gr.TabItem(translations["convert_text"], visible=configs.get("tts_tab", True)):
            gr.Markdown(translations["convert_text_markdown"])
            with gr.Row():
                gr.Markdown(translations["convert_text_markdown_2"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            use_txt = gr.Checkbox(label=translations["input_txt"], value=False, interactive=True)
                            google_tts_check_box = gr.Checkbox(label=translations["googletts"], value=False, interactive=True)
                        prompt = gr.Textbox(label=translations["text_to_speech"], value="", placeholder="Hello Words", lines=3)
                with gr.Column():
                    speed = gr.Slider(label=translations["voice_speed"], info=translations["voice_speed_info"], minimum=-100, maximum=100, value=0, step=1)
                    pitch0 = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
            with gr.Row():
                tts_button = gr.Button(translations["tts_1"], variant="primary", scale=2)
                convert_button0 = gr.Button(translations["tts_2"], variant="secondary", scale=2)
            with gr.Row():
                with gr.Column():
                    txt_input = gr.File(label=translations["drop_text"], file_types=[".txt", ".srt"], visible=use_txt.value)  
                    tts_voice = gr.Dropdown(label=translations["voice"], choices=edgetts, interactive=True, value="vi-VN-NamMinhNeural")
                    tts_pitch = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info_2"], label=translations["pitch"], value=0, interactive=True)
                with gr.Column():
                    with gr.Accordion(translations["model_accordion"], open=True):
                        with gr.Row():
                            model_pth0 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                            model_index0 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refesh1 = gr.Button(translations["refesh"])
                        with gr.Row():
                            index_strength0 = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index0.value != "")
                    with gr.Accordion(translations["output_path"], open=False):
                        export_format0 = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                        output_audio0 = gr.Textbox(label=translations["output_tts"], value="audios/tts.wav", placeholder="audios/tts.wav", info=translations["tts_output"], interactive=True)
                        output_audio1 = gr.Textbox(label=translations["output_tts_convert"], value="audios/tts-convert.wav", placeholder="audios/tts-convert.wav", info=translations["tts_output"], interactive=True)
                    with gr.Accordion(translations["setting"], open=False):
                        with gr.Accordion(translations["f0_method"], open=False):
                            with gr.Group():
                                with gr.Row():
                                    onnx_f0_mode1 = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                                    unlock_full_method3 = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                                method0 = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0+["hybrid"], value="rmvpe", interactive=True)
                                hybrid_method0 = gr.Dropdown(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[pm+yin]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[dio+yin]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]", "hybrid[rmvpe+yin]", "hybrid[harvest+yin]"], value="hybrid[pm+dio]", interactive=True, allow_custom_value=True, visible=method0.value == "hybrid")
                            hop_length0 = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion(translations["f0_file"], open=False):
                            upload_f0_file0 = gr.File(label=translations["upload_f0"], file_types=[".txt"])  
                            f0_file_dropdown0 = gr.Dropdown(label=translations["f0_file_2"], value="", choices=f0_file, allow_custom_value=True, interactive=True)
                            refesh_f0_file0 = gr.Button(translations["refesh"])
                        with gr.Accordion(translations["hubert_model"], open=False):
                            embed_mode1 = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                            embedders0 = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                            custom_embedders0 = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=embedders0.value == "custom")
                        with gr.Group():
                            with gr.Row():
                                formant_shifting1 = gr.Checkbox(label=translations["formantshift"], value=False, interactive=True)  
                                split_audio0 = gr.Checkbox(label=translations["split_audio"], value=False, interactive=True)   
                                cleaner1 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)     
                                autotune3 = gr.Checkbox(label=translations["autotune"], value=False, interactive=True) 
                                checkpointing0 = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)         
                        with gr.Column():
                            f0_autotune_strength0 = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=autotune3.value)
                            clean_strength1 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner1.value)
                            resample_sr0 = gr.Radio(choices=[0]+sample_rate_choice, label=translations["resample"], info=translations["resample_info"], value=0, interactive=True)
                            filter_radius0 = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                            volume_envelope0 = gr.Slider(minimum=0, maximum=1, label=translations["volume_envelope"], info=translations["volume_envelope_info"], value=1, step=0.1, interactive=True)
                            protect0 = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.5, step=0.01, interactive=True)
                        with gr.Row():
                            formant_qfrency1 = gr.Slider(value=1.0, label=translations["formant_qfrency"], info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                            formant_timbre1 = gr.Slider(value=1.0, label=translations["formant_timbre"], info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
            with gr.Row():
                gr.Markdown(translations["output_tts_markdown"])
            with gr.Row():
                tts_voice_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["output_text_to_speech"])
                tts_voice_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["output_file_tts_convert"])
            with gr.Row():
                unlock_full_method3.change(fn=unlock_f0, inputs=[unlock_full_method3], outputs=[method0])
                upload_f0_file0.upload(fn=lambda inp: shutil.move(inp.name, os.path.join("assets", "f0")), inputs=[upload_f0_file0], outputs=[f0_file_dropdown0])
                refesh_f0_file0.click(fn=change_f0_choices, inputs=[], outputs=[f0_file_dropdown0])
            with gr.Row():
                embed_mode1.change(fn=visible_embedders, inputs=[embed_mode1], outputs=[embedders0])
                autotune3.change(fn=visible, inputs=[autotune3], outputs=[f0_autotune_strength0])
                model_pth0.change(fn=get_index, inputs=[model_pth0], outputs=[model_index0])
            with gr.Row():
                cleaner1.change(fn=visible, inputs=[cleaner1], outputs=[clean_strength1])
                method0.change(fn=lambda method, hybrid: [visible(method == "hybrid"), hoplength_show(method, hybrid)], inputs=[method0, hybrid_method0], outputs=[hybrid_method0, hop_length0])
                hybrid_method0.change(fn=hoplength_show, inputs=[method0, hybrid_method0], outputs=[hop_length0])
            with gr.Row():
                refesh1.click(fn=change_models_choices, inputs=[], outputs=[model_pth0, model_index0])
                embedders0.change(fn=lambda embedders: visible(embedders == "custom"), inputs=[embedders0], outputs=[custom_embedders0])
                formant_shifting1.change(fn=lambda a: [visible(a)]*2, inputs=[formant_shifting1], outputs=[formant_qfrency1, formant_timbre1])
            with gr.Row():
                model_index0.change(fn=index_strength_show, inputs=[model_index0], outputs=[index_strength0])
                txt_input.upload(fn=process_input, inputs=[txt_input], outputs=[prompt])
                use_txt.change(fn=visible, inputs=[use_txt], outputs=[txt_input])
            with gr.Row():
                google_tts_check_box.change(fn=change_tts_voice_choices, inputs=[google_tts_check_box], outputs=[tts_voice])
                tts_button.click(
                    fn=TTS, 
                    inputs=[
                        prompt, 
                        tts_voice, 
                        speed, 
                        output_audio0,
                        tts_pitch,
                        google_tts_check_box,
                        txt_input
                    ], 
                    outputs=[tts_voice_audio],
                    api_name="text-to-speech"
                )
                convert_button0.click(
                    fn=convert_tts,
                    inputs=[
                        cleaner1, 
                        autotune3, 
                        pitch0, 
                        clean_strength1, 
                        model_pth0, 
                        model_index0, 
                        index_strength0, 
                        output_audio0, 
                        output_audio1,
                        export_format0,
                        method0, 
                        hybrid_method0, 
                        hop_length0, 
                        embedders0, 
                        custom_embedders0, 
                        resample_sr0, 
                        filter_radius0, 
                        volume_envelope0, 
                        protect0,
                        split_audio0,
                        f0_autotune_strength0,
                        checkpointing0,
                        onnx_f0_mode1,
                        formant_shifting1, 
                        formant_qfrency1, 
                        formant_timbre1,
                        f0_file_dropdown0,
                        embed_mode1
                    ],
                    outputs=[tts_voice_convert],
                    api_name="convert_tts"
                )

        with gr.TabItem(translations["audio_editing"], visible=configs.get("audioldm2", True)):
            gr.Markdown(translations["audio_editing_info"])
            with gr.Row():
                gr.Markdown(translations["audio_editing_markdown"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            save_compute = gr.Checkbox(label=translations["save_compute"], value=True, interactive=True)
                        tar_prompt = gr.Textbox(label=translations["target_prompt"], info=translations["target_prompt_info"], placeholder="Piano and violin cover", lines=5, interactive=True)
                with gr.Column():
                    cfg_scale_src = gr.Slider(value=3, minimum=0.5, maximum=25, label=translations["cfg_scale_src"], info=translations["cfg_scale_src_info"], interactive=True)
                    cfg_scale_tar = gr.Slider(value=12, minimum=0.5, maximum=25, label=translations["cfg_scale_tar"], info=translations["cfg_scale_tar_info"], interactive=True)
            with gr.Row():
                edit_button = gr.Button(translations["editing"], variant="primary")
            with gr.Row():
                with gr.Column():
                    drop_audio_file = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])  
                    display_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
                with gr.Column():
                    with gr.Accordion(translations["input_output"], open=False):
                        with gr.Column():
                            export_audio_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                            input_audiopath = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                            output_audiopath = gr.Textbox(label=translations["output_path"], value="audios/output.wav", placeholder="audios/output.wav", info=translations["output_path_info"], interactive=True)
                        with gr.Column():
                            refesh_audio = gr.Button(translations["refesh"])
                    with gr.Accordion(translations["setting"], open=False):
                        audioldm2_model = gr.Radio(label=translations["audioldm2_model"], info=translations["audioldm2_model_info"], choices=["audioldm2", "audioldm2-large", "audioldm2-music"], value="audioldm2-music", interactive=True)
                        with gr.Row():
                            src_prompt = gr.Textbox(label=translations["source_prompt"], lines=2, interactive=True, info=translations["source_prompt_info"], placeholder="A recording of a happy upbeat classical music piece")
                        with gr.Row():
                            with gr.Column(): 
                                audioldm2_sample_rate = gr.Radio(choices=sample_rate_choice, label=translations["sr"], info=translations["sr_info"], value=44100, interactive=True)
                                t_start = gr.Slider(minimum=15, maximum=85, value=45, step=1, label=translations["t_start"], interactive=True, info=translations["t_start_info"])
                                steps = gr.Slider(value=50, step=1, minimum=10, maximum=300, label=translations["steps_label"], info=translations["steps_info"], interactive=True)
            with gr.Row():
                gr.Markdown(translations["output_audio"])
            with gr.Row():
                output_audioldm2 = gr.Audio(show_download_button=True, interactive=False, label=translations["output_audio"])
            with gr.Row():
                refesh_audio.click(fn=change_audios_choices, inputs=[input_audiopath], outputs=[input_audiopath])
                drop_audio_file.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[drop_audio_file], outputs=[input_audiopath])
                input_audiopath.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audiopath], outputs=[display_audio])
            with gr.Row():
                edit_button.click(
                    fn=run_audioldm2,
                    inputs=[
                        input_audiopath, 
                        output_audiopath, 
                        export_audio_format, 
                        audioldm2_sample_rate, 
                        audioldm2_model, 
                        src_prompt, 
                        tar_prompt, 
                        steps, 
                        cfg_scale_src, 
                        cfg_scale_tar, 
                        t_start, 
                        save_compute
                    ],
                    outputs=[output_audioldm2],
                    api_name="audioldm2"
                )

        with gr.TabItem(translations["audio_effects"], visible=configs.get("effects_tab", True)):
            gr.Markdown(translations["apply_audio_effects"])
            with gr.Row():
                gr.Markdown(translations["audio_effects_edit"])
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        reverb_check_box = gr.Checkbox(label=translations["reverb"], value=False, interactive=True)
                        chorus_check_box = gr.Checkbox(label=translations["chorus"], value=False, interactive=True)
                        delay_check_box = gr.Checkbox(label=translations["delay"], value=False, interactive=True)
                        phaser_check_box = gr.Checkbox(label=translations["phaser"], value=False, interactive=True)
                        compressor_check_box = gr.Checkbox(label=translations["compressor"], value=False, interactive=True)
                        more_options = gr.Checkbox(label=translations["more_option"], value=False, interactive=True)    
            with gr.Row():
                with gr.Accordion(translations["input_output"], open=False):
                    with gr.Row():
                        upload_audio = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])
                    with gr.Row():
                        audio_in_path = gr.Dropdown(label=translations["input_audio"], value="", choices=paths_for_files, info=translations["provide_audio"], interactive=True, allow_custom_value=True)
                        audio_out_path = gr.Textbox(label=translations["output_audio"], value="audios/audio_effects.wav", placeholder="audios/audio_effects.wav", info=translations["provide_output"], interactive=True)
                    with gr.Row():
                        with gr.Column():
                            audio_combination = gr.Checkbox(label=translations["merge_instruments"], value=False, interactive=True)
                            audio_combination_input = gr.Dropdown(label=translations["input_audio"], value="", choices=paths_for_files, info=translations["provide_audio"], interactive=True, allow_custom_value=True, visible=audio_combination.value)
                    with gr.Row():
                        main_vol = gr.Slider(minimum=-80, maximum=80, label=translations["main_volume"], info=translations["main_volume_info"], value=-4, step=1, interactive=True, visible=audio_combination.value)
                        combine_vol = gr.Slider(minimum=-80, maximum=80, label=translations["combination_volume"], info=translations["combination_volume_info"], value=-7, step=1, interactive=True, visible=audio_combination.value)
                    with gr.Row():
                        audio_effects_refesh = gr.Button(translations["refesh"])
                    with gr.Row():
                        audio_output_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
            with gr.Row():
                apply_effects_button = gr.Button(translations["apply"], variant="primary", scale=2)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Accordion(translations["reverb"], open=False, visible=reverb_check_box.value) as reverb_accordion:
                            reverb_freeze_mode = gr.Checkbox(label=translations["reverb_freeze"], info=translations["reverb_freeze_info"], value=False, interactive=True)
                            reverb_room_size = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.15, label=translations["room_size"], info=translations["room_size_info"], interactive=True)
                            reverb_damping = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label=translations["damping"], info=translations["damping_info"], interactive=True)
                            reverb_wet_level = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.2, label=translations["wet_level"], info=translations["wet_level_info"], interactive=True)
                            reverb_dry_level = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.8, label=translations["dry_level"], info=translations["dry_level_info"], interactive=True)
                            reverb_width = gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label=translations["width"], info=translations["width_info"], interactive=True)
                    with gr.Row():
                        with gr.Accordion(translations["chorus"], open=False, visible=chorus_check_box.value) as chorus_accordion:
                            chorus_depth = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["chorus_depth"], info=translations["chorus_depth_info"], interactive=True)
                            chorus_rate_hz = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1.5, label=translations["chorus_rate_hz"], info=translations["chorus_rate_hz_info"], interactive=True)
                            chorus_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["chorus_mix"], info=translations["chorus_mix_info"], interactive=True)
                            chorus_centre_delay_ms = gr.Slider(minimum=0, maximum=50, step=1, value=10, label=translations["chorus_centre_delay_ms"], info=translations["chorus_centre_delay_ms_info"], interactive=True)
                            chorus_feedback = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label=translations["chorus_feedback"], info=translations["chorus_feedback_info"], interactive=True)
                    with gr.Row():
                        with gr.Accordion(translations["delay"], open=False, visible=delay_check_box.value) as delay_accordion:
                            delay_second = gr.Slider(minimum=0, maximum=5, step=0.01, value=0.5, label=translations["delay_seconds"], info=translations["delay_seconds_info"], interactive=True)
                            delay_feedback = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["delay_feedback"], info=translations["delay_feedback_info"], interactive=True)
                            delay_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["delay_mix"], info=translations["delay_mix_info"], interactive=True)
                with gr.Column():
                    with gr.Row():
                        with gr.Accordion(translations["more_option"], open=False, visible=more_options.value) as more_accordion:
                            with gr.Row():
                                fade = gr.Checkbox(label=translations["fade"], value=False, interactive=True)
                                bass_or_treble = gr.Checkbox(label=translations["bass_or_treble"], value=False, interactive=True)
                                limiter = gr.Checkbox(label=translations["limiter"], value=False, interactive=True)
                                resample_checkbox = gr.Checkbox(label=translations["resample"], value=False, interactive=True)
                            with gr.Row():
                                distortion_checkbox = gr.Checkbox(label=translations["distortion"], value=False, interactive=True)
                                gain_checkbox = gr.Checkbox(label=translations["gain"], value=False, interactive=True)
                                bitcrush_checkbox = gr.Checkbox(label=translations["bitcrush"], value=False, interactive=True)
                                clipping_checkbox = gr.Checkbox(label=translations["clipping"], value=False, interactive=True)
                            with gr.Accordion(translations["fade"], open=True, visible=fade.value) as fade_accordion:
                                with gr.Row():
                                    fade_in = gr.Slider(minimum=0, maximum=10000, step=100, value=0, label=translations["fade_in"], info=translations["fade_in_info"], interactive=True)
                                    fade_out = gr.Slider(minimum=0, maximum=10000, step=100, value=0, label=translations["fade_out"], info=translations["fade_out_info"], interactive=True)
                            with gr.Accordion(translations["bass_or_treble"], open=True, visible=bass_or_treble.value) as bass_treble_accordion:
                                with gr.Row():
                                    bass_boost = gr.Slider(minimum=0, maximum=20, step=1, value=0, label=translations["bass_boost"], info=translations["bass_boost_info"], interactive=True)
                                    bass_frequency = gr.Slider(minimum=20, maximum=200, step=10, value=100, label=translations["bass_frequency"], info=translations["bass_frequency_info"], interactive=True)
                                with gr.Row():
                                    treble_boost = gr.Slider(minimum=0, maximum=20, step=1, value=0, label=translations["treble_boost"], info=translations["treble_boost_info"], interactive=True)
                                    treble_frequency = gr.Slider(minimum=1000, maximum=10000, step=500, value=3000, label=translations["treble_frequency"], info=translations["treble_frequency_info"], interactive=True)
                            with gr.Accordion(translations["limiter"], open=True, visible=limiter.value) as limiter_accordion:
                                with gr.Row():
                                    limiter_threashold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-1, label=translations["limiter_threashold_db"], info=translations["limiter_threashold_db_info"], interactive=True)
                                    limiter_release_ms = gr.Slider(minimum=10, maximum=1000, step=1, value=100, label=translations["limiter_release_ms"], info=translations["limiter_release_ms_info"], interactive=True)
                            with gr.Column():
                                pitch_shift_semitones = gr.Slider(minimum=-20, maximum=20, step=1, value=0, label=translations["pitch"], info=translations["pitch_info"], interactive=True)
                                audio_effect_resample_sr = gr.Radio(choices=[0]+sample_rate_choice, value=0, label=translations["resample"], info=translations["resample_info"], interactive=True, visible=resample_checkbox.value)
                                distortion_drive_db = gr.Slider(minimum=0, maximum=50, step=1, value=20, label=translations["distortion"], info=translations["distortion_info"], interactive=True, visible=distortion_checkbox.value)
                                gain_db = gr.Slider(minimum=-60, maximum=60, step=1, value=0, label=translations["gain"], info=translations["gain_info"], interactive=True, visible=gain_checkbox.value)
                                clipping_threashold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-1, label=translations["clipping_threashold_db"], info=translations["clipping_threashold_db_info"], interactive=True, visible=clipping_checkbox.value)
                                bitcrush_bit_depth = gr.Slider(minimum=1, maximum=24, step=1, value=16, label=translations["bitcrush_bit_depth"], info=translations["bitcrush_bit_depth_info"], interactive=True, visible=bitcrush_checkbox.value)
                    with gr.Row():
                        with gr.Accordion(translations["phaser"], open=False, visible=phaser_check_box.value) as phaser_accordion:
                            phaser_depth = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["phaser_depth"], info=translations["phaser_depth_info"], interactive=True)
                            phaser_rate_hz = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1, label=translations["phaser_rate_hz"], info=translations["phaser_rate_hz_info"], interactive=True)
                            phaser_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["phaser_mix"], info=translations["phaser_mix_info"], interactive=True)
                            phaser_centre_frequency_hz = gr.Slider(minimum=50, maximum=5000, step=10, value=1000, label=translations["phaser_centre_frequency_hz"], info=translations["phaser_centre_frequency_hz_info"], interactive=True)
                            phaser_feedback = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label=translations["phaser_feedback"], info=translations["phaser_feedback_info"], interactive=True)
                    with gr.Row():
                        with gr.Accordion(translations["compressor"], open=False, visible=compressor_check_box.value) as compressor_accordion:
                            compressor_threashold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-20, label=translations["compressor_threashold_db"], info=translations["compressor_threashold_db_info"], interactive=True)
                            compressor_ratio = gr.Slider(minimum=1, maximum=20, step=0.1, value=1, label=translations["compressor_ratio"], info=translations["compressor_ratio_info"], interactive=True)
                            compressor_attack_ms = gr.Slider(minimum=0.1, maximum=100, step=0.1, value=10, label=translations["compressor_attack_ms"], info=translations["compressor_attack_ms_info"], interactive=True)
                            compressor_release_ms = gr.Slider(minimum=10, maximum=1000, step=1, value=100, label=translations["compressor_release_ms"], info=translations["compressor_release_ms_info"], interactive=True)   
            with gr.Row():
                gr.Markdown(translations["output_audio"])
            with gr.Row():
                audio_play_input = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
                audio_play_output = gr.Audio(show_download_button=True, interactive=False, label=translations["output_audio"])
            with gr.Row():
                reverb_check_box.change(fn=visible, inputs=[reverb_check_box], outputs=[reverb_accordion])
                chorus_check_box.change(fn=visible, inputs=[chorus_check_box], outputs=[chorus_accordion])
                delay_check_box.change(fn=visible, inputs=[delay_check_box], outputs=[delay_accordion])
            with gr.Row():
                compressor_check_box.change(fn=visible, inputs=[compressor_check_box], outputs=[compressor_accordion])
                phaser_check_box.change(fn=visible, inputs=[phaser_check_box], outputs=[phaser_accordion])
                more_options.change(fn=visible, inputs=[more_options], outputs=[more_accordion])
            with gr.Row():
                fade.change(fn=visible, inputs=[fade], outputs=[fade_accordion])
                bass_or_treble.change(fn=visible, inputs=[bass_or_treble], outputs=[bass_treble_accordion])
                limiter.change(fn=visible, inputs=[limiter], outputs=[limiter_accordion])
                resample_checkbox.change(fn=visible, inputs=[resample_checkbox], outputs=[audio_effect_resample_sr])
            with gr.Row():
                distortion_checkbox.change(fn=visible, inputs=[distortion_checkbox], outputs=[distortion_drive_db])
                gain_checkbox.change(fn=visible, inputs=[gain_checkbox], outputs=[gain_db])
                clipping_checkbox.change(fn=visible, inputs=[clipping_checkbox], outputs=[clipping_threashold_db])
                bitcrush_checkbox.change(fn=visible, inputs=[bitcrush_checkbox], outputs=[bitcrush_bit_depth])
            with gr.Row():
                upload_audio.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[upload_audio], outputs=[audio_in_path])
                audio_in_path.change(fn=lambda audio: audio if audio else None, inputs=[audio_in_path], outputs=[audio_play_input])
                audio_effects_refesh.click(fn=lambda a, b: [change_audios_choices(a), change_audios_choices(b)], inputs=[audio_in_path, audio_combination_input], outputs=[audio_in_path, audio_combination_input])
            with gr.Row():
                more_options.change(fn=lambda: [False]*8, inputs=[], outputs=[fade, bass_or_treble, limiter, resample_checkbox, distortion_checkbox, gain_checkbox, clipping_checkbox, bitcrush_checkbox])
                audio_combination.change(fn=visible, inputs=[audio_combination], outputs=[audio_combination_input])
                audio_combination.change(fn=lambda a: [visible(a)]*2, inputs=[audio_combination], outputs=[main_vol, combine_vol])
            with gr.Row():
                apply_effects_button.click(
                    fn=audio_effects,
                    inputs=[
                        audio_in_path, 
                        audio_out_path, 
                        resample_checkbox, 
                        audio_effect_resample_sr, 
                        chorus_depth, 
                        chorus_rate_hz, 
                        chorus_mix, 
                        chorus_centre_delay_ms, 
                        chorus_feedback, 
                        distortion_drive_db, 
                        reverb_room_size, 
                        reverb_damping, 
                        reverb_wet_level, 
                        reverb_dry_level, 
                        reverb_width, 
                        reverb_freeze_mode, 
                        pitch_shift_semitones, 
                        delay_second, 
                        delay_feedback, 
                        delay_mix, 
                        compressor_threashold_db, 
                        compressor_ratio, 
                        compressor_attack_ms, 
                        compressor_release_ms, 
                        limiter_threashold_db, 
                        limiter_release_ms, 
                        gain_db, 
                        bitcrush_bit_depth, 
                        clipping_threashold_db, 
                        phaser_rate_hz, 
                        phaser_depth, 
                        phaser_centre_frequency_hz, 
                        phaser_feedback, 
                        phaser_mix, 
                        bass_boost, 
                        bass_frequency, 
                        treble_boost, 
                        treble_frequency, 
                        fade_in, 
                        fade_out, 
                        audio_output_format, 
                        chorus_check_box, 
                        distortion_checkbox, 
                        reverb_check_box, 
                        delay_check_box, 
                        compressor_check_box, 
                        limiter, 
                        gain_checkbox, 
                        bitcrush_checkbox, 
                        clipping_checkbox, 
                        phaser_check_box, 
                        bass_or_treble, 
                        fade,
                        audio_combination,
                        audio_combination_input,
                        main_vol,
                        combine_vol
                    ],
                    outputs=[audio_play_output],
                    api_name="audio_effects"
                )

        with gr.TabItem(translations["createdataset"], visible=configs.get("create_dataset_tab", True)):
            gr.Markdown(translations["create_dataset_markdown"])
            with gr.Row():
                gr.Markdown(translations["create_dataset_markdown_2"])
            with gr.Row():
                dataset_url = gr.Textbox(label=translations["url_audio"], info=translations["create_dataset_url"], value="", placeholder="https://www.youtube.com/...", interactive=True)
                output_dataset = gr.Textbox(label=translations["output_data"], info=translations["output_data_info"], value="dataset", placeholder="dataset", interactive=True)
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            separator_reverb = gr.Checkbox(label=translations["dereveb_audio"], value=False, interactive=True)
                            denoise_mdx = gr.Checkbox(label=translations["denoise"], value=False, interactive=True)
                        with gr.Row():
                            kim_vocal_version = gr.Radio(label=translations["model_ver"], info=translations["model_ver_info"], choices=["Version-1", "Version-2"], value="Version-2", interactive=True)
                            kim_vocal_overlap = gr.Radio(label=translations["overlap"], info=translations["overlap_info"], choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True)
                        with gr.Row():    
                            kim_vocal_hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=8192, value=1024, step=1, interactive=True)
                            kim_vocal_batch_size = gr.Slider(label=translations["batch_size"], info=translations["mdx_batch_size_info"], minimum=1, maximum=64, value=1, step=1, interactive=True) 
                        with gr.Row():
                            kim_vocal_segments_size = gr.Slider(label=translations["segments_size"], info=translations["segments_size_info"], minimum=32, maximum=3072, value=256, step=32, interactive=True)
                        with gr.Row():
                            sample_rate0 = gr.Radio(choices=sample_rate_choice, value=44100, label=translations["sr"], info=translations["sr_info"], interactive=True)
                with gr.Column():
                    create_button = gr.Button(translations["createdataset"], variant="primary", scale=2, min_width=4000)
                    with gr.Group():
                        with gr.Row():
                            clean_audio = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                            skip = gr.Checkbox(label=translations["skip"], value=False, interactive=True)
                        with gr.Row():   
                            dataset_clean_strength = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label=translations["clean_strength"], info=translations["clean_strength_info"], interactive=True, visible=clean_audio.value)
                        with gr.Row():
                            skip_start = gr.Textbox(label=translations["skip_start"], info=translations["skip_start_info"], value="", placeholder="0,...", interactive=True, visible=skip.value)
                            skip_end = gr.Textbox(label=translations["skip_end"], info=translations["skip_end_info"], value="", placeholder="0,...", interactive=True, visible=skip.value)
                    create_dataset_info = gr.Textbox(label=translations["create_dataset_info"], value="", interactive=False)
            with gr.Row():
                clean_audio.change(fn=visible, inputs=[clean_audio], outputs=[dataset_clean_strength])
                skip.change(fn=lambda a: [valueEmpty_visible1(a)]*2, inputs=[skip], outputs=[skip_start, skip_end])
            with gr.Row():
                create_button.click(
                    fn=create_dataset,
                    inputs=[
                        dataset_url, 
                        output_dataset, 
                        clean_audio, 
                        dataset_clean_strength, 
                        separator_reverb, 
                        kim_vocal_version, 
                        kim_vocal_overlap, 
                        kim_vocal_segments_size, 
                        denoise_mdx, 
                        skip, 
                        skip_start, 
                        skip_end,
                        kim_vocal_hop_length,
                        kim_vocal_batch_size,
                        sample_rate0
                    ],
                    outputs=[create_dataset_info],
                    api_name="create_dataset"
                )

        with gr.TabItem(translations["training_model"], visible=configs.get("training_tab", True)):
            gr.Markdown(f"## {translations['training_model']}")
            with gr.Row():
                gr.Markdown(translations["training_markdown"])
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            training_name = gr.Textbox(label=translations["modelname"], info=translations["training_model_name"], value="", placeholder=translations["modelname"], interactive=True)
                            training_sr = gr.Radio(label=translations["sample_rate"], info=translations["sample_rate_info"], choices=["32k", "40k", "48k"], value="48k", interactive=True) 
                            training_ver = gr.Radio(label=translations["training_version"], info=translations["training_version_info"], choices=["v1", "v2"], value="v2", interactive=True) 
                            with gr.Row():
                                clean_dataset = gr.Checkbox(label=translations["clear_dataset"], value=False, interactive=True)
                                preprocess_cut = gr.Checkbox(label=translations["split_audio"], value=True, interactive=True)
                                process_effects = gr.Checkbox(label=translations["preprocess_effect"], value=False, interactive=True)
                                checkpointing1 = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)
                                training_f0 = gr.Checkbox(label=translations["training_pitch"], value=True, interactive=True)
                                upload = gr.Checkbox(label=translations["upload_dataset"], value=False, interactive=True)
                            with gr.Row():
                                clean_dataset_strength = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.7, step=0.1, interactive=True, visible=clean_dataset.value)
                        with gr.Column():
                            preprocess_button = gr.Button(translations["preprocess_button"], scale=2)
                            upload_dataset = gr.Files(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"], visible=upload.value)
                            preprocess_info = gr.Textbox(label=translations["preprocess_info"], value="", interactive=False)
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion(label=translations["f0_method"], open=False):
                                with gr.Group():
                                    with gr.Row():
                                        onnx_f0_mode2 = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                                        unlock_full_method4 = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                                    extract_method = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0, value="rmvpe", interactive=True)
                                extract_hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                            with gr.Accordion(label=translations["hubert_model"], open=False):
                                with gr.Group():
                                    embed_mode2 = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                                    extract_embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                                with gr.Row():
                                    extract_embedders_custom = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=extract_embedders.value == "custom")
                        with gr.Column():
                            extract_button = gr.Button(translations["extract_button"], scale=2)
                            extract_info = gr.Textbox(label=translations["extract_info"], value="", interactive=False)
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            total_epochs = gr.Slider(label=translations["total_epoch"], info=translations["total_epoch_info"], minimum=1, maximum=10000, value=300, step=1, interactive=True)
                            save_epochs = gr.Slider(label=translations["save_epoch"], info=translations["save_epoch_info"], minimum=1, maximum=10000, value=50, step=1, interactive=True)
                        with gr.Column():
                            index_button = gr.Button(f"3. {translations['create_index']}", variant="primary", scale=2)
                            training_button = gr.Button(f"4. {translations['training_model']}", variant="primary", scale=2)
                    with gr.Row():
                        with gr.Accordion(label=translations["setting"], open=False):
                            with gr.Row():
                                index_algorithm = gr.Radio(label=translations["index_algorithm"], info=translations["index_algorithm_info"], choices=["Auto", "Faiss", "KMeans"], value="Auto", interactive=True)
                            with gr.Row():
                                custom_dataset = gr.Checkbox(label=translations["custom_dataset"], info=translations["custom_dataset_info"], value=False, interactive=True)
                                overtraining_detector = gr.Checkbox(label=translations["overtraining_detector"], info=translations["overtraining_detector_info"], value=False, interactive=True)
                                clean_up = gr.Checkbox(label=translations["cleanup_training"], info=translations["cleanup_training_info"], value=False, interactive=True)
                                cache_in_gpu = gr.Checkbox(label=translations["cache_in_gpu"], info=translations["cache_in_gpu_info"], value=True, interactive=True)
                            with gr.Column():
                                dataset_path = gr.Textbox(label=translations["dataset_folder"], value="dataset", interactive=True, visible=custom_dataset.value)
                            with gr.Column():
                                threshold = gr.Slider(minimum=1, maximum=100, value=50, step=1, label=translations["threshold"], interactive=True, visible=overtraining_detector.value)
                                with gr.Accordion(translations["setting_cpu_gpu"], open=False):
                                    with gr.Column():
                                        gpu_number = gr.Textbox(label=translations["gpu_number"], value=str("-".join(map(str, range(torch.cuda.device_count()))) if torch.cuda.is_available() else "-"), info=translations["gpu_number_info"], interactive=True)
                                        gpu_info = gr.Textbox(label=translations["gpu_info"], value=get_gpu_info(), info=translations["gpu_info_2"], interactive=False)
                                        cpu_core = gr.Slider(label=translations["cpu_core"], info=translations["cpu_core_info"], minimum=0, maximum=cpu_count(), value=cpu_count(), step=1, interactive=True)          
                                        train_batch_size = gr.Slider(label=translations["batch_size"], info=translations["batch_size_info"], minimum=1, maximum=64, value=8, step=1, interactive=True)
                            with gr.Row():
                                save_only_latest = gr.Checkbox(label=translations["save_only_latest"], info=translations["save_only_latest_info"], value=True, interactive=True)
                                save_every_weights = gr.Checkbox(label=translations["save_every_weights"], info=translations["save_every_weights_info"], value=True, interactive=True)
                                not_use_pretrain = gr.Checkbox(label=translations["not_use_pretrain_2"], info=translations["not_use_pretrain_info"], value=False, interactive=True)
                                custom_pretrain = gr.Checkbox(label=translations["custom_pretrain"], info=translations["custom_pretrain_info"], value=False, interactive=True)
                            with gr.Row():
                                vocoders = gr.Radio(label=translations["vocoder"], info=translations["vocoder_info"], choices=["Default", "MRF-HiFi-GAN", "RefineGAN"], value="Default", interactive=True) 
                            with gr.Row():
                                deterministic = gr.Checkbox(label=translations["deterministic"], info=translations["deterministic_info"], value=False, interactive=True)
                                benchmark = gr.Checkbox(label=translations["benchmark"], info=translations["benchmark_info"], value=False, interactive=True)
                            with gr.Row():
                                optimizer = gr.Radio(label=translations["optimizer"], info=translations["optimizer_info"], value="AdamW", choices=["AdamW", "RAdam"], interactive=True)
                            with gr.Row():
                                model_author = gr.Textbox(label=translations["training_author"], info=translations["training_author_info"], value="", placeholder=translations["training_author"], interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    with gr.Accordion(translations["custom_pretrain_info"], open=False, visible=custom_pretrain.value and not not_use_pretrain.value) as pretrain_setting:
                                        pretrained_D = gr.Dropdown(label=translations["pretrain_file"].format(dg="D"), choices=pretrainedD, value=pretrainedD[0] if len(pretrainedD) > 0 else '', interactive=True, allow_custom_value=True)
                                        pretrained_G = gr.Dropdown(label=translations["pretrain_file"].format(dg="G"), choices=pretrainedG, value=pretrainedG[0] if len(pretrainedG) > 0 else '', interactive=True, allow_custom_value=True)
                                        refesh_pretrain = gr.Button(translations["refesh"], scale=2)
                    with gr.Row():
                        training_info = gr.Textbox(label=translations["train_info"], value="", interactive=False)
                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion(translations["export_model"], open=False):
                                with gr.Row():
                                    model_file= gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                                    index_file = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                                with gr.Row():
                                    refesh_file = gr.Button(f"1. {translations['refesh']}", scale=2)
                                    zip_model = gr.Button(translations["zip_model"], variant="primary", scale=2)
                                with gr.Row():
                                    zip_output = gr.File(label=translations["output_zip"], file_types=[".zip"], interactive=False, visible=False)
            with gr.Row():
                vocoders.change(fn=pitch_guidance_lock, inputs=[vocoders], outputs=[training_f0])
                training_f0.change(fn=vocoders_lock, inputs=[training_f0, vocoders], outputs=[vocoders])
                unlock_full_method4.change(fn=unlock_f0, inputs=[unlock_full_method4], outputs=[extract_method])
            with gr.Row():
                refesh_file.click(fn=change_models_choices, inputs=[], outputs=[model_file, index_file]) 
                zip_model.click(fn=zip_file, inputs=[training_name, model_file, index_file], outputs=[zip_output])                
                dataset_path.change(fn=lambda folder: os.makedirs(folder, exist_ok=True), inputs=[dataset_path], outputs=[])
            with gr.Row():
                upload.change(fn=visible, inputs=[upload], outputs=[upload_dataset]) 
                overtraining_detector.change(fn=visible, inputs=[overtraining_detector], outputs=[threshold]) 
                clean_dataset.change(fn=visible, inputs=[clean_dataset], outputs=[clean_dataset_strength])
            with gr.Row():
                custom_dataset.change(fn=lambda custom_dataset: [visible(custom_dataset), "dataset"],inputs=[custom_dataset], outputs=[dataset_path, dataset_path])
                training_ver.change(fn=unlock_vocoder, inputs=[training_ver, vocoders], outputs=[vocoders])
                vocoders.change(fn=unlock_ver, inputs=[training_ver, vocoders], outputs=[training_ver])
                upload_dataset.upload(
                    fn=lambda files, folder: [shutil.move(f.name, os.path.join(folder, os.path.split(f.name)[1])) for f in files] if folder != "" else gr_warning(translations["dataset_folder1"]),
                    inputs=[upload_dataset, dataset_path], 
                    outputs=[], 
                    api_name="upload_dataset"
                )           
            with gr.Row():
                not_use_pretrain.change(fn=lambda a, b: visible(a and not b), inputs=[custom_pretrain, not_use_pretrain], outputs=[pretrain_setting])
                custom_pretrain.change(fn=lambda a, b: visible(a and not b), inputs=[custom_pretrain, not_use_pretrain], outputs=[pretrain_setting])
                refesh_pretrain.click(fn=change_pretrained_choices, inputs=[], outputs=[pretrained_D, pretrained_G])
            with gr.Row():
                preprocess_button.click(
                    fn=preprocess,
                    inputs=[
                        training_name, 
                        training_sr, 
                        cpu_core,
                        preprocess_cut, 
                        process_effects,
                        dataset_path,
                        clean_dataset,
                        clean_dataset_strength
                    ],
                    outputs=[preprocess_info],
                    api_name="preprocess"
                )
            with gr.Row():
                embed_mode2.change(fn=visible_embedders, inputs=[embed_mode2], outputs=[extract_embedders])
                extract_method.change(fn=hoplength_show, inputs=[extract_method], outputs=[extract_hop_length])
                extract_embedders.change(fn=lambda extract_embedders: visible(extract_embedders == "custom"), inputs=[extract_embedders], outputs=[extract_embedders_custom])
            with gr.Row():
                extract_button.click(
                    fn=extract,
                    inputs=[
                        training_name, 
                        training_ver, 
                        extract_method, 
                        training_f0, 
                        extract_hop_length, 
                        cpu_core,
                        gpu_number,
                        training_sr, 
                        extract_embedders, 
                        extract_embedders_custom,
                        onnx_f0_mode2,
                        embed_mode2
                    ],
                    outputs=[extract_info],
                    api_name="extract"
                )
            with gr.Row():
                index_button.click(
                    fn=create_index,
                    inputs=[
                        training_name, 
                        training_ver, 
                        index_algorithm
                    ],
                    outputs=[training_info],
                    api_name="create_index"
                )
            with gr.Row():
                training_button.click(
                    fn=training,
                    inputs=[
                        training_name, 
                        training_ver, 
                        save_epochs, 
                        save_only_latest, 
                        save_every_weights, 
                        total_epochs, 
                        training_sr,
                        train_batch_size, 
                        gpu_number,
                        training_f0,
                        not_use_pretrain,
                        custom_pretrain,
                        pretrained_G,
                        pretrained_D,
                        overtraining_detector,
                        threshold,
                        clean_up,
                        cache_in_gpu,
                        model_author,
                        vocoders,
                        checkpointing1,
                        deterministic, 
                        benchmark,
                        optimizer
                    ],
                    outputs=[training_info],
                    api_name="training_model"
                )

        with gr.TabItem(translations["fushion"], visible=configs.get("fushion_tab", True)):
            gr.Markdown(translations["fushion_markdown"])
            with gr.Row():
                gr.Markdown(translations["fushion_markdown_2"])
            with gr.Row():
                name_to_save = gr.Textbox(label=translations["modelname"], placeholder="Model.pth", value="", max_lines=1, interactive=True)
            with gr.Row():
                fushion_button = gr.Button(translations["fushion"], variant="primary", scale=4)
            with gr.Column():
                with gr.Row():
                    model_a = gr.File(label=f"{translations['model_name']} 1", file_types=[".pth", ".onnx"]) 
                    model_b = gr.File(label=f"{translations['model_name']} 2", file_types=[".pth", ".onnx"])
                with gr.Row():
                    model_path_a = gr.Textbox(label=f"{translations['model_path']} 1", value="", placeholder="assets/weights/Model_1.pth")
                    model_path_b = gr.Textbox(label=f"{translations['model_path']} 2", value="", placeholder="assets/weights/Model_2.pth")
            with gr.Row():
                ratio = gr.Slider(minimum=0, maximum=1, label=translations["model_ratio"], info=translations["model_ratio_info"], value=0.5, interactive=True)
            with gr.Row():
                output_model = gr.File(label=translations["output_model_path"], file_types=[".pth", ".onnx"], interactive=False, visible=False)
            with gr.Row():
                model_a.upload(fn=lambda model: shutil.move(model.name, os.path.join("assets", "weights")), inputs=[model_a], outputs=[model_path_a])
                model_b.upload(fn=lambda model: shutil.move(model.name, os.path.join("assets", "weights")), inputs=[model_b], outputs=[model_path_b])
            with gr.Row():
                fushion_button.click(
                    fn=fushion_model,
                    inputs=[
                        name_to_save, 
                        model_path_a, 
                        model_path_b, 
                        ratio
                    ],
                    outputs=[name_to_save, output_model],
                    api_name="fushion_model"
                )
                fushion_button.click(fn=lambda: visible(True), inputs=[], outputs=[output_model])  

        with gr.TabItem(translations["read_model"], visible=configs.get("read_tab", True)):
            gr.Markdown(translations["read_model_markdown"])
            with gr.Row():
                gr.Markdown(translations["read_model_markdown_2"])
            with gr.Row():
                model = gr.File(label=translations["drop_model"], file_types=[".pth", ".onnx"]) 
            with gr.Row():
                read_button = gr.Button(translations["readmodel"], variant="primary", scale=2)
            with gr.Column():
                model_path = gr.Textbox(label=translations["model_path"], value="", placeholder="assets/weights/Model.pth", info=translations["model_path_info"], interactive=True)
                output_info = gr.Textbox(label=translations["modelinfo"], value="", interactive=False, scale=6)
            with gr.Row():
                model.upload(fn=lambda model: shutil.move(model.name, os.path.join("assets", "weights")), inputs=[model], outputs=[model_path])
                read_button.click(
                    fn=model_info,
                    inputs=[model_path],
                    outputs=[output_info],
                    api_name="read_model"
                )

        with gr.TabItem(translations["convert_model"], visible=configs.get("onnx_tab", True)):
            gr.Markdown(translations["pytorch2onnx"])
            with gr.Row():
                gr.Markdown(translations["pytorch2onnx_markdown"])
            with gr.Row():
                model_pth_upload = gr.File(label=translations["drop_model"], file_types=[".pth"]) 
            with gr.Row():
                convert_onnx = gr.Button(translations["convert_model"], variant="primary", scale=2)
            with gr.Row():
                model_pth_path = gr.Textbox(label=translations["model_path"], value="", placeholder="assets/weights/Model.pth", info=translations["model_path_info"], interactive=True)
            with gr.Row():
                output_model2 = gr.File(label=translations["output_model_path"], file_types=[".pth", ".onnx"], interactive=False, visible=False)
            with gr.Row():
                model_pth_upload.upload(fn=lambda model_pth_upload: shutil.move(model_pth_upload.name, os.path.join("assets", "weights")), inputs=[model_pth_upload], outputs=[model_pth_path])
                convert_onnx.click(
                    fn=onnx_export,
                    inputs=[model_pth_path],
                    outputs=[output_model2, output_info],
                    api_name="model_onnx_export"
                )
                convert_onnx.click(fn=lambda: visible(True), inputs=[], outputs=[output_model2])  

        with gr.TabItem(translations["downloads"], visible=configs.get("downloads_tab", True)):
            gr.Markdown(translations["download_markdown"])
            with gr.Row():
                gr.Markdown(translations["download_markdown_2"])
            with gr.Row():
                with gr.Accordion(translations["model_download"], open=True):
                    with gr.Row():
                        downloadmodel = gr.Radio(label=translations["model_download_select"], choices=[translations["download_url"], translations["download_from_csv"], translations["search_models"], translations["upload"]], interactive=True, value=translations["download_url"])
                    with gr.Row():
                        gr.Markdown("___")
                    with gr.Column():
                        with gr.Row():
                            url_input = gr.Textbox(label=translations["model_url"], value="", placeholder="https://...", scale=6)
                            download_model_name = gr.Textbox(label=translations["modelname"], value="", placeholder=translations["modelname"], scale=2)
                        url_download = gr.Button(value=translations["downloads"], scale=2)
                    with gr.Column():
                        model_browser = gr.Dropdown(choices=models.keys(), label=translations["model_warehouse"], scale=8, allow_custom_value=True, visible=False)
                        download_from_browser = gr.Button(value=translations["get_model"], scale=2, variant="primary", visible=False)
                    with gr.Column():
                        search_name = gr.Textbox(label=translations["name_to_search"], placeholder=translations["modelname"], interactive=True, scale=8, visible=False)
                        search = gr.Button(translations["search_2"], scale=2, visible=False)
                        search_dropdown = gr.Dropdown(label=translations["select_download_model"], value="", choices=[], allow_custom_value=True, interactive=False, visible=False)
                        download = gr.Button(translations["downloads"], variant="primary", visible=False)
                    with gr.Column():
                        model_upload = gr.File(label=translations["drop_model"], file_types=[".pth", ".onnx", ".index", ".zip"], visible=False)
            with gr.Row():
                with gr.Accordion(translations["download_pretrained_2"], open=False):
                    with gr.Row():
                        pretrain_download_choices = gr.Radio(label=translations["model_download_select"], choices=[translations["download_url"], translations["list_model"], translations["upload"]], value=translations["download_url"], interactive=True)  
                    with gr.Row():
                        gr.Markdown("___")
                    with gr.Column():
                        with gr.Row():
                            pretrainD = gr.Textbox(label=translations["pretrained_url"].format(dg="D"), value="", info=translations["only_huggingface"], placeholder="https://...", interactive=True, scale=4)
                            pretrainG = gr.Textbox(label=translations["pretrained_url"].format(dg="G"), value="", info=translations["only_huggingface"], placeholder="https://...", interactive=True, scale=4)
                        download_pretrain_button = gr.Button(translations["downloads"], scale=2)
                    with gr.Column():
                        with gr.Row():
                            pretrain_choices = gr.Dropdown(label=translations["select_pretrain"], info=translations["select_pretrain_info"], choices=list(fetch_pretrained_data().keys()), value="Titan_Medium", allow_custom_value=True, interactive=True, scale=6, visible=False)
                            sample_rate_pretrain = gr.Dropdown(label=translations["pretrain_sr"], info=translations["pretrain_sr"], choices=["48k", "40k", "32k"], value="48k", interactive=True, visible=False)
                        download_pretrain_choices_button = gr.Button(translations["downloads"], scale=2, variant="primary", visible=False)
                    with gr.Row():
                        pretrain_upload_g = gr.File(label=translations["drop_pretrain"].format(dg="G"), file_types=[".pth"], visible=False)
                        pretrain_upload_d = gr.File(label=translations["drop_pretrain"].format(dg="D"), file_types=[".pth"], visible=False)
            with gr.Row():
                url_download.click(
                    fn=download_model, 
                    inputs=[
                        url_input, 
                        download_model_name
                    ], 
                    outputs=[url_input],
                    api_name="download_model"
                )
                download_from_browser.click(
                    fn=lambda model: download_model(models[model], model), 
                    inputs=[model_browser], 
                    outputs=[model_browser],
                    api_name="download_browser"
                )
            with gr.Row():
                downloadmodel.change(fn=change_download_choices, inputs=[downloadmodel], outputs=[url_input, download_model_name, url_download, model_browser, download_from_browser, search_name, search, search_dropdown, download, model_upload])
                search.click(fn=search_models, inputs=[search_name], outputs=[search_dropdown, download])
                model_upload.upload(fn=save_drop_model, inputs=[model_upload], outputs=[model_upload])
                download.click(
                    fn=lambda model: download_model(model_options[model], model), 
                    inputs=[search_dropdown], 
                    outputs=[search_dropdown],
                    api_name="search_models"
                )
            with gr.Row():
                pretrain_download_choices.change(fn=change_download_pretrained_choices, inputs=[pretrain_download_choices], outputs=[pretrainD, pretrainG, download_pretrain_button, pretrain_choices, sample_rate_pretrain, download_pretrain_choices_button, pretrain_upload_d, pretrain_upload_g])
                pretrain_choices.change(fn=update_sample_rate_dropdown, inputs=[pretrain_choices], outputs=[sample_rate_pretrain])
            with gr.Row():
                download_pretrain_button.click(
                    fn=download_pretrained_model,
                    inputs=[
                        pretrain_download_choices, 
                        pretrainD, 
                        pretrainG
                    ],
                    outputs=[pretrainD],
                    api_name="download_pretrain_link"
                )
                download_pretrain_choices_button.click(
                    fn=download_pretrained_model,
                    inputs=[
                        pretrain_download_choices, 
                        pretrain_choices, 
                        sample_rate_pretrain
                    ],
                    outputs=[pretrain_choices],
                    api_name="download_pretrain_choices"
                )
                pretrain_upload_g.upload(
                    fn=lambda pretrain_upload_g: shutil.move(pretrain_upload_g.name, os.path.join("assets", "models", "pretrained_custom")), 
                    inputs=[pretrain_upload_g], 
                    outputs=[],
                    api_name="upload_pretrain_g"
                )
                pretrain_upload_d.upload(
                    fn=lambda pretrain_upload_d: shutil.move(pretrain_upload_d.name, os.path.join("assets", "models", "pretrained_custom")), 
                    inputs=[pretrain_upload_d], 
                    outputs=[],
                    api_name="upload_pretrain_d"
                )

        with gr.TabItem(translations["f0_extractor_tab"], visible=configs.get("f0_extractor_tab", True)):
            gr.Markdown(translations["f0_extractor_markdown"])
            with gr.Row():
                gr.Markdown(translations["f0_extractor_markdown_2"])
            with gr.Row():
                extractor_button = gr.Button(translations["extract_button"].replace("2. ", ""), variant="primary")
            with gr.Row():
                with gr.Column():
                    upload_audio_file = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])
                    audioplay = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
                with gr.Column():
                    with gr.Accordion(translations["f0_method"], open=False):
                        with gr.Group():
                            onnx_f0_mode3 = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                            f0_method_extract = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0, value="rmvpe", interactive=True)
                    with gr.Accordion(translations["audio_path"], open=True):
                        input_audio_path = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, allow_custom_value=True, interactive=True)
                        refesh_audio_button = gr.Button(translations["refesh"])
            with gr.Row():
                gr.Markdown("___")
            with gr.Row():
                file_output = gr.File(label="", file_types=[".txt"], interactive=False)
                image_output = gr.Image(label="", interactive=False, show_download_button=True)
            with gr.Row():
                upload_audio_file.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[upload_audio_file], outputs=[input_audio_path])
                input_audio_path.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio_path], outputs=[audioplay])
                refesh_audio_button.click(fn=change_audios_choices, inputs=[input_audio_path], outputs=[input_audio_path])
            with gr.Row():
                extractor_button.click(
                    fn=f0_extract,
                    inputs=[
                        input_audio_path,
                        f0_method_extract,
                        onnx_f0_mode3
                    ],
                    outputs=[file_output, image_output],
                    api_name="f0_extract"
                )

        with gr.TabItem(translations["settings"], visible=configs.get("settings_tab", True)):
            gr.Markdown(translations["settings_markdown"])
            with gr.Row():
                gr.Markdown(translations["settings_markdown_2"])
            with gr.Row():
                toggle_button = gr.Button(translations["change_light_dark"], variant="secondary", scale=2)
            with gr.Row():
                with gr.Column():
                    language_dropdown = gr.Dropdown(label=translations["lang"], interactive=True, info=translations["lang_restart"], choices=configs.get("support_language", "vi-VN"), value=language)
                    change_lang = gr.Button(translations["change_lang"], variant="primary", scale=2)
                with gr.Column():
                    theme_dropdown = gr.Dropdown(label=translations["theme"], interactive=True, info=translations["theme_restart"], choices=configs.get("themes", theme), value=theme, allow_custom_value=True)
                    changetheme = gr.Button(translations["theme_button"], variant="primary", scale=2)
            with gr.Row():
                with gr.Column():
                    fp_choice = gr.Radio(choices=["fp16","fp32"], value="fp16" if configs.get("fp16", False) else "fp32", label=translations["precision"], info=translations["precision_info"], interactive=True)
                    fp_button = gr.Button(translations["update_precision"], variant="secondary", scale=2)
                with gr.Column():
                    font_choice = gr.Textbox(label=translations["font"], info=translations["font_info"], value=font, interactive=True)
                    font_button = gr.Button(translations["change_font"])
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(translations["stop"], open=False):
                        separate_stop = gr.Button(translations["stop_separate"])
                        convert_stop = gr.Button(translations["stop_convert"])
                        create_dataset_stop = gr.Button(translations["stop_create_dataset"])
                        audioldm2_stop = gr.Button(translations["stop_audioldm2"])
                        with gr.Accordion(translations["stop_training"], open=False):
                            model_name_stop = gr.Textbox(label=translations["modelname"], info=translations["training_model_name"], value="", placeholder=translations["modelname"], interactive=True)
                            preprocess_stop = gr.Button(translations["stop_preprocess"])
                            extract_stop = gr.Button(translations["stop_extract"])
                            train_stop = gr.Button(translations["stop_training"])
            with gr.Row():
                toggle_button.click(fn=None, js="() => {document.body.classList.toggle('dark')}")
                fp_button.click(fn=change_fp, inputs=[fp_choice], outputs=[fp_choice])
            with gr.Row():
                change_lang.click(fn=change_language, inputs=[language_dropdown], outputs=[])
                changetheme.click(fn=change_theme, inputs=[theme_dropdown], outputs=[])
                font_button.click(fn=change_font, inputs=[font_choice], outputs=[])
            with gr.Row():
                change_lang.click(fn=None, js="setTimeout(function() {location.reload()}, 15000)", inputs=[], outputs=[])
                changetheme.click(fn=None, js="setTimeout(function() {location.reload()}, 15000)", inputs=[], outputs=[])
                font_button.click(fn=None, js="setTimeout(function() {location.reload()}, 15000)", inputs=[], outputs=[])
            with gr.Row():
                separate_stop.click(fn=lambda: stop_pid("separate_pid", None, False), inputs=[], outputs=[])
                convert_stop.click(fn=lambda: stop_pid("convert_pid", None, False), inputs=[], outputs=[])
                create_dataset_stop.click(fn=lambda: stop_pid("create_dataset_pid", None, False), inputs=[], outputs=[])
            with gr.Row():
                preprocess_stop.click(fn=lambda model_name_stop: stop_pid("preprocess_pid", model_name_stop, False), inputs=[model_name_stop], outputs=[])
                extract_stop.click(fn=lambda model_name_stop: stop_pid("extract_pid", model_name_stop, False), inputs=[model_name_stop], outputs=[])
                train_stop.click(fn=lambda model_name_stop: stop_pid("train_pid", model_name_stop, True), inputs=[model_name_stop], outputs=[])
            with gr.Row():
                audioldm2_stop.click(fn=lambda: stop_pid("audioldm2_pid", None, False), inputs=[], outputs=[])

        with gr.TabItem(translations["report_bugs"], visible=configs.get("report_bug_tab", True)):
            gr.Markdown(translations["report_bugs"])
            with gr.Row():
                gr.Markdown(translations["report_bug_info"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        agree_log = gr.Checkbox(label=translations["agree_log"], value=True, interactive=True) 
                        report_text = gr.Textbox(label=translations["error_info"], info=translations["error_info_2"], interactive=True)
                    report_button = gr.Button(translations["report_bugs"], variant="primary", scale=2)
            with gr.Row():
                gr.Markdown(translations["report_info"].format(github=codecs.decode("uggcf://tvguho.pbz/CunzUhlauNau16/Ivrganzrfr-EIP/vffhrf", "rot13")))
            with gr.Row():
                report_button.click(fn=report_bug, inputs=[report_text, agree_log], outputs=[])

    with gr.Row(): 
        gr.Markdown(translations["rick_roll"].format(rickroll=codecs.decode('uggcf://jjj.lbhghor.pbz/jngpu?i=qDj4j9JtKpD', 'rot13')))
    with gr.Row(): 
        gr.Markdown(translations["terms_of_use"])
    with gr.Row():
        gr.Markdown(translations["exemption"])

    logger.info(translations["start_app"])
    logger.info(translations["set_lang"].format(lang=language))

    port = configs.get("app_port", 7860)

    for i in range(configs.get("num_of_restart", 5)):
        try:
            app.queue().launch(
                favicon_path=os.path.join("assets", "ico.png"), 
                server_name=configs.get("server_name", "0.0.0.0"), 
                server_port=port, 
                show_error=configs.get("app_show_error", False), 
                inbrowser="--open" in sys.argv, 
                share="--share" in sys.argv, 
                allowed_paths=allow_disk
            )
            break
        except OSError:
            logger.debug(translations["port"].format(port=port))
            port -= 1
        except Exception as e:
            logger.error(translations["error_occurred"].format(e=e))
            sys.exit(1)