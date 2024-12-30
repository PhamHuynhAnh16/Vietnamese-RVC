import os
import re
import ssl
import sys
import json
import torch
import codecs
import shutil
import logging
import platform
import requests
import warnings
import threading
import logging.handlers

import gradio as gr
import pandas as pd

from time import sleep
from subprocess import Popen
from bs4 import BeautifulSoup
from datetime import datetime
from multiprocessing import cpu_count

sys.path.append(os.getcwd())
from main.configs.config import Config
from main.tools.huggingface import HF_download_file as hgf

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
for l in ["httpx", "gradio", "uvicorn", "httpcore"]:
    logging.getLogger(l).setLevel(logging.ERROR)

config = Config()
python = sys.executable = "python"
translations = config.translations
configs_json = os.path.join("main", "configs", "config.json")
configs = json.load(open(configs_json, "r"))

models, model_options = {}, {}
paths_for_files = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk("audios") for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])
model_name, index_path, delete_index = sorted(list(model for model in os.listdir(os.path.join("assets", "weights")) if model.endswith(".pth") and not model.startswith("G_") and not model.startswith("D_"))), sorted([os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".index")]), sorted([os.path.join("assets", "logs", f) for f in os.listdir(os.path.join("assets", "logs")) if "mute" not in f and os.path.isdir(os.path.join("assets", "logs", f))])
pretrainedD, pretrainedG = ([model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "D" in model], [model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "G" in model])
Allpretrained = sorted([os.path.join("assets", "models", path, model) for path in ["pretrained_v1", "pretrained_v2", "pretrained_custom"] for model in os.listdir(os.path.join("assets", "models", path)) if model.endswith(".pth") and ("D" in model or "G" in model)])
separate_model = sorted([os.path.join("assets", "models", "uvr5", models) for models in os.listdir(os.path.join("assets", "models", "uvr5")) if models.endswith((".th", ".yaml", ".onnx"))])
presets_file = sorted(list(f for f in os.listdir(os.path.join("assets", "presets")) if f.endswith(".json")))
language, theme, edge_tts, google_tts_voice = configs.get("language", "vi-VN"), configs.get("theme", "NoCrypt/miku"), configs.get("edge_tts", ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]), configs.get("google_tts_voice", ["vi", "en"])

if language == "vi-VN":
    import gradio.strings
    gradio.strings.en = {"RUNNING_LOCALLY": "Chạy với liên kết nội bộ:  {}", "RUNNING_LOCALLY_SEPARATED": "Chạy với liên kết nội bộ:  {}://{}:{}", "SHARE_LINK_DISPLAY": "Chạy với liên kết công khai: {}", "COULD_NOT_GET_SHARE_LINK": "\nKhông thể tạo liên kết công khai. Vui lòng kiểm tra kết nối mạng của bạn hoặc trang trạng thái của chúng tôi: https://status.gradio.app.", "COULD_NOT_GET_SHARE_LINK_MISSING_FILE": "\nKhông thể tạo liên kết công khai. Thiếu tập tin: {}. \n\nVui lòng kiểm tra kết nối mạng của bạn. Điều này có thể xảy ra nếu phần mềm chống vi-rút của bạn chặn việc tải xuống tệp này. Bạn có thể cài đặt thủ công bằng cách làm theo các bước sau: \n\n1. Tải tập tin này: {}\n2. Đổi tên tệp đã tải xuống thành: {}\n3. Di chuyển tập tin đến vị trí này: {}", "COLAB_NO_LOCAL": "Không thể hiển thị giao diện cục bộ trên google colab, liên kết công khai đã được tạo.", "PUBLIC_SHARE_TRUE": "\nĐể tạo một liên kết công khai, hãy đặt `share=True` trong `launch()`.", "MODEL_PUBLICLY_AVAILABLE_URL": "Mô hình được cung cấp công khai tại: {} (có thể mất tới một phút để sử dụng được liên kết)", "GENERATING_PUBLIC_LINK": "Đang tạo liên kết công khai (có thể mất vài giây...):", "BETA_INVITE": "\nCảm ơn bạn đã là người dùng Gradio! Nếu bạn có thắc mắc hoặc phản hồi, vui lòng tham gia máy chủ Discord của chúng tôi và trò chuyện với chúng tôi: https://discord.gg/feTf9x3ZSB", "COLAB_DEBUG_TRUE": "Đã phát hiện thấy sổ tay Colab. Ô này sẽ chạy vô thời hạn để bạn có thể xem lỗi và nhật ký. " "Để tắt, hãy đặt debug=False trong launch().", "COLAB_DEBUG_FALSE": "Đã phát hiện thấy sổ tay Colab. Để hiển thị lỗi trong sổ ghi chép colab, hãy đặt debug=True trong launch()", "COLAB_WARNING": "Lưu ý: việc mở Chrome Inspector có thể làm hỏng bản demo trong sổ tay Colab.", "SHARE_LINK_MESSAGE": "\nLiên kết chia sẻ này sẽ hết hạn sau 72 giờ. Để nâng cấp GPU và lưu trữ vĩnh viễn miễn phí, hãy chạy `gradio deploy` từ Terminal để triển khai lên Spaces (https://huggingface.co/spaces)", "INLINE_DISPLAY_BELOW": "Giao diện tải bên dưới..."}

miku_image = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Pbyno_EIP_Cebwrpg_2/erfbyir/znva/zvxh.cat", "rot13")
csv_path = os.path.join("assets", "spreadsheet.csv")

if not os.path.exists(os.path.join("assets", "miku.png")): hgf(miku_image, os.path.join("assets", "miku.png"))
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
    gr.Info(message, duration=5)
    logger.info(message)

def gr_warning(message):
    gr.Warning(message, duration=5)
    logger.warning(message)

def gr_error(message):
    gr.Error(message=message, duration=15)
    logger.error(message)

def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = [f"{i}: {torch.cuda.get_device_name(i)} ({int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4)} GB)" for i in range(ngpu) if torch.cuda.is_available() or ngpu != 0]
    return "\n".join(gpu_infos) if len(gpu_infos) > 0 else translations["no_support_gpu"]

def change_audios_choices(): 
    paths_for_files = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk("audios") for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])
    return {"value": "" if len(paths_for_files) < 1 else paths_for_files[0], "choices": [] if len(paths_for_files) < 1 else paths_for_files, "__type__": "update"}

def change_separate_choices():
    return [{"choices": sorted([os.path.join("assets", "models", "uvr5", models) for models in os.listdir(os.path.join("assets", "models", "uvr5")) if model.endswith((".th", ".yaml", ".onnx"))]), "__type__": "update"}]

def change_models_choices():
    return [{"choices": sorted(list(model for model in os.listdir(os.path.join("assets", "weights")) if model.endswith(".pth") and not model.startswith("G_") and not model.startswith("D_"))), "__type__": "update"}, {"choices": sorted([os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".index")]), "__type__": "update"}]

def change_allpretrained_choices():
    return [{"choices": sorted([os.path.join("assets", "models", path, model) for path in ["pretrained_v1", "pretrained_v2", "pretrained_custom"] for model in os.listdir(os.path.join("assets", "models", path)) if model.endswith(".pth") and ("D" in model or "G" in model)]), "__type__": "update"}]

def change_model_separate_choices(mdx):
    choices = configs.get("demucs_model", "HD_MMI") if not mdx else configs.get("mdx_model", "MDXNET_Main")
    return {"value": choices[0], "choices": choices, "__type__": "update"}

def change_pretrained_choices():
    return [{"choices": sorted([model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "D" in model]), "__type__": "update"}, {"choices": sorted([model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "G" in model]), "__type__": "update"}]

def change_choices_del():
    return [{"choices": sorted(list(model for model in os.listdir(os.path.join("assets", "weights")) if model.endswith(".pth") and not model.startswith("G_") and not model.startswith("D_"))), "__type__": "update"}, {"choices": sorted([os.path.join("assets", "logs", f) for f in os.listdir(os.path.join("assets", "logs")) if "mute" not in f and os.path.isdir(os.path.join("assets", "logs", f))]), "__type__": "update"}]

def change_preset_choices():
    presets = sorted(list(f for f in os.listdir(os.path.join("assets", "presets")) if f.endswith(".json")))
    return {"value": "" if len(presets) < 1 else presets[0], "choices": [] if len(presets) < 1 else presets, "__type__": "update"}

def change_tts_voice_choices(google):
    return {"choices": google_tts_voice if google else edge_tts, "value": google_tts_voice[0] if google else edge_tts[0], "__type__": "update"}

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

def get_sid(model):
    try:
        speakers_id = torch.load(os.path.join("assets", "weights", model), map_location="cpu").get("speakers_id")
        return {"choices": list(range(speakers_id)) if speakers_id else [0], "value": 0, "__type__": "update"} 
    except:
        return {"choices": [0], "value": 0, "__type__": "update"}

def get_index(model):
    return {"value": next((f for f in [os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".index")] if model.split(".")[0] in f), ""), "__type__": "update"}

def index_strength_show(index):
    return {"visible": index != "" and os.path.exists(index), "value": 0.5, "__type__": "update"}

def hoplength_show(method, hybrid_method=None):
    if method in ["crepe-tiny", "crepe", "fcpe"]: visible = True
    elif method == "hybrid":
        methods_str = re.search("hybrid\[(.+)\]", hybrid_method)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        visible = methods[0] in ["crepe-tiny", "crepe", "fcpe"] or methods[1] in ["crepe-tiny", "crepe", "fcpe"]
    else: visible = False
    
    return {"visible": visible, "__type__": "update"}

def visible_1(value):
    return {"visible": value, "__type__": "update"}

def valueFalse_interactive1(inp): 
    return {"value": False, "interactive": inp, "__type__": "update"}

def valueFalse_interactive2(inp1, inp2): 
    return {"value": False, "interactive": inp1 and inp2, "__type__": "update"}

def valueFalse_visible1(inp1): 
    return {"value": False, "visible": inp1, "__type__": "update"}

def valueEmpty_visible1(inp1): 
    return {"value": "", "visible": inp1, "__type__": "update"}

def process_input(file_path):
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
    os.system(f"{python} {os.path.join('main', 'app', 'app.py')} --share")

def change_language(lang):
    with open(configs_json, "r") as f:
        configs = json.load(f)

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

def change_fp(fp):
    gr_info(translations["fp_select"])
    config.set_precision(fp)
    gr_info(translations["fp_select_2"].format(fp=fp))

def zip_file(name, pth, index):
    pth_path = os.path.join("assets", "weights", pth)
    if not pth or not os.path.exists(pth_path) or not pth.endswith(".pth"): return gr_warning(translations["provide_file"].format(filename=translations["model"]))
    if not index or not os.path.exists(index) or not index.endswith(".index"): return gr_warning(translations["provide_file"].format(filename=translations["index"]))

    zip_file_path = os.path.join("assets", name + ".zip")
    gr_info(translations["start"].format(start=translations["zip"]))

    import zipfile
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(pth_path, os.path.basename(pth_path))
        zipf.write(index, os.path.basename(index))

    gr_info(translations["success"])
    return zip_file_path    

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

        for table in tables:
            for row in BeautifulSoup(table, "html.parser").select("tr"):
                name_tag, url_tag = row.find("a", {"class": "fs-5"}), row.find("a", {"class": "btn btn-sm fw-bold btn-light ms-0 p-1 ps-2 pe-2"})
                if name_tag and url_tag: model_options[name_tag.text.replace(".pth", "").replace(".index", "").replace(".zip", "").replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip()] = url_tag["href"].replace("https://easyaivoice.com/run?url=", "")

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
            elif file.endswith(".pth") and "G_" not in file and "D_" not in file:
                pth_path = os.path.join(dest_weights, model_name + ".pth")
                if os.path.exists(pth_path): os.remove(pth_path)

                shutil.move(file_path, pth_path)

def download_url(url):
    if not url: return gr_warning(translations["provide_url"])
    if not os.path.exists("audios"): os.makedirs("audios", exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'noplaylist': True,
            'verbose': False,
        }

        gr_info(translations["start"].format(start=translations["download_music"]))
        import yt_dlp

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            audio_output = os.path.join("audios", re.sub(r'\s+', '-', re.sub(r'[^\w\s\u4e00-\u9fff\uac00-\ud7af\u0400-\u04FF\u1100-\u11FF]', '', ydl.extract_info(url, download=False).get('title', 'video')).strip()))

            if os.path.exists(audio_output): os.remove(audio_output)
            ydl_opts['outtmpl'] = audio_output
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
            ydl.download([url])

        gr_info(translations["success"])
        return [audio_output + ".wav", audio_output + ".wav", translations["success"]]

def download_model(url=None, model=None):
    if not url: return gr_warning(translations["provide_url"])
    if not model: return gr_warning(translations["provide_name_is_save"])

    model = model.replace(".pth", "").replace(".index", "").replace(".zip", "").replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip()
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()
    
    download_dir = os.path.join("download_model")
    weights_dir = os.path.join("assets", "weights")
    logs_dir = os.path.join("assets", "logs")

    if not os.path.exists(download_dir): os.makedirs(download_dir, exist_ok=True)
    if not os.path.exists(weights_dir): os.makedirs(weights_dir, exist_ok=True)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir, exist_ok=True)
    
    try:
        from main.tools import gdown, meganz, mediafire, pixeldrain
        gr_info(translations["start"].format(start=translations["download"]))

        if url.endswith(".pth"): hgf(url, os.path.join(weights_dir, f"{model}.pth"))
        elif url.endswith(".index"):
            model_log_dir = os.path.join(logs_dir, model)
            os.makedirs(model_log_dir, exist_ok=True)
            hgf(url, os.path.join(model_log_dir, f"{model}.index"))
        elif url.endswith(".zip"):
            output_path = hgf(url, os.path.join(download_dir, model + ".zip"))
            shutil.unpack_archive(output_path, download_dir)

            move_files_from_directory(download_dir, weights_dir, logs_dir, model)
        else:
            if "drive.google.com" in url or "drive.usercontent.google.com" in url:
                file_id = None

                if "/file/d/" in url: file_id = url.split("/d/")[1].split("/")[0]
                elif "open?id=" in url: file_id = url.split("open?id=")[1].split("/")[0]
                elif "/download?id=" in url: file_id = url.split("/download?id=")[1].split("&")[0]
                
                if file_id:
                    file = gdown.gdown_download(id=file_id, output=download_dir)
                    if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)
                    move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif "mega.nz" in url:
                meganz.mega_download_url(url, download_dir)

                file_download = next((f for f in os.listdir(download_dir)), None)
                if file_download.endswith(".zip"): shutil.unpack_archive(os.path.join(download_dir, file_download), download_dir)
                move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif "mediafire.com" in url:
                file = mediafire.Mediafire_Download(url, download_dir)
                if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)
                move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif "pixeldrain.com" in url:
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

        if file_name.endswith(".pth") and file_name.endswith(".index"): gr_warning(translations["not_model"])
        else:    
            if file_name.endswith(".zip"):
                shutil.unpack_archive(os.path.join(save_model_temp, file_name), save_model_temp)
                move_files_from_directory(save_model_temp, weight_folder, logs_folder, file_name.replace(".zip", ""))
            elif file_name.endswith(".pth"): 
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
    if choices == translations["list_model"]:
        paths = fetch_pretrained_data()[model][sample_rate]
        pretraineds_custom_path = os.path.join("assets", "models", "pretrained_custom")

        if not os.path.exists(pretraineds_custom_path): os.makedirs(pretraineds_custom_path, exist_ok=True)
        url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cergenvarq_phfgbz/", "rot13") + paths

        gr_info(translations["download_pretrain"])
        file = hgf(url.replace("/blob/", "/resolve/").replace("?download=true", "").strip(), os.path.join(pretraineds_custom_path, paths))

        if file.endswith(".zip"): 
            shutil.unpack_archive(file, pretraineds_custom_path)
            os.remove(file)

        gr_info(translations["success"])
        return translations["success"]
    elif choices == translations["download_url"]:
        if not model: return gr_warning(translations["provide_pretrain"].format(dg="D"))
        if not sample_rate: return gr_warning(translations["provide_pretrain"].format(dg="G"))

        gr_info(translations["download_pretrain"])

        hgf(model.replace("/blob/", "/resolve/").replace("?download=true", "").strip(), pretraineds_custom_path)
        hgf(sample_rate.replace("/blob/", "/resolve/").replace("?download=true", "").strip(), pretraineds_custom_path)

        gr_info(translations["success"])
        return translations["success"]

def hubert_download(hubert):
    if not hubert: 
        gr_warning(translations["provide_hubert"])
        return translations["provide_hubert"]

    hgf(hubert.replace("/blob/", "/resolve/").replace("?download=true", "").strip(), os.path.join("assets", "models", "embedders"))

    gr_info(translations["success"])
    return translations["success"]

def fushion_model(name, pth_1, pth_2, ratio):
    if not name:
        gr_warning(translations["provide_name_is_save"]) 
        return [translations["provide_name_is_save"], None]
    
    if not name.endswith(".pth"): name = name + ".pth"

    if not pth_1 or not os.path.exists(pth_1) or not pth_1.endswith(".pth"):
        gr_warning(translations["provide_file"].format(filename=translations["model"] + " 1"))
        return [translations["provide_file"].format(filename=translations["model"] + " 1"), None]
    
    if not pth_2 or not os.path.exists(pth_2) or not pth_1.endswith(".pth"):
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

        output_model = os.path.join("assets", "weights")
        if not os.path.exists(output_model): os.makedirs(output_model, exist_ok=True)

        torch.save(opt, os.path.join(output_model, name))

        gr_info(translations["success"])
        return [translations["success"], os.path.join(output_model, name)]
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))
        logger.debug(e)
        return [e, None]

def model_info(path):
    if not path or not os.path.exists(path) or os.path.isdir(path) or not path.endswith(".pth"): return gr_warning(translations["provide_file"].format(filename=translations["model"]))
    
    def prettify_date(date_str):
        if date_str == translations["not_found_create_time"]: return None

        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            logger.debug(e)
            return translations["format_not_valid"]
        
    model_data = torch.load(path, map_location=torch.device("cpu"))
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

    gr_info(translations["success"])
    return translations["model_info"].format(model_name=model_name, model_author=model_author, epochs=epochs, steps=steps, version=version, sr=sr, pitch_guidance=pitch_guidance, model_hash=model_hash, creation_date_str=creation_date_str)

def audio_effects(input_path, output_path, resample, resample_sr, chorus_depth, chorus_rate, chorus_mix, chorus_delay, chorus_feedback, distortion_drive, reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level, reverb_width, reverb_freeze_mode, pitch_shift, delay_seconds, delay_feedback, delay_mix, compressor_threshold, compressor_ratio, compressor_attack_ms, compressor_release_ms, limiter_threshold, limiter_release, gain_db, bitcrush_bit_depth, clipping_threshold, phaser_rate_hz, phaser_depth, phaser_centre_frequency_hz, phaser_feedback, phaser_mix, bass_boost_db, bass_boost_frequency, treble_boost_db, treble_boost_frequency, fade_in_duration, fade_out_duration, export_format, chorus, distortion, reverb, delay, compressor, limiter, gain, bitcrush, clipping, phaser, treble_bass_boost, fade_in_out, audio_combination, audio_combination_input):
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
    os.system(f'{python} main/inference/audio_effects.py --input_path "{input_path}" --output_path "{output_path}" --resample {resample} --resample_sr {resample_sr} --chorus_depth {chorus_depth} --chorus_rate {chorus_rate} --chorus_mix {chorus_mix} --chorus_delay {chorus_delay} --chorus_feedback {chorus_feedback} --drive_db {distortion_drive} --reverb_room_size {reverb_room_size} --reverb_damping {reverb_damping} --reverb_wet_level {reverb_wet_level} --reverb_dry_level {reverb_dry_level} --reverb_width {reverb_width} --reverb_freeze_mode {reverb_freeze_mode} --pitch_shift {pitch_shift} --delay_seconds {delay_seconds} --delay_feedback {delay_feedback} --delay_mix {delay_mix} --compressor_threshold {compressor_threshold} --compressor_ratio {compressor_ratio} --compressor_attack_ms {compressor_attack_ms} --compressor_release_ms {compressor_release_ms} --limiter_threshold {limiter_threshold} --limiter_release {limiter_release} --gain_db {gain_db} --bitcrush_bit_depth {bitcrush_bit_depth} --clipping_threshold {clipping_threshold} --phaser_rate_hz {phaser_rate_hz} --phaser_depth {phaser_depth} --phaser_centre_frequency_hz {phaser_centre_frequency_hz} --phaser_feedback {phaser_feedback} --phaser_mix {phaser_mix} --bass_boost_db {bass_boost_db} --bass_boost_frequency {bass_boost_frequency} --treble_boost_db {treble_boost_db} --treble_boost_frequency {treble_boost_frequency} --fade_in_duration {fade_in_duration} --fade_out_duration {fade_out_duration} --export_format {export_format} --chorus {chorus} --distortion {distortion} --reverb {reverb} --pitchshift {pitch_shift != 0} --delay {delay} --compressor {compressor} --limiter {limiter} --gain {gain} --bitcrush {bitcrush} --clipping {clipping} --phaser {phaser} --treble_bass_boost {treble_bass_boost} --fade_in_out {fade_in_out} --audio_combination {audio_combination} --audio_combination_input "{audio_combination_input}"')

    gr_info(translations["success"])
    return output_path 

async def TTS(prompt, voice, speed, output, pitch, google):
    if not prompt:
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

    output_dir = os.path.dirname(output_model) or output_model
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    if not google:
        from main.tools.edge_tts import Communicate
        await Communicate(text=prompt, voice=voice, rate=f"+{speed}%" if speed >= 0 else f"{speed}%", pitch=f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz").save(output)
    else:
        from main.tools.google_tts import google_tts
        google_tts(text=prompt, lang=voice, speed=speed, pitch=pitch, output_file=output)

    gr_info(translations["success"])
    return output

def separator_music(input, output_audio, format, shifts, segments_size, overlap, clean_audio, clean_strength, backing_denoise, separator_model, kara_model, backing, mdx_denoise, reverb, reverb_denoise, backing_reverb, hop_length, batch_size, sample_rate):
    output = os.path.dirname(output_audio) or output_audio

    if not input or not os.path.exists(input) or os.path.isdir(input): 
        gr_warning(translations["input_not_valid"])
        return [None]*4
    
    if not os.path.exists(output): 
        gr_warning(translations["output_not_valid"])
        return [None]*4
    
    filename, _ = os.path.splitext(os.path.basename(input))
    output = os.path.join(output, filename)

    if not os.path.exists(output): os.makedirs(output)
    gr_info(translations["start"].format(start=translations["separator_music"]))

    os.system(f'{python} main/inference/separator_music.py --input_path "{input}" --output_path "{output}" --format {format} --shifts {shifts} --segments_size {segments_size} --overlap {overlap} --mdx_hop_length {hop_length} --mdx_batch_size {batch_size} --clean_audio {clean_audio} --clean_strength {clean_strength} --backing_denoise {backing_denoise} --kara_model {kara_model} --backing {backing} --mdx_denoise {mdx_denoise} --reverb {reverb} --reverb_denoise {reverb_denoise} --backing_reverb {backing_reverb} --model_name "{separator_model}" --sample_rate {sample_rate}')
    gr_info(translations["success"])

    return [os.path.join(output, f"Original_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Original_Vocals.{format}"), os.path.join(output, f"Instruments.{format}"), (os.path.join(output, f"Main_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Main_Vocals.{format}") if backing else None), (os.path.join(output, f"Backing_Vocals_No_Reverb.{format}") if backing_reverb else os.path.join(output, f"Backing_Vocals.{format}") if backing else None)]

def convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, clean_audio, clean_strength, export_format, embedder_model, resample_sr, split_audio, f0_autotune_strength, sid, checkpointing):    
    os.system(f'{python} main/inference/convert.py --pitch {pitch} --filter_radius {filter_radius} --index_rate {index_rate} --volume_envelope {volume_envelope} --protect {protect} --hop_length {hop_length} --f0_method {f0_method} --input_path "{input_path}" --output_path "{output_path}" --pth_path "{pth_path}" --index_path "{index_path}" --f0_autotune {f0_autotune} --clean_audio {clean_audio} --clean_strength {clean_strength} --export_format {export_format} --embedder_model {embedder_model} --resample_sr {resample_sr} --split_audio {split_audio} --f0_autotune_strength {f0_autotune_strength} --sid "{sid}" --checkpointing {checkpointing}')

def convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, input_audio_name, sid, checkpointing):
    model_path = os.path.join("assets", "weights", model)

    if not use_audio:
        if merge_instrument or not_merge_backing or convert_backing or use_original:
            gr_warning(translations["turn_on_use_audio"])
            return [None]*5

    if use_original:
        if convert_backing:
            gr_warning(translations["turn_off_convert_backup"])
            return [None]*5
        elif not_merge_backing:
            gr_warning(translations["turn_off_merge_backup"])
            return [None]*5

    if not model or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith(".pth"):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return [None]*5

    f0method = method if method != "hybrid" else hybrid_method
    embedder_model = embedders if embedders != "custom" else custom_embedders

    if use_audio:
        output_audio = os.path.join("audios", input_audio_name)
        
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
                return [None]*5
            
            input_path = original_vocal
        else:
            main_vocal = get_audio_file('Main_Vocals_No_Reverb.')
            backing_vocal = get_audio_file('Backing_Vocals_No_Reverb.')

            if main_vocal == translations["notfound"]: main_vocal = get_audio_file('Main_Vocals.')
            if not not_merge_backing and backing_vocal == translations["notfound"]: backing_vocal = get_audio_file('Backing_Vocals.')

            if main_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_main_vocal"])
                return [None]*5
            
            if not not_merge_backing and backing_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_backing_vocal"])
                return [None]*5
            
            input_path = main_vocal
            backing_path = backing_vocal

        gr_info(translations["convert_vocal"])

        convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input_path, output_path, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, sid, checkpointing)

        gr_info(translations["convert_success"])

        if convert_backing:
            if os.path.exists(output_backing): os.remove(output_backing)

            gr_info(translations["convert_backup"])

            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, backing_path, output_backing, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, sid, checkpointing)

            gr_info(translations["convert_backup_success"])

        from pydub import AudioSegment

        if not not_merge_backing and not use_original:
            backing_source = output_backing if convert_backing else backing_vocal

            if os.path.exists(output_merge_backup): os.remove(output_merge_backup)

            gr_info(translations["merge_backup"])

            AudioSegment.from_file(output_path).overlay(AudioSegment.from_file(backing_source)).export(output_merge_backup, format=format)

            gr_info(translations["merge_success"])

        if merge_instrument:    
            vocals = output_merge_backup if not not_merge_backing and not use_original else output_path

            if os.path.exists(output_merge_instrument): os.remove(output_merge_instrument)

            gr_info(translations["merge_instruments_process"])

            instruments = get_audio_file('Instruments.')
            
            if instruments == translations["notfound"]: 
                gr_warning(translations["not_found_instruments"])
                output_merge_instrument = None
            else: AudioSegment.from_file(instruments).overlay(AudioSegment.from_file(vocals)).export(output_merge_instrument, format=format)
            
            gr_info(translations["merge_success"])

        return [(None if use_original else output_path), output_backing, (None if not_merge_backing and use_original else output_merge_backup), (output_path if use_original else None), (output_merge_instrument if merge_instrument else None)]
    else:
        if not input or not os.path.exists(input): 
            gr_warning(translations["input_not_valid"])
            return [None]*5
        
        if not output:
            gr_warning(translations["output_not_valid"])
            return [None]*5
        
        if os.path.isdir(input):
            gr_info(translations["is_folder"])

            if not [f for f in os.listdir(input) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]:
                gr_warning(translations["not_found_in_folder"])
                return [None]*5
            
            gr_info(translations["batch_convert"])

            output_dir = os.path.dirname(output)
            output_dir = output if not output_dir else output_dir

            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output_dir, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, sid, checkpointing)
            gr_info(translations["batch_convert_success"])

            return [None]*5
        else:
            output_dir = os.path.dirname(output) or output

            if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(output): os.remove(output)

            gr_info(translations["convert_vocal"])
            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, sid, checkpointing)

            gr_info(translations["convert_success"])
            return [output, None, None, None, None]

def convert_selection(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, sid, checkpointing):
    if use_audio:
        gr_info(translations["search_separate"])
        choice = [f for f in os.listdir("audios") if os.path.isdir(os.path.join("audios", f))]

        gr_info(translations["found_choice"].format(choice=len(choice)))

        if len(choice) == 0: 
            gr_info(translations["separator==0"])

            return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, None, None, None, None, None]
        elif len(choice) == 1:
            main_convert, backing_convert, main_backing, original_convert, vocal_instrument = convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, None, None, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, choice[0], sid, checkpointing)
            return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, main_convert, backing_convert, main_backing, original_convert, vocal_instrument]
        else: return [{"choices": choice, "value": "", "interactive": True, "visible": True, "__type__": "update"}, None, None, None, None, None]
    else:
        main_convert, _, _, _, _ = convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, None, sid, checkpointing)
        return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, main_convert, None, None, None, None]
    
def convert_tts(clean, autotune, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, sid, checkpointing):
    model_path = os.path.join("assets", "weights", model)

    if not model_path or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith(".pth"):
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
    
    if os.path.isdir(output): output = os.path.join(output, f"tts-convert.{format}")

    output_dir = os.path.dirname(output)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output): os.remove(output)

    f0method = method if method != "hybrid" else hybrid_method
    embedder_model = embedders if embedders != "custom" else custom_embedders

    gr_info(translations["convert_vocal"])
    convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, sid, checkpointing)

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

def create_dataset(input_audio, output_dataset, clean_dataset, clean_strength, separator_music, separator_reverb, kim_vocals_version, overlap, segments_size, denoise_mdx, skip, skip_start, skip_end, hop_length, batch_size, sample_rate):
    version = 1 if kim_vocals_version == "Version-1" else 2
    gr_info(translations["start"].format(start=translations["create"]))

    p = Popen(f'{python} main/inference/create_dataset.py --input_audio "{input_audio}" --output_dataset "{output_dataset}" --clean_dataset {clean_dataset} --clean_strength {clean_strength} --separator_music {separator_music} --separator_reverb {separator_reverb} --kim_vocal_version {version} --overlap {overlap} --segments_size {segments_size} --mdx_hop_length {hop_length} --mdx_batch_size {batch_size} --denoise_mdx {denoise_mdx} --skip {skip} --skip_start_audios "{skip_start}" --skip_end_audios "{skip_end}" --sample_rate {sample_rate}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(os.path.join("assets", "logs", "create_dataset.log"), done):
        yield log

def preprocess(model_name, sample_rate, cpu_core, cut_preprocess, process_effects, path, clean_dataset, clean_strength):
    dataset = os.path.join(path)
    sr = int(float(sample_rate.rstrip("k")) * 1000)

    if not model_name: return gr_warning(translations["provide_name"])
    if not any(f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3")) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))): return gr_warning(translations["not_found_data"])
    
    model_dir = os.path.join("assets", "logs", model_name)
    if os.path.exists(model_dir): shutil.rmtree(model_dir, ignore_errors=True)

    p = Popen(f'{python} main/inference/preprocess.py --model_name "{model_name}" --dataset_path "{dataset}" --sample_rate {sr} --cpu_cores {cpu_core} --cut_preprocess {cut_preprocess} --process_effects {process_effects} --clean_dataset {clean_dataset} --clean_strength {clean_strength}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(os.path.join(model_dir, "preprocess.log"), done):
        yield log

def extract(model_name, version, method, pitch_guidance, hop_length, cpu_cores, gpu, sample_rate, embedders, custom_embedders):
    embedder_model = embedders if embedders != "custom" else custom_embedders
    sr = int(float(sample_rate.rstrip("k")) * 1000)

    if not model_name: return gr_warning(translations["provide_name"])

    model_dir = os.path.join("assets", "logs", model_name)
    if not any(os.path.isfile(os.path.join(model_dir, "sliced_audios", f)) for f in os.listdir(os.path.join(model_dir, "sliced_audios"))) or not any(os.path.isfile(os.path.join(model_dir, "sliced_audios_16k", f)) for f in os.listdir(os.path.join(model_dir, "sliced_audios_16k"))): return gr_warning(translations["not_found_data_preprocess"])

    p = Popen(f'{python} main/inference/extract.py --model_name "{model_name}" --rvc_version {version} --f0_method {method} --pitch_guidance {pitch_guidance} --hop_length {hop_length} --cpu_cores {cpu_cores} --gpu {gpu} --sample_rate {sr} --embedder_model {embedder_model}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(os.path.join(model_dir, "extract.log"), done):
        yield log

def create_index(model_name, rvc_version, index_algorithm):
    if not model_name: return gr_warning(translations["provide_name"])
    model_dir = os.path.join("assets", "logs", model_name)

    if not any(os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))): return gr_warning(translations["not_found_data_extract"])

    p = Popen(f'{python} main/inference/create_index.py --model_name "{model_name}" --rvc_version {rvc_version} --index_algorithm {index_algorithm}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(os.path.join(model_dir, "create_index.log"), done):
        yield log

def training(model_name, rvc_version, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sample_rate, batch_size, gpu, pitch_guidance, not_pretrain, custom_pretrained, pretrain_g, pretrain_d, detector, threshold, clean_up, cache, model_author, vocoder, checkpointing):
    sr = int(float(sample_rate.rstrip("k")) * 1000)
    if not model_name: return gr_warning(translations["provide_name"])

    model_dir = os.path.join("assets", "logs", model_name)
    if not any(os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))): return gr_warning(translations["not_found_data_extract"])

    if not not_pretrain:
        if not custom_pretrained: 
            pretrained_selector = {True: {32000: ("f0G32k.pth", "f0D32k.pth"), 40000: ("f0G40k.pth", "f0D40k.pth"), 48000: ("f0G48k.pth", "f0D48k.pth")}, False: {32000: ("G32k.pth", "D32k.pth"), 40000: ("G40k.pth", "D40k.pth"), 48000: ("G48k.pth", "D48k.pth")}}
            pg, pd = pretrained_selector(pitch_guidance)[sr]
        else:
            if not pretrain_g: return gr_warning(translations["provide_pretrained"].format(dg="G"))
            if not pretrain_d: return gr_warning(translations["provide_pretrained"].format(dg="D"))
            
            pg, pd = pretrain_g, pretrain_d

        if not custom_pretrained: pretrained_G, pretrained_D = os.path.join("assets", "models", f"pretrained_{rvc_version}", f"{vocoder if vocoder != 'Default' else ''}{pg}"), os.path.join("assets", "models", f"pretrained_{rvc_version}", f"{vocoder if vocoder != 'Default' else ''}{pd}")
        else: pretrained_G, pretrained_D = os.path.join("assets", "models", f"pretrained_custom", pg), os.path.join("assets", "models", f"pretrained_custom", pd)

        download_version = codecs.decode(f"uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cergenvarq_i{'2' if rvc_version == 'v2' else '1'}/", "rot13")
        
        if not custom_pretrained:
            try:
                if not os.path.exists(pretrained_G):
                    gr_info(translations["download_pretrained"].format(dg="G", rvc_version=rvc_version))
                    hgf(f"{download_version}{pg}", os.path.join("assets", "models", f"pretrained_{rvc_version}", pg))
                        
                if not os.path.exists(pretrained_D):
                    gr_info(translations["download_pretrained"].format(dg="D", rvc_version=rvc_version))
                    hgf(f"{download_version}{pg}", os.path.join("assets", "models", f"pretrained_{rvc_version}", pg))
            except:
                gr_warning(translations["not_use_pretrain_error_download"])
                pretrained_G, pretrained_D = None, None
        else:
            if not os.path.exists(pretrained_G): return gr_warning(translations["not_found_pretrain"].format(dg="G"))
            if not os.path.exists(pretrained_D): return gr_warning(translations["not_found_pretrain"].format(dg="D"))
    else: gr_warning(translations["not_use_pretrain"])

    gr_info(translations["start"].format(start=translations["training"]))

    p = Popen(f'{python} main/inference/train.py --model_name "{model_name}" --rvc_version {rvc_version} --save_every_epoch {save_every_epoch} --save_only_latest {save_only_latest} --save_every_weights {save_every_weights} --total_epoch {total_epoch} --sample_rate {sr} --batch_size {batch_size} --gpu {gpu} --pitch_guidance {pitch_guidance} --overtraining_detector {detector} --overtraining_threshold {threshold} --clean_up {clean_up} --cache_data_in_gpu {cache} --g_pretrained_path "{pretrained_G}" --d_pretrained_path "{pretrained_D}" --model_author "{model_author}" --vocoder {vocoder} --checkpointing {checkpointing}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    if not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)

    for log in log_read(os.path.join(model_dir, "train.log"), done):
        yield log

def stop_pid(pid_file, model_name=None):
    try:
        pid_file_path = os.path.join("assets", f"{pid_file}.txt") if model_name is None else os.path.join("assets", "logs", model_name, f"{pid_file}.txt")

        if not os.path.exists(pid_file_path): return gr_warning(translations["not_found_pid"])
        else:
            with open(pid_file_path, "r") as pid_file:
                pids = [int(pid) for pid in pid_file.readlines()]

            for pid in pids:
                os.kill(pid, 9)

            gr_info(translations["end_pid"])
            if os.path.exists(pid_file_path): os.remove(pid_file_path)
    except:
        pass

def stop_train(model_name):
    try:
        pid_file_path = os.path.join("assets", "logs", model_name, "config.json")

        if not os.path.exists(pid_file_path): return gr_warning(translations["not_found_pid"])
        else:
            with open(pid_file_path, "r") as pid_file:
                pid_data = json.load(pid_file)
                pids = pid_data.get("process_pids", [])

            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)

                json.dump(pid_data, pid_file, indent=4)

            for pid in pids:
                os.kill(pid, 9)

            gr_info(translations["end_pid"])     
            if os.path.exists(pid_file_path): os.remove(pid_file_path)
    except:
        pass

def delete_audios(files):
    if not os.path.exists(files) or os.path.isdir(files): return gr_warning(translations["input_not_valid"])
    else:
        gr_info(translations["clean_audios"])
        os.remove(files)

        for item in os.listdir("audios"):
            item_path = os.path.join("audios", item)

            if os.path.isdir(item_path) and len([f for f in os.listdir(item_path)]) < 1: shutil.rmtree(item_path, ignore_errors=True)  

        gr_info(translations["clean_audios_success"])
        return change_audios_choices()

def delete_separated(files):
    if not os.path.exists(files) or os.path.isdir(files): return gr_warning(translations["input_not_valid"])
    else:
        gr_info(translations["clean_separate"])
        os.remove(files)

        gr_info(translations["clean_separate_success"])
        return change_separate_choices()

def delete_model(model, index):
    files = os.path.join("assets", "weights", model)

    if model:
        if not os.path.exists(files) or not model.endswith(".pth"): return gr_warning(translations["provide_file"].format(filename=translations["model"]))
        else:
            gr_info(translations["clean_model"])
            os.remove(files)
            gr_info(translations["clean_model_success"])
        
    if index:
        if not os.path.exists(index): return gr_warning(translations["provide_file"].format(filename=translations["index"]))
        else:
            gr_info(translations["clean_index"])
            shutil.rmtree(index, ignore_errors=True)
            gr_info(translations["clean_index_success"])

    return change_choices_del()

def delete_pretrained(pretrain):
    if not os.path.exists(pretrain) or os.path.isdir(pretrain): return gr_warning(translations["input_not_valid"])
    else:
        gr_info(translations["clean_pretrain"])
        os.remove(pretrain) 
        gr_info(translations["clean_pretrain_success"])

    return change_allpretrained_choices()

def delete_presets(json_file):
    files = os.path.join("assets", "presets", json_file)

    if not os.path.exists(files) or not json_file.endswith(".json"): return gr_warning(translations["provide_file_settings"])
    else:
        gr_info(translations["clean_presets_2"])
        os.remove(files)
        gr_info(translations["clean_presets_success"])

    return change_preset_choices()

def delete_all_audios():
    dir = "audios"

    if len(os.listdir(dir)) < 1: return gr_warning(translations["not_found_in_folder"])
    else:
        gr_info(translations["clean_all_audios"])

        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir, exist_ok=True)

        gr_info(translations["clean_all_audios_success"])
    return {"choices": [], "value": "", "__type__": "update"}

def delete_all_separated():
    dir = os.path.join("assets", "models", "uvr5")

    if len(os.listdir(dir)) < 1: return gr_warning(translations["not_found_separate_model"])
    else:
        gr_info(translations["clean_all_separate_model"])

        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir, exist_ok=True)

        gr_info(translations["clean_all_separate_model_success"])
    return {"choices": [], "value": "", "__type__": "update"}

def delete_all_model():
    model = os.listdir(os.path.join("assets", "weights"))
    index = list(f for f in os.listdir(os.path.join("assets", "logs")) if os.path.isdir(os.path.join("assets", "logs", f)) and f != "mute")

    if len(model) < 1: return gr_warning(translations["not_found"].format(name=translations["model"]))
    if len(index) < 1: return gr_warning(translations["not_found"].format(name=translations["index"]))

    gr_info(translations["start_clean_model"])

    for f in model:
        file = os.path.join("assets", "weights", f)
        if os.path.exists(file) and f.endswith(".pth"): os.remove(file)

    for f in index:
        file = os.path.join("assets", "logs", f)
        if os.path.exists(file): shutil.rmtree(file, ignore_errors=True)

    gr_info(translations["clean_all_models_success"])
    return [{"choices": [], "value": "", "__type__": "update"}]*2

def delete_all_pretrained():
    Allpretrained = [os.path.join("assets", "models", path, model) for path in ["pretrained_v1", "pretrained_v2", "pretrained_custom"] for model in os.listdir(os.path.join("assets", "models", path)) if model.endswith(".pth") and ("D" in model or "G" in model)]

    if len(Allpretrained) < 1: return gr_warning(translations["not_found_pretrained"])
    else:
        gr_info(translations["clean_all_pretrained"])
        for f in Allpretrained:
            if os.path.exists(f): os.remove(f)

        gr_info(translations["clean_all_pretrained_success"])
    return {"choices": [], "value": "", "__type__": "update"}

def delete_all_presets():
    dir = os.path.join("assets", "presets")

    if len(os.listdir(dir)) < 1: return gr_warning(translations["not_found_presets"])
    else:
        gr_info(translations["clean_all_presets"])

        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir, exist_ok=True)

        gr_info(translations["clean_all_presets_success"])
    return {"choices": [], "value": "", "__type__": "update"}

def delete_all_log():
    log_path = [os.path.join(root, f) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for f in files if f.endswith(".log")]

    if len(log_path) < 1: return gr_warning(translations["not_found_log"])
    else:
        gr_info(translations["clean_all_log"])

        for f in log_path:
            if os.path.exists(f): os.remove(f)

        open(os.path.join("assets", "logs", "app.log"), "w", encoding="utf-8")
        gr_info(translations["clean_all_log_success"])

def delete_all_predictors():
    dir = os.path.join("assets", "models", "predictors")

    if len(os.listdir(dir)) < 1: return gr_warning(translations["not_found_predictors"])
    else:
        gr_info(translations["clean_all_predictors"])

        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir, exist_ok=True)

        gr_info(translations["clean_all_predictors_success"])
    return {"choices": [], "value": "", "__type__": "update"}

def delete_all_embedders():
    dir = os.path.join("assets", "models", "embedders")

    if len(os.listdir(dir)) < 1: return gr_warning(translations["not_found_embedders"])
    else:
        gr_info(translations["clean_all_embedders"])

        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir, exist_ok=True)

        gr_info(translations["clean_all_embedders_success"])
    return {"choices": [], "value": "", "__type__": "update"}

def delete_dataset(name):
    if not name or not os.path.exists(name) or not os.path.isdir(name): return gr_warning(translations["provide_folder"])
    else:
        if len(os.listdir(name)) < 1: gr_warning(translations["empty_folder"])
        else:
            gr_info(translations["clean_dataset"])

            shutil.rmtree(name, ignore_errors=True)
            os.makedirs(name, exist_ok=True)

            gr_info(translations["clean_dataset_success"])

def load_presets(presets, cleaner, autotune, pitch, clean_strength, index_strength, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength):
    if not presets: return gr_warning(translations["provide_file_settings"])
    with open(os.path.join("assets", "presets", presets)) as f:
        file = json.load(f)

    gr_info(translations["load_presets"].format(presets=presets))
    return file.get("cleaner", cleaner), file.get("autotune", autotune), file.get("pitch", pitch), file.get("clean_strength", clean_strength), file.get("index_strength", index_strength), file.get("resample_sr", resample_sr), file.get("filter_radius", filter_radius), file.get("volume_envelope", volume_envelope), file.get("protect", protect), file.get("split_audio", split_audio), file.get("f0_autotune_strength", f0_autotune_strength)

def save_presets(name, cleaner, autotune, pitch, clean_strength, index_strength, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, cleaner_chbox, autotune_chbox, pitch_chbox, index_strength_chbox, resample_sr_chbox, filter_radius_chbox, volume_envelope_chbox, protect_chbox, split_audio_chbox):  
    if not name: return gr_warning(translations["provide_filename_settings"])
    if not any([cleaner_chbox, autotune_chbox, pitch_chbox, index_strength_chbox, resample_sr_chbox, filter_radius_chbox, volume_envelope_chbox, protect_chbox, split_audio_chbox]): return gr_warning(translations["choose1"])

    settings = {}
    for checkbox, data in [(cleaner_chbox, {"cleaner": cleaner, "clean_strength": clean_strength}), (autotune_chbox, {"autotune": autotune, "f0_autotune_strength": f0_autotune_strength}), (pitch_chbox, {"pitch": pitch}), (index_strength_chbox, {"index_strength": index_strength}), (resample_sr_chbox, {"resample_sr": resample_sr}), (filter_radius_chbox, {"filter_radius": filter_radius}), (volume_envelope_chbox, {"volume_envelope": volume_envelope}), (protect_chbox, {"protect": protect}), (split_audio_chbox, {"split_audio": split_audio})]:
        if checkbox: settings.update(data)

    with open(os.path.join("assets", "presets", name + ".json"), "w") as f:
        json.dump(settings, f, indent=4)

    gr_info(translations["export_settings"])

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
                    with open(report_path, "w", encoding="utf-8") as w:
                        w.write(str(r.read()))
                        w.write("\n")
        except Exception as e:
            gr_error(translations["error_read_log"])
            logger.debug(e)

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            requests.post(report_url, json={"embeds": [{"title": "Báo Cáo Lỗi", "description": f"Mô tả lỗi: {error_info}", "color": 15158332, "author": {"name": "Vietnamese_RVC", "icon_url": miku_image, "url": codecs.decode("uggcf://tvguho.pbz/CunzUhlauNau16/Ivrganzrfr-EIP/gerr/znva","rot13")}, "thumbnail": {"url": codecs.decode("uggcf://p.grabe.pbz/7dADJbv-36fNNNNq/grabe.tvs", "rot13")}, "fields": [{"name": "Số Lượng Gỡ Lỗi", "value": content.count("DEBUG")}, {"name": "Số Lượng Thông Tin", "value": content.count("INFO")}, {"name": "Số Lượng Cảnh Báo", "value": content.count("WARNING")}, {"name": "Số Lượng Lỗi", "value": content.count("ERROR")}], "footer": {"text": f"Tên Máy: {platform.uname().node} - Hệ Điều Hành: {platform.system()}-{platform.version()}\nThời Gian Báo Cáo Lỗi: {datetime.now()}."}}]})

            with open(report_path, "rb", encoding="utf-8") as f:
                requests.post(report_url, files={"file": f})
        except Exception as e:
            gr_error(translations["error_send"])
            logger.debug(e)
        finally:
            if os.path.exists(report_path): os.remove(report_path)
    else: requests.post(report_url, json={"embeds": [{"title": "Báo Cáo Lỗi", "description": error_info}]})



with gr.Blocks(title="📱 Vietnamese-RVC GUI BY ANH", theme=theme) as app:
    gr.HTML(translations["display_title"])
    with gr.Row(): 
        gr.Markdown(translations["rick_roll"].format(rickroll=codecs.decode('uggcf://jjj.lbhghor.pbz/jngpu?i=qDj4j9JtKpD', 'rot13')))
    with gr.Row(): 
        gr.Markdown(translations["terms_of_use"])
    with gr.Row():
        gr.Markdown(translations["exemption"])

    with gr.Tabs():      
        with gr.TabItem(translations["separator_tab"], visible=configs.get("separator_tab", True)):
            gr.Markdown(f"## {translations['separator_tab']}")
            with gr.Row(): 
                gr.Markdown(translations["4_part"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():       
                            cleaner = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)       
                            backing = gr.Checkbox(label=translations["separator_backing"], value=False, interactive=True)
                            denoise = gr.Checkbox(label=translations["denoise_backing"], value=False, interactive=False)
                            separator_denoise = gr.Checkbox(label=translations["denoise_mdx"], value=False, interactive=False)       
                            mdx_model = gr.Checkbox(label=translations["use_mdx"], value=False, interactive=True)
                            reverb = gr.Checkbox(label=translations["dereveb_audio"], value=False, interactive=True)
                            backing_reverb = gr.Checkbox(label=translations["dereveb_backing"], value=False, interactive=False)
                            reverb_denoise = gr.Checkbox(label=translations["denoise_dereveb"], value=False, interactive=False)                   
                        with gr.Row():
                            separator_model = gr.Dropdown(label=translations["separator_model"], value="HT-Normal", choices=["HT-Normal", "HT-Tuned", "HD_MMI", "HT_6S"], interactive=True, visible=True)
                            separator_backing_model = gr.Dropdown(label=translations["separator_backing_model"], value="Version-1", choices=["Version-1", "Version-2"], interactive=True, visible=False)
                with gr.Column():
                    separator_button = gr.Button(translations["separator_tab"], variant="primary", scale=2)
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            shifts = gr.Slider(label=translations["shift"], info=translations["shift_info"], minimum=1, maximum=20, value=2, step=1, interactive=True)
                            segment_size = gr.Slider(label=translations["segments_size"], info=translations["segments_size_info"], minimum=32, maximum=3072, value=256, step=32, interactive=True)
                        with gr.Row():
                            mdx_batch_size = gr.Slider(label=translations["batch_size"], info=translations["mdx_batch_size_info"], minimum=1, maximum=64, value=1, step=1, interactive=True, visible=False)
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            overlap = gr.Radio(label=translations["overlap"], info=translations["overlap_info"], choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True)
                        with gr.Row():
                            mdx_hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=8192, value=1024, step=1, interactive=True, visible=False)
            with gr.Row():
                with gr.Column():
                    input = gr.File(label=translations["drop_audio"], file_types=['audio'])    
                    with gr.Accordion(translations["use_url"], open=False):
                        url = gr.Textbox(label=translations["url_audio"], value="", placeholder="https://www.youtube.com/...", scale=6)
                        download_button = gr.Button(translations["downloads"])
                with gr.Column():
                    clean_strength = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
                    sample_rate1 = gr.Slider(minimum=0, maximum=96000, step=1, value=44100, label=translations["sr"], info=translations["sr_info"], interactive=True)
                    with gr.Accordion(translations["input_output"]):
                        format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                        input_audio = gr.Dropdown(label=translations["audio_path"], value="" if len(paths_for_files) < 1 else paths_for_files[0], choices=paths_for_files, allow_custom_value=True, interactive=True)
                        refesh_separator = gr.Button(translations["refesh"])
                        output_separator = gr.Textbox(label=translations["output_folder"], value="audios", placeholder="audios", info=translations["output_folder_info"], interactive=True)
                    audio_input = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
            with gr.Row():
                gr.Markdown(translations["output_separator"])
            with gr.Row():
                instruments_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["instruments"])
                original_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["original_vocal"])
                main_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["main_vocal"], visible=False)
                backing_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["backing_vocal"], visible=False)
            with gr.Row():
                backing.change(fn=lambda a, b, c: [visible_1(a or b or c), visible_1(a or b or c)], inputs=[backing, mdx_model, reverb], outputs=[mdx_batch_size, mdx_hop_length])
                mdx_model.change(fn=lambda a, b, c: [visible_1(a or b or c), visible_1(a or b or c)], inputs=[backing, mdx_model, reverb], outputs=[mdx_batch_size, mdx_hop_length])
                reverb.change(fn=lambda a, b, c: [visible_1(a or b or c), visible_1(a or b or c)], inputs=[backing, mdx_model, reverb], outputs=[mdx_batch_size, mdx_hop_length])
            with gr.Row():
                backing.change(fn=visible_1, inputs=[backing], outputs=[separator_backing_model])
                backing.change(fn=visible_1, inputs=[backing], outputs=[main_vocals])
                backing.change(fn=visible_1, inputs=[backing], outputs=[backing_vocals])
            with gr.Row():
                backing.change(fn=valueFalse_interactive1, inputs=[backing], outputs=[denoise])
                backing.change(fn=valueFalse_interactive2,  inputs=[backing, reverb], outputs=[backing_reverb])
            with gr.Row():
                reverb.change(fn=valueFalse_interactive1, inputs=[reverb], outputs=[reverb_denoise])
                reverb.change(fn=valueFalse_interactive2, inputs=[backing, reverb], outputs=[backing_reverb])
            with gr.Row():
                mdx_model.change(fn=valueFalse_interactive1, inputs=[mdx_model], outputs=[separator_denoise])
                mdx_model.change(fn=change_model_separate_choices, inputs=[mdx_model], outputs=[separator_model])
                mdx_model.change(fn=lambda inp: visible_1(not inp), inputs=[mdx_model], outputs=[shifts])
            with gr.Row():
                input_audio.change(fn=lambda audio: audio if audio else None, inputs=[input_audio], outputs=[audio_input])
                cleaner.change(fn=visible_1, inputs=[cleaner], outputs=[clean_strength])
                input.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[input], outputs=[input_audio])
                refesh_separator.click(fn=change_audios_choices, inputs=[], outputs=[input_audio])
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
                        separator_denoise, 
                        reverb, 
                        reverb_denoise,
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
                            use_original = gr.Checkbox(label=translations["convert_original"], value=False, interactive=True, visible=False) 
                            convert_backing = gr.Checkbox(label=translations["convert_backing"], value=False, interactive=True, visible=False)   
                            not_merge_backing = gr.Checkbox(label=translations["not_merge_backing"], value=False, interactive=True, visible=False)
                            merge_instrument = gr.Checkbox(label=translations["merge_instruments"], value=False, interactive=True, visible=False) 
                    with gr.Row(): 
                        with gr.Row():
                            audio_select = gr.Dropdown(label=translations["select_separate"], choices=[], value="", interactive=True, allow_custom_value=True, visible=False)
                            convert_button_2 = gr.Button(translations["convert_audio"], visible=False)
                    with gr.Row():
                        pitch = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                        clean_strength0 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
                with gr.Column():
                    convert_button = gr.Button(translations["convert_audio"], variant="primary", scale=2)
            with gr.Row():
                with gr.Column():
                    input0 = gr.File(label=translations["drop_audio"], file_types=['audio'])  
                    play_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
                with gr.Column():
                    with gr.Accordion(translations["model_accordion"], open=True):
                        with gr.Row():
                            model_pth = gr.Dropdown(label=translations["model_name"], choices=sorted(model_name), value=sorted(model_name)[0] if len(sorted(model_name)) > 0 else '', interactive=True, allow_custom_value=True)
                            model_index = gr.Dropdown(label=translations["index_path"], choices=sorted(index_path), value=sorted(index_path)[0] if len(sorted(index_path)) > 0 else '', interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refesh = gr.Button(translations["refesh"])
                        with gr.Row():
                            sid = gr.Dropdown(label=translations["sid"], choices=[], value=0, interactive=True, allow_custom_value=True)
                        with gr.Row():
                            index_strength = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=lambda: os.path.exists(model_index))
                    with gr.Accordion(translations["input_output"], open=False):
                        with gr.Column():
                            export_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                            input_audio0 = gr.Dropdown(label=translations["audio_path"], value="" if len(paths_for_files) < 1 else paths_for_files[0], choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                            output_audio = gr.Textbox(label=translations["output_path"], value="audios/output.wav", placeholder="audios/output.wav", info=translations["output_path_info"], interactive=True)
                        with gr.Column():
                            refesh0 = gr.Button(translations["refesh"])
                    with gr.Accordion(translations["setting"], open=False):
                        with gr.Accordion(translations["f0_method"], open=False):
                            method = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=["pm", "dio", "crepe-tiny", "crepe", "fcpe", "rmvpe", "harvest", "hybrid"], value="rmvpe", interactive=True)
                            hybrid_method = gr.Radio(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[rmvpe+harvest]"], value="hybrid[pm+dio]", interactive=True, visible=False)
                            hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion(translations["hubert_model"], open=False):
                            embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "Hidden_Rabbit_last", "portuguese_hubert_base"], value="contentvec_base", interactive=True)
                            custom_embedders = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=False)
                        with gr.Accordion(translations["use_presets"], open=False):
                            with gr.Row():
                                presets_name = gr.Dropdown(label=translations["file_preset"], choices=sorted(presets_file), value=sorted(presets_file)[0] if len(sorted(presets_file)) > 0 else '', interactive=True, allow_custom_value=True)
                                with gr.Row():
                                    load_click = gr.Button(translations["load_file"], variant=["primary"])
                                    refesh_click = gr.Button(translations["refesh"])
                            with gr.Accordion(translations["export_file"], open=False):
                                with gr.Row():
                                    with gr.Row():
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
                                with gr.Row():
                                    with gr.Row():
                                        name_to_save_file = gr.Textbox(label=translations["filename_to_save"])
                                        save_file_button = gr.Button(translations["export_file"])
                            with gr.Row():
                                upload_presets = gr.File(label=translations["upload_presets"], file_types=['json'])  
                        with gr.Column():
                            with gr.Group():
                                split_audio = gr.Checkbox(label=translations["split_audio"], value=False, interactive=True)
                            f0_autotune_strength = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=False)
                            resample_sr = gr.Slider(minimum=0, maximum=96000, label=translations["resample"], info=translations["resample_info"], value=0, step=1, interactive=True)
                            filter_radius = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                            volume_envelope = gr.Slider(minimum=0, maximum=1, label=translations["volume_envelope"], info=translations["volume_envelope_info"], value=1, step=0.1, interactive=True)
                            protect = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.33, step=0.01, interactive=True)
            with gr.Row():
                gr.Markdown(translations["output_convert"])
            with gr.Row():
                main_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["main_convert"])
                backing_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["convert_backing"], visible=False)
                main_backing = gr.Audio(show_download_button=True, interactive=False, label=translations["main_or_backing"], visible=False)  
            with gr.Row():
                original_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["convert_original"], visible=False)
                vocal_instrument = gr.Audio(show_download_button=True, interactive=False, label=translations["voice_or_instruments"], visible=False)  
            with gr.Row():
                refesh_click.click(fn=change_preset_choices, inputs=[], outputs=[presets_name])
                save_file_button.click(fn=change_preset_choices, inputs=[], outputs=[presets_name])
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
                        f0_autotune_strength
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
                        f0_autotune_strength
                    ]
                )
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
                        split_audio_chbox
                    ],
                    outputs=[]
                )
            with gr.Row():
                upload_presets.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("assets", "presets")), inputs=[upload_presets], outputs=[presets_name])
                autotune.change(fn=visible_1, inputs=[autotune], outputs=[f0_autotune_strength])
            with gr.Row():
                use_audio.change(fn=visible_1, inputs=[use_audio], outputs=[main_backing])
                use_audio.change(fn=lambda audio: [valueFalse_interactive1(audio), valueFalse_interactive1(audio), valueFalse_interactive1(audio), valueFalse_interactive1(audio)], inputs=[use_audio], outputs=[use_original, convert_backing, not_merge_backing, merge_instrument])
            with gr.Row():
                use_audio.change(fn=visible_1, inputs=[use_audio], outputs=[use_original]); use_audio.change(fn=visible_1, inputs=[use_audio], outputs=[convert_backing])
                use_audio.change(fn=visible_1, inputs=[use_audio], outputs=[not_merge_backing]); use_audio.change(fn=visible_1, inputs=[use_audio], outputs=[merge_instrument])
                use_audio.change(fn=lambda audio: [visible_1(not audio), visible_1(not audio), visible_1(not audio), visible_1(not audio)], inputs=[use_audio], outputs=[input_audio0, output_audio, input0, play_audio])
            with gr.Row():
                convert_backing.change(fn=visible_1, inputs=[convert_backing], outputs=[backing_convert])
                convert_backing.change(fn=change_backing_choices, inputs=[convert_backing, not_merge_backing], outputs=[use_original])
            with gr.Row():
                use_original.change(fn=lambda original: [valueFalse_interactive1(not original), valueFalse_interactive1(not original)], inputs=[use_original], outputs=[convert_backing, not_merge_backing])
                use_original.change(fn=lambda audio, original: [visible_1(original), visible_1(not original), visible_1(audio and not original)], inputs=[use_audio, use_original], outputs=[original_convert, main_convert, main_backing])
            with gr.Row():
                cleaner0.change(fn=visible_1, inputs=[cleaner0], outputs=[clean_strength0])
                merge_instrument.change(fn=visible_1, inputs=[merge_instrument], outputs=[vocal_instrument])
            with gr.Row():
                not_merge_backing.change(fn=lambda audio, merge: visible_1(audio and not merge), inputs=[use_audio, not_merge_backing], outputs=[main_backing])
                not_merge_backing.change(fn=change_backing_choices, inputs=[convert_backing, not_merge_backing], outputs=[use_original])
            with gr.Row():
                method.change(fn=lambda method: visible_1(True if method == "hybrid" else False), inputs=[method], outputs=[hybrid_method])
                method.change(fn=hoplength_show, inputs=[method, hybrid_method], outputs=[hop_length])
            with gr.Row():
                hybrid_method.change(fn=hoplength_show, inputs=[method, hybrid_method], outputs=[hop_length])
                refesh.click(fn=change_models_choices, inputs=[], outputs=[model_pth, model_index])
                model_pth.change(fn=get_index, inputs=[model_pth], outputs=[model_index])
            with gr.Row():
                input0.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[input0], outputs=[input_audio0])
                input_audio0.change(fn=lambda audio: audio if audio else None, inputs=[input_audio0], outputs=[play_audio])
                model_pth.change(fn=get_sid, inputs=[model_pth], outputs=[sid])
            with gr.Row():
                embedders.change(fn=lambda embedders: visible_1(True if embedders == "custom" else False), inputs=[embedders], outputs=[custom_embedders])
                refesh0.click(fn=change_audios_choices, inputs=[], outputs=[input_audio0])
            with gr.Row():
                model_index.change(fn=index_strength_show, inputs=[model_index], outputs=[index_strength])
                audio_select.change(fn=lambda: visible_1(True), inputs=[], outputs=[convert_button_2])
                convert_button_2.click(fn=lambda: [visible_1(False), visible_1(False)], inputs=[], outputs=[audio_select, convert_button_2])
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
                        sid,
                        checkpointing
                    ],
                    outputs=[audio_select, main_convert, backing_convert, main_backing, original_convert, vocal_instrument],
                    api_name="convert_selection"
                )
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
                        sid,
                        checkpointing
                    ],
                    outputs=[main_convert, backing_convert, main_backing, original_convert, vocal_instrument],
                    api_name="convert_audio"
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
                            google_tts = gr.Checkbox(label=translations["googletts"], value=False, interactive=True)
                        prompt = gr.Textbox(label=translations["text_to_speech"], value="", placeholder="Hello Words", lines=2)
                    with gr.Row():
                        speed = gr.Slider(label=translations["voice_speed"], info=translations["voice_speed_info"], minimum=-100, maximum=100, value=0, step=1)
                        pitch0 = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                with gr.Column():
                    tts_button = gr.Button(translations["tts_1"], variant="primary", scale=2)
                    convert_button0 = gr.Button(translations["tts_2"], variant="secondary", scale=2)
            with gr.Row():
                with gr.Column():
                    tts_voice = gr.Dropdown(label=translations["voice"], choices=edge_tts, interactive=True, value="vi-VN-NamMinhNeural")
                    tts_pitch = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info_2"], label=translations["pitch"], value=0, interactive=True)
                    txt_input = gr.File(label=translations["drop_text"], file_types=['txt'], visible=False)  
                with gr.Column():
                    with gr.Accordion(translations["model_accordion"], open=True):
                        with gr.Row():
                            model_pth0 = gr.Dropdown(label=translations["model_name"], choices=sorted(model_name), value=sorted(model_name)[0] if len(sorted(model_name)) > 0 else '', interactive=True, allow_custom_value=True)
                            model_index0 = gr.Dropdown(label=translations["index_path"], choices=sorted(index_path), value=sorted(index_path)[0] if len(sorted(index_path)) > 0 else '', interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refesh1 = gr.Button(translations["refesh"])
                        with gr.Row():
                            sid0 = gr.Dropdown(label=translations["sid"], choices=[], value=0, interactive=True, allow_custom_value=True)
                        with gr.Row():
                            index_strength0 = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=lambda: os.path.exists(model_index0))
                    with gr.Accordion(translations["output_path"], open=False):
                        export_format0 = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                        output_audio0 = gr.Textbox(label=translations["output_tts"], value="audios/tts.wav", placeholder="audios/tts.wav", info=translations["tts_output"], interactive=True)
                        output_audio1 = gr.Textbox(label=translations["output_tts_convert"], value="audios/tts-convert.wav", placeholder="audios/tts-convert.wav", info=translations["tts_output"], interactive=True)
                    with gr.Accordion(translations["setting"], open=False):
                        with gr.Accordion(translations["f0_method"], open=False):
                            method0 = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=["pm", "dio", "crepe-tiny", "crepe", "fcpe", "rmvpe", "harvest", "hybrid"], value="rmvpe", interactive=True)
                            hybrid_method0 = gr.Radio(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[rmvpe+harvest]"], value="hybrid[pm+dio]", interactive=True, visible=False)
                            hop_length0 = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion(translations["hubert_model"], open=False):
                            embedders0 = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "Hidden_Rabbit_last", "portuguese_hubert_base"], value="contentvec_base", interactive=True)
                            custom_embedders0 = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=False)
                        with gr.Group():
                            with gr.Row():
                                split_audio0 = gr.Checkbox(label=translations["split_audio"], value=False, interactive=True)   
                                cleaner1 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)       
                        with gr.Row():
                            autotune3 = gr.Checkbox(label=translations["autotune"], value=False, interactive=True) 
                            checkpointing0 = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)         
                        with gr.Column():
                            f0_autotune_strength0 = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=False)
                            clean_strength1 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
                            resample_sr0 = gr.Slider(minimum=0, maximum=96000, label=translations["resample"], info=translations["resample_info"], value=0, step=1, interactive=True)
                            filter_radius0 = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                            volume_envelope0 = gr.Slider(minimum=0, maximum=1, label=translations["volume_envelope"], info=translations["volume_envelope_info"], value=1, step=0.1, interactive=True)
                            protect0 = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.33, step=0.01, interactive=True)
            with gr.Row():
                gr.Markdown(translations["output_tts_markdown"])
            with gr.Row():
                tts_voice_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["output_text_to_speech"])
                tts_voice_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["output_file_tts_convert"])
            with gr.Row():
                autotune3.change(fn=visible_1, inputs=[autotune3], outputs=[f0_autotune_strength0])
                model_pth0.change(fn=get_sid, inputs=[model_pth0], outputs=[sid0])
            with gr.Row():
                cleaner1.change(fn=visible_1, inputs=[cleaner1], outputs=[clean_strength1])
                method0.change(fn=lambda method: visible_1(True if method == "hybrid" else False), inputs=[method0], outputs=[hybrid_method0])
                method0.change(fn=hoplength_show, inputs=[method0, hybrid_method0], outputs=[hop_length0])
                hybrid_method0.change(fn=hoplength_show, inputs=[method0, hybrid_method0], outputs=[hop_length0])
            with gr.Row():
                refesh1.click(fn=change_models_choices, inputs=[], outputs=[model_pth0, model_index0])
                model_pth0.change(fn=get_index, inputs=[model_pth0], outputs=[model_index0])
                embedders0.change(fn=lambda embedders: visible_1(True if embedders == "custom" else False), inputs=[embedders0], outputs=[custom_embedders0])
            with gr.Row():
                model_index0.change(fn=index_strength_show, inputs=[model_index0], outputs=[index_strength0])
                txt_input.upload(fn=process_input, inputs=[txt_input], outputs=[prompt])
                use_txt.change(fn=visible_1, inputs=[use_txt], outputs=[txt_input])
            with gr.Row():
                google_tts.change(fn=change_tts_voice_choices, inputs=[google_tts], outputs=[tts_voice])
                tts_button.click(
                    fn=TTS, 
                    inputs=[
                        prompt, 
                        tts_voice, 
                        speed, 
                        output_audio0,
                        tts_pitch,
                        google_tts
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
                        sid0,
                        checkpointing0
                    ],
                    outputs=[tts_voice_convert],
                    api_name="convert_tts"
                )

        with gr.TabItem(translations["audio_effects"], visible=configs.get("effects_tab", True)):
            gr.Markdown(translations["apply_audio_effects"])
            with gr.Row():
                gr.Markdown(translations["audio_effects_edit"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            reverb_check_box = gr.Checkbox(label=translations["reverb"], value=False, interactive=True)
                            chorus_check_box = gr.Checkbox(label=translations["chorus"], value=False, interactive=True)
                            delay_check_box = gr.Checkbox(label=translations["delay"], value=False, interactive=True)
                        with gr.Row():
                            more_options = gr.Checkbox(label=translations["more_option"], value=False, interactive=True)    
                            phaser_check_box = gr.Checkbox(label=translations["phaser"], value=False, interactive=True)
                            compressor_check_box = gr.Checkbox(label=translations["compressor"], value=False, interactive=True)
                with gr.Column():
                    apply_effects_button = gr.Button(translations["apply"], variant="primary", scale=2)
            with gr.Row():
                with gr.Row():
                    with gr.Accordion(translations["input_output"], open=False):
                        with gr.Row():
                            upload_audio = gr.File(label=translations["drop_audio"], file_types=['audio'])
                        with gr.Row():
                            audio_in_path = gr.Dropdown(label=translations["input_audio"], value="" if len(paths_for_files) < 1 else paths_for_files[0], choices=paths_for_files, info=translations["provide_audio"], interactive=True, allow_custom_value=True)
                            audio_out_path = gr.Textbox(label=translations["output_audio"], value="audios/audio_effects.wav", placeholder="audios/audio_effects.wav", info=translations["provide_output"], interactive=True)
                        with gr.Row():
                            with gr.Column():
                                audio_combination = gr.Checkbox(label=translations["merge_instruments"], value=False, interactive=True)
                                audio_combination_input = gr.Dropdown(label=translations["input_audio"], value="" if len(paths_for_files) < 1 else paths_for_files[0], choices=paths_for_files, info=translations["provide_audio"], interactive=True, allow_custom_value=True, visible=False)
                        with gr.Row():
                            audio_output_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                            audio_effects_refesh = gr.Button(translations["refesh"])
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Accordion(translations["reverb"], open=False, visible=False) as reverb_accordion:
                            reverb_freeze_mode = gr.Checkbox(label=translations["reverb_freeze"], info=translations["reverb_freeze_info"], value=False, interactive=True)
                            reverb_room_size = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["room_size"], info=translations["room_size_info"], interactive=True)
                            reverb_damping = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["damping"], info=translations["damping_info"], interactive=True)
                            reverb_wet_level = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.3, label=translations["wet_level"], info=translations["wet_level_info"], interactive=True)
                            reverb_dry_level = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label=translations["dry_level"], info=translations["dry_level_info"], interactive=True)
                            reverb_width = gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label=translations["width"], info=translations["width_info"], interactive=True)
                    with gr.Row():
                        with gr.Accordion(translations["chorus"], open=False, visible=False) as chorus_accordion:
                            chorus_depth = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["chorus_depth"], info=translations["chorus_depth_info"], interactive=True)
                            chorus_rate_hz = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1.5, label=translations["chorus_rate_hz"], info=translations["chorus_rate_hz_info"], interactive=True)
                            chorus_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["chorus_mix"], info=translations["chorus_mix_info"], interactive=True)
                            chorus_centre_delay_ms = gr.Slider(minimum=0, maximum=50, step=1, value=10, label=translations["chorus_centre_delay_ms"], info=translations["chorus_centre_delay_ms_info"], interactive=True)
                            chorus_feedback = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label=translations["chorus_feedback"], info=translations["chorus_feedback_info"], interactive=True)
                    with gr.Row():
                        with gr.Accordion(translations["delay"], open=False, visible=False) as delay_accordion:
                            delay_second = gr.Slider(minimum=0, maximum=5, step=0.01, value=0.5, label=translations["delay_seconds"], info=translations["delay_seconds_info"], interactive=True)
                            delay_feedback = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["delay_feedback"], info=translations["delay_feedback_info"], interactive=True)
                            delay_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["delay_mix"], info=translations["delay_mix_info"], interactive=True)
                with gr.Column():
                    with gr.Row():
                        with gr.Accordion(translations["more_option"], open=False, visible=False) as more_accordion:
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
                            with gr.Accordion(translations["fade"], open=True, visible=False) as fade_accordion:
                                with gr.Row():
                                    fade_in = gr.Slider(minimum=0, maximum=10000, step=100, value=0, label=translations["fade_in"], info=translations["fade_in_info"], interactive=True)
                                    fade_out = gr.Slider(minimum=0, maximum=10000, step=100, value=0, label=translations["fade_out"], info=translations["fade_out_info"], interactive=True)
                            with gr.Accordion(translations["bass_or_treble"], open=True, visible=False) as bass_treble_accordion:
                                with gr.Row():
                                    bass_boost = gr.Slider(minimum=0, maximum=20, step=1, value=0, label=translations["bass_boost"], info=translations["bass_boost_info"], interactive=True)
                                    bass_frequency = gr.Slider(minimum=20, maximum=200, step=10, value=100, label=translations["bass_frequency"], info=translations["bass_frequency_info"], interactive=True)
                                with gr.Row():
                                    treble_boost = gr.Slider(minimum=0, maximum=20, step=1, value=0, label=translations["treble_boost"], info=translations["treble_boost_info"], interactive=True)
                                    treble_frequency = gr.Slider(minimum=1000, maximum=10000, step=500, value=3000, label=translations["treble_frequency"], info=translations["treble_frequency_info"], interactive=True)
                            with gr.Accordion(translations["limiter"], open=True, visible=False) as limiter_accordion:
                                with gr.Row():
                                    limiter_threashold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-1, label=translations["limiter_threashold_db"], info=translations["limiter_threashold_db_info"], interactive=True)
                                    limiter_release_ms = gr.Slider(minimum=10, maximum=1000, step=1, value=100, label=translations["limiter_release_ms"], info=translations["limiter_release_ms_info"], interactive=True)
                            with gr.Column():
                                pitch_shift_semitones = gr.Slider(minimum=-20, maximum=20, step=1, value=0, label=translations["pitch"], info=translations["pitch_info"], interactive=True)
                                audio_effect_resample_sr = gr.Slider(minimum=0, maximum=96000, step=1, value=0, label=translations["resample"], info=translations["resample_info"], interactive=True, visible=False)
                                distortion_drive_db = gr.Slider(minimum=0, maximum=50, step=1, value=20, label=translations["distortion"], info=translations["distortion_info"], interactive=True, visible=False)
                                gain_db = gr.Slider(minimum=-60, maximum=60, step=1, value=0, label=translations["gain"], info=translations["gain_info"], interactive=True, visible=False)
                                clipping_threashold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-1, label=translations["clipping_threashold_db"], info=translations["clipping_threashold_db_info"], interactive=True, visible=False)
                                bitcrush_bit_depth = gr.Slider(minimum=1, maximum=24, step=1, value=16, label=translations["bitcrush_bit_depth"], info=translations["bitcrush_bit_depth_info"], interactive=True, visible=False)
                    with gr.Row():
                        with gr.Accordion(translations["phaser"], open=False, visible=False) as phaser_accordion:
                            phaser_depth = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["phaser_depth"], info=translations["phaser_depth_info"], interactive=True)
                            phaser_rate_hz = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1, label=translations["phaser_rate_hz"], info=translations["phaser_rate_hz_info"], interactive=True)
                            phaser_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["phaser_mix"], info=translations["phaser_mix_info"], interactive=True)
                            phaser_centre_frequency_hz = gr.Slider(minimum=50, maximum=5000, step=10, value=1000, label=translations["phaser_centre_frequency_hz"], info=translations["phaser_centre_frequency_hz_info"], interactive=True)
                            phaser_feedback = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label=translations["phaser_feedback"], info=translations["phaser_feedback_info"], interactive=True)
                    with gr.Row():
                        with gr.Accordion(translations["compressor"], open=False, visible=False) as compressor_accordion:
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
                reverb_check_box.change(fn=visible_1, inputs=[reverb_check_box], outputs=[reverb_accordion])
                chorus_check_box.change(fn=visible_1, inputs=[chorus_check_box], outputs=[chorus_accordion])
                delay_check_box.change(fn=visible_1, inputs=[delay_check_box], outputs=[delay_accordion])
            with gr.Row():
                compressor_check_box.change(fn=visible_1, inputs=[compressor_check_box], outputs=[compressor_accordion])
                phaser_check_box.change(fn=visible_1, inputs=[phaser_check_box], outputs=[phaser_accordion])
                more_options.change(fn=visible_1, inputs=[more_options], outputs=[more_accordion])
            with gr.Row():
                fade.change(fn=visible_1, inputs=[fade], outputs=[fade_accordion])
                bass_or_treble.change(fn=visible_1, inputs=[bass_or_treble], outputs=[bass_treble_accordion])
                limiter.change(fn=visible_1, inputs=[limiter], outputs=[limiter_accordion])
                resample_checkbox.change(fn=visible_1, inputs=[resample_checkbox], outputs=[audio_effect_resample_sr])
            with gr.Row():
                distortion_checkbox.change(fn=visible_1, inputs=[distortion_checkbox], outputs=[distortion_drive_db])
                gain_checkbox.change(fn=visible_1, inputs=[gain_checkbox], outputs=[gain_db])
                clipping_checkbox.change(fn=visible_1, inputs=[clipping_checkbox], outputs=[clipping_threashold_db])
                bitcrush_checkbox.change(fn=visible_1, inputs=[bitcrush_checkbox], outputs=[bitcrush_bit_depth])
            with gr.Row():
                upload_audio.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[upload_audio], outputs=[audio_in_path])
                audio_in_path.change(fn=lambda audio: audio if audio else None, inputs=[audio_in_path], outputs=[audio_play_input])
                audio_effects_refesh.click(fn=change_audios_choices, inputs=[], outputs=[audio_in_path])
                audio_effects_refesh.click(fn=change_audios_choices, inputs=[], outputs=[audio_combination_input])
            with gr.Row():
                more_options.change(fn=lambda: [False]*4, inputs=[], outputs=[fade, bass_or_treble, limiter, resample_checkbox])
                more_options.change(fn=lambda: [False]*4, inputs=[], outputs=[distortion_checkbox, gain_checkbox, clipping_checkbox, bitcrush_checkbox])
                audio_combination.change(fn=visible_1, inputs=[audio_combination], outputs=[audio_combination_input])
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
                        audio_combination_input
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
            with gr.Row():
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            with gr.Row():
                                separator_audio = gr.Checkbox(label=translations["separator_tab"], value=False, interactive=True)
                                clean_audio = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                                skip = gr.Checkbox(label=translations["skip"], value=False, interactive=True)
                        with gr.Row():
                            sample_rate0 = gr.Slider(minimum=0, maximum=96000, step=1, value=44100, label=translations["sr"], info=translations["sr_info"], interactive=True)
                            dataset_clean_strength = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label=translations["clean_strength"], info=translations["clean_strength_info"], interactive=True, visible=False)
                    with gr.Column():
                        create_button = gr.Button(translations["createdataset"], variant="primary", scale=2)
            with gr.Row():
                with gr.Column():
                    with gr.Group(visible=False) as separator_dataset:
                        with gr.Row():
                            separator_reverb = gr.Checkbox(label=translations["dereveb_audio"], value=False, interactive=False)
                            denoise_mdx = gr.Checkbox(label=translations["denoise"], value=False, interactive=False)
                        with gr.Row():
                            kim_vocal_version = gr.Radio(label=translations["model_ver"], info=translations["model_ver_info"], choices=["Version-1", "Version-2"], value="Version-2", interactive=True)
                            kim_vocal_overlap = gr.Radio(label=translations["overlap"], info=translations["overlap_info"], choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True)
                        with gr.Row():
                            kim_vocal_segments_size = gr.Slider(label=translations["segments_size"], info=translations["segments_size_info"], minimum=32, maximum=3072, value=256, step=32, interactive=True)
                            kim_vocal_hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=8192, value=1024, step=1, interactive=True)
                        with gr.Row():
                            kim_vocal_batch_size = gr.Slider(label=translations["batch_size"], info=translations["mdx_batch_size_info"], minimum=1, maximum=64, value=1, step=1, interactive=True)
                    with gr.Row():
                        create_dataset_info = gr.Textbox(label=translations["create_dataset_info"], value="", interactive=False)
                with gr.Row():
                    with gr.Column():
                        output_dataset = gr.Textbox(label=translations["output_data"], info=translations["output_data_info"], value="dataset", placeholder="dataset", interactive=True)
                        with gr.Row():
                            skip_start = gr.Textbox(label=translations["skip_start"], info=translations["skip_start_info"], value="", placeholder="0,...", interactive=True, visible=False)
                            skip_end = gr.Textbox(label=translations["skip_end"], info=translations["skip_end_info"], value="", placeholder="0,...", interactive=True, visible=False)
            with gr.Row():
                separator_audio.change(fn=visible_1, inputs=[separator_audio], outputs=[separator_dataset])
                separator_audio.change(fn=valueFalse_interactive1, inputs=[separator_audio], outputs=[separator_reverb])
                separator_audio.change(fn=valueFalse_interactive1, inputs=[separator_audio], outputs=[denoise_mdx])
            with gr.Row():
                clean_audio.change(fn=visible_1, inputs=[clean_audio], outputs=[dataset_clean_strength])
                skip.change(fn=valueEmpty_visible1, inputs=[skip], outputs=[skip_start])
                skip.change(fn=valueEmpty_visible1, inputs=[skip], outputs=[skip_end])
            with gr.Row():
                create_button.click(
                    fn=create_dataset,
                    inputs=[
                        dataset_url, 
                        output_dataset, 
                        clean_audio, 
                        dataset_clean_strength, 
                        separator_audio, 
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
                            training_sr = gr.Radio(label=translations["sample_rate"], info=translations["sample_rate_info"], choices=["32k", "40k", "44.1k", "48k"], value="48k", interactive=True) 
                            training_ver = gr.Radio(label=translations["training_version"], info=translations["training_version_info"], choices=["v1", "v2"], value="v2", interactive=True) 
                            with gr.Row():
                                clean_dataset = gr.Checkbox(label=translations["clear_dataset"], value=False, interactive=True)
                                checkpointing1 = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)
                                training_f0 = gr.Checkbox(label=translations["training_pitch"], info=translations["training_pitch_info"], value=True, interactive=True)
                                upload = gr.Checkbox(label=translations["upload"], info=translations["upload_dataset"], value=False, interactive=True)
                                preprocess_cut = gr.Checkbox(label=translations["split_audio"], info=translations["preprocess_split"], value=True, interactive=True)
                                process_effects = gr.Checkbox(label=translations["preprocess_effect"], info=translations["preprocess_effect_info"], value=False, interactive=True)
                            with gr.Row():
                                clean_dataset_strength = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.7, step=0.1, interactive=True, visible=False)
                        with gr.Column():
                            preprocess_button = gr.Button(translations["preprocess_button"], scale=2)
                            upload_dataset = gr.Files(label=translations["drop_audio"], file_types=['audio'], visible=False)
                            preprocess_info = gr.Textbox(label=translations["preprocess_info"], value="", interactive=False)
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            extract_method = gr.Radio(label=translations["f0_method"], info=translations["f0_method"], choices=["pm", "dio", "crepe", "crepe-tiny", "fcpe", "rmvpe", "harvest"], value="pm", interactive=True)
                            extract_hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                            with gr.Accordion(label=translations["hubert_model"], open=False):
                                extract_embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "Hidden_Rabbit_last", "portuguese_hubert_base"], value="contentvec_base", interactive=True)
                                with gr.Row():
                                    extract_embedders_custom = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=False)
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
                                cache_in_gpu = gr.Checkbox(label=translations["cache_in_gpu"], info=translations["cache_in_gpu_info"], value=False, interactive=True)
                            with gr.Column():
                                dataset_path = gr.Textbox(label=translations["dataset_folder"], value="dataset", interactive=True, visible=False)
                            with gr.Column():
                                threshold = gr.Slider(minimum=1, maximum=100, value=50, step=1, label=translations["threshold"], interactive=True, visible=False)
                                with gr.Accordion(translations["setting_cpu_gpu"], open=False):
                                    with gr.Column():
                                        gpu_number = gr.Textbox(label=translations["gpu_number"], value=str("-".join(map(str, range(torch.cuda.device_count()))) if torch.cuda.is_available() else "-"), info=translations["gpu_number_info"], interactive=True)
                                        gpu_info = gr.Textbox(label=translations["gpu_info"], value=get_gpu_info(), info=translations["gpu_info_2"], interactive=False)
                                        cpu_core = gr.Slider(label=translations["cpu_core"], info=translations["cpu_core_info"], minimum=0, maximum=cpu_count(), value=cpu_count(), step=1, interactive=True)          
                                        train_batch_size = gr.Slider(label=translations["batch_size"], info=translations["batch_size_info"], minimum=1, maximum=64, value=8, step=1, interactive=True)
                            with gr.Row():
                                with gr.Row():
                                    save_only_latest = gr.Checkbox(label=translations["save_only_latest"], info=translations["save_only_latest_info"], value=True, interactive=True)
                                    save_every_weights = gr.Checkbox(label=translations["save_every_weights"], info=translations["save_every_weights_info"], value=True, interactive=True)
                                    not_use_pretrain = gr.Checkbox(label=translations["not_use_pretrain_2"], info=translations["not_use_pretrain_info"], value=False, interactive=True)
                                    custom_pretrain = gr.Checkbox(label=translations["custom_pretrain"], info=translations["custom_pretrain_info"], value=False, interactive=True)
                            with gr.Row():
                                with gr.Row():
                                    model_author = gr.Textbox(label=translations["training_author"], info=translations["training_author_info"], value="", placeholder=translations["training_author"], interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    with gr.Accordion(translations["custom_pretrain_info"], open=False, visible=False) as pretrain_setting:
                                        pretrained_D = gr.Dropdown(label=translations["pretrain_file"].format(dg="D"), choices=sorted(pretrainedD), value=sorted(pretrainedD)[0] if len(sorted(pretrainedD)) > 0 else '', interactive=True, allow_custom_value=True, visible=False)
                                        pretrained_G = gr.Dropdown(label=translations["pretrain_file"].format(dg="G"), choices=sorted(pretrainedG), value=sorted(pretrainedG)[0] if len(sorted(pretrainedG)) > 0 else '', interactive=True, allow_custom_value=True, visible=False)
                                        refesh_pretrain = gr.Button(translations["refesh_pretrain"], scale=2, visible=False)
                    with gr.Row():
                        training_info = gr.Textbox(label=translations["train_info"], value="", interactive=False)
                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion(translations["export_model"], open=False):
                                with gr.Row():
                                    model_file= gr.Dropdown(label=translations["model_name"], choices=sorted(model_name), value=sorted(model_name)[0] if len(sorted(model_name)) > 0 else '', interactive=True, allow_custom_value=True)
                                    index_file = gr.Dropdown(label=translations["index_path"], choices=sorted(index_path), value=sorted(index_path)[0] if len(sorted(index_path)) > 0 else '', interactive=True, allow_custom_value=True)
                                with gr.Row():
                                    refesh_file = gr.Button(f"1. {translations['refesh']}", scale=2)
                                    zip_model = gr.Button(translations["zip_model"], variant="primary", scale=2)
                                with gr.Row():
                                    zip_output = gr.File(label=translations["output_zip"], file_types=['zip'], interactive=False, visible=False)
            with gr.Row():
                refesh_file.click(fn=change_models_choices, inputs=[], outputs=[model_file, index_file]) 
                zip_model.click(fn=lambda: visible_1(True), inputs=[], outputs=[zip_output])           
                zip_model.click(fn=zip_file, inputs=[training_name, model_file, index_file], outputs=[zip_output])                
            with gr.Row():
                dataset_path.change(
                    fn=lambda folder: os.makedirs(folder, exist_ok=True),
                    inputs=[dataset_path],
                    outputs=[],
                    api_name="create_folder"
                )
                upload.change(fn=visible_1, inputs=[upload], outputs=[upload_dataset]) 
                overtraining_detector.change(fn=visible_1, inputs=[overtraining_detector], outputs=[threshold]) 
                clean_dataset.change(fn=visible_1, inputs=[clean_dataset], outputs=[clean_dataset_strength])
            with gr.Row():
                custom_dataset.change(fn=lambda custom_dataset: [visible_1(custom_dataset), "dataset"],inputs=[custom_dataset], outputs=[dataset_path, dataset_path])
                upload_dataset.upload(
                    fn=lambda files, folder: [shutil.move(f.name, os.path.join(folder, os.path.split(f.name)[1])) for f in files] if folder != "" else gr_warning('Vui lòng nhập tên thư mục dữ liệu'),
                    inputs=[upload_dataset, dataset_path], 
                    outputs=[], 
                    api_name="upload_dataset"
                )           
            with gr.Row():
                not_use_pretrain.change(fn=lambda a, b: [visible_1(a and not b), visible_1(a and not b), visible_1(a and not b), visible_1(a and not b)], inputs=[custom_pretrain, not_use_pretrain], outputs=[pretrained_D, pretrained_G, refesh_pretrain, pretrain_setting])
                custom_pretrain.change(fn=lambda a, b: [visible_1(a and not b), visible_1(a and not b), visible_1(a and not b), visible_1(a and not b)], inputs=[custom_pretrain, not_use_pretrain], outputs=[pretrained_D, pretrained_G, refesh_pretrain, pretrain_setting])
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
                extract_method.change(fn=hoplength_show, inputs=[extract_method], outputs=[extract_hop_length])
                extract_embedders.change(fn=lambda extract_embedders: visible_1(True if extract_embedders == "custom" else False), inputs=[extract_embedders], outputs=[extract_embedders_custom])
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
                        extract_embedders_custom
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
                        checkpointing1
                    ],
                    outputs=[training_info],
                    api_name="training_model"
                )

        with gr.TabItem(translations["fushion"], visible=configs.get("fushion_tab", True)):
            gr.Markdown(translations["fushion_markdown"])
            with gr.Row():
                gr.Markdown(translations["fushion_markdown_2"])
            with gr.Row():
                with gr.Column():
                    name_to_save = gr.Textbox(label=translations["modelname"], placeholder="Model.pth", value="", max_lines=1, interactive=True)
                with gr.Column():
                    fushion_button = gr.Button(translations["fushion"], variant="primary", scale=4)
            with gr.Column():
                with gr.Row():
                    model_a = gr.File(label=f"{translations['model_name']} 1", file_types=['pth']) 
                    model_b = gr.File(label=f"{translations['model_name']} 2", file_types=['pth'])
                with gr.Row():
                    model_path_a = gr.Textbox(label=f"{translations['model_path']} 1", value="", placeholder="assets/weights/Model_1.pth")
                    model_path_b = gr.Textbox(label=f"{translations['model_path']} 2", value="", placeholder="assets/weights/Model_2.pth")
            with gr.Row():
                ratio = gr.Slider(minimum=0, maximum=1, label=translations["model_ratio"], info=translations["model_ratio_info"], value=0.5, interactive=True)
            with gr.Row():
                output_model = gr.File(label=translations["output_model_path"], visible=False)
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
                fushion_button.click(fn=lambda: visible_1(True), inputs=[], outputs=[output_model])  

        with gr.TabItem(translations["read_model"], visible=configs.get("read_tab", True)):
            gr.Markdown(translations["read_model_markdown"])
            with gr.Row():
                gr.Markdown(translations["read_model_markdown_2"])
            with gr.Row():
                with gr.Column():
                    model = gr.File(label=translations["drop_model"], file_types=['pth']) 
                with gr.Column():
                    read_button = gr.Button(translations["readmodel"], variant="primary", scale=2)
            with gr.Column():
                model_path = gr.Textbox(label=translations["download_url"], value="", info=translations["model_path_info"], interactive=True)
                output_info = gr.Textbox(label=translations["modelinfo"], value="", interactive=False, scale=6)
            with gr.Row():
                model.upload(fn=lambda model: shutil.move(model.name, os.path.join("assets", "weights")), inputs=[model], outputs=[model_path])
                read_button.click(
                    fn=model_info,
                    inputs=[model_path],
                    outputs=[output_info],
                    api_name="read_model"
                )

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
                    with gr.Row():
                        url_input = gr.Textbox(label=translations["model_url"], value="", placeholder="https://...", scale=6, visible=True)
                        download_model_name = gr.Textbox(label=translations["modelname"], value="", placeholder=translations["modelname"], scale=2, visible=True)
                        url_download = gr.Button(value=translations["downloads"], scale=2, visible=True)
                    with gr.Row():
                        model_browser = gr.Dropdown(choices=models.keys(), label=translations["model_warehouse"], scale=8, allow_custom_value=True, visible=False)
                        download_from_browser = gr.Button(value=translations["get_model"], scale=2, variant="primary", visible=False)
                    with gr.Row():
                        model_upload = gr.File(label=translations["drop_model"], file_types=['pth', 'index', 'zip'], visible=False)
                    with gr.Column():
                        with gr.Row():
                            search_name = gr.Textbox(label=translations["name_to_search"], placeholder=translations["modelname"], interactive=True, scale=8, visible=False)
                            search = gr.Button(translations["search_2"], scale=2, visible=False)
                        with gr.Row():
                            search_dropdown = gr.Dropdown(label=translations["select_download_model"], value="", choices=[], allow_custom_value=True, interactive=False, visible=False)
                            download = gr.Button(translations["downloads"], variant="primary", visible=False)
            with gr.Row():
                with gr.Accordion(translations["download_pretrained_2"], open=False):
                    with gr.Row():
                        pretrain_download_choices = gr.Radio(label=translations["model_download_select"], choices=[translations["download_url"], translations["list_model"], translations["upload"]], value=translations["download_url"], interactive=True)  
                    with gr.Row():
                        gr.Markdown("___")
                    with gr.Row():
                        pretrainD = gr.Textbox(label=translations["pretrained_url"].format(dg="D"), value="", info=translations["only_huggingface"], placeholder="https://...", interactive=True, scale=4, visible=True)
                        pretrainG = gr.Textbox(label=translations["pretrained_url"].format(dg="G"), value="", info=translations["only_huggingface"], placeholder="https://...", interactive=True, scale=4, visible=True)
                        download_pretrain_button = gr.Button(translations["downloads"], scale=2)
                    with gr.Row():
                        pretrain_choices = gr.Dropdown(label=translations["select_pretrain"], info=translations["select_pretrain_info"], choices=list(fetch_pretrained_data().keys()), value="Titan_Medium", allow_custom_value=True, interactive=True, scale=6, visible=False)
                        sample_rate_pretrain = gr.Dropdown(label=translations["pretrain_sr"], choices=["48k", "40k", "32k"], value="48k", interactive=True, visible=False)
                        download_pretrain_choices_button = gr.Button(translations["downloads"], scale=2, variant="primary", visible=False)
                    with gr.Row():
                        pretrain_upload_g = gr.File(label=translations["drop_pretrain"].format(dg="G"), file_types=['pth'], visible=False)
                        pretrain_upload_d = gr.File(label=translations["drop_pretrain"].format(dg="D"), file_types=['pth'], visible=False)
            with gr.Row():
                with gr.Accordion(translations["hubert_download"], open=False):
                    with gr.Row():
                        hubert_url = gr.Textbox(label=translations["hubert_url"], value="", info=translations["only_huggingface"], placeholder="https://...", interactive=True, scale=8)
                        hubert_button = gr.Button(translations["downloads"], scale=2, variant="primary")
                    with gr.Row():
                        hubert_input = gr.File(label=translations["drop_hubert"], file_types=['pt'])    
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
            with gr.Row():
                hubert_button.click(
                    fn=hubert_download,
                    inputs=[hubert_url],
                    outputs=[hubert_url],
                    api_name="hubert_download"
                )
                hubert_input.upload(
                    fn=lambda hubert: shutil.move(hubert.name, os.path.join("assets", "models", "embedders")), 
                    inputs=[hubert_input], 
                    outputs=[],
                    api_name="upload_hubert"
                )

        with gr.TabItem(translations["settings"], visible=configs.get("settings_tab", True)):
            gr.Markdown(translations["settings_markdown"])
            with gr.Row():
                gr.Markdown(translations["settings_markdown_2"])
            with gr.Row():
                with gr.Column():
                    language_dropdown = gr.Dropdown(label=translations["lang"], interactive=True, info=translations["lang_restart"], choices=configs.get("support_language", "vi-VN"), value=language)
                    change_lang = gr.Button(translations["change_lang"], variant=["primary"], scale=2)
                with gr.Column():
                    toggle_button = gr.Button(translations["change_light_dark"], variant=["secondary"], scale=2)
            with gr.Row():
                with gr.Column():
                    fp_select = gr.Radio(label=translations["fp_train"], info=translations["fp_info"], value="fp32", choices=["fp16", "fp32"], interactive=True)
                    fp_button = gr.Button(translations["fp_button"], variant=["primary"], scale=2)
                with gr.Column():
                    theme_dropdown = gr.Dropdown(label=translations["theme"], interactive=True, info=translations["theme_restart"], choices=configs.get("themes", theme), value=theme, allow_custom_value=True)
                    changetheme = gr.Button(translations["theme_button"], variant=["primary"], scale=2)
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(translations["stop"], open=False):
                        separate_stop = gr.Button(translations["stop_separate"])
                        convert_stop = gr.Button(translations["stop_convert"])
                        create_dataset_stop = gr.Button(translations["stop_create_dataset"])
                        with gr.Accordion(translations["stop_training"], open=False):
                            model_name_stop = gr.Textbox(label=translations["modelname"], info=translations["training_model_name"], value="", placeholder=translations["modelname"], interactive=True)
                            preprocess_stop = gr.Button(translations["stop_preprocess"])
                            extract_stop = gr.Button(translations["stop_extract"])
                            train_stop = gr.Button(translations["stop_training"])
                with gr.Column():
                    with gr.Accordion(translations["cleaner"], open=False):
                        with gr.Accordion(translations["clean_audio"], open=False):
                            with gr.Row():
                                audio_file_select = gr.Dropdown(label=translations["audio_path"], value="" if len(paths_for_files) < 1 else paths_for_files[0], choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                                refesh_audio_select = gr.Button(translations["refesh"])
                            with gr.Row():
                                delete_all_audio = gr.Button(translations["clean_all"])
                                delete_audio = gr.Button(translations["clean_file"], variant=["primary"])
                        with gr.Accordion(translations["clean_models"], open=False):
                            with gr.Row():
                                model_select = gr.Dropdown(label=translations["model_name"], choices=sorted(model_name), value=sorted(model_name)[0] if len(sorted(model_name)) > 0 else '', interactive=True, allow_custom_value=True)
                                index_select = gr.Dropdown(label=translations["index_path"], choices=sorted(delete_index), value=sorted(delete_index)[0] if len(sorted(delete_index)) > 0 else '', interactive=True, allow_custom_value=True)
                            with gr.Row():
                                refesh_model_select = gr.Button(translations["refesh"])
                            with gr.Row():
                                delete_all_model_button = gr.Button(translations["clean_all"])
                                delete_model_button = gr.Button(translations["clean_file"], variant=["primary"])
                        with gr.Accordion(translations["clean_pretrained"], open=False):
                            with gr.Row():
                                pretrain_select = gr.Dropdown(label=translations["pretrain_file"].format(dg=" "), choices=sorted(Allpretrained), value=sorted(Allpretrained)[0] if len(sorted(Allpretrained)) > 0 else '', interactive=True, allow_custom_value=True)
                                refesh_pretrain_select = gr.Button(translations["refesh"])
                            with gr.Row():
                                delete_all_pretrain = gr.Button(translations["clean_all"])
                                delete_pretrain = gr.Button(translations["clean_file"], variant=["primary"])
                        with gr.Accordion(translations["clean_separated"], open=False):
                            with gr.Row():
                                separate_select = gr.Dropdown(label=translations["separator_model"], choices=sorted(separate_model), value=sorted(separate_model)[0] if len(sorted(separate_model)) > 0 else '', interactive=True, allow_custom_value=True)
                                refesh_separate_select = gr.Button(translations["refesh"])
                            with gr.Row():
                                delete_all_separate = gr.Button(translations["clean_all"])
                                delete_separate = gr.Button(translations["clean_file"], variant=["primary"])
                        with gr.Accordion(translations["clean_presets"], open=False):
                            with gr.Row():
                                presets_select = gr.Dropdown(label=translations["file_preset"], choices=sorted(presets_file), value=sorted(presets_file)[0] if len(sorted(presets_file)) > 0 else '', interactive=True, allow_custom_value=True)
                                refesh_presets_select = gr.Button(translations["refesh"])
                            with gr.Row():
                                delete_all_presets_button = gr.Button(translations["clean_all"])
                                delete_presets_button = gr.Button(translations["clean_file"], variant=["primary"])
                        with gr.Accordion(translations["clean_datasets"], open=False):
                            dataset_folder_name = gr.Textbox(label=translations["dataset_folder"], value="dataset", interactive=True)
                            delete_dataset_button = gr.Button(translations["clean_dataset_folder"], variant=["primary"])
                        with gr.Row():
                            clean_log = gr.Button(translations["clean_log"], variant=["primary"])
                            clean_predictor = gr.Button(translations["clean_predictors"], variant=["primary"])
                            clean_embedders = gr.Button(translations["clean_embed"], variant=["primary"])
            with gr.Row():
                toggle_button.click(fn=None, js="() => {document.body.classList.toggle('dark')}")
                fp_button.click(fn=change_fp, inputs=[fp_select], outputs=[])
            with gr.Row():
                change_lang.click(fn=change_language, inputs=[language_dropdown], outputs=[])
                changetheme.click(fn=change_theme, inputs=[theme_dropdown], outputs=[])
            with gr.Row():
                change_lang.click(fn=None, js="setTimeout(function() {location.reload()}, 15000)", inputs=[], outputs=[])
                changetheme.click(fn=None, js="setTimeout(function() {location.reload()}, 15000)", inputs=[], outputs=[])
            with gr.Row():
                separate_stop.click(fn=lambda: stop_pid("separate_pid", None), inputs=[], outputs=[])
                convert_stop.click(fn=lambda: stop_pid("convert_pid", None), inputs=[], outputs=[])
                create_dataset_stop.click(fn=lambda: stop_pid("create_dataset_pid", None), inputs=[], outputs=[])
            with gr.Row():
                preprocess_stop.click(fn=lambda model_name_stop: stop_pid("preprocess_pid", model_name_stop), inputs=[model_name_stop], outputs=[])
                extract_stop.click(fn=lambda model_name_stop: stop_pid("extract_pid", model_name_stop), inputs=[model_name_stop], outputs=[])
                train_stop.click(fn=lambda model_name_stop: stop_train(model_name_stop), inputs=[model_name_stop], outputs=[])
            with gr.Row():
                refesh_audio_select.click(fn=change_audios_choices, inputs=[], outputs=[audio_file_select])
                delete_all_audio.click(fn=delete_all_audios, inputs=[], outputs=[audio_file_select])
                delete_audio.click(fn=delete_audios, inputs=[audio_file_select], outputs=[audio_file_select])
            with gr.Row():
                refesh_model_select.click(fn=change_choices_del, inputs=[], outputs=[model_select, index_select])
                delete_all_model_button.click(fn=delete_all_model, inputs=[], outputs=[model_select, index_select])
                delete_model_button.click(fn=delete_model, inputs=[model_select, index_select], outputs=[model_select, index_select])
            with gr.Row():
                refesh_pretrain_select.click(fn=change_allpretrained_choices, inputs=[], outputs=[pretrain_select])
                delete_all_pretrain.click(fn=delete_all_pretrained, inputs=[], outputs=[pretrain_select])
                delete_pretrain.click(fn=delete_pretrained, inputs=[pretrain_select], outputs=[pretrain_select])
            with gr.Row():
                refesh_separate_select.click(fn=change_separate_choices, inputs=[], outputs=[separate_select])
                delete_all_separate.click(fn=delete_all_separated, inputs=[], outputs=[separate_select])
                delete_separate.click(fn=delete_separated, inputs=[separate_select], outputs=[separate_select])
            with gr.Row():
                refesh_presets_select.click(fn=change_preset_choices, inputs=[], outputs=[presets_select])
                delete_all_presets_button.click(fn=delete_all_presets, inputs=[], outputs=[presets_select])
                delete_presets_button.click(fn=delete_presets, inputs=[presets_select], outputs=[presets_select])
            with gr.Row():
                delete_dataset_button.click(fn=delete_dataset, inputs=[dataset_folder_name], outputs=[])
            with gr.Row():
                clean_log.click(fn=delete_all_log, inputs=[], outputs=[])
                clean_predictor.click(fn=delete_all_predictors, inputs=[], outputs=[])
                clean_embedders.click(fn=delete_all_embedders, inputs=[], outputs=[])

        with gr.TabItem(translations["report_bugs"]):
            gr.Markdown(translations["report_bugs"])
            with gr.Row():
                gr.Markdown(translations["report_bug_info"])
            with gr.Row():
                with gr.Column():
                    agree_log = gr.Checkbox(label=translations["agree_log"], value=True, interactive=True) 
                    report_text = gr.Textbox(label=translations["error_info"], info=translations["error_info_2"], interactive=True)
                with gr.Column():
                    report_button = gr.Button(translations["report_bugs"], variant=["primary"], scale=2)
            with gr.Row():
                gr.Markdown(translations["report_info"].format(github=codecs.decode("uggcf://tvguho.pbz/CunzUhlauNau16/Ivrganzrfr-EIP/vffhrf", "rot13")))
            with gr.Row():
                report_button.click(fn=report_bug, inputs=[report_text, agree_log], outputs=[])

        with gr.TabItem(translations["source"]):
            gr.Markdown(translations["source_info"])
            with gr.Row():
                gr.Markdown("___")
            with gr.Row():
                gr.Markdown(translations["credits"].format(author=codecs.decode("uggcf://tvguho.pbz/CunzUhlauNau16", "rot13"), applio=codecs.decode("uggcf://tvguho.pbz/VNUvfcnab/Nccyvb/gerr/znva?gno=ernqzr-bi-svyr", "rot13"), ai_hispano=codecs.decode("uggcf://tvguho.pbz/VNUvfcnab", "rot13"), rvc_webui=codecs.decode("uggcf://tvguho.pbz/EIP-Cebwrpg/Ergevriny-onfrq-Ibvpr-Pbairefvba-JroHV?gno=ernqzr-bi-svyr", "rot13"), rvc_boss=codecs.decode("uggcf://tvguho.pbz/EIP-Obff", "rot13"), python_audio_separator=codecs.decode("uggcf://tvguho.pbz/abznqxnenbxr/clguba-nhqvb-frcnengbe?gno=ernqzr-bi-svyr", "rot13"), andrew_beveridge=codecs.decode("uggcf://tvguho.pbz/orirenqo", "rot13")))
    
    logger.info(translations["start_app"])
    logger.info(translations["set_lang"].format(lang=language))

    port = configs.get("app_port", 7860)
    for i in range(configs["num_of_restart"]):
        try:
            app.queue().launch(favicon_path=os.path.join("assets", "miku.png"), server_name=configs.get("server_name", "0.0.0.0"), server_port=port, show_error=configs.get("app_show_error", False), inbrowser="--open" in sys.argv, share="--share" in sys.argv)
            break
        except OSError:
            logger.debug(translations["port"].format(port=port))
            port -= 1
        except Exception as e:
            logger.error(translations["error_occurred"].format(e=e))