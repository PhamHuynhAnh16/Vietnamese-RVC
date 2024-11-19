import os
import re
import sys
import json
import torch
import codecs
import yt_dlp
import shutil
import zipfile
import logging
import platform
import edge_tts
import requests
import warnings
import threading

import gradio as gr
import pandas as pd

from time import sleep
from datetime import datetime
from pydub import AudioSegment
from subprocess import Popen, run
from collections import OrderedDict
from multiprocessing import cpu_count

now_dir = os.getcwd()
sys.path.append(now_dir)

from main.configs.config import Config
from main.tools import gdown, meganz, mediafire, pixeldrain


logging.getLogger("wget").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

config = Config()
python = sys.executable
translations = config.translations

model_name = []
index_path = []

pretrainedD = []
pretrainedG = []

models = {}
model_options = {}


miku_image = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Pbyno_EIP_Cebwrpg_2/erfbyir/znva/zvxh.cat", "rot13")

model_search_csv = codecs.decode("uggcf://qbpf.tbbtyr.pbz/fcernqfurrgf/q/1gNHnDeRULtEfz1Yieaw14USUQjWJy0Oq9k0DrCrjApb/rkcbeg?sbezng=pfi&tvq=1977693859", "rot13")
model_search_api = codecs.decode("rlWuoTpvBvWVHmV1AvVfVaE5pPV6VxcKIPW9.rlWcp3ZvBvWmqKOuLzSmMFVfVaWyMvV6VzAdqTMkrzczMTygM3O2pUqbrzk2Vvjvpz9fMFV6VzSho24vYPWcLKDvBwR3ZwL5ZwLkZmDfVzI4pPV6ZwN0ZwHjZwRmAU0.BlQKyuiU6Q-VfUvJuCNTHgfCTTHiJDlaskHrDjsLGbR", "rot13")

pretrained_json = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Pbyno_EIP_Cebwrpg_2/enj/znva/cergenva_pubvprf.wfba", "rot13")
hugging_face_codecs = codecs.decode("uggcf://uhttvatsnpr.pb", "rot13")

pretrained_v1_link = codecs.decode("uggcf://uhttvatsnpr.pb/VNUvfcnab/Nccyvb/erfbyir/znva/Erfbheprf/cergenvarq_i1/", "rot13")
pretrained_v2_link = codecs.decode("uggcf://uhttvatsnpr.pb/yw1995/IbvprPbairefvbaJroHV/erfbyir/znva/cergenvarq_i2/", "rot13")

configs_json = os.path.join("main", "configs", "config.json")

with open(configs_json, "r") as f:
    configs = json.load(f)

theme = configs["theme"]
server_name = configs["server_name"]
port = configs["app_port"]
show_error = configs["app_show_error"]
share = configs["share"]
tts_voice = configs["tts_voice"] 


if not theme: theme = "NoCrypt/miku"
if not server_name: server_name = "0.0.0.0"
if not port: port = 7860
if not tts_voice: tts_voice = ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]


if not os.path.exists(os.path.join("assets", "miku.png")): run(["wget", "-q", "--show-progress", "--no-check-certificate", miku_image, "-P", os.path.join("assets")], check=True)


for model in os.listdir(os.path.join("assets", "weights")):
    if model.endswith(".pth") and not model.startswith("G_") and not model.startswith("D_"): model_name.append(model)


for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False):
    for name in files:
        if name.endswith(".index"): index_path.append(os.path.join(root, name))


for model in os.listdir(os.path.join("assets", "model", "pretrained_custom")):
    if model.endswith(".pth") and "D" in model: pretrainedD.append(model)
    if model.endswith(".pth") and "G" in model: pretrainedG.append(model)


if os.path.exists("spreadsheet.csv"): cached_data = pd.read_csv("spreadsheet.csv") 
else:
    cached_data = pd.read_csv(model_search_csv)
    cached_data.to_csv("spreadsheet.csv", index=False)


for _, row in cached_data.iterrows():
    filename = row['Filename']
    url = None

    for value in row.values:
        if isinstance(value, str) and "huggingface" in value:
            url = value
            break

    if url: models[filename] = url



def get_number_of_gpus():
    return "-".join(map(str, range(torch.cuda.device_count()))) if torch.cuda.is_available() else "-"


def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = []


    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            mem = int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4)
            gpu_infos.append(f"{i}: {gpu_name} ({mem} GB)") 

    return "\n".join(gpu_infos) if len(gpu_infos) > 0 else translations["no_support_gpu"]


def change_choices_pretrained():
    pretrainedD = []
    pretrainedG = []


    for model in os.listdir(os.path.join("assets", "model", "pretrained_custom")):
        if model.endswith(".pth") and "D" in model: pretrainedD.append(model)

    for model in os.listdir(os.path.join("assets", "model", "pretrained_custom")):
        if model.endswith(".pth") and "G" in model: pretrainedG.append(model)

    return [{"choices": sorted(pretrainedD), "__type__": "update"}, {"choices": sorted(pretrainedG), "__type__": "update"}]


def change_choices():
    model_name = []
    index_path = []


    for name in os.listdir(os.path.join("assets", "weights")):
        if name.endswith(".pth"): model_name.append(name)

    for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False):
        for name in files:
            if name.endswith(".index"): index_path.append(f"{root}/{name}")


    return [{"choices": sorted(model_name), "__type__": "update"}, {"choices": sorted(index_path), "__type__": "update"}]


def get_index(model):
    return {"value": next((f for f in [os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".index")] if model.split(".")[0] in f), ""), "__type__": "update"}


def visible_1(value):
    return {"visible": value, "__type__": "update"}


def interactive_1(value):
    return {"interactive": value, "__type__": "update"}


def valueFalse_interactive1(inp): 
    return {"value": False, "interactive": inp, "__type__": "update"}


def valueFalse_interactive2(inp1, inp2): 
    return {"value": False, "interactive": inp1 and inp2, "__type__": "update"}


def valueFalse_visible1(inp1): 
    return {"value": False, "visible": inp1, "__type__": "update"}


def valueEmpty_visible1(inp1): 
    return {"value": "", "visible": inp1, "__type__": "update"}


def refesh_audio(): 
    paths_for_files = [os.path.abspath(os.path.join("audios", f)) for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a')]

    return {"value": "" if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files[0], "choices": [] if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files, "__type__": "update"}


def backing_change(backing, merge):
    if backing or merge: return {"value": False, "interactive": False, "__type__": "update"}
    elif not backing or not merge: return  {"interactive": True, "__type__": "update"}


def model_separator_change(mdx):
    if not mdx: choices = ["HT-Normal", "HT-Tuned", "HD_MMI", "HT_6S"]
    else: choices = ["Main_340", "Main_390", "Main_406", "Main_427", "Main_438", "Inst_full_292", "Inst_HQ_1", "Inst_HQ_2", "Inst_HQ_3", "Inst_HQ_4", "Kim_Vocal_1", "Kim_Vocal_2", "Kim_Inst", "Inst_187_beta", "Inst_82_beta", "Inst_90_beta", "Voc_FT", "Crowd_HQ", "Inst_1", "Inst_2", "Inst_3", "MDXNET_1_9703", "MDXNET_2_9682", "MDXNET_3_9662", "Inst_Main", "MDXNET_Main", "MDXNET_9482"]


    return {"value": choices[0], "choices": choices, "__type__": "update"}


def hoplength_show(method, hybrid_method=None):
    if method in ["crepe-tiny", "crepe", "fcpe"]: visible = True
    elif method == "hybrid":
        methods_str = re.search("hybrid\[(.+)\]", hybrid_method)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        visible = methods[0] in ["crepe-tiny", "crepe", "fcpe"] or methods[1] in ["crepe-tiny", "crepe", "fcpe"]
    else: visible = False
    

    return {"visible": visible, "__type__": "update"}


def process_input(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        file_contents = file.read()

    gr.Info(translations["upload_success"].format(name=translations["text"]))


    return file_contents


def download_change(select):
    selects = [False]*10


    if select == translations["download_url"]: selects[0] = selects[1] = selects[2] = True
    elif select == translations["download_from_csv"]:  selects[3] = selects[4] = True
    elif select == translations["download_from_applio"]: selects[5] = selects[6] = True
    elif select == translations["upload"]: selects[9] = True
    else: gr.Warning(translations["option_not_valid"])
    

    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]


def fetch_pretrained_data():
    response = requests.get(pretrained_json)
    response.raise_for_status()

    return response.json()


def download_pretrained_change(select):
    selects = [False]*8


    if select == translations["download_url"]: selects[0] = selects[1] = selects[2] = True
    elif select == translations["list_model"]: selects[3] = selects[4] = selects[5] = True
    elif select == translations["upload"]: selects[6] = selects[7] = True
    else: gr.Warning(translations["option_not_valid"])


    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]


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
    
    if platform.system() == "Windows": os.system("cls")
    else: os.system("clear")

    app.close()
    os.system(f"{python} {os.path.join(now_dir, 'main', 'app', 'app.py')}")


def change_language(lang):
    gr.Info(translations["30s"])

    with open(configs_json, "r") as f:
        configs = json.load(f)

    configs["language"] = lang

    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)


def change_theme(theme):
    gr.Info(translations["30s"])

    with open(configs_json, "r") as f:
        configs = json.load(f)

    configs["theme"] = theme

    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)


def change_fp(fp):
    gr.Info(translations["fp_select"])

    config.set_precision(fp)

    gr.Info(translations["fp_select_2"].format(fp=fp))


def pretrained_selector(pitch_guidance):
    if pitch_guidance:
        return {
            32000: (
                "f0G32k.pth",
                "f0D32k.pth",
            ),
            40000: (
                "f0G40k.pth",
                "f0D40k.pth",
            ),
            48000: (
                "f0G48k.pth",
                "f0D48k.pth",
            ),
        }
    else:
        return {
            32000: (
                "G32k.pth",
                "D32k.pth",
            ),
            40000: (
                "G40k.pth",
                "D40k.pth",
            ),
            48000: (
                "G48k.pth",
                "D48k.pth",
            ),
        }


def zip_file(name, pth, index):
    pth_path = os.path.join("assets", "weights", pth)

    if not pth or not os.path.exists(pth_path) or not pth.endswith(".pth"): return gr.Warning(translations["provide_file"].format(filename=translations["model"]))
    if not index or not os.path.exists(index) or not index.endswith(".pth"): return gr.Warning(translations["provide_file"].format(filename=translations["index"]))
    

    zip_file_path = os.path.join("assets", name + ".zip")

    gr.Info(translations["start"].format(start=translations["zip"]))
    
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(pth_path, os.path.basename(pth_path))
        zipf.write(index, os.path.basename(index))


    gr.Info(translations["success"])
    return zip_file_path    


def search_models(name):
    gr.Info(translations["start"].format(start="search"))
    url = f"https://cjtfqzjfdimgpvpwhzlv.supabase.co/rest/v1/models?name=ilike.%25{name}%25&order=created_at.desc&limit=15"
    
    response = requests.get(url, headers={"apikey": model_search_api})
    data = response.json()


    if len(data) == 0:
        gr.Info(translations["not_found"].format(name=name))

        return [None]*2
    else:
        model_options.clear()
        model_options.update({item["name"] + " " + item["epochs"] + "e": item["link"] for item in data})

        gr.Info(translations["found"].format(results=len(model_options)))
        return [{"value": "", "choices": model_options, "interactive": True, "visible": True, "__type__": "update"}, {"value": translations["downloads"], "visible": True, "__type__": "update"}]


def move_files_from_directory(src_dir, dest_weights, dest_logs, model_name):
    for root, _, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)


            if file.endswith(".index"):
                model_log_dir = os.path.join(dest_logs, model_name)
                os.makedirs(model_log_dir, exist_ok=True)
                
                filepath = os.path.join(model_log_dir, file.replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').strip())
                if os.path.exists(filepath): os.remove(filepath)

                shutil.move(file_path, filepath)
            elif file.endswith(".pth") and "G_" not in file and "D_" not in file:
                pth_path = os.path.join(dest_weights, model_name + ".pth")
                if os.path.exists(pth_path): os.remove(pth_path)

                shutil.move(file_path, pth_path)


def download_url(url):
    if not url: return gr.Warning(translations["provide_url"])
    if not os.path.exists("audios"): os.makedirs("audios", exist_ok=True)


    audio_output = os.path.join("audios", "audio.wav")

    if os.path.exists(audio_output): os.remove(audio_output)


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join("audios", "audio"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'noplaylist': True,
            'verbose': False,
        }

        gr.Info(translations["start"].format(start=translations["download_music"]))

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])


        gr.Info(translations["success"])
        return [audio_output, audio_output, translations["success"]]


def download_model(url=None, model=None):
    if not url: return gr.Warning(translations["provide_url"])
    if not model: return gr.Warning(translations["provide_name_is_save"])


    model = model.replace('.pth', '').replace('.index', '').replace('.zip', '').replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').strip()
    url = url.replace('/blob/', '/resolve/').replace('?download=true', '').strip()
    
    download_dir = os.path.join("download_model")
    weights_dir = os.path.join("assets", "weights")
    logs_dir = os.path.join("assets", "logs")

    if not os.path.exists(download_dir): os.makedirs(download_dir, exist_ok=True)
    if not os.path.exists(weights_dir): os.makedirs(weights_dir, exist_ok=True)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir, exist_ok=True)
    

    try:
        gr.Info(translations["start"].format(start=translations["download"]))

        if url.endswith('.pth'):
            run(["wget", "-q", "--show-progress", "--no-check-certificate", url, "-O", os.path.join(weights_dir, f"{model}.pth")], check=True)
        elif url.endswith('.index'):
            model_log_dir = os.path.join(logs_dir, model)
            os.makedirs(model_log_dir, exist_ok=True)
            run(["wget", "-q", "--show-progress", "--no-check-certificate", url, "-O", os.path.join(model_log_dir, f"{model}.index")], check=True)
        elif url.endswith('.zip'):
            dest_path = os.path.join(download_dir, model + ".zip")
            run(["wget", "-q", "--show-progress", "--no-check-certificate", url, "-O", dest_path], check=True)
            shutil.unpack_archive(dest_path, download_dir)

            move_files_from_directory(download_dir, weights_dir, logs_dir, model)
        else:
            if 'drive.google.com' in url:
                file_id = None

                if '/file/d/' in url: file_id = url.split('/d/')[1].split('/')[0]
                elif 'open?id=' in url: file_id = url.split('open?id=')[1].split('/')[0]
                
                if file_id:
                    file = gdown.gdown_download(id=file_id, output_dir=download_dir)
                    if file.endswith('.zip'): shutil.unpack_archive(os.path.join(download_dir, file), download_dir)

                    move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif 'mega.nz' in url:
                meganz.mega_download_url(url, download_dir)

                file_download = next((f for f in os.listdir(download_dir)), None)
                if file_download.endswith(".zip"): shutil.unpack_archive(os.path.join(download_dir, file_download), download_dir)

                move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif 'mediafire.com' in url:
                file = mediafire.Mediafire_Download(url, download_dir)
                if file.endswith('.zip'): shutil.unpack_archive(file, download_dir)

                move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif 'pixeldrain.com' in url:
                file = pixeldrain.pixeldrain(url, download_dir)
                if file.endswith('.zip'): shutil.unpack_archive(file, download_dir)

                move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            else:
                gr.Warning(translations["not_support_url"])
                return translations["not_support_url"]
        
        gr.Info(translations["success"])
        return translations["success"]
    except Exception as e:
        gr.Error(message=translations["error_occurred"].format(e=e))

        print(translations["error_occurred"].format(e=e))
        return translations["error_occurred"].format(e=e)
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)


def extract_name_model(filename):
    match = re.search(r"([A-Za-z]+)(?=_v|\.|$)", filename)

    return match.group(1) if match else None


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

        if file_name.endswith(".pth") and file_name.endswith(".index"): gr.Warning(translations["not_model"])
        else:    
            if file_name.endswith(".zip"):
                shutil.unpack_archive(os.path.join(save_model_temp, file_name), save_model_temp)
                move_files_from_directory(save_model_temp, weight_folder, logs_folder, file_name.replace(".zip", ""))
            elif file_name.endswith(".pth"): 
                output_file = os.path.join(weight_folder, file_name)
                if os.path.exists(output_file): os.remove(output_file)

                shutil.move(os.path.join(save_model_temp, file_name), output_file)
            elif file_name.endswith(".index"):
                model_logs = os.path.join(logs_folder, extract_name_model(file_name))

                if not os.path.exists(model_logs): os.makedirs(model_logs, exist_ok=True)
                shutil.move(os.path.join(save_model_temp, file_name), model_logs)
            else: 
                gr.Warning(translations["unable_analyze_model"])
                return None
        
        gr.Info(translations["upload_success"].format(name=translations["model"]))
        return None
    except Exception as e:
        gr.Error(message=translations["error_occurred"].format(e=e))

        print(translations["error_occurred"].format(e=e))
        return None
    finally:
        shutil.rmtree(save_model_temp, ignore_errors=True)


def download_pretrained_model(choices, model, sample_rate):
    if choices == translations["list_model"]:
        data = fetch_pretrained_data()
        paths = data[model][sample_rate]


        pretraineds_custom_path = os.path.join("assets", "model", "pretrained_custom")

        if not os.path.exists(pretraineds_custom_path): os.makedirs(pretraineds_custom_path, exist_ok=True)

        d_url = hugging_face_codecs + f"/{paths['D']}"
        g_url = hugging_face_codecs + f"/{paths['G']}"

        gr.Info(translations["download_pretrain"])

        run(["wget", "-q", "--show-progress", "--no-check-certificate", d_url.replace('/blob/', '/resolve/').replace('?download=true', '').strip(), "-P", os.path.join(pretraineds_custom_path)], check=True)
        run(["wget", "-q", "--show-progress", "--no-check-certificate", g_url.replace('/blob/', '/resolve/').replace('?download=true', '').strip(), "-P", os.path.join(pretraineds_custom_path)], check=True)

        gr.Info(translations["success"])
        return translations["success"]
    elif choices == translations["download_url"]:
        if not model: return gr.Warning(translations["provide_pretrain"].format(dg="D"))
        if not sample_rate: return gr.Warning(translations["provide_pretrain"].format(dg="G"))


        gr.Info(translations["download_pretrain"])

        run(["wget", "-q", "--show-progress", "--no-check-certificate", model, "-P", os.path.join(pretraineds_custom_path)], check=True)
        run(["wget", "-q", "--show-progress", "--no-check-certificate", sample_rate, "-P", os.path.join(pretraineds_custom_path)], check=True)

        gr.Info(translations["success"])
        return translations["success"]
    

def hubert_download(hubert):
    if not hubert: 
        gr.Warning(translations["provide_hubert"])
        return translations["provide_hubert"]
    

    run(["wget", "-q", "--show-progress", "--no-check-certificate", hubert.replace('/blob/', '/resolve/').replace('?download=true', '').strip(), "-P", os.path.join("assets", "model", "embedders")], check=True)

    gr.Info(translations["success"])
    return translations["success"]


def fushion_model(name, pth_1, pth_2, ratio):
    if not name:
        gr.Warning(translations["provide_name_is_save"]) 
        return [translations["provide_name_is_save"], None]
    
    if not name.endswith(".pth"): name = name + ".pth"
    

    if not pth_1 or not os.path.exists(pth_1) or not pth_1.endswith(".pth"):
        gr.Warning(translations["provide_file"].format(filename=translations["model"] + " 1"))
        return [translations["provide_file"].format(filename=translations["model"] + " 1"), None]
    
    if not pth_2 or not os.path.exists(pth_2) or not pth_1.endswith(".pth"):
        gr.Warning(translations["provide_file"].format(filename=translations["model"] + " 2"))
        return [translations["provide_file"].format(filename=translations["model"] + " 2"), None]
    

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
            gr.Warning(translations["sr_not_same"])
            return [translations["sr_not_same"], None]

        cfg = ckpt1["config"]
        cfg_f0 = ckpt1["f0"]
        cfg_version = ckpt1["version"]
        cfg_sr = ckpt1["sr"]

        ckpt1 = extract(ckpt1) if "model" in ckpt1 else ckpt1["weight"]
        ckpt2 = extract(ckpt2) if "model" in ckpt2 else ckpt2["weight"]

        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())): 
            gr.Warning(translations["architectures_not_same"])
            return [translations["architectures_not_same"], None]
         
        gr.Info(translations["start"].format(start=translations["fushion_model"]))

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

        torch.save(opt, os.path.join(output_model, f"{name}.pth"))


        gr.Info(translations["success"])
        return [translations["success"], output_model]
    except Exception as e:
        gr.Error(message=translations["error_occurred"].format(e=e))

        print(translations["error_occurred"].format(e=e))
        return [e, None]


def model_info(path):
    if not path or not os.path.exists(path) or os.path.isdir(path) or not path.endswith(".pth"): return gr.Warning(translations["provide_file"].format(filename=translations["model"]))
    

    def prettify_date(date_str):
        if date_str == translations["not_found_create_time"]: return None

        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return translations["format_not_valid"]
        

    model_data = torch.load(path, map_location=torch.device("cpu"))

    gr.Info(translations["read_info"])

    epochs = model_data.get("epoch", None)

    if epochs is None: 
        epochs = model_data.get("info", None)
        epoch = epochs.replace("epoch", "").replace("e", "").isdigit()

        if epoch and epochs is None: epochs = translations["not_found"].format(name=translations["epoch"])
        
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

    gr.Info(translations["success"])

    return translations["model_info"].format(model_name=model_name, model_author=model_author, epochs=epochs, steps=steps, version=version, sr=sr, pitch_guidance=pitch_guidance, model_hash=model_hash, creation_date_str=creation_date_str)


def audio_effects(input_path, output_path, resample, resample_sr, chorus_depth, chorus_rate, chorus_mix, chorus_delay, chorus_feedback, distortion_drive, reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level, reverb_width, reverb_freeze_mode, pitch_shift, delay_seconds, delay_feedback, delay_mix, compressor_threshold, compressor_ratio, compressor_attack_ms, compressor_release_ms, limiter_threshold, limiter_release, gain_db, bitcrush_bit_depth, clipping_threshold, phaser_rate_hz, phaser_depth, phaser_centre_frequency_hz, phaser_feedback, phaser_mix, bass_boost_db, bass_boost_frequency, treble_boost_db, treble_boost_frequency, fade_in_duration, fade_out_duration, export_format, chorus, distortion, reverb, delay, compressor, limiter, gain, bitcrush, clipping, phaser, treble_bass_boost, fade_in_out):
    if not input_path or not os.path.exists(input_path) or os.path.isdir(input_path): 
        gr.Warning(translations["input_not_valid"])
        return None
        
    if not output_path:
        gr.Warning(translations["output_not_valid"])
        return None
    
    if os.path.isdir(output_path): output_path = os.path.join(output_path, f"audio_effects.{export_format}")
    

    output_dir = os.path.dirname(output_path)
    output_dir = output_path if not output_dir else output_dir

    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_path): os.remove(output_path)
    

    gr.Info(translations["start"].format(start=translations["apply_effect"]))

    pitchshift = pitch_shift != 0

    cmd = f'{python} main/inference/audio_effects.py --input_path "{input_path}" --output_path "{output_path}" --resample {resample} --resample_sr {resample_sr} --chorus_depth {chorus_depth} --chorus_rate {chorus_rate} --chorus_mix {chorus_mix} --chorus_delay {chorus_delay} --chorus_feedback {chorus_feedback} --drive_db {distortion_drive} --reverb_room_size {reverb_room_size} --reverb_damping {reverb_damping} --reverb_wet_level {reverb_wet_level} --reverb_dry_level {reverb_dry_level} --reverb_width {reverb_width} --reverb_freeze_mode {reverb_freeze_mode} --pitch_shift {pitch_shift} --delay_seconds {delay_seconds} --delay_feedback {delay_feedback} --delay_mix {delay_mix} --compressor_threshold {compressor_threshold} --compressor_ratio {compressor_ratio} --compressor_attack_ms {compressor_attack_ms} --compressor_release_ms {compressor_release_ms} --limiter_threshold {limiter_threshold} --limiter_release {limiter_release} --gain_db {gain_db} --bitcrush_bit_depth {bitcrush_bit_depth} --clipping_threshold {clipping_threshold} --phaser_rate_hz {phaser_rate_hz} --phaser_depth {phaser_depth} --phaser_centre_frequency_hz {phaser_centre_frequency_hz} --phaser_feedback {phaser_feedback} --phaser_mix {phaser_mix} --bass_boost_db {bass_boost_db} --bass_boost_frequency {bass_boost_frequency} --treble_boost_db {treble_boost_db} --treble_boost_frequency {treble_boost_frequency} --fade_in_duration {fade_in_duration} --fade_out_duration {fade_out_duration} --export_format {export_format} --chorus {chorus} --distortion {distortion} --reverb {reverb} --pitchshift {pitchshift} --delay {delay} --compressor {compressor} --limiter {limiter} --gain {gain} --bitcrush {bitcrush} --clipping {clipping} --phaser {phaser} --treble_bass_boost {treble_bass_boost} --fade_in_out {fade_in_out}'
    os.system(cmd)


    gr.Info(translations["success"])

    return output_path 


async def TTS(prompt, voice, speed, output):
    if not prompt:
        gr.Warning(translations["enter_the_text"])
        return None
    
    if not voice:
        gr.Warning(translations["choose_voice"])
        return None
    
    if not output: 
        gr.Warning(translations["output_not_valid"])
        return None
    
    if os.path.isdir(output): output = os.path.join(output, f"output_tts.wav")


    gr.Info(translations["convert"].format(name=translations["text"]))

    output_dir = os.path.dirname(output)
    output_dir = output if not output_dir else output_dir

    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    await edge_tts.Communicate(text=prompt, voice=voice, rate=f"+{speed}%" if speed >= 0 else f"{speed}%").save(output)


    gr.Info(translations["success"])

    return output


def separator_music(input, output_audio, format, shifts, segments_size, overlap, clean_audio, clean_strength, backing_denoise, separator_model, kara_model, backing, mdx, mdx_denoise, reverb, reverb_denoise, backing_reverb, hop_length, batch_size):
    output = os.path.dirname(output_audio)
    output = output_audio if not output else output
    

    if not input or not os.path.exists(input) or os.path.isdir(input): 
        gr.Warning(translations["input_not_valid"])
        return [None]*4
    
    if not os.path.exists(output): 
        gr.Warning(translations["output_not_valid"])
        return [None]*4


    gr.Info(translations["start"].format(start=translations["separator_music"]))

    cmd = f'{python} main/inference/separator_music.py --input_path "{input}" --output_path "{output}" --format {format} --shifts {shifts} --segments_size {segments_size} --overlap {overlap} --mdx_hop_length {hop_length} --mdx_batch_size {batch_size} --clean_audio {clean_audio} --clean_strength {clean_strength} --backing_denoise {backing_denoise} --kara_model {kara_model} --backing {backing} --mdx {mdx} --mdx_denoise {mdx_denoise} --reverb {reverb} --reverb_denoise {reverb_denoise} --backing_reverb {backing_reverb}'


    if separator_model == "HT-Normal" or separator_model == "HT-Tuned" or separator_model == "HD_MMI" or separator_model == "HT_6S": cmd += f' --demucs_model {separator_model}'
    else: cmd += f' --mdx_model {separator_model}'

    os.system(cmd)
    
    gr.Info(translations["success"])

    if not os.path.exists(output): os.makedirs(output)


    original_output = os.path.join(output, f"Original_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Original_Vocals.{format}")
    instrument_output = os.path.join(output, f"Instruments.{format}")
    main_output = os.path.join(output, f"Main_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Main_Vocals.{format}")
    backing_output = os.path.join(output, f"Backing_Vocals_No_Reverb.{format}") if backing_reverb else os.path.join(output, f"Backing_Vocals.{format}")


    if backing: return [original_output, instrument_output, main_output, backing_output]
    else: return [original_output, instrument_output, None, None]


def convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, clean_audio, clean_strength, export_format, embedder_model, upscale_audio, resample_sr, batch_process, batch_size, split_audio, f0_autotune_strength):
    cmd = f'{python} main/inference/convert.py --pitch {pitch} --filter_radius {filter_radius} --index_rate {index_rate} --volume_envelope {volume_envelope} --protect {protect} --hop_length {hop_length} --f0_method {f0_method} --input_path "{input_path}" --output_path "{output_path}" --pth_path {pth_path} --index_path {index_path} --f0_autotune {f0_autotune} --clean_audio {clean_audio} --clean_strength {clean_strength} --export_format {export_format} --embedder_model {embedder_model} --upscale_audio {upscale_audio} --resample_sr {resample_sr} --batch_process {batch_process} --batch_size {batch_size} --split_audio {split_audio} --f0_autotune_strength {f0_autotune_strength}'

    os.system(cmd)


def convert_audio(clean, upscale, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, batch_process, batch_size, split_audio, f0_autotune_strength):
    def get_audio_file(label):
        matching_files = [f for f in os.listdir("audios") if label in f]
        if not matching_files: return translations["notfound"]
        
        return os.path.join("audios", matching_files[0])


    model_path = os.path.join("assets", "weights", model)

    if not use_audio:
        if merge_instrument or not_merge_backing or convert_backing or use_original:
            gr.Warning(translations["turn_on_use_audio"])
            return [None]*5

    if use_original:
        if convert_backing:
            gr.Warning(translations["turn_off_convert_backup"])
            return [None]*5
        elif not_merge_backing:
            gr.Warning(translations["turn_off_merge_backup"])
            return [None]*5

    if not model or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith(".pth"):
        gr.Warning(translations["provide_file"].format(filename=translations["model"]))
        return [None]*5
    
    if not index or not os.path.exists(index) or os.path.isdir(index) or not index.endswith(".index"):
        gr.Warning(translations["provide_file"].format(filename=translations["index"]))
        return [None]*5


    f0method = method if method != "hybrid" else hybrid_method
    
    embedder_model = embedders if embedders != "custom" else custom_embedders


    output_path = os.path.join("audios", f"Convert_Vocals.{format}")
    output_backing = os.path.join("audios", f"Convert_Backing.{format}")
    output_merge_backup = os.path.join("audios", f"Vocals+Backing.{format}")
    output_merge_instrument = os.path.join("audios", f"Vocals+Instruments.{format}")


    if use_audio:
        if os.path.exists("audios"): os.makedirs("audios", exist_ok=True)
        if os.path.exists(output_path): os.remove(output_path)


        if use_original:
            original_vocal = get_audio_file('Original_Vocals_No_Reverb.')

            if original_vocal == translations["notfound"]: original_vocal = get_audio_file('Original_Vocals.')

            if original_vocal == translations["notfound"]: 
                gr.Warning(translations["not_found_original_vocal"])
                return [None]*5
            
            input_path = original_vocal
        else:
            main_vocal = get_audio_file('Main_Vocals_No_Reverb.')
            backing_vocal = get_audio_file('Backing_Vocals_No_Reverb.')

            if main_vocal == translations["notfound"]: main_vocal = get_audio_file('Main_Vocals.')
            if not not_merge_backing and backing_vocal == translations["notfound"]: backing_vocal = get_audio_file('Backing_Vocals.')

            if main_vocal == translations["notfound"]: 
                gr.Warning(translations["not_found_main_vocal"])
                return [None]*5
            
            if not not_merge_backing and backing_vocal == translations["notfound"]: 
                gr.Warning(translations["not_found_backing_vocal"])
                return [None]*5
            
            input_path = main_vocal
            backing_path = backing_vocal


        gr.Info(translations["convert_vocal"])

        convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input_path, output_path, model_path, index, autotune, clean, clean_strength, format, embedder_model, upscale, resample_sr, batch_process, batch_size, split_audio, f0_autotune_strength)

        gr.Info(translations["convert_success"])

        if convert_backing:
            if os.path.exists(output_backing): os.remove(output_backing)

            gr.Info(translations["convert_backup"])

            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, backing_path, output_backing, model_path, index, autotune, clean, clean_strength, format, embedder_model, upscale, resample_sr, batch_process, batch_size, split_audio, f0_autotune_strength)

            gr.Info(translations["convert_backup_success"])

        if not not_merge_backing and not use_original:
            backing_source = output_backing if convert_backing else backing_vocal

            if os.path.exists(output_merge_backup): os.remove(output_merge_backup)


            gr.Info(translations["merge_backup"])

            AudioSegment.from_file(output_path).overlay(AudioSegment.from_file(backing_source)).export(output_merge_backup, format=format)

            gr.Info(translations["merge_success"])

        if merge_instrument:    
            vocals = output_merge_backup if not not_merge_backing and not use_original else output_path

            if os.path.exists(output_merge_instrument): os.remove(output_merge_instrument)

            gr.Info(translations["merge_instruments_process"])

            instruments = get_audio_file('Instruments.')
            
            if instruments == translations["notfound"]: 
                gr.Warning(translations["not_found_instruments"])
                output_merge_instrument = None
            else: AudioSegment.from_file(instruments).overlay(AudioSegment.from_file(vocals)).export(output_merge_instrument, format=format)
            
            gr.Info(translations["merge_success"])

        return [(None if use_original else output_path), output_backing, (None if not_merge_backing and use_original else output_merge_backup), (output_path if use_original else None), (output_merge_instrument if merge_instrument else None)]
    else:
        if not input or not os.path.exists(input): 
            gr.Warning(translations["input_not_valid"])
            return [None]*5
        
        if not output:
            gr.Warning(translations["output_not_valid"])
            return [None]*5
        

        if os.path.isdir(input):
            gr.Info(translations["is_folder"])

            if not [f for f in os.listdir(input) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]:
                gr.Warning(translations["not_found_in_folder"])
                return [None]*5
            
            gr.Info(translations["batch_convert"])

            output_dir = os.path.dirname(output)
            output_dir = output if not output_dir else output_dir

            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output_dir, model_path, index, autotune, clean, clean_strength, format, embedder_model, upscale, resample_sr, batch_process, batch_size, split_audio, f0_autotune_strength)

            gr.Info(translations["batch_convert_success"])

            return [None]*5
        else:
            output_dir = os.path.dirname(output)
            output_dir = output if not output_dir else output_dir

            if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

            if os.path.exists(output): os.remove(output)


            gr.Info(translations["convert_vocal"])

            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, upscale, resample_sr, batch_process, batch_size, split_audio, f0_autotune_strength)

            gr.Info(translations["convert_success"])

            return [output, None, None, None, None]


def convert_tts(clean, upscale, autotune, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, batch_process, batch_size, split_audio, f0_autotune_strength):
    model_path = os.path.join("assets", "weights", model)

    if not model_path or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith(".pth"):
        gr.Warning(translations["provide_file"].format(filename=translations["model"]))
        return None
    
    if not index or not os.path.exists(index) or os.path.isdir(index) or not index.endswith(".index"):
        gr.Warning(translations["provide_file"].format(filename=translations["index"]))
        return None

    if not input or not os.path.exists(input): 
        gr.Warning(translations["input_not_valid"])
        return None
    
    if os.path.isdir(input): 
        input_audio = [f for f in os.listdir(input) if "output_tts" in f and f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]
        
        if not input_audio:
            gr.Warning(translations["not_found_in_folder"])
            return None
        
        input = os.path.join(input, input_audio[0])
    
    if not output:
        gr.Warning(translations["output_not_valid"])
        return None
    
    if os.path.isdir(output): output = os.path.join(output, f"output_tts-convert.{format}")
    

    output_dir = os.path.dirname(output)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output): os.remove(output)

    f0method = method if method != "hybrid" else hybrid_method
    
    embedder_model = embedders if embedders != "custom" else custom_embedders

    gr.Info(translations["convert_vocal"])

    convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, upscale, resample_sr, batch_process, batch_size, split_audio, f0_autotune_strength)


    gr.Info(translations["convert_success"])
    return output


def create_dataset(input_audio, output_dataset, resample, resample_sr, clean_dataset, clean_strength, separator_music, separator_reverb, kim_vocals_version, overlap, segments_size, denoise_mdx, skip, skip_start, skip_end, hop_length, batch_size):
    version = 1 if kim_vocals_version == "Version-1" else 2

    cmd = f'{python} main/inference/create_dataset.py --input_audio "{input_audio}" --output_dataset "{output_dataset}" --resample {resample} --resample_sr {resample_sr} --clean_dataset {clean_dataset} --clean_strength {clean_strength} --separator_music {separator_music} --separator_reverb {separator_reverb} --kim_vocal_version {version} --overlap {overlap} --segments_size {segments_size} --mdx_hop_length {hop_length} --mdx_batch_size {batch_size} --denoise_mdx {denoise_mdx} --skip {skip} --skip_start_audios "{skip_start}" --skip_end_audios "{skip_end}"'

    gr.Info(translations["start"].format(start=translations["create"]))


    p = Popen(cmd, shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    create_dataset_log = os.path.join("assets", "logs", "create_dataset.log")

    f = open(create_dataset_log, "w", encoding="utf-8")
    f.close()


    while 1:
        with open(create_dataset_log, "r", encoding='utf-8') as f:
            yield (f.read())

        sleep(1)
        if done[0]: break

    with open(create_dataset_log, "r", encoding='utf-8') as f:
        log = f.read()

    yield log


def preprocess(model_name, sample_rate, cpu_core, cut_preprocess, process_effects, path, clean_dataset, clean_strength):
    dataset = os.path.join(path)
    sr = int(sample_rate.rstrip("k")) * 1000

    if not model_name: return gr.Warning(translations["provide_name"])
    if len([f for f in os.listdir(os.path.join(dataset)) if os.path.isfile(os.path.join(dataset, f)) and f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]) < 1: return gr.Warning(translations["not_found_data"])


    cmd = f'{python} main/inference/preprocess.py --model_name "{model_name}" --dataset_path "{dataset}" --sample_rate {sr} --cpu_cores {cpu_core} --cut_preprocess {cut_preprocess} --process_effects {process_effects} --clean_dataset {clean_dataset} --clean_strength {clean_strength}'


    p = Popen(cmd, shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    model_dir = os.path.join("assets", "logs", model_name)
    preprocess_log = os.path.join(model_dir, model_name, "preprocess.log")

    os.makedirs(model_dir, exist_ok=True)

    f = open(preprocess_log, "w", encoding="utf-8")
    f.close()


    while 1:
        with open(preprocess_log, "r", encoding='utf-8') as f:
            yield (f.read())

        sleep(1)
        if done[0]: break

    with open(preprocess_log, "r", encoding='utf-8') as f:
        log = f.read()

    yield log


def extract(model_name, version, method, pitch_guidance, hop_length, cpu_cores, gpu, sample_rate, embedders, custom_embedders):
    embedder_model = embedders if embedders != "custom" else custom_embedders
    model_dir = os.path.join("assets", "logs", model_name)

    sr = int(sample_rate.rstrip("k")) * 1000

    if not model_name: return gr.Warning(translations["provide_name"])

    if len([f for f in os.listdir(os.path.join(model_dir, "sliced_audios")) if os.path.isfile(os.path.join(model_dir, "sliced_audios", f))]) < 1 or len([f for f in os.listdir(os.path.join(model_dir, "sliced_audios_16k")) if os.path.isfile(os.path.join(model_dir, "sliced_audios_16k", f))]) < 1: return gr.Warning(translations["not_found_data_preprocess"])


    cmd = f'{python} main/inference/extract.py --model_name "{model_name}" --rvc_version {version} --f0_method {method} --pitch_guidance {pitch_guidance} --hop_length {hop_length} --cpu_cores {cpu_cores} --gpu {gpu} --sample_rate {sr} --embedder_model {embedder_model}'


    p = Popen(cmd, shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    extract_log = os.path.join(model_dir, model_name, "extract.log")

    os.makedirs(model_dir, exist_ok=True)

    f = open(extract_log, "w", encoding="utf-8")
    f.close()


    while 1:
        with open(extract_log, "r", encoding='utf-8') as f:
            yield (f.read())

        sleep(1)
        if done[0]: break

    with open(extract_log, "r", encoding='utf-8') as f:
        log = f.read()

    yield log


def create_index(model_name, rvc_version, index_algorithm):
    if not model_name: return gr.Warning(translations["provide_name"])
    model_dir = os.path.join("assets", "logs", model_name)

    if len([f for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted")) if os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f))]) < 1: return gr.Warning(translations["not_found_data_extract"])


    cmd = f'{python} main/inference/create_index.py --model_name "{model_name}" --rvc_version {rvc_version} --index_algorithm {index_algorithm}'


    p = Popen(cmd, shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    create_index_log = os.path.join(model_dir, model_name, "create_index.log")

    os.makedirs(model_dir, exist_ok=True)

    f = open(create_index_log, "w", encoding="utf-8")
    f.close()


    while 1:
        with open(create_index_log, "r", encoding='utf-8') as f:
            yield (f.read())

        sleep(1)
        if done[0]: break

    with open(create_index_log, "r", encoding='utf-8') as f:
        log = f.read()

    yield log


def training(model_name, rvc_version, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sample_rate, batch_size, gpu, pitch_guidance, not_pretrain, custom_pretrained, pretrain_g, pretrain_d, detector, threshold, sync_graph, cache, model_author):
    sr = int(sample_rate.rstrip("k")) * 1000
    model_dir = os.path.join("assets", "logs", model_name)

    if not model_name: return gr.Warning(translations["provide_name"])
    if len([f for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted")) if os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f))]) < 1: return gr.Warning(translations["not_found_data_extract"])

    cmd = f'{python} main/inference/train.py --model_name "{model_name}" --rvc_version {rvc_version} --save_every_epoch {save_every_epoch} --save_only_latest {save_only_latest} --save_every_weights {save_every_weights} --total_epoch {total_epoch} --sample_rate {sr} --batch_size {batch_size} --gpu {gpu} --pitch_guidance {pitch_guidance} --overtraining_detector {detector} --overtraining_threshold {threshold} --sync_graph {sync_graph} --cache_data_in_gpu {cache}'


    if not not_pretrain:
        if not custom_pretrained: pg, pd = pretrained_selector(pitch_guidance)[sr]
        else:
            if not pretrain_g: return gr.Warning(translations["provide_pretrained"].format(dg="G"))
            if not pretrain_d: return gr.Warning(translations["provide_pretrained"].format(dg="D"))
            
            pg = pretrain_g
            pd = pretrain_d

        if not custom_pretrained:
            pretrained_G = os.path.join("assets", "model", f"pretrained_{rvc_version}", pg)
            pretrained_D = os.path.join("assets", "model", f"pretrained_{rvc_version}", pd)
        else:
            pretrained_G = os.path.join("assets", "model", f"pretrained_custom", pg)
            pretrained_D = os.path.join("assets", "model", f"pretrained_custom", pd)


        download_version = pretrained_v2_link if rvc_version == "v2" else pretrained_v1_link
        
        if not custom_pretrained:
            if not os.path.exists(pretrained_G):
                gr.Info(translations["download_pretrained"].format(dg="G", rvc_version=rvc_version))
                run(["wget", "-q", "--show-progress", "-q", "--show-progress", "--no-check-certificate", f"{download_version}{pg}", "-P", os.path.join("assets", "model", f"pretrained_{rvc_version}")], check=True)
                
            if not os.path.exists(pretrained_D):
                gr.Info(translations["download_pretrained"].format(dg="D", rvc_version=rvc_version))
                run(["wget", "-q", "--show-progress", "-q", "--show-progress", "--no-check-certificate", f"{download_version}{pd}", "-P", os.path.join("assets", "model", f"pretrained_{rvc_version}")], check=True)
        else:
            if not os.path.exists(pretrained_G): return gr.Warning(translations["not_found_pretrain"].format(dg="G"))
            if not os.path.exists(pretrained_D): return gr.Warning(translations["not_found_pretrain"].format(dg="D"))

        cmd += f" --g_pretrained_path {pretrained_G} --d_pretrained_path {pretrained_D}"
    else: gr.Warning(translations["not_use_pretrain"])

    if model_author: cmd += f'--model_author {model_author}'

    gr.Info(translations["start"].format(start=translations["training"]))

    p = Popen(cmd, shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    if not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)

    train_log = os.path.join(model_dir, "train.log")

    f = open(train_log, "w", encoding="utf-8")
    f.close()


    while 1:
        with open(train_log, "r", encoding='utf-8') as f:
            yield (f.read())

        sleep(1)
        if done[0]: break

    with open(train_log, "r", encoding='utf-8') as f:
        log = f.read()

    yield log



with gr.Blocks(title="📱 Vietnamese-RVC GUI BY ANH", theme=theme) as app:
    gr.HTML(translations["display_title"])
    with gr.Row(): 
        gr.Markdown(translations["rick_roll"].format(rickroll=codecs.decode('uggcf://jjj.lbhghor.pbz/jngpu?i=qDj4j9JtKpD', 'rot13')))
    with gr.Row(): 
        gr.Markdown(translations["terms_of_use"])
    with gr.Row():
        gr.Markdown(translations["exemption"])

    with gr.Tabs():
        paths_for_files = lambda path: [os.path.abspath(os.path.join(path, f)) for f in os.listdir(path) if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a')]

        with gr.TabItem(translations["separator_tab"]):
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
                            segment_size = gr.Slider(label=translations["segments_size"], info=translations["segments_size_info"], minimum=32, maximum=4000, value=256, step=8, interactive=True)
                        with gr.Row():
                            mdx_batch_size = gr.Slider(label=translations["batch_size"], info=translations["mdx_batch_size_info"], minimum=1, maximum=64, value=1, step=1, interactive=True, visible=False)
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            overlap = gr.Radio(label=translations["overlap"], info=translations["overlap_info"], choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True)
                            format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac"], value="wav", interactive=True)
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
                    with gr.Accordion(translations["input_output"]):
                        input_audio = gr.Dropdown(label=translations["audio_path"], value="" if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios")[0], choices=[] if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios"), allow_custom_value=True, interactive=True)
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
                backing.change(fn=interactive_1, inputs=[backing], outputs=[denoise])
                backing.change(fn=valueFalse_interactive2,  inputs=[backing, reverb], outputs=[backing_reverb])
            with gr.Row():
                reverb.change(fn=interactive_1, inputs=[reverb], outputs=[reverb_denoise])
                reverb.change(fn=valueFalse_interactive2, inputs=[backing, reverb], outputs=[backing_reverb])
            with gr.Row():
                mdx_model.change(fn=interactive_1, inputs=[mdx_model], outputs=[separator_denoise])
                mdx_model.change(fn=model_separator_change, inputs=[mdx_model], outputs=[separator_model])
                mdx_model.change(fn=lambda inp: visible_1(not inp), inputs=[mdx_model], outputs=[shifts])
            with gr.Row():
                input_audio.change(fn=lambda audio: audio if audio else None, inputs=[input_audio], outputs=[audio_input])
                cleaner.change(fn=visible_1, inputs=[cleaner], outputs=[clean_strength])
                input.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[input], outputs=[input_audio])
                refesh_separator.click(fn=refesh_audio, inputs=[], outputs=[input_audio])
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
                        mdx_model, 
                        separator_denoise, 
                        reverb, 
                        reverb_denoise,
                        backing_reverb,
                        mdx_hop_length,
                        mdx_batch_size
                    ],
                    outputs=[original_vocals, instruments_audio, main_vocals, backing_vocals],
                    api_name='separator_music'
                )

        with gr.TabItem(translations["convert_audio"]):
            gr.Markdown(f"## {translations['convert_audio']}")
            with gr.Row():
                gr.Markdown(translations["convert_info"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            cleaner0 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                            upscale = gr.Checkbox(label=translations["upscale_audio"], value=False, interactive=True)
                            autotune = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
                            use_audio = gr.Checkbox(label=translations["use_audio"], value=False, interactive=True)
                            use_original = gr.Checkbox(label=translations["convert_original"], value=False, interactive=True, visible=False) 
                            convert_backing = gr.Checkbox(label=translations["convert_backing"], value=False, interactive=True, visible=False)   
                            not_merge_backing = gr.Checkbox(label=translations["not_merge_backing"], value=False, interactive=True, visible=False)
                            merge_instrument = gr.Checkbox(label=translations["merge_instruments"], value=False, interactive=True, visible=False)  
                    with gr.Row():
                        pitch = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                        clean_strength0 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
                with gr.Column():
                    convert_button = gr.Button(translations["convert_audio"], variant="primary", scale=4)
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
                            index_strength = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True)
                    with gr.Accordion(translations["input_output"], open=False):
                        with gr.Column():
                            export_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "m4a"], value="wav", interactive=True)
                            input_audio0 = gr.Dropdown(label=translations["audio_path"], value="" if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios")[0], choices=[] if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios"), info="Nhập đường dẫn đến tệp âm thanh", allow_custom_value=True, interactive=True)
                            output_audio = gr.Textbox(label=translations["output_path"], value="audios/output.wav", placeholder="audios/output.wav", info=translations["output_path_info"], interactive=True)
                        with gr.Column():
                            refesh0 = gr.Button(translations["refesh"])
                    with gr.Accordion(translations["setting"], open=False):
                        with gr.Accordion(translations["f0_method"], open=False):
                            method = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=["pm", "dio", "crepe-tiny", "crepe", "fcpe", "rmvpe", "harvest", "hybrid"], value="rmvpe", interactive=True)
                            hybrid_method = gr.Radio(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[rmvpe+harvest]"], value="hybrid[pm+dio]", interactive=True, visible=False)
                            hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion(translations["hubert_model"], open=False):
                            embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "custom"], value="contentvec_base", interactive=True)
                            custom_embedders = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=False)
                        with gr.Column():
                            with gr.Group():
                                with gr.Row():
                                    split_audio = gr.Checkbox(label=translations["split_audio"], info=translations["split_audio_info"], value=False, interactive=True)
                                    batch_process = gr.Checkbox(label=translations["batch_process"], info=translations["batch_process_info"], value=False, interactive=True, visible=False)
                                with gr.Row():
                                    batch_size = gr.Slider(minimum=1, maximum=10, label=translations["batch_size"], info=translations["batch_size_info"], value=1, step=1, interactive=True, visible=False)
                            f0_autotune_strength = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=False)
                            resample_sr = gr.Slider(minimum=0, maximum=48000, label=translations["resample"], info=translations["resample_info"], value=0, step=1, interactive=True)
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
                split_audio.change(fn=valueFalse_visible1, inputs=[split_audio], outputs=[batch_process])
                batch_process.change(fn=visible_1, inputs=[batch_process], outputs=[batch_size])
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
                convert_backing.change(fn=backing_change, inputs=[convert_backing, not_merge_backing], outputs=[use_original])
            with gr.Row():
                use_original.change(fn=lambda original: [valueFalse_interactive1(not original), valueFalse_interactive1(not original)], inputs=[use_original], outputs=[convert_backing, not_merge_backing])
                use_original.change(fn=lambda audio, original: [visible_1(original), visible_1(not original), visible_1(audio and not original)], inputs=[use_audio, use_original], outputs=[original_convert, main_convert, main_backing])
            with gr.Row():
                cleaner0.change(fn=visible_1, inputs=[cleaner0], outputs=[clean_strength0])
                merge_instrument.change(fn=visible_1, inputs=[merge_instrument], outputs=[vocal_instrument])
            with gr.Row():
                not_merge_backing.change(fn=lambda audio, merge: visible_1(audio and not merge), inputs=[use_audio, not_merge_backing], outputs=[main_backing])
                not_merge_backing.change(fn=backing_change, inputs=[convert_backing, not_merge_backing], outputs=[use_original])
            with gr.Row():
                method.change(fn=lambda method: visible_1(True if method == "hybrid" else False), inputs=[method], outputs=[hybrid_method])
                method.change(fn=hoplength_show, inputs=[method, hybrid_method], outputs=[hop_length])
            with gr.Row():
                hybrid_method.change(fn=hoplength_show, inputs=[method, hybrid_method], outputs=[hop_length])
                refesh.click(fn=change_choices, inputs=[], outputs=[model_pth, model_index])
                model_pth.change(fn=get_index, inputs=[model_pth], outputs=[model_index])
            with gr.Row():
                input0.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[input0], outputs=[input_audio0])
                input_audio0.change(fn=lambda audio: audio if audio else None, inputs=[input_audio0], outputs=[play_audio])
            with gr.Row():
                embedders.change(fn=lambda embedders: visible_1(True if embedders == "custom" else False), inputs=[embedders], outputs=[custom_embedders])
                refesh0.click(fn=refesh_audio, inputs=[], outputs=[input_audio0])
            with gr.Row():
                convert_button.click(
                    fn=convert_audio,
                    inputs=[
                        cleaner0,
                        upscale,
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
                        batch_process,
                        batch_size,
                        split_audio,
                        f0_autotune_strength
                    ],
                    outputs=[main_convert, backing_convert, main_backing, original_convert, vocal_instrument],
                    api_name="convert_audio"
                )

        with gr.TabItem(translations["convert_text"]):
            gr.Markdown(translations["convert_text_markdown"])
            with gr.Row():
                gr.Markdown(translations["convert_text_markdown_2"])
            with gr.Row():
                with gr.Column():
                    use_txt = gr.Checkbox(label=translations["input_txt"], value=False, interactive=True)
                    prompt = gr.Textbox(label=translations["text_to_speech"], value="", placeholder="Hello Words", lines=2)
                    with gr.Row():
                        speed = gr.Slider(label=translations["voice_speed"], info=translations["voice_speed_info"], minimum=-100, maximum=100, value=0, step=1)
                        pitch0 = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                with gr.Column():
                    tts_button = gr.Button(translations["tts_1"], variant="primary", scale=2)
                    convert_button0 = gr.Button(translations["tts_2"], variant="secondary", scale=2)
            with gr.Row():
                with gr.Column():
                    tts_voice = gr.Dropdown(label=translations["voice"], choices=tts_voice, interactive=True, value="vi-VN-NamMinhNeural")
                    txt_input = gr.File(label=translations["drop_text"], file_types=['txt'], visible=False)  
                with gr.Column():
                    with gr.Accordion(translations["model_accordion"], open=True):
                        with gr.Row():
                            model_pth0 = gr.Dropdown(label=translations["model_name"], choices=sorted(model_name), value=sorted(model_name)[0] if len(sorted(model_name)) > 0 else '', interactive=True, allow_custom_value=True)
                            model_index0 = gr.Dropdown(label=translations["index_path"], choices=sorted(index_path), value=sorted(index_path)[0] if len(sorted(index_path)) > 0 else '', interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refesh1 = gr.Button(translations["refesh"])
                        with gr.Row():
                            index_strength0 = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True)
                    with gr.Accordion(translations["output_path"], open=False):
                        export_format0 = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "m4a"], value="wav", interactive=True)
                        output_audio0 = gr.Textbox(label=translations["output_tts"], value="audios/tts.wav", placeholder="audios/tts.wav", info=translations["tts_output"], interactive=True)
                        output_audio1 = gr.Textbox(label=translations["output_tts_convert"], value="audios/tts-convert.wav", placeholder="audios/tts-convert.wav", info=translations["tts_output"], interactive=True)
                    with gr.Accordion(translations["setting"], open=False):
                        with gr.Accordion(translations["f0_method"], open=False):
                            method0 = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=["pm", "dio", "crepe-tiny", "crepe", "fcpe", "rmvpe", "harvest", "hybrid"], value="rmvpe", interactive=True)
                            hybrid_method0 = gr.Radio(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[rmvpe+harvest]"], value="hybrid[pm+dio]", interactive=True, visible=False)
                            hop_length0 = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion(translations["hubert_model"], open=False):
                            embedders0 = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "custom"], value="contentvec_base", interactive=True)
                            custom_embedders0 = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=False)
                        with gr.Group():
                            with gr.Row():
                                split_audio0 = gr.Checkbox(label=translations["split_audio"], info=translations["split_audio_info"], value=False, interactive=True)
                                batch_process0 = gr.Checkbox(label=translations["batch_process"], info=translations["batch_process_info"], value=False, interactive=True, visible=False)
                            with gr.Row():
                                    batch_size0 = gr.Slider(minimum=1, maximum=10, label=translations["batch_size"], info=translations["batch_size_info"], value=1, step=1, interactive=True, visible=False)
                        with gr.Row():
                            cleaner1 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                            upscale2 = gr.Checkbox(label=translations["upscale_audio"], value=False, interactive=True)
                            autotune3 = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)          
                        with gr.Column():
                            f0_autotune_strength0 = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=False)
                            clean_strength1 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
                            resample_sr0 = gr.Slider(minimum=0, maximum=48000, label=translations["resample"], info=translations["resample_info"], value=0, step=1, interactive=True)
                            filter_radius0 = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                            volume_envelope0 = gr.Slider(minimum=0, maximum=1, label=translations["volume_envelope"], info=translations["volume_envelope_info"], value=1, step=0.1, interactive=True)
                            protect0 = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.33, step=0.01, interactive=True)
            with gr.Row():
                gr.Markdown(translations["output_tts_markdown"])
            with gr.Row():
                tts_voice_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["output_text_to_speech"])
                tts_voice_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["output_file_tts_convert"])
            with gr.Row():
                batch_process0.change(fn=visible_1, inputs=[batch_process0], outputs=[batch_size0])
                split_audio0.change(fn=valueFalse_visible1, inputs=[split_audio0], outputs=[batch_process0])
                autotune3.change(fn=visible_1, inputs=[autotune3], outputs=[f0_autotune_strength0])
            with gr.Row():
                cleaner1.change(fn=visible_1, inputs=[cleaner1], outputs=[clean_strength1])
                method0.change(fn=lambda method: visible_1(True if method == "hybrid" else False), inputs=[method0], outputs=[hybrid_method0])
                method0.change(fn=hoplength_show, inputs=[method0, hybrid_method0], outputs=[hop_length0])
                hybrid_method0.change(fn=hoplength_show, inputs=[method0, hybrid_method0], outputs=[hop_length0])
            with gr.Row():
                refesh1.click(fn=change_choices, inputs=[], outputs=[model_pth0, model_index0])
                model_pth0.change(fn=get_index, inputs=[model_pth0], outputs=[model_index0])
                embedders0.change(fn=lambda embedders: visible_1(True if embedders == "custom" else False), inputs=[embedders0], outputs=[custom_embedders0])
            with gr.Row():
                txt_input.upload(fn=process_input, inputs=[txt_input], outputs=[prompt])
                use_txt.change(fn=visible_1, inputs=[use_txt], outputs=[txt_input])
            with gr.Row():
                tts_button.click(
                    fn=TTS, 
                    inputs=[
                        prompt, 
                        tts_voice, 
                        speed, 
                        output_audio0
                    ], 
                    outputs=[tts_voice_audio],
                    api_name="text-to-speech"
                )
                convert_button0.click(
                    fn=convert_tts,
                    inputs=[
                        cleaner1, 
                        upscale2, 
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
                        batch_process0,
                        batch_size0,
                        split_audio0,
                        f0_autotune_strength0
                    ],
                    outputs=[tts_voice_convert],
                    api_name="convert_tts"
                )

        with gr.TabItem(translations["audio_effects"]):
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
                            audio_in_path = gr.Dropdown(label=translations["input_audio"], value="" if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios")[0], choices=[] if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios"), info="Nhập đường dẫn đầu vào âm thanh", interactive=True, allow_custom_value=True)
                            audio_out_path = gr.Textbox(label=translations["output_audio"], value="audios/audio_effects.wav", placeholder="audios/audio_effects.wav", info=translations["provide_output"], interactive=True)
                        with gr.Row():
                            audio_output_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac"], value="wav", interactive=True)
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
                                audio_effect_resample_sr = gr.Slider(minimum=0, maximum=48000, step=1, value=0, label=translations["resample"], info=translations["resample_info"], interactive=True, visible=False)
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
                audio_effects_refesh.click(fn=refesh_audio, inputs=[], outputs=[audio_in_path])
            with gr.Row():
                more_options.change(fn=lambda: [False]*4, inputs=[], outputs=[fade, bass_or_treble, limiter, resample_checkbox])
                more_options.change(fn=lambda: [False]*4, inputs=[], outputs=[distortion_checkbox, gain_checkbox, clipping_checkbox, bitcrush_checkbox])
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
                        fade
                    ],
                    outputs=[audio_play_output],
                    api_name="audio_effects"
                )

        with gr.TabItem(translations["createdataset"]):
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
                                separator_reverb = gr.Checkbox(label=translations["dereveb_audio"], value=False, interactive=False)
                                denoise_mdx = gr.Checkbox(label=translations["denoise"], value=False, interactive=False)
                            with gr.Row():
                                clean_audio = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                                resample = gr.Checkbox(label=translations["resample"], value=False, interactive=True)
                                skip = gr.Checkbox(label=translations["skip"], value=False, interactive=True)
                        with gr.Row():
                            resample_sample_rate = gr.Slider(minimum=0, maximum=48000, step=1, value=0, label=translations["resample"], info=translations["resample_info"], interactive=True, visible=False)
                            dataset_clean_strength = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label=translations["clean_strength"], info=translations["clean_strength_info"], interactive=True, visible=False)
                    with gr.Column():
                        create_button = gr.Button(translations["createdataset"], variant="primary", scale=2)
            with gr.Row():
                with gr.Column():
                    with gr.Group(visible=False) as separator_dataset:
                        with gr.Row() as kim_vocal_row:
                            kim_vocal_version = gr.Radio(label=translations["model_ver"], info=translations["model_ver_info"], choices=["Version-1", "Version-2"], value="Version-2", interactive=True, visible=False)
                            kim_vocal_overlap = gr.Radio(label=translations["overlap"], info=translations["overlap_info"], choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True, visible=False)
                        with gr.Row() as kim_vocal_row_2:
                            kim_vocal_segments_size = gr.Slider(label=translations["segments_size"], info=translations["segments_size_info"], minimum=32, maximum=4000, value=256, step=8, interactive=True, visible=False)
                            kim_vocal_hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=8192, value=1024, step=1, interactive=True, visible=False)
                        with gr.Row() as kim_vocal_row_3:
                            kim_vocal_batch_size = gr.Slider(label=translations["batch_size"], info=translations["mdx_batch_size_info"], minimum=1, maximum=64, value=1, step=1, interactive=True, visible=False)
                    with gr.Row():
                        create_dataset_info = gr.Textbox(label=translations["create_dataset_info"], value="", interactive=False)
                with gr.Row():
                    with gr.Column():
                        output_dataset = gr.Textbox(label=translations["output_data"], info=translations["output_data_info"], value="dataset", placeholder="dataset", interactive=True)
                        with gr.Row():
                            skip_start = gr.Textbox(label=translations["skip_start"], info=translations["skip_start_info"], value="", placeholder="0,...", interactive=True, visible=False)
                            skip_end = gr.Textbox(label=translations["skip_end"], info=translations["skip_end_info"], value="", placeholder="0,...", interactive=True, visible=False)
            with gr.Row():
                separator_audio.change(fn=interactive_1, inputs=[separator_audio], outputs=[separator_reverb])
                separator_audio.change(fn=interactive_1, inputs=[separator_audio], outputs=[denoise_mdx])
                separator_audio.change(fn=visible_1, inputs=[separator_audio], outputs=[separator_dataset])
            with gr.Row():
                separator_audio.change(fn=visible_1, inputs=[separator_audio], outputs=[kim_vocal_row])
                separator_audio.change(fn=visible_1, inputs=[separator_audio], outputs=[kim_vocal_row_2])
                separator_audio.change(fn=visible_1, inputs=[separator_audio], outputs=[kim_vocal_row_3])
            with gr.Row():
                resample.change(fn=visible_1, inputs=[resample], outputs=[resample_sample_rate])
                clean_audio.change(fn=visible_1, inputs=[clean_audio], outputs=[dataset_clean_strength])
            with gr.Row():
                skip.change(fn=valueEmpty_visible1, inputs=[skip], outputs=[skip_start])
                skip.change(fn=valueEmpty_visible1, inputs=[skip], outputs=[skip_end])
            with gr.Row():
                create_button.click(
                    fn=create_dataset,
                    inputs=[
                        dataset_url, 
                        output_dataset, 
                        resample, 
                        resample_sample_rate, 
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
                        kim_vocal_batch_size
                    ],
                    outputs=[create_dataset_info],
                    api_name="create_dataset"
                )

        with gr.TabItem(translations["training_model"]):
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
                                training_f0 = gr.Checkbox(label=translations["training_pitch"], info=translations["training_pitch_info"], value=True, interactive=True)
                                upload = gr.Checkbox(label=translations["upload"], info=translations["upload_dataset"], value=False, interactive=True)
                                preprocess_cut = gr.Checkbox(label=translations["split_audio"], info=translations["preprocess_split"], value=False, interactive=True)
                                process_effects = gr.Checkbox(label=translations["preprocess_effect"], info=translations["preprocess_effect_info"], value=False, interactive=True)
                            with gr.Column():
                                clean_dataset = gr.Checkbox(label=translations["clear_dataset"], info=translations["clear_dataset_info"], value=False, interactive=True)
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
                                extract_embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "custom"], value="contentvec_base", interactive=True)
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
                                sync_graph = gr.Checkbox(label=translations["sync_graph"], info=translations["sync_graph_info"], value=False, interactive=True)
                                cache_in_gpu = gr.Checkbox(label=translations["cache_in_gpu"], info=translations["cache_in_gpu_info"], value=False, interactive=True)
                            with gr.Column():
                                dataset_path = gr.Textbox(label=translations["dataset_folder"], value="dataset", interactive=True, visible=False)
                            with gr.Column():
                                threshold = gr.Slider(minimum=1, maximum=100, value=50, step=1, label=translations["threshold"], interactive=True, visible=False)
                                with gr.Accordion(translations["setting_cpu_gpu"], open=False):
                                    with gr.Column():
                                        gpu_number = gr.Textbox(label=translations["gpu_number"], value=str(get_number_of_gpus()), info=translations["gpu_number_info"], interactive=True)
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
                refesh_file.click(fn=change_choices, inputs=[], outputs=[model_file, index_file]) 
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
                    fn=lambda files, folder: [shutil.move(f.name, os.path.join(folder, os.path.split(f.name)[1])) for f in files] if folder != "" else gr.Warning('Vui lòng nhập tên thư mục dữ liệu'),
                    inputs=[upload_dataset, dataset_path], 
                    outputs=[], 
                    api_name="upload_dataset"
                )           
            with gr.Row():
                not_use_pretrain.change(fn=lambda a, b: [visible_1(a and not b), visible_1(a and not b), visible_1(a and not b), visible_1(a and not b)], inputs=[custom_pretrain, not_use_pretrain], outputs=[pretrained_D, pretrained_G, refesh_pretrain, pretrain_setting])
                custom_pretrain.change(fn=lambda a, b: [visible_1(a and not b), visible_1(a and not b), visible_1(a and not b), visible_1(a and not b)], inputs=[custom_pretrain, not_use_pretrain], outputs=[pretrained_D, pretrained_G, refesh_pretrain, pretrain_setting])
                refesh_pretrain.click(fn=change_choices_pretrained, inputs=[], outputs=[pretrained_D, pretrained_G])
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
                        sync_graph,
                        cache_in_gpu,
                        model_author
                    ],
                    outputs=[training_info],
                    api_name="training_model"
                )

        with gr.TabItem(translations["fushion"]):
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

        with gr.TabItem(translations["read_model"]):
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

        with gr.TabItem(translations["downloads"]):
            gr.Markdown(translations["download_markdown"])
            with gr.Row():
                gr.Markdown(translations["download_markdown_2"])
            with gr.Row():
                with gr.Accordion(translations["model_download"], open=True):
                    with gr.Row():
                        downloadmodel = gr.Radio(label=translations["model_download_select"], choices=[translations["download_url"], translations["download_from_csv"], translations["download_from_applio"], translations["upload"]], interactive=True, value=translations["download_url"])
                    with gr.Row():
                        gr.Markdown("___")
                    with gr.Row():
                        url_input = gr.Textbox(label=translations["model_url"], value="", placeholder="https://...", scale=6, visible=True)
                        model_name = gr.Textbox(label=translations["modelname"], value="", placeholder=translations["modelname"], scale=2, visible=True)
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
                with gr.Accordion(translations["download_pretrainec"], open=False):
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
                        model_name
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
                downloadmodel.change(fn=download_change, inputs=[downloadmodel], outputs=[url_input, model_name, url_download, model_browser, download_from_browser, search_name, search, search_dropdown, download, model_upload])
                search.click(fn=search_models, inputs=[search_name], outputs=[search_dropdown, download])
                model_upload.upload(fn=save_drop_model, inputs=[model_upload], outputs=[model_upload])
                download.click(
                    fn=lambda model: download_model(model_options[model], model), 
                    inputs=[search_dropdown], 
                    outputs=[search_dropdown],
                    api_name="download_applio"
                )
            with gr.Row():
                pretrain_download_choices.change(fn=download_pretrained_change, inputs=[pretrain_download_choices], outputs=[pretrainD, pretrainG, download_pretrain_button, pretrain_choices, sample_rate_pretrain, download_pretrain_choices_button, pretrain_upload_d, pretrain_upload_g])
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
                    fn=lambda pretrain_upload_g: shutil.move(pretrain_upload_g.name, os.path.join("assets", "model", "pretrained_custom")), 
                    inputs=[pretrain_upload_g], 
                    outputs=[],
                    api_name="upload_pretrain_g"
                )
                pretrain_upload_d.upload(
                    fn=lambda pretrain_upload_d: shutil.move(pretrain_upload_d.name, os.path.join("assets", "model", "pretrained_custom")), 
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
                    fn=lambda hubert: shutil.move(hubert.name, os.path.join("assets", "model", "embedders")), 
                    inputs=[hubert_input], 
                    outputs=[],
                    api_name="upload_hubert"
                )

        with gr.TabItem(translations["settings"]):
            gr.Markdown(translations["settings_markdown"])
            with gr.Row():
                gr.Markdown(translations["settings_markdown_2"])
            with gr.Row():
                with gr.Column():
                    language_dropdown = gr.Dropdown(label=translations["lang"], interactive=True, info=translations["lang_restart"], choices=configs["support_language"], value=configs["language"])
                    change_lang = gr.Button(translations["change_lang"], variant=["primary"], scale=2)
                with gr.Column():
                    toggle_button = gr.Button(translations["change_light_dark"], variant=["secondary"], scale=2)
            with gr.Row():
                with gr.Column():
                    fp_select = gr.Radio(label=translations["fp_train"], info=translations["fp_info"], value="fp32", choices=["fp16", "fp32"], interactive=True)
                    fp_button = gr.Button(translations["fp_button"], variant=["primary"], scale=2)
                with gr.Column():
                    theme_dropdown = gr.Dropdown(label=translations["theme"], interactive=True, info=translations["theme_restart"], choices=configs["themes"], value=configs["theme"], allow_custom_value=True)
                    changetheme = gr.Button(translations["theme_button"], variant=["primary"], scale=2)
            with gr.Row():
                toggle_button.click(fn=None, js="""() => {document.body.classList.toggle('dark')}""")
                fp_button.click(fn=change_fp, inputs=[fp_select], outputs=[])
            with gr.Row():
                change_lang.click(fn=change_language, inputs=[language_dropdown], outputs=[])
                change_lang.click(fn=restart_app, inputs=[], outputs=[])
            with gr.Row():
                changetheme.click(fn=change_theme, inputs=[theme_dropdown], outputs=[])
                changetheme.click(fn=restart_app, inputs=[], outputs=[])
            with gr.Row():
                change_lang.click(fn=None, js="""setTimeout(function() {location.reload()}, 30000)""", inputs=[], outputs=[])
                changetheme.click(fn=None, js="""setTimeout(function() {location.reload()}, 30000)""", inputs=[], outputs=[])

        with gr.TabItem(translations["source"]):
            gr.Markdown(translations["source_info"])
            with gr.Row():
                gr.Markdown("___")
            with gr.Row():
                gr.Markdown(translations["credits"].format(author=codecs.decode("uggcf://tvguho.pbz/CunzUhlauNau16", "rot13"), applio=codecs.decode("uggcf://tvguho.pbz/VNUvfcnab/Nccyvb/gerr/znva?gno=ernqzr-bi-svyr", "rot13"), ai_hispano=codecs.decode("uggcf://tvguho.pbz/VNUvfcnab", "rot13"), rvc_webui=codecs.decode("uggcf://tvguho.pbz/EIP-Cebwrpg/Ergevriny-onfrq-Ibvpr-Pbairefvba-JroHV?gno=ernqzr-bi-svyr", "rot13"), rvc_boss=codecs.decode("uggcf://tvguho.pbz/EIP-Obff", "rot13"), python_audio_separator=codecs.decode("uggcf://tvguho.pbz/abznqxnenbxr/clguba-nhqvb-frcnengbe?gno=ernqzr-bi-svyr", "rot13"), andrew_beveridge=codecs.decode("uggcf://tvguho.pbz/orirenqo", "rot13")))
    
    print(translations["set_lang"].format(lang=configs["language"]))

    for i in range(configs["num_of_restart"]):
        try:
            app.queue().launch(
                favicon_path=os.path.join("assets", "miku.png"), 
                server_name=server_name, 
                server_port=port, 
                show_error=show_error, 
                inbrowser=False, 
                share=share
            )
            break
        except OSError:
            port -= 1
        except Exception as e:
            raise RuntimeError(translations["error_occurred"].format(e=e))