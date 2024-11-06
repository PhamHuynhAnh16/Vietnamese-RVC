import os
import re
import sys
import torch
import codecs
import yt_dlp
import shutil
import zipfile
import logging
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

from main.tools import gdown, meganz, mediafire


logging.getLogger("wget").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

python = sys.executable = "python"

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


if not os.path.exists(os.path.join("assets", "miku.png")): run(["wget", "-q", "--show-progress", "--no-check-certificate", miku_image, "-P", os.path.join("assets")], check=True)


tts_voice = [
    'af-ZA-AdriNeural', 'af-ZA-WillemNeural', 'sq-AL-AnilaNeural', 
    'sq-AL-IlirNeural', 'am-ET-AmehaNeural', 'am-ET-MekdesNeural', 
    'ar-DZ-AminaNeural', 'ar-DZ-IsmaelNeural', 'ar-BH-AliNeural', 
    'ar-BH-LailaNeural', 'ar-EG-SalmaNeural', 'ar-EG-ShakirNeural', 
    'ar-IQ-BasselNeural', 'ar-IQ-RanaNeural', 'ar-JO-SanaNeural', 
    'ar-JO-TaimNeural', 'ar-KW-FahedNeural', 'ar-KW-NouraNeural', 
    'ar-LB-LaylaNeural', 'ar-LB-RamiNeural', 'ar-LY-ImanNeural', 
    'ar-LY-OmarNeural', 'ar-MA-JamalNeural', 'ar-MA-MounaNeural', 
    'ar-OM-AbdullahNeural', 'ar-OM-AyshaNeural', 'ar-QA-AmalNeural', 
    'ar-QA-MoazNeural', 'ar-SA-HamedNeural', 'ar-SA-ZariyahNeural', 
    'ar-SY-AmanyNeural', 'ar-SY-LaithNeural', 'ar-TN-HediNeural', 
    'ar-TN-ReemNeural', 'ar-AE-FatimaNeural', 'ar-AE-HamdanNeural', 
    'ar-YE-MaryamNeural', 'ar-YE-SalehNeural', 'az-AZ-BabekNeural', 
    'az-AZ-BanuNeural', 'bn-BD-NabanitaNeural', 'bn-BD-PradeepNeural', 
    'bn-IN-BashkarNeural', 'bn-IN-TanishaaNeural', 'bs-BA-GoranNeural', 
    'bs-BA-VesnaNeural', 'bg-BG-BorislavNeural', 'bg-BG-KalinaNeural', 
    'my-MM-NilarNeural', 'my-MM-ThihaNeural', 'ca-ES-EnricNeural', 
    'ca-ES-JoanaNeural', 'zh-HK-HiuGaaiNeural', 'zh-HK-HiuMaanNeural', 
    'zh-HK-WanLungNeural', 'zh-CN-XiaoxiaoNeural', 'zh-CN-XiaoyiNeural', 
    'zh-CN-YunjianNeural', 'zh-CN-YunxiNeural', 'zh-CN-YunxiaNeural', 
    'zh-CN-YunyangNeural', 'zh-CN-liaoning-XiaobeiNeural', 'zh-TW-HsiaoChenNeural', 
    'zh-TW-YunJheNeural', 'zh-TW-HsiaoYuNeural', 'zh-CN-shaanxi-XiaoniNeural', 
    'hr-HR-GabrijelaNeural', 'hr-HR-SreckoNeural', 'cs-CZ-AntoninNeural', 
    'cs-CZ-VlastaNeural', 'da-DK-ChristelNeural', 'da-DK-JeppeNeural', 
    'nl-BE-ArnaudNeural', 'nl-BE-DenaNeural', 'nl-NL-ColetteNeural', 
    'nl-NL-FennaNeural', 'nl-NL-MaartenNeural', 'en-AU-NatashaNeural', 
    'en-AU-WilliamNeural', 'en-CA-ClaraNeural', 'en-CA-LiamNeural', 
    'en-HK-SamNeural', 'en-HK-YanNeural', 'en-IN-NeerjaExpressiveNeural', 
    'en-IN-NeerjaNeural', 'en-IN-PrabhatNeural', 'en-IE-ConnorNeural', 
    'en-IE-EmilyNeural', 'en-KE-AsiliaNeural', 'en-KE-ChilembaNeural', 
    'en-NZ-MitchellNeural', 'en-NZ-MollyNeural', 'en-NG-AbeoNeural', 
    'en-NG-EzinneNeural', 'en-PH-JamesNeural', 'en-PH-RosaNeural', 
    'en-SG-LunaNeural', 'en-SG-WayneNeural', 'en-ZA-LeahNeural', 
    'en-ZA-LukeNeural', 'en-TZ-ElimuNeural', 'en-TZ-ImaniNeural', 
    'en-GB-LibbyNeural', 'en-GB-MaisieNeural', 'en-GB-RyanNeural', 
    'en-GB-SoniaNeural', 'en-GB-ThomasNeural', 'en-US-AvaMultilingualNeural', 
    'en-US-AndrewMultilingualNeural', 'en-US-EmmaMultilingualNeural', 
    'en-US-BrianMultilingualNeural', 'en-US-AvaNeural', 'en-US-AndrewNeural', 
    'en-US-EmmaNeural', 'en-US-BrianNeural', 'en-US-AnaNeural', 'en-US-AriaNeural', 
    'en-US-ChristopherNeural', 'en-US-EricNeural', 'en-US-GuyNeural', 
    'en-US-JennyNeural', 'en-US-MichelleNeural', 'en-US-RogerNeural', 
    'en-US-SteffanNeural', 'et-EE-AnuNeural', 'et-EE-KertNeural', 
    'fil-PH-AngeloNeural', 'fil-PH-BlessicaNeural', 'fi-FI-HarriNeural', 
    'fi-FI-NooraNeural', 'fr-BE-CharlineNeural', 'fr-BE-GerardNeural', 
    'fr-CA-ThierryNeural', 'fr-CA-AntoineNeural', 'fr-CA-JeanNeural', 
    'fr-CA-SylvieNeural', 'fr-FR-VivienneMultilingualNeural', 'fr-FR-RemyMultilingualNeural', 
    'fr-FR-DeniseNeural', 'fr-FR-EloiseNeural', 'fr-FR-HenriNeural', 
    'fr-CH-ArianeNeural', 'fr-CH-FabriceNeural', 'gl-ES-RoiNeural', 
    'gl-ES-SabelaNeural', 'ka-GE-EkaNeural', 'ka-GE-GiorgiNeural', 
    'de-AT-IngridNeural', 'de-AT-JonasNeural', 'de-DE-SeraphinaMultilingualNeural', 
    'de-DE-FlorianMultilingualNeural', 'de-DE-AmalaNeural', 'de-DE-ConradNeural', 
    'de-DE-KatjaNeural', 'de-DE-KillianNeural', 'de-CH-JanNeural', 
    'de-CH-LeniNeural', 'el-GR-AthinaNeural', 'el-GR-NestorasNeural', 
    'gu-IN-DhwaniNeural', 'gu-IN-NiranjanNeural', 'he-IL-AvriNeural', 
    'he-IL-HilaNeural', 'hi-IN-MadhurNeural', 'hi-IN-SwaraNeural', 
    'hu-HU-NoemiNeural', 'hu-HU-TamasNeural', 'is-IS-GudrunNeural', 
    'is-IS-GunnarNeural', 'id-ID-ArdiNeural', 'id-ID-GadisNeural', 
    'ga-IE-ColmNeural', 'ga-IE-OrlaNeural', 'it-IT-GiuseppeNeural', 
    'it-IT-DiegoNeural', 'it-IT-ElsaNeural', 'it-IT-IsabellaNeural', 
    'ja-JP-KeitaNeural', 'ja-JP-NanamiNeural', 'jv-ID-DimasNeural', 
    'jv-ID-SitiNeural', 'kn-IN-GaganNeural', 'kn-IN-SapnaNeural', 
    'kk-KZ-AigulNeural', 'kk-KZ-DauletNeural', 'km-KH-PisethNeural', 
    'km-KH-SreymomNeural', 'ko-KR-HyunsuNeural', 'ko-KR-InJoonNeural', 
    'ko-KR-SunHiNeural', 'lo-LA-ChanthavongNeural', 'lo-LA-KeomanyNeural', 
    'lv-LV-EveritaNeural', 'lv-LV-NilsNeural', 'lt-LT-LeonasNeural', 
    'lt-LT-OnaNeural', 'mk-MK-AleksandarNeural', 'mk-MK-MarijaNeural', 
    'ms-MY-OsmanNeural', 'ms-MY-YasminNeural', 'ml-IN-MidhunNeural', 
    'ml-IN-SobhanaNeural', 'mt-MT-GraceNeural', 'mt-MT-JosephNeural', 
    'mr-IN-AarohiNeural', 'mr-IN-ManoharNeural', 'mn-MN-BataaNeural', 
    'mn-MN-YesuiNeural', 'ne-NP-HemkalaNeural', 'ne-NP-SagarNeural', 
    'nb-NO-FinnNeural', 'nb-NO-PernilleNeural', 'ps-AF-GulNawazNeural', 
    'ps-AF-LatifaNeural', 'fa-IR-DilaraNeural', 'fa-IR-FaridNeural', 
    'pl-PL-MarekNeural', 'pl-PL-ZofiaNeural', 'pt-BR-ThalitaNeural', 
    'pt-BR-AntonioNeural', 'pt-BR-FranciscaNeural', 'pt-PT-DuarteNeural', 
    'pt-PT-RaquelNeural', 'ro-RO-AlinaNeural', 'ro-RO-EmilNeural', 
    'ru-RU-DmitryNeural', 'ru-RU-SvetlanaNeural', 'sr-RS-NicholasNeural', 
    'sr-RS-SophieNeural', 'si-LK-SameeraNeural', 'si-LK-ThiliniNeural', 
    'sk-SK-LukasNeural', 'sk-SK-ViktoriaNeural', 'sl-SI-PetraNeural', 
    'sl-SI-RokNeural', 'so-SO-MuuseNeural', 'so-SO-UbaxNeural', 
    'es-AR-ElenaNeural', 'es-AR-TomasNeural', 'es-BO-MarceloNeural', 
    'es-BO-SofiaNeural', 'es-CL-CatalinaNeural', 'es-CL-LorenzoNeural', 
    'es-ES-XimenaNeural', 'es-CO-GonzaloNeural', 'es-CO-SalomeNeural', 
    'es-CR-JuanNeural', 'es-CR-MariaNeural', 'es-CU-BelkysNeural', 
    'es-CU-ManuelNeural', 'es-DO-EmilioNeural', 'es-DO-RamonaNeural', 
    'es-EC-AndreaNeural', 'es-EC-LuisNeural', 'es-SV-LorenaNeural', 
    'es-SV-RodrigoNeural', 'es-GQ-JavierNeural', 'es-GQ-TeresaNeural', 
    'es-GT-AndresNeural', 'es-GT-MartaNeural', 'es-HN-CarlosNeural', 
    'es-HN-KarlaNeural', 'es-MX-DaliaNeural', 'es-MX-JorgeNeural', 
    'es-NI-FedericoNeural', 'es-NI-YolandaNeural', 'es-PA-MargaritaNeural', 
    'es-PA-RobertoNeural', 'es-PY-MarioNeural', 'es-PY-TaniaNeural', 
    'es-PE-AlexNeural', 'es-PE-CamilaNeural', 'es-PR-KarinaNeural', 
    'es-PR-VictorNeural', 'es-ES-AlvaroNeural', 'es-ES-ElviraNeural', 
    'es-US-AlonsoNeural', 'es-US-PalomaNeural', 'es-UY-MateoNeural', 
    'es-UY-ValentinaNeural', 'es-VE-PaolaNeural', 'es-VE-SebastianNeural', 
    'su-ID-JajangNeural', 'su-ID-TutiNeural', 'sw-KE-RafikiNeural', 
    'sw-KE-ZuriNeural', 'sw-TZ-DaudiNeural', 'sw-TZ-RehemaNeural', 
    'sv-SE-MattiasNeural', 'sv-SE-SofieNeural', 'ta-IN-PallaviNeural', 
    'ta-IN-ValluvarNeural', 'ta-MY-KaniNeural', 'ta-MY-SuryaNeural', 
    'ta-SG-AnbuNeural', 'ta-SG-VenbaNeural', 'ta-LK-KumarNeural', 
    'ta-LK-SaranyaNeural', 'te-IN-MohanNeural', 'te-IN-ShrutiNeural', 
    'th-TH-NiwatNeural', 'th-TH-PremwadeeNeural', 'tr-TR-AhmetNeural', 
    'tr-TR-EmelNeural', 'uk-UA-OstapNeural', 'uk-UA-PolinaNeural', 
    'ur-IN-GulNeural', 'ur-IN-SalmanNeural', 'ur-PK-AsadNeural', 
    'ur-PK-UzmaNeural', 'uz-UZ-MadinaNeural', 'uz-UZ-SardorNeural', 
    'vi-VN-HoaiMyNeural', 'vi-VN-NamMinhNeural', 'cy-GB-AledNeural', 
    'cy-GB-NiaNeural', 'zu-ZA-ThandoNeural', 'zu-ZA-ThembaNeural'
]


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

    return "\n".join(gpu_infos) if len(gpu_infos) > 0 else "Thật không may, không có GPU tương thích để hỗ trợ việc đào tạo của bạn."


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

    gr.Info(f"Đã tải lên tệp văn bản thành công")

    return file_contents


def download_change(select):
    selects = [False]*10

    if select == "Tải từ đường dẫn liên kết": selects[0] = selects[1] = selects[2] = True
    elif select == "Tải từ kho mô hình csv":  selects[3] = selects[4] = True
    elif select == "Tải mô hình từ Applio": selects[5] = selects[6] = True
    elif select == "Tải lên": selects[9] = True
    else: gr.Warning("Tùy chọn không hợp lệ")
    
    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]


def fetch_pretrained_data():
    response = requests.get(pretrained_json)
    response.raise_for_status()

    return response.json()


def download_pretrained_change(select):
    selects = [False]*8

    if select == "Đường dẫn mô hình": selects[0] = selects[1] = selects[2] = True
    elif select == "Danh sách mô hình": selects[3] = selects[4] = selects[5] = True
    elif select == "Tải lên": selects[6] = selects[7] = True
    else: gr.Warning("Tùy chọn không hợp lệ")

    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]


def update_sample_rate_dropdown(model):
    data = fetch_pretrained_data()

    if model != "Hoàn thành": return {"choices": list(data[model].keys()), "value": list(data[model].keys())[0], "__type__": "update"}


def if_done(done, p):
    while 1:
        if p.poll() is None: sleep(0.5)
        else: break

    done[0] = True


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

    if not pth or not os.path.exists(pth_path): return gr.Warning("Vui lòng cung cấp tệp mô hình hợp lệ!")
    if not index or not os.path.exists(index): return gr.Warning("Vui lòng cung cấp tệp chỉ mục hợp lệ")
    
    zip_file_path = os.path.join("assets", name + ".zip")

    gr.Info("Bắt đầu nén tệp...")
    
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(pth_path, os.path.basename(pth_path))
        zipf.write(index, os.path.basename(index))


    gr.Info("Hoàn thành")
    return zip_file_path    


def search_models(name):
    gr.Info("Bắt đầu tìm kiếm...")
    url = f"https://cjtfqzjfdimgpvpwhzlv.supabase.co/rest/v1/models?name=ilike.%25{name}%25&order=created_at.desc&limit=15"
    
    response = requests.get(url, headers={"apikey": model_search_api})
    data = response.json()


    if len(data) == 0:
        gr.Info(f"Không tìm thấy {name}")

        return [None, None]
    else:
        model_options.clear()
        model_options.update({item["name"] + " " + item["epochs"] + "e": item["link"] for item in data})

        gr.Info(f"Đã tìm thấy {len(model_options)} kết quả")
        return [{"value": "", "choices": model_options, "interactive": True, "visible": True, "__type__": "update"}, {"value": "Tải xuống", "visible": True, "__type__": "update"}]


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
    if not url: return gr.Warning("Vui lòng nhập đường dẫn liên kết")
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

        gr.Info("Bắt đầu tải nhạc...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        gr.Info("Hoàn thành!")
        return [audio_output, audio_output, "Hoàn thành"]


def download_model(url=None, model=None):
    if not url: return gr.Warning("Vui lòng cung cấp đường dẫn liên kết mô hình")
    if not model: return gr.Warning("Vui lòng nhập tên mô hình để lưu")

    model = model.replace('.pth', '').replace('.index', '').replace('.zip', '').replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').strip()
    url = url.replace('/blob/', '/resolve/').replace('?download=true', '').strip()
    
    download_dir = os.path.join("download_model")
    weights_dir = os.path.join("assets", "weights")
    logs_dir = os.path.join("assets", "logs")

    if not os.path.exists(download_dir): os.makedirs(download_dir, exist_ok=True)
    if not os.path.exists(weights_dir): os.makedirs(weights_dir, exist_ok=True)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir, exist_ok=True)
    
    try:
        gr.Info("Bắt đầu tải xuống...")

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
            else:
                gr.Warning("Liên kết mô hình của bạn không được hỗ trợ")
                return "Liên kết mô hình của bạn không được hỗ trợ"
        
        gr.Info("Hoàn thành")
        return "Hoàn thành"
    except Exception as e:
        gr.Warning(f"Đã xảy ra lỗi: {e}")

        print(f"Đã xảy ra lỗi: {e}")
        return f"Đã xảy ra lỗi: {e}"
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

        if file_name.endswith(".pth") and file_name.endswith(".index"): gr.Warning("Tệp bạn vừa tải lên không phải là tệp mô hình!")
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
                gr.Warning("Không phân tích được mô hình!")
                return None
        
        gr.Info(f"Đã tải lên thành công {file_name}")
        return None
    except Exception as e:
        gr.Warning(f"Đã xảy ra lỗi {e}")

        print(f"Đã xảy ra lỗi {e}")
        return None
    finally:
        shutil.rmtree(save_model_temp, ignore_errors=True)


def download_pretrained_model(choices, model, sample_rate):
    if choices == "Danh sách mô hình":
        data = fetch_pretrained_data()
        paths = data[model][sample_rate]

        pretraineds_custom_path = os.path.join("assets", "model", "pretrained_custom")

        if not os.path.exists(pretraineds_custom_path): os.makedirs(pretraineds_custom_path, exist_ok=True)

        d_url = hugging_face_codecs + f"/{paths['D']}"
        g_url = hugging_face_codecs + f"/{paths['G']}"

        gr.Info("Tải xuống huấn luyện trước...")

        run(["wget", "-q", "--show-progress", "--no-check-certificate", d_url.replace('/blob/', '/resolve/').replace('?download=true', '').strip(), "-P", os.path.join(pretraineds_custom_path)], check=True)
        run(["wget", "-q", "--show-progress", "--no-check-certificate", g_url.replace('/blob/', '/resolve/').replace('?download=true', '').strip(), "-P", os.path.join(pretraineds_custom_path)], check=True)

        gr.Info("Hoàn thành")
        return "Hoàn thành"
    elif choices == "Đường dẫn mô hình":
        if not model: return gr.Warning("Vui lòng cung cấp đường dẫn mô hình huấn luyện trước D")
        if not sample_rate: return gr.Warning("Vui lòng cung cấp đường dẫn mô hình huấn luyện trước G")

        gr.Info("Tải xuống huấn luyện trước...")

        run(["wget", "-q", "--show-progress", "--no-check-certificate", model, "-P", os.path.join(pretraineds_custom_path)], check=True)
        run(["wget", "-q", "--show-progress", "--no-check-certificate", sample_rate, "-P", os.path.join(pretraineds_custom_path)], check=True)

        gr.Info("Hoàn thành")
        return "Hoàn thành"
    

def hubert_download(hubert):
    if not hubert: 
        gr.Warning("Vui lòng đưa đường dẫn liên kết đến mô hình học cách nói")
        return "Vui lòng đưa đường dẫn liên kết đến mô hình học cách nói"
    
    run(["wget", "-q", "--show-progress", "--no-check-certificate", hubert.replace('/blob/', '/resolve/').replace('?download=true', '').strip(), "-P", os.path.join("assets", "model", "embedders")], check=True)

    gr.Info("Hoàn Thành!")
    return "Hoàn Thành!"


def fushion_model(name, pth_1, pth_2, ratio):
    if not name:
        gr.Warning("Vui lòng cung cấp tên") 
        return ["Vui lòng cung cấp tên", None]
    
    if not pth_1 or not os.path.exists(pth_1):
        gr.Warning("Vui lòng cung cấp mô hình 1")
        return ["Vui lòng cung cấp mô hình 1", None]
    
    if not pth_2 or not os.path.exists(pth_2):
        gr.Warning("Vui lòng cung cấp mô hình 2")
        return ["Vui lòng cung cấp mô hình 2", None]
    
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
            gr.Warning("Tốc độ lấy mẫu của hai mô hình không giống nhau")
            return ["Tốc độ lấy mẫu của hai mô hình không giống nhau", None]

        cfg = ckpt1["config"]
        cfg_f0 = ckpt1["f0"]
        cfg_version = ckpt1["version"]
        cfg_sr = ckpt1["sr"]

        ckpt1 = extract(ckpt1) if "model" in ckpt1 else ckpt1["weight"]
        ckpt2 = extract(ckpt2) if "model" in ckpt2 else ckpt2["weight"]

        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())): 
            gr.Warning("Không thể hợp nhất các mô hình. Các kiến ​​trúc mô hình không giống nhau")
            return ["Không thể hợp nhất các mô hình. Các kiến ​​trúc mô hình không giống nhau", None]
         
        gr.Info("Bắt đầu dung hợp mô hình...")

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
        opt["infos"] = f"Mô hình được {name} được dung hợp từ {pth_1} và {pth_2} với ratio {ratio}"

        output_model = os.path.join("assets", "weights")

        if not os.path.exists(output_model): os.makedirs(output_model, exist_ok=True)

        torch.save(opt, os.path.join(output_model, f"{name}.pth"))

        gr.Info("Hoàn thành")
        return ["Hoàn thành", output_model]
    except Exception as error:
        gr.Warning(f"Đã xảy ra lỗi khi hợp nhất các mô hình: {error}")

        print(f"Đã xảy ra lỗi khi hợp nhất các mô hình: {error}")
        return [error, None]


def model_info(path):
    if not path or not os.path.exists(path): gr.Warning("Không tìm thấy mô hình!")
    
    def prettify_date(date_str):
        if date_str == "Không tìm thấy thời gian tạo": return None

        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return "Định dạng không hợp lệ"
        
    model_data = torch.load(path, map_location=torch.device("cpu"))

    gr.Info(f"Các mô hình được huấn luyện trên các ứng dụng khác nhau có thể đem lại các thông tin khác nhau hoặc không thể đọc!")

    epochs = model_data.get("epoch", None)

    if epochs is None: 
        epochs = model_data.get("info", None)
        epoch = epochs.replace("epoch", "").replace("e", "").isdigit()

        if epoch and epochs is None: epochs = "Không tìm thấy kỷ nguyên"
        
    steps = model_data.get("step", "Không tìm thấy")

    sr = model_data.get("sr", "Không tìm thấy tốc độ lấy mẫu")
    f0 = model_data.get("f0", "Không tìm thấy huấn luyện cao độ")

    version = model_data.get("version", "Không tìm thấy phiên bản")
    creation_date = model_data.get("creation_date", "Không tìm thấy thời gian tạo")
    model_hash = model_data.get("model_hash", "Không tìm thấy")

    pitch_guidance = "Được huấn luyện cao độ" if f0 == 1 else "Không được huấn luyện cao độ"

    creation_date_str = prettify_date(creation_date) if creation_date else "Không tìm thấy thời gian tạo"

    gr.Info("Hoàn thành")

    return (
        f"Kỷ nguyên: {epochs}\n"
        f"Số bước: {steps}\n"
        f"Phiên bản của mô hình: {version}\n"
        f"Tốc độ lấy mẫu: {sr}\n"
        f"Huấn luyện cao độ: {pitch_guidance}\n"
        f"Hash (ID): {model_hash}\n"
        f"Thời gian tạo: {creation_date_str}\n"
    )


def audio_effects(input_path, output_path, resample, resample_sr, chorus_depth, chorus_rate, chorus_mix, chorus_delay, chorus_feedback, distortion_drive, reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level, reverb_width, reverb_freeze_mode, pitch_shift, delay_seconds, delay_feedback, delay_mix, compressor_threshold, compressor_ratio, compressor_attack_ms, compressor_release_ms, limiter_threshold, limiter_release, gain_db, bitcrush_bit_depth, clipping_threshold, phaser_rate_hz, phaser_depth, phaser_centre_frequency_hz, phaser_feedback, phaser_mix, bass_boost_db, bass_boost_frequency, treble_boost_db, treble_boost_frequency, fade_in_duration, fade_out_duration, export_format, chorus, distortion, reverb, delay, compressor, limiter, gain, bitcrush, clipping, phaser, treble_bass_boost, fade_in_out):
    if not input_path or not os.path.exists(input_path): 
        gr.Warning("Vui lòng nhập đầu vào hợp lệ!")
        return None
        
    if not output_path:
        gr.Warning("Vui lòng nhập đầu ra!")
        return None
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_path): os.remove(output_path)
    
    gr.Info("Bắt đầu áp dụng hiệu ứng...")

    pitchshift = pitch_shift != 0

    cmd = f"{python} main/inference/audio_effects.py --input_path {input_path} --output_path {output_path} --resample {resample} --resample_sr {resample_sr} --chorus_depth {chorus_depth} --chorus_rate {chorus_rate} --chorus_mix {chorus_mix} --chorus_delay {chorus_delay} --chorus_feedback {chorus_feedback} --drive_db {distortion_drive} --reverb_room_size {reverb_room_size} --reverb_damping {reverb_damping} --reverb_wet_level {reverb_wet_level} --reverb_dry_level {reverb_dry_level} --reverb_width {reverb_width} --reverb_freeze_mode {reverb_freeze_mode} --pitch_shift {pitch_shift} --delay_seconds {delay_seconds} --delay_feedback {delay_feedback} --delay_mix {delay_mix} --compressor_threshold {compressor_threshold} --compressor_ratio {compressor_ratio} --compressor_attack_ms {compressor_attack_ms} --compressor_release_ms {compressor_release_ms} --limiter_threshold {limiter_threshold} --limiter_release {limiter_release} --gain_db {gain_db} --bitcrush_bit_depth {bitcrush_bit_depth} --clipping_threshold {clipping_threshold} --phaser_rate_hz {phaser_rate_hz} --phaser_depth {phaser_depth} --phaser_centre_frequency_hz {phaser_centre_frequency_hz} --phaser_feedback {phaser_feedback} --phaser_mix {phaser_mix} --bass_boost_db {bass_boost_db} --bass_boost_frequency {bass_boost_frequency} --treble_boost_db {treble_boost_db} --treble_boost_frequency {treble_boost_frequency} --fade_in_duration {fade_in_duration} --fade_out_duration {fade_out_duration} --export_format {export_format} --chorus {chorus} --distortion {distortion} --reverb {reverb} --pitchshift {pitchshift} --delay {delay} --compressor {compressor} --limiter {limiter} --gain {gain} --bitcrush {bitcrush} --clipping {clipping} --phaser {phaser} --treble_bass_boost {treble_bass_boost} --fade_in_out {fade_in_out}"
    os.system(cmd)

    gr.Info("Hoàn thành")

    return output_path 


async def TTS(prompt, voice, speed, output):
    if not prompt:
        gr.Warning("Vui lòng nhập văn bản để đọc!")
        return None
    
    if not voice:
        gr.Warning("Vui lòng chọn giọng!")
        return None
    
    if not output: 
        gr.Warning("Vui lòng cung cấp đầu ra!")
        return None

    gr.Info("Chuyển đổi văn bản...")

    output_dir = os.path.dirname(output)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    await edge_tts.Communicate(text=prompt, voice=voice, rate=f"+{speed}%" if speed >= 0 else f"{speed}%").save(output)

    gr.Info("Hoàn thành")

    return output


def separator_music(input, output, format, shifts, segments_size, overlap, clean_audio, clean_strength, backing_denoise, separator_model, kara_model, backing, mdx, mdx_denoise, reverb, reverb_denoise, backing_reverb, hop_length, batch_size):
    output = os.path.dirname(output)
    
    if not input or not os.path.exists(input): 
        gr.Warning("Vui lòng nhập đầu vào hợp lệ!")
        return [None, None, None, None]
    
    if not os.path.exists(output): 
        gr.Warning("Không tìm thấy thư mục đầu ra!")
        return [None, None, None, None]

    gr.Info("Bắt đầu tách nhạc...")

    cmd = f'{python} main/inference/separator_music.py --input_path {input} --output_path {output} --format {format} --shifts {shifts} --segments_size {segments_size} --overlap {overlap} --mdx_hop_length {hop_length} --mdx_batch_size {batch_size} --clean_audio {clean_audio} --clean_strength {clean_strength} --backing_denoise {backing_denoise} --kara_model {kara_model} --backing {backing} --mdx {mdx} --mdx_denoise {mdx_denoise} --reverb {reverb} --reverb_denoise {reverb_denoise} --backing_reverb {backing_reverb}'

    if separator_model == "HT-Normal" or separator_model == "HT-Tuned" or separator_model == "HD_MMI" or separator_model == "HT_6S": cmd += f' --demucs_model {separator_model}'
    else: cmd += f' --mdx_model {separator_model}'

    os.system(cmd)
    
    gr.Info("Hoàn thành")

    if not os.path.exists(output): os.makedirs(output)

    original_output = os.path.join(output, f"Original_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Original_Vocals.{format}")
    instrument_output = os.path.join(output, f"Instruments.{format}")
    main_output = os.path.join(output, f"Main_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Main_Vocals.{format}")
    backing_output = os.path.join(output, f"Backing_Vocals_No_Reverb.{format}") if backing_reverb else os.path.join(output, f"Backing_Vocals.{format}")

    if backing: return [original_output, instrument_output, main_output, backing_output]
    else: return [original_output, instrument_output, None, None]


def convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, clean_audio, clean_strength, export_format, embedder_model, upscale_audio, resample_sr, use_threads, max_threads, split_audio, f0_autotune_strength):
    if os.path.exists(output_path): os.remove(output_path)

    cmd = f"{python} main/inference/convert.py --pitch {pitch} --filter_radius {filter_radius} --index_rate {index_rate} --volume_envelope {volume_envelope} --protect {protect} --hop_length {hop_length} --f0_method {f0_method} --input_path {input_path} --output_path {output_path} --pth_path {pth_path} --index_path {index_path} --f0_autotune {f0_autotune} --clean_audio {clean_audio} --clean_strength {clean_strength} --export_format {export_format} --embedder_model {embedder_model} --upscale_audio {upscale_audio} --resample_sr {resample_sr} --use_threads {use_threads} --max_threads {max_threads} --split_audio {split_audio} --f0_autotune_strength {f0_autotune_strength}"
    os.system(cmd)


def convert_audio(clean, upscale, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, use_threads, max_threads, split_audio, f0_autotune_strength):
    def get_audio_file(label):
        matching_files = [f for f in os.listdir("audios") if label in f]
        if not matching_files: return "Không tìm thấy"
        
        return os.path.join("audios", matching_files[0])

    model_path = os.path.join("assets", "weights", model)

    if not use_audio:
        if merge_instrument or not_merge_backing or convert_backing or use_original:
            gr.Warning("Vui lòng bật sử dụng âm thanh vừa tách để sử dụng")
            return [None, None, None, None, None]

    if use_original:
        if convert_backing:
            gr.Warning("Tắt chuyển đổi giọng bè để có thể sử dụng giọng gốc")
            return [None, None, None, None, None]
        elif not_merge_backing:
            gr.Warning("Tắt không kết hợp giọng bè để có thể sử dụng giọng gốc")
            return [None, None, None, None, None]

    if not model or not os.path.exists(model_path):
        gr.Warning("Không tìm thấy mô hình")
        return [None, None, None, None, None]
    
    if not model or not os.path.exists(index):
        gr.Warning("Không tìm thấy chỉ mục")
        return [None, None, None, None, None]

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

            if original_vocal == "Không tìm thấy": original_vocal = get_audio_file('Original_Vocals.')

            if original_vocal == "Không tìm thấy": 
                gr.Warning("Không tìm thấy giọng gốc!")
                return [None, None, None, None, None]
            
            input_path = original_vocal
        else:
            main_vocal = get_audio_file('Main_Vocals_No_Reverb.')
            backing_vocal = get_audio_file('Backing_Vocals_No_Reverb.')

            if main_vocal == "Không tìm thấy": main_vocal = get_audio_file('Main_Vocals.')
            if not not_merge_backing and backing_vocal == "Không tìm thấy": backing_vocal = get_audio_file('Backing_Vocals.')

            if main_vocal == "Không tìm thấy": 
                gr.Warning("Không tìm thấy giọng chính!")
                return [None, None, None, None, None]
            
            if not not_merge_backing and backing_vocal == "Không tìm thấy": 
                gr.Warning("Không tìm thấy giọng bè!")
                return [None, None, None, None, None]
            
            input_path = main_vocal
            backing_path = backing_vocal

        gr.Info("Đang chuyển đổi giọng nói...")

        convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input_path, output_path, model_path, index, autotune, clean, clean_strength, format, embedder_model, upscale, resample_sr, use_threads, max_threads, split_audio, f0_autotune_strength)

        gr.Info("Đã Hoàn thành chuyển đổi giọng nói!")

        if convert_backing:
            if os.path.exists(output_backing): os.remove(output_backing)

            gr.Info("Đang chuyển đổi giọng bè...")

            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, backing_path, output_backing, model_path, index, autotune, clean, clean_strength, format, embedder_model, upscale, resample_sr, use_threads, max_threads, split_audio, f0_autotune_strength)

            gr.Info("Đã Hoàn thành chuyển đổi giọng bè!")

        if not not_merge_backing and not use_original:
            backing_source = output_backing if convert_backing else backing_vocal

            if os.path.exists(output_merge_backup): os.remove(output_merge_backup)

            gr.Info("Kết hợp giọng với giọng bè...")

            AudioSegment.from_file(output_path).overlay(AudioSegment.from_file(backing_source)).export(output_merge_backup, format=format)

            gr.Info("Kết hợp Hoàn thành")

        if merge_instrument:    
            vocals = output_path if not_merge_backing and use_original else output_merge_backup   

            if os.path.exists(output_merge_instrument): os.remove(output_merge_instrument)

            gr.Info("Kết hợp giọng với nhạc nền...")

            instruments = get_audio_file('Instruments.')
            
            if instruments == "Không tìm thấy": 
                gr.Warning("Không tìm thấy nhạc nền")
                output_merge_instrument = None
            else: AudioSegment.from_file(instruments).overlay(AudioSegment.from_file(vocals)).export(output_merge_instrument, format=format)
            
            gr.Info("Kết hợp Hoàn thành")

        return [(None if use_original else output_path), output_backing, (None if not_merge_backing and use_original else output_merge_backup), (output_path if use_original else None), (output_merge_instrument if merge_instrument else None)]
    else:
        if not input or not os.path.exists(input): 
            gr.Warning("Vui lòng nhập đầu vào hợp lệ!")
            return [None, None, None, None, None]
        
        if not output:
            gr.Warning("Vui lòng nhập đầu ra!")
            return [None, None, None, None, None]
        
        output_dir = os.path.dirname(output)
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(output): os.remove(output)

        gr.Info("Đang chuyển đổi giọng nói...")

        convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, upscale, resample_sr, use_threads, max_threads, split_audio, f0_autotune_strength)

        gr.Info("Kết hợp Hoàn thành")

        return [output, None, None, None, None]


def convert_tts(clean, upscale, autotune, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, use_threads, max_threads, split_audio, f0_autotune_strength):
    model_path = os.path.join("assets", "weights", model)

    if not model_path or not os.path.exists(model_path):
        gr.Warning("Không tìm thấy mô hình")
        return None
    
    if not model_path or not os.path.exists(index):
        gr.Warning("Không tìm thấy chỉ mục")
        return None

    if not input or not os.path.exists(input): 
        gr.Warning("Vui lòng nhập đầu vào hợp lệ!")
        return None
        
    if not output:
        gr.Warning("Vui lòng nhập đầu ra!")
        return None
    
    output_dir = os.path.dirname(output)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output): os.remove(output)

    f0method = method if method != "hybrid" else hybrid_method
    
    embedder_model = embedders if embedders != "custom" else custom_embedders

    gr.Info("Đang chuyển đổi giọng nói...")

    convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, upscale, resample_sr, use_threads, max_threads, split_audio, f0_autotune_strength)

    gr.Info("Đã Hoàn thành chuyển đổi giọng nói")
    return output


def create_dataset(input_audio, output_dataset, resample, resample_sr, clean_dataset, clean_strength, separator_music, separator_reverb, kim_vocals_version, overlap, segments_size, denoise_mdx, skip, skip_start, skip_end, hop_length, batch_size):
    version = 1 if kim_vocals_version == "Version-1" else 2

    cmd = f'{python} main/inference/create_dataset.py --input_audio "{input_audio}" --output_dataset {output_dataset} --resample {resample} --resample_sr {resample_sr} --clean_dataset {clean_dataset} --clean_strength {clean_strength} --separator_music {separator_music} --separator_reverb {separator_reverb} --kim_vocal_version {version} --overlap {overlap} --segments_size {segments_size} --mdx_hop_length {hop_length} --mdx_batch_size {batch_size} --denoise_mdx {denoise_mdx} --skip {skip} --skip_start_audios "{skip_start}" --skip_end_audios "{skip_end}"'

    gr.Info("Bắt đầu tạo...")

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

    if not model_name: return gr.Warning("Vui lòng cung cấp tên")
    if len([f for f in os.listdir(os.path.join(dataset)) if os.path.isfile(os.path.join(dataset, f)) and f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]) < 1: return gr.Warning("Không tìm thấy dữ liệu")

    cmd = f'{python} main/inference/preprocess.py --model_name {model_name} --dataset_path {dataset} --sample_rate {sr} --cpu_cores {cpu_core} --cut_preprocess {cut_preprocess} --process_effects {process_effects} --clean_dataset {clean_dataset} --clean_strength {clean_strength}'

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

    if not model_name: return gr.Warning("Vui lòng cung cấp tên")

    if len([f for f in os.listdir(os.path.join(model_dir, "sliced_audios")) if os.path.isfile(os.path.join(model_dir, "sliced_audios", f))]) < 1 or len([f for f in os.listdir(os.path.join(model_dir, "sliced_audios_16k")) if os.path.isfile(os.path.join(model_dir, "sliced_audios_16k", f))]) < 1: return gr.Warning("Không tìm thấy dữ liệu được xử lý, vui lòng xử lý lại âm thanh")

    cmd = f'{python} main/inference/extract.py --model_name {model_name} --rvc_version {version} --f0_method {method} --pitch_guidance {pitch_guidance} --hop_length {hop_length} --cpu_cores {cpu_cores} --gpu {gpu} --sample_rate {sr} --embedder_model {embedder_model}'

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
    if not model_name: return gr.Warning("Vui lòng cung cấp tên")
    model_dir = os.path.join("assets", "logs", model_name)

    if len([f for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted")) if os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f))]) < 1: return gr.Warning("Không tìm thấy dữ liệu được trích xuất, vui lòng trích xuất lại âm thanh")

    cmd = f'{python} main/inference/create_index.py --model_name {model_name} --rvc_version {rvc_version} --index_algorithm {index_algorithm}'

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


def training(model_name, rvc_version, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sample_rate, batch_size, gpu, pitch_guidance, not_pretrain, custom_pretrained, pretrain_g, pretrain_d, detector, threshold, sync_graph, cache):
    sr = int(sample_rate.rstrip("k")) * 1000
    model_dir = os.path.join("assets", "logs", model_name)

    if not model_name: return gr.Warning("Vui lòng cung cấp tên")
    if len([f for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted")) if os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f))]) < 1: return gr.Warning("Không tìm thấy dữ liệu được trích xuất, vui lòng trích xuất lại âm thanh")

    cmd = f'{python} main/inference/train.py --model_name {model_name} --rvc_version {rvc_version} --save_every_epoch {save_every_epoch} --save_only_latest {save_only_latest} --save_every_weights {save_every_weights} --total_epoch {total_epoch} --sample_rate {sr} --batch_size {batch_size} --gpu {gpu} --pitch_guidance {pitch_guidance} --overtraining_detector {detector} --overtraining_threshold {threshold} --sync_graph {sync_graph} --cache_data_in_gpu {cache}'

    if not not_pretrain:
        if not custom_pretrained: pg, pd = pretrained_selector(pitch_guidance)[sr]
        else:
            if not pretrain_g: return gr.Warning("Vui lòng nhập huấn luyện G")
            if not pretrain_d: return gr.Warning("Vui lòng nhập huấn luyện D")
            
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
                gr.Info(f"Tải xuống huấn luyện trước G{rvc_version} gốc")
                run(["wget", "-q", "--show-progress", "-q", "--show-progress", "--no-check-certificate", f"{download_version}{pg}", "-P", os.path.join("assets", "model", f"pretrained_{rvc_version}")], check=True)
                
            if not os.path.exists(pretrained_D):
                gr.Info(f"Tải xuống huấn luyện trước D{rvc_version} gốc")
                run(["wget", "-q", "--show-progress", "-q", "--show-progress", "--no-check-certificate", f"{download_version}{pd}", "-P", os.path.join("assets", "model", f"pretrained_{rvc_version}")], check=True)
        else:
            if not os.path.exists(pretrained_G): return gr.Warning("Không tìm thấy huấn luyện trước G")
            if not os.path.exists(pretrained_D): return gr.Warning("Không tìm thấy huấn luyện trước D")

        cmd += f" --g_pretrained_path {pretrained_G} --d_pretrained_path {pretrained_D}"
    else: gr.Warning("Sẽ không có huấn luyện trước được sử dụng")

    gr.Info("Bắt đầu huấn luyện...")

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





with gr.Blocks(title = "📱 RVC GUI BY ANH", theme = 'NoCrypt/miku') as app:
    gr.HTML("<h1> 🎵 Giao diện chuyển đổi và huấn luyện mô hình giọng nói được tạo bởi Anh 🎵 <h1>")
    with gr.Row(): 
        gr.Markdown(f"Bấm vào đây nếu bạn muốn bị Rick Roll:) ---> [RickRoll]({codecs.decode('uggcf://jjj.lbhghor.pbz/jngpu?i=qDj4j9JtKpD', 'rot13')})")
    with gr.Row(): 
        gr.Markdown("**Vui lòng không sử dụng Dự án với bất kỳ mục đích nào vi phạm đạo đức, pháp luật, hoặc gây tổn hại đến cá nhân, tổ chức...**")
    with gr.Row():
        gr.Markdown("**Trong trường hợp người sử dụng không tuân thủ các điều khoản hoặc vi phạm, tôi sẽ không chịu trách nhiệm về bất kỳ khiếu nại, thiệt hại, hay trách nhiệm pháp lý nào, dù là trong hợp đồng, do sơ suất, hay các lý do khác, phát sinh từ, ngoài, hoặc liên quan đến phần mềm, việc sử dụng phần mềm hoặc các giao dịch khác liên quan đến phần mềm.**")
    with gr.Tabs():
        paths_for_files = lambda path: [os.path.abspath(os.path.join(path, f)) for f in os.listdir(path) if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a')]

        with gr.TabItem("Tách Nhạc"):
            gr.Markdown("## Tách Nhạc")
            with gr.Row(): 
                gr.Markdown("Một hệ thống tách nhạc đơn giản có thể tách được 4 phần: Nhạc, giọng, giọng chính, giọng bè")
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():       
                            cleaner = gr.Checkbox(label="Làm sạch âm thanh", value=False, interactive=True)       
                            backing = gr.Checkbox(label="Tách giọng bè", value=False, interactive=True)
                            denoise = gr.Checkbox(label="Khữ giọng bè", value=False, interactive=False)
                            separator_denoise = gr.Checkbox(label="Khữ tách MDX", value=False, interactive=False)       
                            mdx_model = gr.Checkbox(label="Sử dụng MDX", value=False, interactive=True)
                            reverb = gr.Checkbox(label="Tách vang giọng", value=False, interactive=True)
                            backing_reverb = gr.Checkbox(label="Tách vang bè", value=False, interactive=False)
                            reverb_denoise = gr.Checkbox(label="Khữ tách vang", value=False, interactive=False)                   
                        with gr.Row():
                            separator_model = gr.Dropdown(label="Mô hình tách nhạc", value="HT-Normal", choices=["HT-Normal", "HT-Tuned", "HD_MMI", "HT_6S"], interactive=True, visible=True)
                            separator_backing_model = gr.Dropdown(label="Mô hình tách bè", value="Version-1", choices=["Version-1", "Version-2"], interactive=True, visible=False)
                with gr.Column():
                    separator_button = gr.Button("Tách Nhạc", variant="primary", scale=2)
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            shifts = gr.Slider(label="Số lượng dự đoán", info="Càng cao chất lượng càng tốt nhưng lâu nhưng tốn tài nguyên", minimum=1, maximum=20, value=2, step=1, interactive=True)
                            segment_size = gr.Slider(label="Kích Thước Phân Đoạn", info="Càng cao chất lượng càng tốt nhưng tốn tài nguyên", minimum=32, maximum=4000, value=256, step=8, interactive=True)
                        with gr.Row():
                            mdx_batch_size = gr.Slider(label="Kích thước lô", info="Số lượng mẫu được xử lý cùng một lúc. Việc chia thành các lô giúp tối ưu hóa quá trình tính toán. Lô quá lớn có thể làm tràn bộ nhớ, khi lô quá nhỏ sẽ làm giảm hiệu quả dùng tài nguyên", minimum=1, maximum=64, value=1, step=1, interactive=True, visible=False)
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            overlap = gr.Radio(label="Chồng chéo", info="Số lượng chồng chéo giữa các cửa sổ dự đoán", choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True)
                            format = gr.Radio(label="Định dạng âm thanh", info="Định dạng âm thanh khi xuất tệp âm thanh ra", choices=["wav", "mp3", "flac"], value="wav", interactive=True)
                        with gr.Row():
                            mdx_hop_length = gr.Slider(label="Hop length", info="Biểu thị khoảng thời gian di chuyển cửa sổ phân tích trên tín hiệu âm thanh khi thực hiện các phép biến đổi. Giá trị nhỏ hơn tăng độ chi tiết nhưng tốn tài nguyên tính toán hơn", minimum=1, maximum=8192, value=1024, step=1, interactive=True, visible=False)
            with gr.Row():
                with gr.Column():
                    input = gr.File(label="Thả âm thanh vào đây", file_types=['audio'])    
                    with gr.Accordion("Sử dụng link youtube", open=False):
                        url = gr.Textbox(label="Đường dẫn liên kết đến âm thanh", value="", placeholder="https://www.youtube.com/...", scale=6)
                        download_button = gr.Button("Tải Xuống")
                with gr.Column():
                    clean_strength = gr.Slider(label="Sức mạnh làm sạch âm thanh", info="Sức mạnh của bộ làm sạch âm thanh để lọc giọng hát khi xuất", minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
                    with gr.Accordion("Đầu vào, đầu ra âm thanh"):
                        input_audio = gr.Dropdown(label="Đường dẫn âm thanh", value="" if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios")[0], choices=[] if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios"), allow_custom_value=True, interactive=True)
                        refesh_separator = gr.Button("Tải lại")
                        output_separator = gr.Textbox(label="Đường dẫn thư mục đầu ra âm thanh", value="audios", placeholder="audios", info="Nhập đường dẫn thư mục âm thanh sẽ xuất ra ở đó", interactive=True)
                    audio_input = gr.Audio(show_download_button=True, interactive=False, label="Đầu vào âm thanh")
            with gr.Row():
                gr.Markdown("Âm thanh đã được tách")
            with gr.Row():
                instruments_audio = gr.Audio(show_download_button=True, interactive=False, label="Nhạc nền")
                original_vocals = gr.Audio(show_download_button=True, interactive=False, label="Giọng gốc")
                main_vocals = gr.Audio(show_download_button=True, interactive=False, label="Giọng chính", visible=False)
                backing_vocals = gr.Audio(show_download_button=True, interactive=False, label="Giọng bè", visible=False)
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
                input_audio.change(fn=lambda audio: audio, inputs=[input_audio], outputs=[audio_input])
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

        with gr.TabItem("Chuyển Đổi"):
            gr.Markdown("## Chuyển Đổi Âm Thanh")
            with gr.Row():
                gr.Markdown("Chuyển đổi âm thanh bằng mô hình giọng nói đã được huấn luyện")
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            cleaner0 = gr.Checkbox(label="Làm sạch âm thanh", value=False, interactive=True)
                            upscale = gr.Checkbox(label="Tăng chất lượng", value=False, interactive=True)
                            autotune = gr.Checkbox(label="Tự động điều chỉnh", value=False, interactive=True)
                            use_audio = gr.Checkbox(label="Sử dụng âm thanh vừa tách", value=False, interactive=True)
                            use_original = gr.Checkbox(label="Chuyển đổi giọng gốc", value=False, interactive=True, visible=False) 
                            convert_backing = gr.Checkbox(label="Chuyển đổi giọng bè", value=False, interactive=True, visible=False)   
                            not_merge_backing = gr.Checkbox(label="Không kết hợp giọng bè", value=False, interactive=True, visible=False)
                            merge_instrument = gr.Checkbox(label="Kết hợp nhạc nền", value=False, interactive=True, visible=False)  
                    with gr.Row():
                        pitch = gr.Slider(minimum=-20, maximum=20, step=1, info="Khuyến cáo: chỉnh lên 12 để chuyển giọng nam thành nữ và ngược lại", label="Cao độ", value=0, interactive=True)
                        clean_strength0 = gr.Slider(label="Sức mạnh làm sạch âm thanh", info="Sức mạnh của bộ làm sạch âm thanh để lọc giọng hát khi xuất", minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
                with gr.Column():
                    convert_button = gr.Button("Chuyển Đổi", variant="primary", scale=4)
            with gr.Row():
                with gr.Column():
                    input0 = gr.File(label="Thả âm thanh vào đây", file_types=['audio'])  
                    play_audio = gr.Audio(show_download_button=True, interactive=False, label="Đầu vào âm thanh")
                with gr.Column():
                    with gr.Accordion("Mô hình và chỉ mục", open=True):
                        with gr.Row():
                            model_pth = gr.Dropdown(label="Tệp mô hình", choices=sorted(model_name), value=sorted(model_name)[0] if len(sorted(model_name)) > 0 else '', interactive=True, allow_custom_value=True)
                            model_index = gr.Dropdown(label="Tệp chỉ mục", choices=sorted(index_path), value=sorted(index_path)[0] if len(sorted(index_path)) > 0 else '', interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refesh = gr.Button("Tải lại")
                        with gr.Row():
                            index_strength = gr.Slider(label="Ảnh hưởng của chỉ mục", info="Càng cao ảnh hưởng càng lớn. Tuy nhiên, việc chọn giá trị thấp hơn có thể giảm hiện tượng giả trong âm thanh", minimum=0, maximum=1, value=0.5, step=0.01, interactive=True)
                    with gr.Accordion("Đầu vào, đầu ra âm thanh", open=False):
                        with gr.Column():
                            export_format = gr.Radio(label="Định dạng", info="Định dạng khi xuất tệp âm thanh ra", choices=["wav", "mp3", "flac", "ogg", "m4a"], value="wav", interactive=True)
                            input_audio0 = gr.Dropdown(label="Đường dẫn đầu vào âm thanh", value="" if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios")[0], choices=[] if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios"), info="Nhập đường dẫn đến tệp âm thanh", allow_custom_value=True, interactive=True)
                            output_audio = gr.Textbox(label="Đường dẫn đầu ra âm thanh", value="audios/output.wav", placeholder="audios/output.wav", info="Nhập đường dẫn đầu ra(cứ để định dạng .wav khi chuyển đổi nó tự sửa)", interactive=True)
                        with gr.Column():
                            refesh0 = gr.Button("Tải lại")
                    with gr.Accordion("Cài đặt chung", open=False):
                        with gr.Accordion("Phương pháp trích xuất", open=False):
                            method = gr.Radio(label="Phương pháp trích xuất", info="Phương pháp để trích xuất dữ liệu âm thanh để cho mô hình nói", choices=["pm", "dio", "crepe-tiny", "crepe", "fcpe", "rmvpe", "harvest", "hybrid"], value="rmvpe", interactive=True)
                            hybrid_method = gr.Radio(label="Phương pháp trích xuất HYBRID", info="Sự kết hợp của hai loại trích xuất khác nhau", choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[rmvpe+harvest]"], value="hybrid[pm+dio]", interactive=True, visible=False)
                            hop_length = gr.Slider(label="Hop length", info="Khoảng cách giữa các khung âm thanh khi xử lý tín hiệu. Giá trị lớn xử lý nhẹ hơn, chi tiết giảm, giá trị nhỏ chi tiết cao, xử lý nặng", minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion("Mô hình học cách nói", open=False):
                            embedders = gr.Radio(label="Mô hình học cách nói", info="Mô hình được huấn luyện trước để giúp cho mô hình học cách nói cách ngắt hơi", choices=["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "custom"], value="contentvec_base", interactive=True)
                            custom_embedders = gr.Textbox(label="Tên của mô hình", info="Nếu bạn có msô hình riêng chỉ cần tải và nhập tên của mô hình vào đây", value="", placeholder="hubert_base", interactive=True, visible=False)
                        with gr.Column():
                            with gr.Row():
                                split_audio = gr.Checkbox(label="Cắt âm thanh", info="Cắt âm thanh ra các phần nhỏ để chuyển đổi có thể giúp tăng tốc độ", value=False, interactive=True)
                                use_threads = gr.Checkbox(label="Xử lý đa luồng", info="Xử lý đa luồng có thể giảm thời gian huấn luyện nhưng có thể bị quá tải", value=False, interactive=True, visible=False)
                            max_threads = gr.Slider(minimum=1, maximum=10, label="Số lượng Xử lý tối đa", info="Số lượng Xử lý tối đa cùng lúc", value=1, step=1, interactive=True, visible=False)
                            f0_autotune_strength = gr.Slider(minimum=0, maximum=1, label="Mức độ điều chỉnh", info="Mức độ điều chỉnh của điều chỉnh tự động", value=1, step=0.1, interactive=True, visible=False)
                            resample_sr = gr.Slider(minimum=0, maximum=48000, label="Lấy mẫu lại", info="Lấy mẫu lại sau xử lý đến tốc độ lấy mẫu cuối cùng, 0 có nghĩa là không lấy mẫu lại", value=0, step=1, interactive=True)
                            filter_radius = gr.Slider(minimum=0, maximum=7, label="Lọc trung vị", info="Nếu giá trị lớn hơn ba sẽ áp dụng tính năng lọc trung vị. Giá trị đại diện cho bán kính bộ lọc và có thể làm giảm hơi thở hoặc tắt thở.", value=3, step=1, interactive=True)
                            volume_envelope = gr.Slider(minimum=0, maximum=1, label="Đường bao âm thanh", info="Sử dụng đường bao âm lượng của đầu vào để thay thế hoặc trộn với đường bao âm lượng của đầu ra. Càng gần 1 thì đường bao đầu ra càng được sử dụng nhiều", value=1, step=0.1, interactive=True)
                            protect = gr.Slider(minimum=0, maximum=1, label="Bảo vệ phụ âm", info="Bảo vệ các phụ âm riêng biệt và âm thanh thở ngăn chặn việc rách điện âm và các hiện tượng giả khác. Việc chỉnh tối đa sẽ bảo vệ toàn diện. Việc giảm giá trị này có thể giảm độ bảo vệ, đồng thời có khả năng giảm thiểu hiệu ứng lập chỉ mục", value=0.33, step=0.01, interactive=True)
            with gr.Row():
                gr.Markdown("Âm thanh đã được chuyển đổi")
            with gr.Row():
                main_convert = gr.Audio(show_download_button=True, interactive=False, label="Chuyển đổi giọng chính")
                backing_convert = gr.Audio(show_download_button=True, interactive=False, label="Chuyển đổi giọng bè", visible=False)
                main_backing = gr.Audio(show_download_button=True, interactive=False, label="Giọng chính + Giọng bè", visible=False)  
            with gr.Row():
                original_convert = gr.Audio(show_download_button=True, interactive=False, label="Chuyển đổi giọng gốc", visible=False)
                vocal_instrument = gr.Audio(show_download_button=True, interactive=False, label="Giọng + Nhạc nền", visible=False)  
            with gr.Row():
                split_audio.change(fn=valueFalse_visible1, inputs=[split_audio], outputs=[use_threads])
                use_threads.change(fn=visible_1, inputs=[use_threads], outputs=[max_threads])
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
                input_audio0.change(fn=lambda audio: audio, inputs=[input_audio0], outputs=[play_audio])
            with gr.Row():
                embedders.change(fn=lambda embedders: visible_1(True if embedders == "custom" else False), inputs=[embedders], outputs=[custom_embedders])
                refesh0.click(fn=lambda: refesh_audio, inputs=[], outputs=[input_audio0])
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
                        use_threads,
                        max_threads,
                        split_audio,
                        f0_autotune_strength
                    ],
                    outputs=[main_convert, backing_convert, main_backing, original_convert, vocal_instrument],
                    api_name="convert_audio"
                )

        with gr.TabItem("Chuyển Đổi Văn Bản"):
            gr.Markdown("## Chuyển Đổi Văn Bản Thành Giọng Nói")
            with gr.Row():
                gr.Markdown("Chuyển văn bản thành giọng nói và đọc lại bằng mô hình giọng nói được huấn luyện")
            with gr.Row():
                with gr.Column():
                    use_txt = gr.Checkbox(label="Nhập dữ liệu từ tệp văn bản txt", value=False, interactive=True)
                    prompt = gr.Textbox(label="Văn bản cần đọc", value="", placeholder="Xin chào thế giới!", lines=2)
                    with gr.Row():
                        speed = gr.Slider(label="Tốc độ đọc", info="Tốc độ đọc của giọng nói", minimum=-100, maximum=100, value=0, step=1)
                        pitch0 = gr.Slider(minimum=-20, maximum=20, step=1, info="Khuyến cáo: chỉnh lên 12 để chuyển giọng nam thành nữ và ngược lại", label="Cao độ", value=0, interactive=True)
                with gr.Column():
                    tts_button = gr.Button("1. Chuyển Đổi Văn Bản", variant="primary", scale=2)
                    convert_button0 = gr.Button("2. Chuyển Đổi Giọng Nói", variant="secondary", scale=2)
            with gr.Row():
                with gr.Column():
                    tts_voice = gr.Dropdown(
                        label="Giọng nói của mô hình chuyển đổi",  
                        choices=tts_voice, 
                        interactive=True, 
                        value="vi-VN-NamMinhNeural"
                    )
                    txt_input = gr.File(label="Thả tệp văn bản vào đây", file_types=['txt'], visible=False)  
                with gr.Column():
                    with gr.Accordion("Mô hình và chỉ mục", open=True):
                        with gr.Row():
                            model_pth0 = gr.Dropdown(label="Tệp mô hình", choices=sorted(model_name), value=sorted(model_name)[0] if len(sorted(model_name)) > 0 else '', interactive=True, allow_custom_value=True)
                            model_index0 = gr.Dropdown(label="Tệp chỉ mục", choices=sorted(index_path), value=sorted(index_path)[0] if len(sorted(index_path)) > 0 else '', interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refesh1 = gr.Button("Tải lại")
                        with gr.Row():
                            index_strength0 = gr.Slider(label="Ảnh hưởng của chỉ mục", info="Càng cao ảnh hưởng càng lớn. Tuy nhiên, việc chọn giá trị thấp hơn có thể giảm hiện tượng giả trong âm thanh", minimum=0, maximum=1, value=0.5, step=0.01, interactive=True)
                    with gr.Accordion("đầu ra âm thanh", open=False):
                        export_format0 = gr.Radio(label="Định dạng", info="Định dạng khi xuất tệp âm thanh ra", choices=["wav", "mp3", "flac", "ogg", "m4a"], value="wav", interactive=True)
                        output_audio0 = gr.Textbox(label="Đường dẫn đầu ra giọng nói", value="audios/tts.wav", placeholder="audios/tts.wav", info="Nhập đường dẫn đầu ra", interactive=True)
                        output_audio1 = gr.Textbox(label="Đường dẫn đầu ra giọng chuyển đổi", value="audios/tts-convert.wav", placeholder="audios/tts-convert.wav", info="Nhập đường dẫn đầu ra(cứ để định dạng .wav khi chuyển đổi nó tự sửa)", interactive=True)
                    with gr.Accordion("Cài đặt chung", open=False):
                        with gr.Accordion("Phương pháp trích xuất", open=False):
                            method0 = gr.Radio(label="Phương pháp trích xuất", info="Phương pháp để trích xuất dữ liệu âm thanh để cho mô hình nói", choices=["pm", "dio", "crepe-tiny", "crepe", "fcpe", "rmvpe", "harvest", "hybrid"], value="rmvpe", interactive=True)
                            hybrid_method0 = gr.Radio(label="Phương pháp trích xuất HYBRID", info="Sự kết hợp của hai loại trích xuất khác nhau", choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[rmvpe+harvest]"], value="hybrid[pm+dio]", interactive=True, visible=False)
                            hop_length0 = gr.Slider(label="Hop length", info="Khoảng cách giữa các khung âm thanh khi xử lý tín hiệu. Giá trị lớn xử lý nhẹ hơn, chi tiết giảm, giá trị nhỏ chi tiết cao, xử lý nặng", minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion("Mô hình học cách nói", open=False):
                            embedders0 = gr.Radio(label="Mô hình học cách nói", info="Mô hình được huấn luyện trước để giúp cho mô hình học cách nói cách ngắt hơi", choices=["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "custom"], value="contentvec_base", interactive=True)
                            custom_embedders0 = gr.Textbox(label="Tên của mô hình", info="Nếu bạn có mô hình riêng chỉ cần tải và nhập tên của mô hình vào đây", value="", placeholder="hubert_base", interactive=True, visible=False)
                        with gr.Row():
                            split_audio0 = gr.Checkbox(label="Cắt âm thanh", info="Cắt âm thanh ra các phần nhỏ để chuyển đổi có thể giúp tăng tốc độ", value=False, interactive=True)
                            use_threads0 = gr.Checkbox(label="Xử lý đa luồng", info="Xử lý đa luồng có thể giảm thời gian huấn luyện", value=False, interactive=True, visible=False)
                        with gr.Row():
                            cleaner1 = gr.Checkbox(label="Làm sạch âm thanh", value=False, interactive=True)
                            upscale2 = gr.Checkbox(label="Tăng chất lượng", value=False, interactive=True)
                            autotune3 = gr.Checkbox(label="Tự động điều chỉnh", value=False, interactive=True)
                        with gr.Column():
                            max_threads0 = gr.Slider(minimum=1, maximum=10, label="Số lượng Xử lý tối đa", info="Số lượng Xử lý tối đa cùng lúc", value=1, step=1, interactive=True, visible=False)
                            f0_autotune_strength0 = gr.Slider(minimum=0, maximum=1, label="Mức độ điều chỉnh", info="Mức độ điều chỉnh của điều chỉnh tự động", value=1, step=0.1, interactive=True, visible=False)
                            clean_strength1 = gr.Slider(label="Sức mạnh làm sạch âm thanh", info="Sức mạnh của bộ làm sạch âm thanh để lọc giọng hát khi xuất", minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
                            resample_sr0 = gr.Slider(minimum=0, maximum=48000, label="Lấy mẫu lại", info="Lấy mẫu lại sau xử lý đến tốc độ lấy mẫu cuối cùng, 0 có nghĩa là không lấy mẫu lại", value=0, step=1, interactive=True)
                            filter_radius0 = gr.Slider(minimum=0, maximum=7, label="Lọc trung vị", info="Nếu giá trị lớn hơn ba sẽ áp dụng tính năng lọc trung vị. Giá trị đại diện cho bán kính bộ lọc và có thể làm giảm hơi thở hoặc tắt thở.", value=3, step=1, interactive=True)
                            volume_envelope0 = gr.Slider(minimum=0, maximum=1, label="Đường bao âm thanh", info="Sử dụng đường bao âm lượng của đầu vào để thay thế hoặc trộn với đường bao âm lượng của đầu ra. Càng gần 1 thì đường bao đầu ra càng được sử dụng nhiều", value=1, step=0.1, interactive=True)
                            protect0 = gr.Slider(minimum=0, maximum=1, label="Bảo vệ phụ âm", info="Bảo vệ các phụ âm riêng biệt và âm thanh thở ngăn chặn việc rách điện âm và các hiện tượng giả khác. Việc chỉnh tối đa sẽ bảo vệ toàn diện. Việc giảm giá trị này có thể giảm độ bảo vệ, đồng thời có khả năng giảm thiểu hiệu ứng lập chỉ mục", value=0.33, step=0.01, interactive=True)
            with gr.Row():
                gr.Markdown("Âm thanh chưa được chuyển đổi và âm thanh đã được chuyển đổi")
            with gr.Row():
                tts_voice_audio = gr.Audio(show_download_button=True, interactive=False, label="Giọng được tạo bởi chuyển đổi văn bản thành giọng nói")
                tts_voice_convert = gr.Audio(show_download_button=True, interactive=False, label="Giọng được chuyển đổi bởi mô hình")
            with gr.Row():
                use_threads0.change(fn=visible_1, inputs=[use_threads0], outputs=[max_threads0])
                split_audio0.change(fn=valueFalse_visible1, inputs=[split_audio0], outputs=[use_threads0])
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
                        use_threads0,
                        max_threads0,
                        split_audio0,
                        f0_autotune_strength0
                    ],
                    outputs=[tts_voice_convert],
                    api_name="convert_tts"
                )

        with gr.TabItem("Hiệu Ứng Âm Thanh"):
            gr.Markdown("## Áp Dụng Thêm Hiệu Ứng Cho Âm Thanh")
            with gr.Row():
                gr.Markdown("Chỉnh sửa thêm hiệu ứng cho âm thanh")
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            reverb_check_box = gr.Checkbox(label="Hiệu ứng vọng âm", value=False, interactive=True)
                            chorus_check_box = gr.Checkbox(label="Hiệu ứng hòa âm", value=False, interactive=True)
                            delay_check_box = gr.Checkbox(label="Hiệu ứng độ trễ", value=False, interactive=True)
                        with gr.Row():
                            more_options = gr.Checkbox(label="Tùy chọn thêm", value=False, interactive=True)    
                            phaser_check_box = gr.Checkbox(label="Hiệu ứng xoay pha", value=False, interactive=True)
                            compressor_check_box = gr.Checkbox(label="Hiệu ứng nén", value=False, interactive=True)
                with gr.Column():
                    apply_effects_button = gr.Button("Áp dụng", variant="primary", scale=2)
            with gr.Row():
                with gr.Row():
                    with gr.Accordion("Đầu vào, đầu ra của âm thanh", open=False):
                        with gr.Row():
                            upload_audio = gr.File(label="Thả tệp âm thanh vào đây", file_types=['audio'])
                        with gr.Row():
                            audio_in_path = gr.Dropdown(label="Đầu vào âm thanh", value="" if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios")[0], choices=[] if len(list(f for f in os.listdir("audios") if os.path.splitext(f)[1] in ('.mp3', '.wav', '.flac', '.ogg', '.m4a'))) < 1 else paths_for_files("audios"), info="Nhập đường dẫn đầu vào âm thanh", interactive=True, allow_custom_value=True)
                            audio_out_path = gr.Textbox(label="Đầu ra âm thanh", value="audios/audio_effects.wav", placeholder="audios/audio_effects.wav", info="Nhập đường dẫn đầu ra(cứ để .wav khi áp dụng sẽ tự sửa)", interactive=True)
                        with gr.Row():
                            audio_output_format = gr.Radio(label="Định dạng âm thanh", info="Định dạng âm thanh khi xuất tệp âm thanh ra", choices=["wav", "mp3", "flac"], value="wav", interactive=True)
                            audio_effects_refesh = gr.Button("Tải lại")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Accordion("Hiệu ứng vọng âm", open=False, visible=False) as reverb_accordion:
                            reverb_freeze_mode = gr.Checkbox(label="Chế độ đóng băng", info="Tạo hiệu ứng vang liên tục khi bật chế độ này", value=False, interactive=True)
                            reverb_room_size = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Kích thước phòng", info="Điều chỉnh không gian của phòng để tạo độ vang", interactive=True)
                            reverb_damping = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Giảm âm", info="Điều chỉnh độ hút âm, kiểm soát mức độ vang", interactive=True)
                            reverb_wet_level = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.3, label="Mức độ tín hiệu vang", info="Điều chỉnh mức độ của tín hiệu có hiệu ứng vọng âm", interactive=True)
                            reverb_dry_level = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="Mức độ tín hiệu gốc", info="Điều chỉnh mức độ của tín hiệu không có hiệu ứng", interactive=True)
                            reverb_width = gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label="Chiều rộng âm thanh", info="Điều chỉnh độ rộng của không gian âm thanh", interactive=True)
                    with gr.Row():
                        with gr.Accordion("Hiệu ứng hòa âm", open=False, visible=False) as chorus_accordion:
                            chorus_depth = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Độ sâu", info="Điều chỉnh cường độ hòa âm, tạo ra cảm giác rộng cho âm thanh", interactive=True)
                            chorus_rate_hz = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1.5, label="Tần số", info="Điều chỉnh tốc độ dao động của hòa âm", interactive=True)
                            chorus_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Trộn tín hiệu", info="Điều chỉnh mức độ trộn giữa âm gốc và âm có hiệu ứng", interactive=True)
                            chorus_centre_delay_ms = gr.Slider(minimum=0, maximum=50, step=1, value=10, label="Đỗ trễ trung tâm (mili giây)", info="Khoảng thời gian trễ giữa các kênh stereo để tạo hiệu ứng hòa âm", interactive=True)
                            chorus_feedback = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label="Phản hồi", info="Điều chỉnh lượng tín hiệu hiệu ứng được quay lại vào tín hiệu gốc", interactive=True)
                    with gr.Row():
                        with gr.Accordion("Hiệu ứng độ trễ", open=False, visible=False) as delay_accordion:
                            delay_second = gr.Slider(minimum=0, maximum=5, step=0.01, value=0.5, label="Thời gian trễ", info="Điều chỉnh khoảng thời gian trễ giữa âm gốc và âm có hiệu ứng", interactive=True)
                            delay_feedback = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Phản hồi độ trễ", info="Điều chỉnh lượng tín hiệu được quay lại, tạo hiệu ứng lặp lại", interactive=True)
                            delay_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Trộn tín hiệu độ trễ", info="Điều chỉnh mức độ trộn giữa âm gốc và âm trễ", interactive=True)
                with gr.Column():
                    with gr.Row():
                        with gr.Accordion("Tùy chọn thêm", open=False, visible=False) as more_accordion:
                            with gr.Row():
                                fade = gr.Checkbox(label="Hiệu ứng mờ dần", value=False, interactive=True)
                                bass_or_treble = gr.Checkbox(label="Âm trầm và âm cao", value=False, interactive=True)
                                limiter = gr.Checkbox(label="Giới hạn ngưỡng", value=False, interactive=True)
                                resample_checkbox = gr.Checkbox(label="Lấy mẫu lại", value=False, interactive=True)
                            with gr.Row():
                                distortion_checkbox = gr.Checkbox(label="Hiệu ứng nhiễu âm", value=False, interactive=True)
                                gain_checkbox = gr.Checkbox(label="Cường độ âm", value=False, interactive=True)
                                bitcrush_checkbox = gr.Checkbox(label="Hiệu ứng giảm bits", value=False, interactive=True)
                                clipping_checkbox = gr.Checkbox(label="Hiệu ứng méo âm", value=False, interactive=True)
                            with gr.Accordion("Hiệu ứng mờ dần", open=True, visible=False) as fade_accordion:
                                with gr.Row():
                                    fade_in = gr.Slider(minimum=0, maximum=10000, step=100, value=0, label="Hiệu ứng mờ dần vào (mili giây)", info="Thời gian mà âm thanh sẽ tăng dần từ mức 0 đến mức bình thường", interactive=True)
                                    fade_out = gr.Slider(minimum=0, maximum=10000, step=100, value=0, label="Hiệu ứng mờ dần ra (mili giây)", info="thời gian mà âm thanh sẽ giảm dần từ xuống 0", interactive=True)
                            with gr.Accordion("Âm trầm và âm cao", open=True, visible=False) as bass_treble_accordion:
                                with gr.Row():
                                    bass_boost = gr.Slider(minimum=0, maximum=20, step=1, value=0, label="Độ khuếch đại âm trầm (db)", info="mức độ tăng cường âm trầm", interactive=True)
                                    bass_frequency = gr.Slider(minimum=20, maximum=200, step=10, value=100, label="Tần số cắt của bộ lọc thông thấp (Hz)", info="tần số mà âm thanh bắt đầu bị giảm. Tần số thấp hơn sẽ làm âm trầm rõ hơn", interactive=True)
                                with gr.Row():
                                    treble_boost = gr.Slider(minimum=0, maximum=20, step=1, value=0, label="Độ khuếch đại âm cao (db)", info="mức độ tăng cường âm cao", interactive=True)
                                    treble_frequency = gr.Slider(minimum=1000, maximum=10000, step=500, value=3000, label="Tần số cắt của bộ lọc thông cao (Hz)", info="âm thanh dưới tần số này sẽ bị lọc bỏ. Tần số càng cao thì chỉ giữ lại âm càng cao", interactive=True)
                            with gr.Accordion("Giới hạn ngưỡng", open=True, visible=False) as limiter_accordion:
                                with gr.Row():
                                    limiter_threashold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-1, label="Ngưỡng giới hạn", info="Giới hạn mức độ âm thanh tối đa, ngăn không cho vượt quá ngưỡng", interactive=True)
                                    limiter_release_ms = gr.Slider(minimum=10, maximum=1000, step=1, value=100, label="Thời gian thả", info="Khoảng thời gian để âm thanh trở lại sau khi bị giới hạn", interactive=True)
                            with gr.Column():
                                pitch_shift_semitones = gr.Slider(minimum=-20, maximum=20, step=1, value=0, label="Cao độ", info="Điều chỉnh cao độ của âm thanh, mỗi bán cung tương ứng với nữa nốt nhạc", interactive=True)
                                audio_effect_resample_sr = gr.Slider(minimum=0, maximum=48000, step=1, value=0, label="Tốc độ lấy mẫu lại", info="Lấy mẫu lại sau khi áp dụng hiệu ứng đến tốc độ lấy mẫu cuối cùng, 0 có nghĩa là không lấy mẫu lại", interactive=True, visible=False)
                                distortion_drive_db = gr.Slider(minimum=0, maximum=50, step=1, value=20, label="Hiệu ứng nhiễu âm", info="Điều chỉnh mức độ nhiễu âm, tạo hiệu ứng méo tiếng", interactive=True, visible=False)
                                gain_db = gr.Slider(minimum=-60, maximum=60, step=1, value=0, label="Cường độ âm", info="Tăng giảm âm lượng của tín hiệu", interactive=True, visible=False)
                                clipping_threashold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-1, label="Ngưỡng cắt", info="Cắt bớt tín hiệu vượt quá ngưỡng, tạo âm thanh méo", interactive=True, visible=False)
                                bitcrush_bit_depth = gr.Slider(minimum=1, maximum=24, step=1, value=16, label="Độ sâu bit", info="Giảm chất lượng âm thanh bằng cách giảm số bit, tạo hiệu ứng âm thanh bị méo", interactive=True, visible=False)
                    with gr.Row():
                        with gr.Accordion("Hiệu ứng xoay pha", open=False, visible=False) as phaser_accordion:
                            phaser_depth = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Độ sâu", info="Điều chỉnh độ sâu của hiệu ứng, ảnh hưởng đến cường độ của hiệu ứng xoay pha", interactive=True)
                            phaser_rate_hz = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1, label="Tần số", info="Điều chỉnh tốc độ của hiệu ứng hiệu ứng xoay pha", interactive=True)
                            phaser_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Trộn tín hiệu", info="Điều chỉnh mức độ trộn giữa tín hiệu gốc và tín hiệu đã qua xử lý", interactive=True)
                            phaser_centre_frequency_hz = gr.Slider(minimum=50, maximum=5000, step=10, value=1000, label="Tần số trung tâm", info="Tần số trung tâm của hiệu ứng xoay pha, ảnh hưởng đến tần số bị điều chỉnh", interactive=True)
                            phaser_feedback = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label="Phản hồi", info="Điều chỉnh lượng phản hồi tín hiệu, tạo cảm giác xoay pha mạnh hoặc nhẹ", interactive=True)
                    with gr.Row():
                        with gr.Accordion("Hiệu ứng nén", open=False, visible=False) as compressor_accordion:
                            compressor_threashold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-20, label="Ngưỡng nén", info="Ngưỡng mức âm thanh sẽ bị nén khi vượt qua ngưỡng này", interactive=True)
                            compressor_ratio = gr.Slider(minimum=1, maximum=20, step=0.1, value=1, label="Tỉ lệ nén", info="Điều chỉnh mức độ nén âm thanh khi vượt qua ngưỡng", interactive=True)
                            compressor_attack_ms = gr.Slider(minimum=0.1, maximum=100, step=0.1, value=10, label="Thời gian tấn công (mili giây)", info="Khoảng thời gian nén bắt đầu tác dụng sau khi âm thanh vượt ngưỡng", interactive=True)
                            compressor_release_ms = gr.Slider(minimum=10, maximum=1000, step=1, value=100, label="Thời gian thả", info="Thời gian để âm thanh trở lại trạng thái bình thường sau khi bị nén", interactive=True)   
            with gr.Row():
                audio_play_input = gr.Audio(show_download_button=True, interactive=False, label="Đầu vào âm thanh")
                audio_play_output = gr.Audio(show_download_button=True, interactive=False, label="Đầu ra âm thanh")
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
                audio_in_path.change(fn=lambda audio: audio, inputs=[audio_in_path], outputs=[audio_play_input])
                audio_effects_refesh.click(fn=lambda: refesh_audio, inputs=[], outputs=[audio_in_path])
            with gr.Row():
                more_options.change(fn=lambda: [False, False, False, False], inputs=[], outputs=[fade, bass_or_treble, limiter, resample_checkbox])
                more_options.change(fn=lambda: [False, False, False, False], inputs=[], outputs=[distortion_checkbox, gain_checkbox, clipping_checkbox, bitcrush_checkbox])
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

        with gr.TabItem("Tạo dữ liệu"):
            gr.Markdown("## Tạo Dữ Liệu Huấn Luyện Từ Youtube")
            with gr.Row():
                gr.Markdown("Xử lý và tạo tập tin dữ liệu huấn luyện bằng đường dẫn youtube")
            with gr.Row():
                dataset_url = gr.Textbox(label="Đường dẫn liên kết âm thanh", info="Đường dẫn liên kết đến âm thanh(sử dụng dấu , để sử dụng nhiều liên kết)", value="", placeholder="https://www.youtube.com/...", interactive=True)
            with gr.Row():
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            with gr.Row():
                                separator_audio = gr.Checkbox(label="Tách nhạc", value=False, interactive=True)
                                separator_reverb = gr.Checkbox(label="Tách vang", value=False, interactive=False)
                                denoise_mdx = gr.Checkbox(label="Khử nhiễu mô hình", value=False, interactive=False)
                            with gr.Row():
                                clean_audio = gr.Checkbox(label="Làm sạch dữ liệu", value=False, interactive=True)
                                resample = gr.Checkbox(label="Lấy mẫu lại", value=False, interactive=True)
                                skip = gr.Checkbox(label="Bỏ qua giây", value=False, interactive=True)
                        with gr.Row():
                            resample_sample_rate = gr.Slider(minimum=0, maximum=48000, step=1, value=0, label="Tốc độ lấy mẫu lại", info="Tốc độ lấy mẫu lại dữ liệu sau khi tạo xong dữ liệu huấn luyện", interactive=True, visible=False)
                            dataset_clean_strength = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label="Sức mạnh làm sạch", info="Sức mạnh của bộ làm sạch âm thanh để lọc âm thanh sau khi tạo xong dữ liệu huấn luyện", interactive=True, visible=False)
                    with gr.Column():
                        create_button = gr.Button("Tạo dữ liệu", variant="primary", scale=2)
            with gr.Row():
                with gr.Column():
                    with gr.Group(visible=False) as separator_dataset:
                        with gr.Row():
                            kim_vocal_version = gr.Radio(label="Phiên bản tách giọng", info="Phiên bản của mô hình tách nhạc để tách giọng", choices=["Version-1", "Version-2"], value="Version-2", interactive=True, visible=False)
                            kim_vocal_overlap = gr.Radio(label="Chồng chéo", info="Số lượng chồng chéo giữa các cửa sổ dự đoán", choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True, visible=False)
                        with gr.Row():
                            kim_vocal_segments_size = gr.Slider(label="Kích Thước Phân Đoạn", info="Càng cao chất lượng càng tốt nhưng tốn tài nguyên", minimum=32, maximum=4000, value=256, step=8, interactive=True, visible=False)
                            kim_vocal_hop_length = gr.Slider(label="Hop length", info="Biểu thị khoảng thời gian di chuyển cửa sổ phân tích trên tín hiệu âm thanh khi thực hiện các phép biến đổi. Giá trị nhỏ hơn tăng độ chi tiết nhưng tốn tài nguyên tính toán hơn", minimum=1, maximum=8192, value=1024, step=1, interactive=True, visible=False)
                        with gr.Row():
                            kim_vocal_batch_size = gr.Slider(label="Kích thước lô", info="Số lượng mẫu được xử lý cùng một lúc. Việc chia thành các lô giúp tối ưu hóa quá trình tính toán. Lô quá lớn có thể làm tràn bộ nhớ, khi lô quá nhỏ sẽ làm giảm hiệu quả dùng tài nguyên", minimum=1, maximum=64, value=1, step=1, interactive=True, visible=False)
                    with gr.Row():
                        create_dataset_info = gr.Textbox(label="Thông tin tạo dữ liệu", value="", interactive=False)
                with gr.Row():
                    with gr.Column():
                        output_dataset = gr.Textbox(label="Đầu ra dữ liệu", info="Đầu ra dữ liệu sau khi tạo xong dữ liệu", value="dataset", placeholder="dataset", interactive=True)
                        with gr.Row():
                            skip_start = gr.Textbox(label="Bỏ qua phần đầu", info="Bỏ qua số giây đầu của âm thanh, dùng dấu , để sử dụng cho nhiều âm thanh", value="", placeholder="0,...", interactive=True, visible=False)
                            skip_end = gr.Textbox(label="Bỏ qua phần cuối", info="Bỏ qua số giây cuối của âm thanh, dùng dấu , để sử dụng cho nhiều âm thanh", value="", placeholder="0,...", interactive=True, visible=False)
            with gr.Row():
                separator_audio.change(fn=interactive_1, inputs=[separator_audio], outputs=[separator_reverb])
                separator_audio.change(fn=interactive_1, inputs=[separator_audio], outputs=[denoise_mdx])
                separator_audio.change(fn=visible_1, inputs=[separator_audio], outputs=[separator_dataset])
            with gr.Row():
                separator_audio.change(fn=visible_1, inputs=[separator_audio], outputs=[kim_vocal_version])
                separator_audio.change(fn=visible_1, inputs=[separator_audio], outputs=[kim_vocal_overlap])
                separator_audio.change(fn=visible_1, inputs=[separator_audio], outputs=[kim_vocal_segments_size])
            with gr.Row():
                separator_audio.change(fn=visible_1, inputs=[separator_audio], outputs=[kim_vocal_hop_length])
                separator_audio.change(fn=visible_1, inputs=[separator_audio], outputs=[kim_vocal_batch_size])
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

        with gr.TabItem("Huấn Luyện"):
            gr.Markdown("## Huấn Luyện Mô Hình")
            with gr.Row():
                gr.Markdown("Huấn luyện và đào tạo mô hình giọng nói bằng một lượng dữ liệu giọng nói")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            training_name = gr.Textbox(label="Tên của mô hình", info="Tên của mô hình khi huấn luyện(không sử dụng ký tự đặc biệt hay dấu cách)", value="", placeholder="Tên mô hình", interactive=True)
                            training_sr = gr.Radio(label="Tỉ lệ lấy mẫu", info="Tỉ lệ lấy mẫu của mô hình", choices=["32k", "40k", "48k"], value="48k", interactive=True) 
                            training_ver = gr.Radio(label="Phiên bản mô hình", info="Phiên bản mô hình khi huấn luyện", choices=["v1", "v2"], value="v2", interactive=True) 
                            with gr.Row():
                                training_f0 = gr.Checkbox(label="Huấn luyện cao độ", info="Huấn luyện cao độ cho mô hình", value=True, interactive=True)
                                upload = gr.Checkbox(label="Tải lên", info="Tải lên dữ liệu huấn luyện", value=False, interactive=True)
                                preprocess_cut = gr.Checkbox(label="Cắt âm thanh", info="Nên tắt nếu dữ liệu đã được xử lý", value=False, interactive=True)
                                process_effects = gr.Checkbox(label="Hiệu ứng quá trình", info="Nên tắt nếu dữ liệu đã được xử lý", value=False, interactive=True)
                            with gr.Column():
                                clean_dataset = gr.Checkbox(label="Làm sạch dữ liệu", info="Làm sạch các đoạn dữ liệu", value=False, interactive=True)
                                clean_dataset_strength = gr.Slider(label="Mức làm sạch", info="Mức độ làm sạch các đoạn dữ liệu", minimum=0, maximum=1, value=0.7, step=0.1, interactive=True, visible=False)
                        with gr.Column():
                            preprocess_button = gr.Button("1. Xử lý dữ liệu", scale=2)
                            upload_dataset = gr.Files(label="Thả dữ liệu vào đây", file_types=['audio'], visible=False)
                            preprocess_info = gr.Textbox(label="Thông tin phần xử lý trước", value="", interactive=False)
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            extract_method = gr.Radio(label="Phương pháp trích xuất", info="Phương pháp trích xuất dữ liệu huấn luyện", choices=["pm", "dio", "crepe", "crepe-tiny", "fcpe", "rmvpe", "harvest"], value="pm", interactive=True)
                            extract_hop_length = gr.Slider(label="Hop length", info="Khoảng cách giữa các khung âm thanh khi xử lý tín hiệu. Giá trị lớn xử lý nhẹ hơn, chi tiết giảm, giá trị nhỏ chi tiết cao, xử lý nặng", minimum=0, maximum=512, value=128, step=1, interactive=True, visible=False)
                            with gr.Accordion(label="Mô hình học cách nói", open=False):
                                extract_embedders = gr.Radio(label="Mô hình học cách nói", info="Mô hình được huấn luyện trước để giúp cho mô hình học cách nói cách ngắt hơi", choices=["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "custom"], value="contentvec_base", interactive=True)
                                with gr.Row():
                                    extract_embedders_custom = gr.Textbox(label="Tên của mô hình", info="Nếu bạn có mô hình riêng chỉ cần tải và nhập tên của mô hình vào đây", value="", placeholder="hubert_base", interactive=True, visible=False)
                        with gr.Column():
                            extract_button = gr.Button("2. Trích xuất dữ liệu", scale=2)
                            extract_info = gr.Textbox(label="Thông tin phần trích xuất dữ liệu", value="", interactive=False)
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            total_epochs = gr.Slider(label="Tổng số kỷ nguyên", info="Tổng số kỷ nguyên huấn luyện đào tạo", minimum=1, maximum=10000, value=300, step=1, interactive=True)
                            save_epochs = gr.Slider(label="Tần suất lưu", info="Tần suất lưu mô hình khi huấn luyện, giúp việc huấn luyện lại mô hình", minimum=1, maximum=10000, value=50, step=1, interactive=True)
                        with gr.Column():
                            index_button = gr.Button("3. Tạo chỉ mục", variant="primary", scale=2)
                            training_button = gr.Button("4. Huấn luyện", variant="primary", scale=2)
                    with gr.Row():
                        with gr.Accordion(label="Cài đặt chung", open=False):
                            with gr.Row():
                                index_algorithm = gr.Radio(label="Thuật toán chỉ mục", info="Thuật toán tạo chỉ mục", choices=["Auto", "Faiss", "KMeans"], value="Auto", interactive=True)
                            with gr.Row():
                                custom_dataset = gr.Checkbox(label="Tùy chọn thư mục", info="Tùy chọn thư mục dữ liệu huấn luyện", value=False, interactive=True)
                                overtraining_detector = gr.Checkbox(label="Kiểm tra quá sức", info="Kiểm tra huấn luyện mô hình quá sức", value=False, interactive=True)
                                sync_graph = gr.Checkbox(label="Đồng bộ biểu đồ", info="Đồng bộ biểu đồ huấn luyện", value=False, interactive=True)
                                cache_in_gpu = gr.Checkbox(label="Lưu mô hình vào đệm", info="Lưu mô hình vào bộ nhớ đệm gpu", value=False, interactive=True)
                            with gr.Column():
                                dataset_path = gr.Textbox(label="Thư mục chứa dữ liệu", value="dataset", interactive=True, visible=False)
                            with gr.Column():
                                threshold = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Ngưỡng huấn luyện quá sức", interactive=True, visible=False)
                                with gr.Accordion("Tùy chọn CPU/GPU", open=False):
                                    with gr.Column():
                                        gpu_number = gr.Textbox(label="Số gpu được sử dụng", value=str(get_number_of_gpus()), info="Số của GPU được sử dụng trong huấn luyện", interactive=True)
                                        gpu_info = gr.Textbox(label="Thông tin của GPU", value=get_gpu_info(), info="Thông tin của GPU được sử dụng trong huấn luyện", interactive=False)
                                        cpu_core = gr.Slider(label="Số lõi xử lý có thể sử dụng", info="Số lõi cpu được sử dụng trong việc huấn luyện", minimum=0, maximum=cpu_count(), value=cpu_count(), step=1, interactive=True)
                                        batch_size = gr.Slider(label="Kích thước lô", info="Số lượng mẫu xử lý đồng thời trong một lần huấn luyện. Cao có thể gây tràn bộ nhợ", minimum=1, maximum=64, value=8, step=1, interactive=True)
                            with gr.Row():
                                with gr.Row():
                                    save_only_latest = gr.Checkbox(label="Chỉ lưu mới nhất", info="Chỉ lưu mô hình D và G mới nhất", value=True, interactive=True)
                                    save_every_weights = gr.Checkbox(label="Lưu mọi mô hình", info="Lưu mọi mô hình sau mỗi lượt kỷ nguyên", value=True, interactive=True)
                                    not_use_pretrain = gr.Checkbox(label="Không dùng huấn luyện", info="Không dùng huấn luyện trước", value=False, interactive=True)
                                    custom_pretrain = gr.Checkbox(label="Tùy chỉnh huấn luyện", info="Tùy chỉnh huấn luyện trước", value=False, interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    with gr.Accordion("Tùy chọn huấn luyện trước", open=False, visible=False) as pretrain_setting:
                                        pretrained_D = gr.Dropdown(label="Tệp mô hình huấn luyện trước D", choices=sorted(pretrainedD), value=sorted(pretrainedD)[0] if len(sorted(pretrainedD)) > 0 else '', interactive=True, allow_custom_value=True, visible=False)
                                        pretrained_G = gr.Dropdown(label="Tệp mô hình huấn luyện trước G", choices=sorted(pretrainedG), value=sorted(pretrainedG)[0] if len(sorted(pretrainedG)) > 0 else '', interactive=True, allow_custom_value=True, visible=False)
                                        refesh_pretrain = gr.Button("Tải lại huấn luyện trước", scale=2, visible=False)
                    with gr.Row():
                        training_info = gr.Textbox(label="Thông tin phần huấn luyện", value="", interactive=False)
                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion("5. Xuất Mô hình", open=False):
                                with gr.Row():
                                    model_file= gr.Dropdown(label="Tệp mô hình", choices=sorted(model_name), value=sorted(model_name)[0] if len(sorted(model_name)) > 0 else '', interactive=True, allow_custom_value=True)
                                    index_file = gr.Dropdown(label="Tệp chỉ mục", choices=sorted(index_path), value=sorted(index_path)[0] if len(sorted(index_path)) > 0 else '', interactive=True, allow_custom_value=True)
                                with gr.Row():
                                    refesh_file = gr.Button("1. Tải lại", scale=2)
                                    zip_model = gr.Button("2. Nén mô hình", variant="primary", scale=2)
                                with gr.Row():
                                    zip_output = gr.File(label="Đầu ra tệp khi nén", file_types=['zip'], interactive=False, visible=False)
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
                        batch_size, 
                        gpu_number,
                        training_f0,
                        not_use_pretrain,
                        custom_pretrain,
                        pretrained_G,
                        pretrained_D,
                        overtraining_detector,
                        threshold,
                        sync_graph,
                        cache_in_gpu
                    ],
                    outputs=[training_info],
                    api_name="training_model"
                )

        with gr.TabItem("Dung Hợp"):
            gr.Markdown("## Dung Hợp Hai Mô Hình Với Nhau")
            with gr.Row():
                gr.Markdown("Dung hợp hai mô hình giọng nói lại với nhau để tạo thành một mô hình duy nhất")
            with gr.Row():
                with gr.Column():
                    name_to_save = gr.Textbox(label="Tên mô hình khi lưu", placeholder="Tên mô hình", value="", max_lines=1, interactive=True)
                with gr.Column():
                    fushion_button = gr.Button("Dung Hợp", variant="primary", scale=4)
            with gr.Column():
                with gr.Row():
                    model_a = gr.File(label="Mô hình A", file_types=['pth']) 
                    model_b = gr.File(label="Mô hình B", file_types=['pth'])
                with gr.Row():
                    model_path_a = gr.Textbox(label="Đường dẫn mô hình A", value="", placeholder="Đường dẫn")
                    model_path_b = gr.Textbox(label="Đường dẫn mô hình B", value="", placeholder="Đường dẫn")
            with gr.Row():
                ratio = gr.Slider(minimum=0, maximum=1, label="Tỉ lệ mô hình", info="Chỉnh hướng về bên nào sẽ làm cho mô hình giống với bên đó", value=0.5, interactive=True)
            with gr.Row():
                output_model = gr.File(label="Đầu ra mô hình", visible=False)
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

        with gr.TabItem("Đọc Thông Tin"):
            gr.Markdown("## Đọc Thông Tin Của Mô Hình")
            with gr.Row():
                gr.Markdown("đọc các thông tin được ghi trong mô hình")
            with gr.Row():
                with gr.Column():
                    model = gr.File(label="Thả mô hình vào đây", file_types=['pth']) 
                with gr.Column():
                    read_button = gr.Button("Đọc mô hình", variant="primary", scale=2)
            with gr.Column():
                model_path = gr.Textbox(label="Đường dẫn mô hình", value="", info="Nhập đường dẫn đến tệp mô hình", interactive=True)
                output_info = gr.Textbox(label="Thông Tin Mô Hình", value="", interactive=False, scale=6)
            with gr.Row():
                model.upload(fn=lambda model: shutil.move(model.name, os.path.join("assets", "weights")), inputs=[model], outputs=[model_path])
                read_button.click(
                    fn=model_info,
                    inputs=[model_path],
                    outputs=[output_info],
                    api_name="read_model"
                )

        with gr.TabItem("Tải Xuống"):
            gr.Markdown("## Tải Xuống Mô Hình")
            with gr.Row():
                gr.Markdown("Tải xuống mô hình giọng nói, mô hình huấn luyện trước, mô hình học cách nói")
            with gr.Row():
                with gr.Accordion("Tải xuống mô hình giọng nói", open=True):
                    with gr.Row():
                        downloadmodel = gr.Radio(label="Chọn cách tải mô hình", choices=["Tải từ đường dẫn liên kết", "Tải từ kho mô hình csv", "Tải mô hình từ Applio", "Tải lên"], interactive=True, value="Tải từ đường dẫn liên kết")
                    with gr.Row():
                        gr.Markdown("___")
                    with gr.Row():
                        url_input = gr.Textbox(label="Đường dẫn liên kết đến mô hình", value="", placeholder="https://...", scale=6, visible=True)
                        model_name = gr.Textbox(label="Tên mô hình để lưu", value="", placeholder="Tên của mô hình", scale=2, visible=True)
                        url_download = gr.Button(value="Tải xuống", scale=2, visible=True)
                    with gr.Row():
                        model_browser = gr.Dropdown(choices=models.keys(), label="Kho mô hình", scale=8, allow_custom_value=True, visible=False)
                        download_from_browser = gr.Button(value="Nhận mô hình", scale=2, variant="primary", visible=False)
                    with gr.Row():
                        model_upload = gr.File(label="Thả mô hình vào đây", file_types=['pth', 'index', 'zip'], visible=False)
                    with gr.Column():
                        with gr.Row():
                            search_name = gr.Textbox(label="Tên để tìm kiếm", placeholder="Tên mô hình", interactive=True, scale=8, visible=False)
                            search = gr.Button("Tìm kiếm", scale=2, visible=False)
                        with gr.Row():
                            search_dropdown = gr.Dropdown(label="Chọn mô hình đã được tìm kiếm(Bấm vào để chọn)", value="", choices=[], allow_custom_value=True, interactive=False, visible=False)
                            download = gr.Button("Tải xuống", variant="primary", visible=False)
            with gr.Row():
                with gr.Accordion("Tải xuống mô hình huấn luyện trước", open=False):
                    with gr.Row():
                        pretrain_download_choices = gr.Radio(label="Chọn cách tải mô hình", choices=["Đường dẫn mô hình", "Danh sách mô hình", "Tải lên"], value="Link mô hình", interactive=True)  
                    with gr.Row():
                        gr.Markdown("___")
                    with gr.Row():
                        pretrainD = gr.Textbox(label="Đường dẫn liên kết đến mô hình huấn luyện trước D", value="", info="Chỉ hỗ trợ huggingface.co", placeholder="https://...", interactive=True, scale=4, visible=True)
                        pretrainG = gr.Textbox(label="Đường dẫn liên kết đến mô hình huấn luyện trước G", value="", info="Chỉ hỗ trợ huggingface.co", placeholder="https://...", interactive=True, scale=4, visible=True)
                        download_pretrain_button = gr.Button("Tải xuống", scale=2)
                    with gr.Row():
                        pretrain_choices = gr.Dropdown(label="Chọn mô hình huấn luyện trước", info="Chọn mô hình huấn luyện trước để cài đặt về", choices=list(fetch_pretrained_data().keys()), value="Titan_Medium", allow_custom_value=True, interactive=True, scale=6, visible=False)
                        sample_rate_pretrain = gr.Dropdown(label="Tốc độ lấy mẫu của mô hình", choices=["48k", "40k", "32k"], value="48k", interactive=True, visible=False)
                        download_pretrain_choices_button = gr.Button("Tải xuống", scale=2, variant="primary", visible=False)
                    with gr.Row():
                        pretrain_upload_g = gr.File(label="Thả mô hình huấn luyện trước G vào đây", file_types=['pth'], visible=False)
                        pretrain_upload_d = gr.File(label="Thả mô hình huấn luyện trước D vào đây", file_types=['pth'], visible=False)
            with gr.Row():
                with gr.Accordion("Tải xuống mô hình học cách nói", open=False):
                    with gr.Row():
                        hubert_url = gr.Textbox(label="Đường dẫn liên kết tới mô hình học cách nói", value="", info="Chỉ hỗ trợ huggingface.co", placeholder="https://...", interactive=True, scale=8)
                        hubert_button = gr.Button("Tải xuống", scale=2, variant="primary")
                    with gr.Row():
                        hubert_input = gr.File(label="Thả mô hình học cách nói vào đây", file_types=['pt'])    
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
                
        with gr.TabItem("Nguồn Gốc"):
            gr.Markdown("## Nguồn Gốc Và Tác Giả Của Dự án")
            with gr.Row():
                gr.Markdown("___")
            with gr.Row():
                gr.Markdown(f"""
 
                    **Dự án này được nấu bởi [Phạm Huỳnh Anh]({codecs.decode("uggcf://tvguho.pbz/CunzUhlauNau16", "rot13")})**

                    **Dự án được nấu dựa trên một số dự án chính như:**

                    **Chuyển đổi, Xử lý, Trích xuất, Huấn luyện, Đọc mô hình, dung hợp mô hình, mô hình huấn luyện, kho mô hình...: [Applio]({codecs.decode("uggcf://tvguho.pbz/VNUvfcnab/Nccyvb/gerr/znva?gno=ernqzr-bi-svyr", "rot13")}) của nhóm [AI Hispano]({codecs.decode("uggcf://tvguho.pbz/VNUvfcnab", "rot13")})**

                    **Phương pháp trích xuất, cách hiển thị thông tin, cách ghi nhật ký, mô hình huấn luyện...: [Retrieval-based-Voice-Conversion-WebUI]({codecs.decode("uggcf://tvguho.pbz/EIP-Cebwrpg/Ergevriny-onfrq-Ibvpr-Pbairefvba-JroHV?gno=ernqzr-bi-svyr", "rot13")}) của tác giả [RVC BOSS]({codecs.decode("uggcf://tvguho.pbz/EIP-Obff", "rot13")})**
                    
                    **Mô hình tách nhạc MDX-Net và Demucs: [Python-audio-separator]({codecs.decode("uggcf://tvguho.pbz/abznqxnenbxr/clguba-nhqvb-frcnengbe?gno=ernqzr-bi-svyr", "rot13")}) của tác giả [Andrew Beveridge]({codecs.decode("uggcf://tvguho.pbz/orirenqo", "rot13")})**

                """)

    app.queue().launch(favicon_path=os.path.join("assets", "miku.png"), server_port=7860, show_error=True, inbrowser=True, share=True)