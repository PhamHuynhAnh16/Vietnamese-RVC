import os
import re
import sys
import codecs
import librosa
import logging
import onnxruntime

import numpy as np
import torch.nn as nn
import soundfile as sf

from pydub import AudioSegment
from transformers import HubertModel

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.library.architectures import fairseq
from main.app.variables import translations, configs

for l in ["httpx", "httpcore"]:
    logging.getLogger(l).setLevel(logging.ERROR)

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

def check_predictors(method, f0_onnx=False):
    if f0_onnx: method += "-onnx"

    def download(predictors):
        if not os.path.exists(os.path.join(configs["predictors_path"], predictors)): huggingface.HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13") + predictors, os.path.join(configs["predictors_path"], predictors))

    model_dict = {**dict.fromkeys(["rmvpe", "rmvpe-legacy"], "rmvpe.pt"), **dict.fromkeys(["rmvpe-onnx", "rmvpe-legacy-onnx"], "rmvpe.onnx"), **dict.fromkeys(["fcpe"], "fcpe.pt"), **dict.fromkeys(["fcpe-legacy"], "fcpe_legacy.pt"), **dict.fromkeys(["fcpe-onnx"], "fcpe.onnx"), **dict.fromkeys(["fcpe-legacy-onnx"], "fcpe_legacy.onnx"), **dict.fromkeys(["crepe-full", "mangio-crepe-full"], "crepe_full.pth"), **dict.fromkeys(["crepe-full-onnx", "mangio-crepe-full-onnx"], "crepe_full.onnx"), **dict.fromkeys(["crepe-large", "mangio-crepe-large"], "crepe_large.pth"), **dict.fromkeys(["crepe-large-onnx", "mangio-crepe-large-onnx"], "crepe_large.onnx"), **dict.fromkeys(["crepe-medium", "mangio-crepe-medium"], "crepe_medium.pth"), **dict.fromkeys(["crepe-medium-onnx", "mangio-crepe-medium-onnx"], "crepe_medium.onnx"), **dict.fromkeys(["crepe-small", "mangio-crepe-small"], "crepe_small.pth"), **dict.fromkeys(["crepe-small-onnx", "mangio-crepe-small-onnx"], "crepe_small.onnx"), **dict.fromkeys(["crepe-tiny", "mangio-crepe-tiny"], "crepe_tiny.pth"), **dict.fromkeys(["crepe-tiny-onnx", "mangio-crepe-tiny-onnx"], "crepe_tiny.onnx")}

    if "hybrid" in method:
        methods_str = re.search("hybrid\[(.+)\]", method)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        for method in methods:
            if method in model_dict: download(model_dict[method])
    elif method in model_dict: download(model_dict[method])

def check_embedders(hubert, embedders_mode="fairseq"):
    huggingface_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13")
    if embedders_mode == "spin": embedders_mode, hubert = "transformers", "spin"

    if hubert in ["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "portuguese_hubert_base", "spin"]:
        if embedders_mode == "fairseq": hubert += ".pt"
        elif embedders_mode == "onnx": hubert += ".onnx"

        model_path = os.path.join(configs["embedders_path"], hubert)

        if embedders_mode == "fairseq": 
            if not os.path.exists(model_path): huggingface.HF_download_file("".join([huggingface_url, "fairseq/", hubert]), model_path)
        elif embedders_mode == "onnx": 
            if not os.path.exists(model_path): huggingface.HF_download_file("".join([huggingface_url, "onnx/", hubert]), model_path)
        elif embedders_mode == "transformers":
            url, hubert = ("transformers/", hubert) if hubert != "spin" else ("spin", "")

            bin_file = os.path.join(model_path, "model.safetensors")
            config_file = os.path.join(model_path, "config.json")

            os.makedirs(model_path, exist_ok=True)

            if not os.path.exists(bin_file): huggingface.HF_download_file("".join([huggingface_url, url, hubert, "/model.safetensors"]), bin_file)
            if not os.path.exists(config_file): huggingface.HF_download_file("".join([huggingface_url, url, hubert, "/config.json"]), config_file)
        else: raise ValueError(translations["option_not_valid"])
    
def check_spk_diarization(model_size):
    whisper_model = os.path.join(configs["speaker_diarization_path"], "models", f"{model_size}.pt")
    if not os.path.exists(whisper_model): huggingface.HF_download_file("".join([codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/", "rot13"), model_size, ".pt"]), whisper_model)

    speechbrain_path = os.path.join(configs["speaker_diarization_path"], "models", "speechbrain")
    if not os.path.exists(speechbrain_path): os.makedirs(speechbrain_path, exist_ok=True)

    for f in ["classifier.ckpt", "config.json", "embedding_model.ckpt", "hyperparams.yaml", "mean_var_norm_emb.ckpt"]:
        speechbrain_model = os.path.join(speechbrain_path, f)

        if not os.path.exists(speechbrain_model): huggingface.HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/fcrrpuoenva/", "rot13") + f, speechbrain_model)

def check_audioldm2(model):
    for f in ["feature_extractor", "language_model", "projection_model", "scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae", "vocoder"]:
        folder_path = os.path.join(configs["audioldm2_model_path"], model, f)

        if not os.path.exists(folder_path): os.makedirs(folder_path, exist_ok=True)

    for f in ["feature_extractor/preprocessor_config.json","language_model/config.json","language_model/model.safetensors","model_index.json","projection_model/config.json","projection_model/diffusion_pytorch_model.safetensors","scheduler/scheduler_config.json","text_encoder/config.json","text_encoder/model.safetensors","text_encoder_2/config.json","text_encoder_2/model.safetensors","tokenizer/merges.txt","tokenizer/special_tokens_map.json","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json","tokenizer/vocab.json","tokenizer_2/special_tokens_map.json","tokenizer_2/spiece.model","tokenizer_2/tokenizer.json","tokenizer_2/tokenizer_config.json","unet/config.json","unet/diffusion_pytorch_model.safetensors","vae/config.json","vae/diffusion_pytorch_model.safetensors","vocoder/config.json","vocoder/model.safetensors"]:
        model_path = os.path.join(configs["audioldm2_model_path"], model, f)

        if not os.path.exists(model_path): huggingface.HF_download_file("".join([codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/nhqvbyqz/", "rot13"), model, "/", f]), model_path)

def load_audio(logger, file, sample_rate=16000, formant_shifting=False, formant_qfrency=0.8, formant_timbre=0.8):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file): raise FileNotFoundError(translations["not_found"].format(name=file))

        try:
            logger.debug(translations['read_sf'])
            audio, sr = sf.read(file, dtype=np.float32)
        except:
            logger.debug(translations['read_librosa'])
            audio, sr = librosa.load(file, sr=None)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != sample_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq")

        if formant_shifting:
            from main.library.algorithm.stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(1024, 32, sample_rate)
            audio = pitchshifter.shiftpitch(audio, factors=1, quefrency=formant_qfrency * 1e-3, distortion=formant_timbre)
    except Exception as e:
        raise RuntimeError(f"{translations['errors_loading_audio']}: {e}")
    
    return audio.flatten()

def pydub_load(input_path, volume = None):
    try:
        if input_path.endswith(".wav"): audio = AudioSegment.from_wav(input_path)
        elif input_path.endswith(".mp3"): audio = AudioSegment.from_mp3(input_path)
        elif input_path.endswith(".ogg"): audio = AudioSegment.from_ogg(input_path)
        else: audio = AudioSegment.from_file(input_path)
    except:
        audio = AudioSegment.from_file(input_path)
        
    return audio if volume is None else audio + volume

def load_embedders_model(embedder_model, embedders_mode="fairseq", providers=None):
    if embedders_mode == "fairseq": embedder_model += ".pt"
    elif embedders_mode == "onnx": embedder_model += ".onnx"
    elif embedders_mode == "spin": embedders_mode, embedder_model = "transformers", "spin"

    embedder_model_path = os.path.join(configs["embedders_path"], embedder_model)
    if not os.path.exists(embedder_model_path): raise FileNotFoundError(f"{translations['not_found'].format(name=translations['model'])}: {embedder_model}")

    try:
        if embedders_mode == "fairseq":
            embed_suffix = ".pt"
            hubert_model = fairseq.load_model(embedder_model_path)
        elif embedders_mode == "onnx":
            sess_options = onnxruntime.SessionOptions()
            sess_options.log_severity_level = 3
            embed_suffix = ".onnx"
            hubert_model = onnxruntime.InferenceSession(embedder_model_path, sess_options=sess_options, providers=providers)
        elif embedders_mode == "transformers":      
            embed_suffix = ".safetensors"
            hubert_model = HubertModelWithFinalProj.from_pretrained(embedder_model_path)
        else: raise ValueError(translations["option_not_valid"])
    except Exception as e:
        raise RuntimeError(translations["read_model_error"].format(e=e))

    return hubert_model, embed_suffix

def cut(audio, sr, db_thresh=-60, min_interval=250):
    from main.inference.preprocess.slicer2 import Slicer2

    slicer = Slicer2(sr=sr, threshold=db_thresh, min_interval=min_interval)
    return slicer.slice2(audio)

def restore(segments, total_len, dtype=np.float32):
    out = []
    last_end = 0

    for start, end, processed_seg in segments:
        if start > last_end: out.append(np.zeros(start - last_end, dtype=dtype))

        out.append(processed_seg)
        last_end = end

    if last_end < total_len: out.append(np.zeros(total_len - last_end, dtype=dtype))
    return np.concatenate(out, axis=-1)

def get_providers():
    ort_providers = onnxruntime.get_available_providers()

    if "CUDAExecutionProvider" in ort_providers: providers = ["CUDAExecutionProvider"]
    elif "CoreMLExecutionProvider" in ort_providers: providers = ["CoreMLExecutionProvider"]
    else: providers = ["CPUExecutionProvider"]

    return providers