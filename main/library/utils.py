import os
import re
import gc
import sys
import torch
import codecs
import logging
import warnings

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.library.backends import directml, opencl
from main.library.algorithm.faisssearch import IndexWrapper
from main.app.variables import translations, configs, config, logger, embedders_model, whisper_model

if not config.debug_mode:
    warnings.filterwarnings("ignore")
    for l in ["httpx", "httpcore"]:
        logging.getLogger(l).setLevel(logging.ERROR)

def check_assets(f0_method, hubert, predictor_onnx=False, embedders_mode="fairseq"):
    predictors_url = codecs.decode(
        "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", 
        "rot13"
    )
    embedders_url = codecs.decode(
        "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", 
        "rot13"
    )

    def download_predictor(predictor):
        model_path = os.path.join(configs["predictors_path"], predictor)

        if not os.path.exists(model_path): 
            huggingface.HF_download_file(
                predictors_url + predictor, 
                model_path
            )

        return os.path.exists(model_path)

    def download_embedder(embedders_mode, hubert):
        model_path = (
            os.path.join(
                configs["speaker_diarization_path"], 
                "models", 
                hubert
            )
        ) if embedders_mode == "whisper" else (
            os.path.join(
                configs["embedders_path"], 
                hubert
            )
        )

        if embedders_mode != "transformers" and not os.path.exists(model_path): 
            if embedders_mode == "whisper":
                huggingface.HF_download_file(
                    "".join([
                        codecs.decode(
                            "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/", 
                            "rot13"
                        ), 
                        hubert
                    ]), 
                    model_path
                )
            else:
                huggingface.HF_download_file(
                    "".join([
                        embedders_url, "fairseq/" if embedders_mode == "fairseq" else "onnx/", 
                        hubert
                    ]), 
                    model_path
                )
        elif embedders_mode == "transformers":
            bin_file = os.path.join(model_path, "model.safetensors")
            config_file = os.path.join(model_path, "config.json")

            os.makedirs(model_path, exist_ok=True)

            if not os.path.exists(bin_file): 
                huggingface.HF_download_file(
                    "".join([embedders_url, "transformers/", hubert, "/model.safetensors"]), 
                    bin_file
                )

            if not os.path.exists(config_file): 
                huggingface.HF_download_file(
                    "".join([embedders_url, "transformers/", hubert, "/config.json"]), 
                    config_file
                )

            return os.path.exists(bin_file) and os.path.exists(config_file)

        return os.path.exists(model_path)

    def get_modelname(f0_method, predictor_onnx=False):
        suffix = ".onnx" if predictor_onnx else (".pt" if "crepe" not in f0_method else ".pth")

        if "rmvpe" in f0_method:
            modelname = (
                "hpa-rmvpe-76000" 
                if "previous" in f0_method else 
                "hpa-rmvpe-112000"
            ) if "hpa" in f0_method else (
                "rmvpe-mix" 
                if "mix" in f0_method else 
                "rmvpe"
            )
        elif "fcpe" in f0_method:
            modelname = (
                "fcpe_legacy" 
                if "legacy" in f0_method else 
                "fcpe"
            ) if "previous" in f0_method or "legacy" in f0_method else "ddsp_200k"
        elif "crepe" in f0_method:
            modelname = "crepe_" + f0_method.replace("mangio-", "").split("-")[1]
        elif "penn" in f0_method:
            modelname = "fcn"
        elif "djcm" in f0_method:
            modelname = "djcm" + ("-svs" if "svs" in f0_method else "")
        elif "pesto" in f0_method:
            modelname = "pesto"
        elif "swift" in f0_method:
            return "swift.onnx"
        else:
            return None
        
        return modelname + suffix
    
    results = []
    count = configs.get("num_of_restart", 5)

    for _ in range(count):
        if "hybrid" in f0_method:
            methods_str = re.search(r"hybrid\[(.+)\]", f0_method)

            if methods_str: 
                methods = [
                    f0_method.strip() 
                    for f0_method in methods_str.group(1).split("+")
                ]

            for method in methods:
                modelname = get_modelname(method, predictor_onnx)

                if modelname is not None: 
                    results.append(
                        download_predictor(modelname)
                    )
        else: 
            modelname = get_modelname(f0_method, predictor_onnx)

            if modelname is not None: 
                results.append(
                    download_predictor(modelname)
                )

        if hubert in embedders_model + whisper_model:
            if embedders_mode != "transformers": hubert += ".pt" if embedders_mode in ["fairseq", "whisper"] else ".onnx"

            results.append(
                download_embedder(
                    embedders_mode, 
                    hubert
                )
            )

        if all(results): return
        else: results = []

    logger.warning(translations["check_assets_error"].format(count=count))
    sys.exit(1)
    
def check_spk_diarization(model_size, speechbrain=True):
    whisper_model = os.path.join(configs["speaker_diarization_path"], "models", f"{model_size}.pt")

    if not os.path.exists(whisper_model) and model_size is not None: 
        huggingface.HF_download_file(
            "".join([
                codecs.decode(
                    "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/", 
                    "rot13"
                ), 
                model_size, 
                ".pt"
            ]), 
            whisper_model
        )

    speechbrain_path = os.path.join(configs["speaker_diarization_path"], "models", "speechbrain")
    if not os.path.exists(speechbrain_path): os.makedirs(speechbrain_path, exist_ok=True)

    if speechbrain:
        for f in [
            "classifier.ckpt", 
            "config.json", 
            "embedding_model.ckpt", 
            "hyperparams.yaml", 
            "mean_var_norm_emb.ckpt"
        ]:
            speechbrain_model = os.path.join(speechbrain_path, f)

            if not os.path.exists(speechbrain_model): 
                huggingface.HF_download_file(
                    codecs.decode(
                        "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/fcrrpuoenva/", 
                        "rot13"
                    ) + f, 
                    speechbrain_model
                )

def check_upscaler():
    upscaler_model = os.path.join("assets", "models", "upscalers", "upscalers.pth")
    if not os.path.exists(upscaler_model):
        huggingface.HF_download_file(
            codecs.decode(
                "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/hcfpnyref/hcfpnyref.cgu",
                "rot13"
            ),
            upscaler_model
        )

def load_embedders_model(embedder_model, embedders_mode="fairseq"):
    if embedders_mode in ["fairseq", "whisper"] and not embedder_model.endswith(".pt"): embedder_model += ".pt"
    elif embedders_mode == "onnx" and not embedder_model.endswith(".onnx"): embedder_model += ".onnx"

    if os.path.exists(embedder_model): embedder_model_path = embedder_model
    else:
        embedder_model_path = (
            os.path.join(
                configs["speaker_diarization_path"], 
                "models", 
                embedder_model
            )
        ) if embedders_mode == "whisper" else (
            os.path.join(
                configs["embedders_path"], 
                embedder_model
            )
        )
    
    if not os.path.exists(embedder_model_path): 
        raise FileNotFoundError(
            f"{translations['not_found'].format(name=translations['model'])}: {embedder_model}"
        )

    try:
        if embedders_mode == "fairseq":
            from main.library.embedders.fairseq import load_model

            hubert_model = load_model(
                embedder_model_path
            )
        elif embedders_mode == "onnx":
            from main.library.embedders.onnx import HubertModelONNX

            hubert_model = HubertModelONNX(
                embedder_model_path, 
                config.providers
            )
        elif embedders_mode == "transformers":
            from main.library.embedders.transformers import HubertModelWithFinalProj

            hubert_model = HubertModelWithFinalProj.from_pretrained(
                embedder_model_path
            )
        elif embedders_mode == "whisper":
            from main.library.embedders.ppg import WhisperModel

            hubert_model = WhisperModel(
                embedder_model_path
            )
        else: raise ValueError(translations["option_not_valid"])
    except Exception as e:
        raise RuntimeError(translations["read_model_error"].format(e=e))

    return hubert_model

def extract_features(model, feats, version, mix=False, mix_layers=9, mix_ratio=0.5):
    if hasattr(model, "_finalproj"): model._finalproj = version == "v1"

    logits = model.extract_features(
        **{
            "source": feats, 
            "output_layer": 9 if version == "v1" else 12
        }
    )[0]

    if mix:
        mix_logits = model.extract_features(
            **{
                "source": feats, 
                "output_layer": mix_layers
            }
        )[0]

        logits = mix_ratio * mix_logits + (1 - mix_ratio) * logits

    feats = model.final_proj(logits) if version == "v1" else logits

    return feats

def clear_gpu_cache():
    gc.collect()

    if config.device.startswith("cuda"): torch.cuda.empty_cache()
    elif config.device.startswith("xpu"): torch.xpu.empty_cache()
    elif config.device.startswith("mps"): torch.mps.empty_cache()
    elif config.device.startswith("privateuseone"): directml.empty_cache()
    elif config.device.startswith("ocl"): opencl.pytorch_ocl.empty_cache()

def circular_write(new_data, target):
    offset = new_data.shape[0]

    target[: -offset] = target[offset :].detach().clone()
    target[-offset :] = new_data

    return target

def load_faiss_index(index_path, nprobe=1):
    index = IndexWrapper(index_path, nprobe=nprobe, device=config.device, is_half=config.is_half, faiss_cpu=configs.get("faiss_cpu", False) or config.device.startswith(("privateuseone", "ocl")))
    big_npy, _ = index.read_index_tensor()

    return index, big_npy

def load_model(model_path, weights_only=True, log_severity_level=3):
    if not os.path.isfile(model_path): return None

    if model_path.endswith(".pth"): 
        return torch.load(
            model_path, 
            map_location="cpu", 
            weights_only=weights_only
        )
    else: 
        from main.library.onnx.wrapper import ONNXRVC

        return ONNXRVC(
            model_path, 
            config.providers, 
            log_severity_level=log_severity_level
        )

def strtobool(value):
    value = value.lower()

    if value in ('y', 'yes', 't', 'true', 'on', '1'): return 1
    elif value in ('n', 'no', 'f', 'false', 'off', '0'): return 0
    else: raise ValueError

def check_ffmpeg():
    try:
        if not sys.platform == "win32": return
        ffmpeg_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/sszcrt/", "rot13")

        if not os.path.exists("ffmpeg.exe"): huggingface.HF_download_file(ffmpeg_url + "ffmpeg.exe", output_path="ffmpeg.exe")
        if not os.path.exists("ffprobe.exe"): huggingface.HF_download_file(ffmpeg_url + "ffprobe.exe", output_path="ffprobe.exe")
    except:
        logger.error(translations["ffmpeg_error"])
        return

if configs.get("ffmpeg_download", True): check_ffmpeg()