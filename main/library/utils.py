import os
import re
import sys
import torch
import codecs
import logging
import warnings

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.library.backends import directml, opencl
from main.library.algorithm.faisssearch import IndexWrapper
from main.app.variables import translations, configs, config, logger, embedders_model

if not config.debug_mode:
    warnings.filterwarnings("ignore")
    for l in ["httpx", "httpcore"]:
        logging.getLogger(l).setLevel(logging.ERROR)

def check_assets(f0_method, hubert, predictor_onnx=False, embedders_mode="fairseq"):
    """
    Checks and downloads necessary asset models (predictors and embedders) if they do not exist.

    Args:
        f0_method (str): The pitch extraction method (e.g., 'rmvpe', 'fcpe', 'crepe', 'hybrid[...]').
        hubert (str): Name of the Hubert embedder model.
        predictor_onnx (bool, optional): Whether to use the ONNX version of the predictor. Defaults to False.
        embedders_mode (str, optional): Mode of the embedder ('fairseq', 'onnx', 'transformers'). Defaults to "fairseq".

    Raises:
        SystemExit: If assets cannot be fully downloaded after the maximum allowed retry attempts.
    """

    # Force ONNX mode and 8-bit quantization if specified in global configuration
    if config.int8: predictor_onnx, embedders_mode = True, "onnx"

    # Decode repository URLs from ROT13 obfuscation
    predictors_url = codecs.decode(
        "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", 
        "rot13"
    )
    embedders_url = codecs.decode(
        "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", 
        "rot13"
    )

    def download_predictor(predictor):
        """Downloads a specific predictor model from HuggingFace if missing."""

        model_path = os.path.join(configs["predictors_path"], predictor)

        if not os.path.exists(model_path): 
            huggingface.HF_download_file(
                predictors_url + predictor, 
                model_path
            )

        return os.path.exists(model_path)

    def download_embedder(embedders_mode, hubert):
        """Downloads a specific embedder model or its components based on the mode."""

        model_path = os.path.join(configs["embedders_path"], hubert)
        
        # Handle standalone Fairseq or ONNX models
        if embedders_mode != "transformers" and not os.path.exists(model_path): 
            huggingface.HF_download_file(
                "".join([
                    embedders_url, 
                    "fairseq/" if embedders_mode == "fairseq" else "onnx/", 
                    hubert
                ]), 
                model_path
            )
        elif embedders_mode == "transformers": # Handle HuggingFace Transformers models which require weights and configuration files
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
        """Resolves the exact file name and suffix of the model based on F0 method."""

        suffix = ("-int8.onnx" if config.int8 else ".onnx") if predictor_onnx else (".pt" if "crepe" not in f0_method else ".pth")
        # Resolve model name based on keyword matching in F0 method string
        if "rmvpe" in f0_method:
            modelname = (
                "hpa-v4" if "v4" in f0_method else (
                    "hpa-rmvpe-76000" 
                    if "previous" in f0_method else 
                    "hpa-rmvpe-112000"
                )
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
            modelname = "swift"
            suffix = ("-int8.onnx" if config.int8 else ".onnx")
        else:
            return None
        
        return modelname + suffix
    
    results = []
    count = configs.get("num_of_restart", 5)
    # Retry loop for robust asset downloading
    for _ in range(count):
        # Extract and download all sub-methods if hybrid F0 is utilized
        if "hybrid" in f0_method:
            methods_str = re.search(r"hybrid\[(.+)\]", f0_method)

            if methods_str: 
                methods = [
                    f0_method.strip() 
                    for f0_method in methods_str.group(1).split("+")
                ]

            for method in methods:
                modelname = get_modelname(method, predictor_onnx)
                if modelname is not None: results.append(download_predictor(modelname))
        else: 
            modelname = get_modelname(f0_method, predictor_onnx)
            if modelname is not None: results.append(download_predictor(modelname))

        # Format append extensions for Hubert embedder based on quantization and mode
        if hubert in embedders_model:
            if config.int8 and embedders_mode == "onnx": hubert += "-int8"
            if embedders_mode != "transformers": hubert += ".pt" if embedders_mode == "fairseq" else ".onnx"

            results.append(download_embedder(embedders_mode, hubert))

        # Successfully downloaded all targeted assets
        if all(results): return
        else: results = []

    # If the block fails to download everything successfully within allowed restarts
    logger.warning(translations["check_assets_error"].format(count=count))
    sys.exit(1)
    
def check_spk_diarization(model_size, speechbrain=True):
    """
    Checks and downloads Whisper and SpeechBrain models required for speaker diarization.

    Args:
        model_size (str, optional): Size name of the Whisper model (e.g., 'tiny', 'base').
        speechbrain (bool, optional): Whether to download SpeechBrain models. Defaults to True.
    """

    whisper_model = os.path.join(configs["speaker_diarization_path"], "models", f"{model_size}.pt")
    # Download Whisper segmenter model if missing
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
    # Download all necessary configuration and checkpoint components for SpeechBrain
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
    """Checks and downloads the upscaler model asset if it is not present locally."""

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
    """
    Loads an audio embedder model (Hubert) from disk using the specified mode wrapper.

    Args:
        embedder_model (str): File path or model name identifier.
        embedders_mode (str, optional): Loading wrapper type ('fairseq', 'onnx', 'transformers'). Defaults to "fairseq".

    Raises:
        FileNotFoundError: If the resolved path of the model file does not exist.
        ValueError: If an unsupported embedder mode option is supplied.
        RuntimeError: Wrapper initialization or reading error.

    Returns:
        Any: Initialized wrapper instance of the loaded Hubert embedder model.
    """

    if config.int8: 
        embedders_mode = "onnx"
        embedder_model += "-int8"

    # Append standard extension suffixes if missing
    if embedders_mode == "fairseq" and not embedder_model.endswith(".pt"): embedder_model += ".pt"
    elif embedders_mode == "onnx" and not embedder_model.endswith(".onnx"): embedder_model += ".onnx"

    # Determine absolute path location
    if os.path.exists(embedder_model): embedder_model_path = embedder_model
    else: embedder_model_path = os.path.join(configs["embedders_path"], embedder_model)
    
    if not os.path.exists(embedder_model_path): raise FileNotFoundError(translations['not_found'].format(name=embedder_model))

    try:
        # Construct and instantiate model structure according to the specified backend framework
        if embedders_mode == "fairseq":
            from main.library.embedders.fairseq import load_model

            hubert_model = load_model(
                embedder_model_path
            )
        elif embedders_mode == "onnx":
            from main.library.embedders.onnx import HubertModelONNX

            hubert_model = HubertModelONNX(
                embedder_model_path, 
                config.providers,
                config.device
            )
        elif embedders_mode == "transformers":
            from main.library.embedders.transformers import HubertModelWithFinalProj

            hubert_model = HubertModelWithFinalProj.from_pretrained(
                embedder_model_path
            )
        else: raise ValueError(translations["option_not_valid"])
    except Exception as e:
        raise RuntimeError(translations["read_model_error"].format(e=e))

    return hubert_model

def extract_features(model, feats, version, mix=False, mix_layers=9, mix_ratio=0.5):
    """
    Extracts features from the hidden states of an embedder model with multi-layer mixing support.

    Args:
        model (Any): The instantiated Hubert embedder model.
        feats (torch.Tensor): Input tensor containing audio features.
        version (str): Model version identifier ('v1' or 'v2').
        mix (bool, optional): Whether to mix alternative layer representations. Defaults to False.
        mix_layers (int, optional): The target layer index to blend when mix is enabled. Defaults to 9.
        mix_ratio (float, optional): Weight ratio given to the alternate mixed layer. Defaults to 0.5.

    Returns:
        torch.Tensor: The final projected feature tensor embeddings.
    """

    # Toggle project status depending on RVC/SVC model architecture version
    if hasattr(model, "_final_proj"): model._final_proj = version == "v1"
    # Extract logits from standard target layers (9 for v1, 12 for v2)
    logits = model.extract_features(feats, output_layer=9 if version == "v1" else 12)[0]

    if mix: # Handle cross-layer representation mixing interpolation
        mix_logits = model.extract_features(feats, output_layer=mix_layers)[0]
        logits = mix_ratio * mix_logits + (1 - mix_ratio) * logits

    # Compute optional final projections
    feats = model.final_proj(logits) if version == "v1" else logits
    return feats

def clear_gpu_cache():
    """Clears empty memory blocks across various compute backends (CUDA, XPU, MPS, DirectML, OpenCL)."""

    if config.device.startswith("cuda"): torch.cuda.empty_cache()
    elif config.device.startswith("xpu"): torch.xpu.empty_cache()
    elif config.device.startswith("mps"): torch.mps.empty_cache()
    elif config.device.startswith("privateuseone"): directml.empty_cache()
    elif config.device.startswith("ocl"): opencl.empty_cache()

def circular_write(new_data, target):
    """
    Performs a circular array buffer shift operation, writing new data to the tail.

    Args:
        new_data (torch.Tensor): The 1D/2D incoming tensor slice to append.
        target (torch.Tensor): The main persistent tensor buffer.

    Returns:
        torch.Tensor: Updated main persistent target tensor buffer.
    """

    offset = new_data.shape[0]
    # Shift remaining elements leftwards to create buffer room at the end
    target[: -offset] = target[offset :].detach().clone()
    # Write fresh chunk data into the freed up index tail slots
    target[-offset :] = new_data

    return target

def phase_vocoder(a, b, fade_out, fade_in):
    """
    Performs a phase vocoder crossfade between two audio segments.

    This function blends segment `a` and segment `b` by aligning their phases 
    in the frequency domain to prevent phase cancellation during the transition,
    while applying the provided fade-out and fade-in windows.

    Args:
        a (torch.Tensor): The first audio segment.
        b (torch.Tensor): The second audio segment.
        fade_out (torch.Tensor): The fade-out window applied to segment `a`.
        fade_in (torch.Tensor): The fade-in window applied to segment `b`.

    Returns:
        torch.Tensor: The crossfaded audio segment.
    """

    # Compute the analysis window as the geometric mean of fade curves
    window = (fade_out * fade_in).sqrt()
    # Transform both windowed segments to the frequency domain (Real FFT)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    # Calculate the combined magnitude spectrum
    absab = fa.abs() + fb.abs()
    n = a.shape[0]

    # Compensate for the energy of negative frequencies (except DC and Nyquist components)
    if n % 2 == 0: absab[1:-1] *= 2
    else: absab[1:] *= 2

    # Extract initial phase and calculate the raw phase difference
    phia = fa.angle()
    deltaphase = fb.angle() - phia

    # Reconstruct the signal using a combination of time-domain crossfade 
    # and phase-aligned sinusoidal synthesis (Phase Vocoder)
    return (
        a * (fade_out ** 2) + b * (fade_in ** 2) + (
            absab * (
                (
                    # Base frequency grid for each bin
                    2 * torch.pi * torch.arange(n // 2 + 1).to(a) + 
                    # Phase unwrapping (wrapping delta phase to the [-pi, pi] range)
                    (deltaphase - 2 * torch.pi * (deltaphase / 2 / torch.pi + 0.5).floor()) 
                ) * (torch.arange(n).unsqueeze(-1).to(a) / n) + phia # Continuous phase evolution over time
            ).cos()
        ).sum(-1) * window / n # Sum the sinusoidal components (IFFT equivalent) and normalize
    )

def load_faiss_index(index_path, nprobe=1, cpu_mode=False):
    """
    Loads a FAISS index from file path with explicit configurations.

    Args:
        index_path (str): File system path pointing to the FAISS index file.
        nprobe (int, optional): Number of cell centroids to query during search. Defaults to 1.
        cpu_mode (bool, optional): Force index evaluation exclusively on host CPU. Defaults to False.

    Returns:
        Tuple[IndexWrapper, Any]: Wrapped FAISS structural object instance along with index array tensors.
    """

    index = IndexWrapper(index_path, nprobe=nprobe, device=config.device, is_half=config.is_half, faiss_cpu=configs.get("faiss_cpu", False) or cpu_mode)
    big_npy, _ = index.read_index_tensor()

    return index, big_npy

def load_model(model_path, weights_only=True, log_severity_level=3):
    """
    Loads an RVC/SVC inference model file from disk supporting PyTorch weights (.pth) and ONNX formats.

    Args:
        model_path (str): File path containing model structure.
        weights_only (bool, optional): PyTorch specific safe-unpickling flag. Defaults to True.
        log_severity_level (int, optional): ONNX initialization logging verbosity. Defaults to 3.

    Returns:
        Model: Instantiated model wrapper or state dictionary tensor mapping.
    """

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
    """Parses a generic string expression to its categorical boolean equivalent (1 or 0).

    Args:
        value (str): Textual representation flag input.

    Raises:
        ValueError: If string value does not match any preset boolean maps.

    Returns:
        int: Returns 1 for positive flags and 0 for negative flags.
    """

    value = value.lower()

    if value in ('y', 'yes', 't', 'true', 'on', '1'): return 1
    elif value in ('n', 'no', 'f', 'false', 'off', '0'): return 0
    else: raise ValueError("Invalid format!")

def check_ffmpeg():
    """Downloads static standard ffmpeg and ffprobe binary utilities on Windows if missing."""

    try:
        if not sys.platform == "win32": return
        ffmpeg_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/sszcrt/", "rot13")

        if not os.path.exists("ffmpeg.exe"): huggingface.HF_download_file(ffmpeg_url + "ffmpeg.exe", output_path="ffmpeg.exe")
        if not os.path.exists("ffprobe.exe"): huggingface.HF_download_file(ffmpeg_url + "ffprobe.exe", output_path="ffprobe.exe")
    except:
        logger.error(translations["ffmpeg_error"])
        return

# Trigger conditional lazy execution sequence checking on script initialization
if configs.get("ffmpeg_download", True): check_ffmpeg()