import os
import sys
import time
import tqdm
import torch
import shutil
import logging
import argparse
import warnings
import onnxruntime
import logging.handlers
import concurrent.futures

import numpy as np
import torch.multiprocessing as mp

from random import shuffle
from distutils.util import strtobool

sys.path.append(os.getcwd())

from main.configs.config import Config
from main.library.predictors.Generator import Generator
from main.library.utils import check_predictors, check_embedders, load_audio, load_embedders_model

logger = logging.getLogger(__name__)
config = Config()
translations = config.translations
logger.propagate = False

warnings.filterwarnings("ignore")
for l in ["torch", "faiss", "httpx", "httpcore", "faiss.loader", "numba.core", "urllib3", "matplotlib"]:
    logging.getLogger(l).setLevel(logging.ERROR)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--rvc_version", type=str, default="v2")
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--pitch_guidance", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--cpu_cores", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="-")
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--embedder_model", type=str, default="contentvec_base")
    parser.add_argument("--f0_onnx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mode", type=str, default="fairseq")

    return parser.parse_args()

def generate_config(rvc_version, sample_rate, model_path):
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path): shutil.copy(os.path.join("main", "configs", rvc_version, f"{sample_rate}.json"), config_save_path)

def generate_filelist(pitch_guidance, model_path, rvc_version, sample_rate, embedders_mode = "fairseq"):
    gt_wavs_dir, feature_dir = os.path.join(model_path, "sliced_audios"), os.path.join(model_path, f"{rvc_version}_extracted")
    f0_dir, f0nsf_dir = None, None
    if pitch_guidance: f0_dir, f0nsf_dir = os.path.join(model_path, "f0"), os.path.join(model_path, "f0_voiced")
    gt_wavs_files, feature_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir)), set(name.split(".")[0] for name in os.listdir(feature_dir))
    names = gt_wavs_files & feature_files & set(name.split(".")[0] for name in os.listdir(f0_dir)) & set(name.split(".")[0] for name in os.listdir(f0nsf_dir)) if pitch_guidance else gt_wavs_files & feature_files
    options = []
    mute_base_path = os.path.join("assets", "logs", "mute")

    for name in names:
        options.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|0" if pitch_guidance else f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|0")

    mute_audio_path, mute_feature_path = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"), os.path.join(mute_base_path, f"{rvc_version}_extracted", f"mute{'_spin' if embedders_mode == 'spin' else ''}.npy")
    for _ in range(2):
        options.append(f"{mute_audio_path}|{mute_feature_path}|{os.path.join(mute_base_path, 'f0', 'mute.wav.npy')}|{os.path.join(mute_base_path, 'f0_voiced', 'mute.wav.npy')}|0" if pitch_guidance else f"{mute_audio_path}|{mute_feature_path}|0")

    shuffle(options)
    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))

def setup_paths(exp_dir, version = None):
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    if version:
        out_path = os.path.join(exp_dir, f"{version}_extracted")
        os.makedirs(out_path, exist_ok=True)
        return wav_path, out_path
    else:
        output_root1, output_root2 = os.path.join(exp_dir, "f0"), os.path.join(exp_dir, "f0_voiced")
        os.makedirs(output_root1, exist_ok=True); os.makedirs(output_root2, exist_ok=True)
        return wav_path, output_root1, output_root2

def get_providers():
    ort_providers = onnxruntime.get_available_providers()

    if "CUDAExecutionProvider" in ort_providers: providers = ["CUDAExecutionProvider"]
    elif "CoreMLExecutionProvider" in ort_providers: providers = ["CoreMLExecutionProvider"]
    else: providers = ["CPUExecutionProvider"]

    return providers

class FeatureInput:
    def __init__(self, sample_rate=16000, hop_size=160, is_half=False, device=config.device):
        self.fs = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        self.is_half = is_half
        self.f0_gen = Generator(self.fs, self.hop, self.f0_min, self.f0_max, self.is_half, self.device, get_providers(), False)

    def compute_f0(self, np_arr, f0_method, hop_length, f0_onnx=False):
        self.f0_gen.hop_length, self.f0_gen.f0_onnx_mode = hop_length, f0_onnx
        return self.f0_gen.calculator(f0_method, np_arr, None, 0)

    def coarse_f0(self, f0):
        return np.rint(np.clip(((1127 * np.log(1 + f0 / 700)) - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1, 1, self.f0_bin - 1)).astype(int)

    def process_file(self, file_info, f0_method, hop_length, f0_onnx):
        inp_path, opt_path1, opt_path2, file_inp = file_info
        if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"): return

        try:
            feature_pit = self.compute_f0(load_audio(logger, file_inp, self.fs), f0_method, hop_length, f0_onnx)
            if isinstance(feature_pit, tuple): feature_pit = feature_pit[0]
            np.save(opt_path2, feature_pit, allow_pickle=False)
            np.save(opt_path1, self.coarse_f0(feature_pit), allow_pickle=False)
        except Exception as e:
            raise RuntimeError(f"{translations['extract_file_error']} {inp_path}: {e}")

    def process_files(self, files, f0_method, hop_length, f0_onnx, device, is_half, threads):
        self.device = device
        self.is_half = is_half

        def worker(file_info):
            self.process_file(file_info, f0_method, hop_length, f0_onnx)

        with tqdm.tqdm(total=len(files), ncols=100, unit="p", leave=True) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                for _ in concurrent.futures.as_completed([executor.submit(worker, f) for f in files]):
                    pbar.update(1)

def run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, devices, f0_onnx, is_half):
    input_root, *output_roots = setup_paths(exp_dir)
    output_root1, output_root2 = output_roots if len(output_roots) == 2 else (output_roots[0], None)
    paths = [(os.path.join(input_root, name), os.path.join(output_root1, name) if output_root1 else None, os.path.join(output_root2, name) if output_root2 else None, os.path.join(input_root, name)) for name in sorted(os.listdir(input_root)) if "spec" not in name]
    start_time = time.time()
    logger.info(translations["extract_f0_method"].format(num_processes=num_processes, f0_method=f0_method))

    feature_input = FeatureInput()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        concurrent.futures.wait([executor.submit(feature_input.process_files, paths[i::len(devices)], f0_method, hop_length, f0_onnx, devices[i], is_half, num_processes // len(devices)) for i in range(len(devices))])

    logger.info(translations["extract_f0_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

def extract_features(model, feats, version):
    return torch.as_tensor(model.run([model.get_outputs()[0].name, model.get_outputs()[1].name], {"feats": feats.detach().cpu().numpy()})[0 if version == "v1" else 1], dtype=torch.float32, device=feats.device)

def process_file_embedding(files, embedder_model, embedders_mode, device, version, is_half, threads):
    model, embed_suffix = load_embedders_model(embedder_model, embedders_mode, providers=get_providers())
    if embed_suffix != ".onnx": model = model.to(device).to(torch.float16 if is_half else torch.float32).eval()
    threads = max(1, threads)

    def worker(file_info):
        file, out_path = file_info
        out_file_path = os.path.join(out_path, os.path.basename(file.replace("wav", "npy")))
        if os.path.exists(out_file_path): return
        feats = torch.from_numpy(load_audio(logger, file, 16000)).to(device).to(torch.float16 if is_half else torch.float32).view(1, -1)

        with torch.no_grad():
            if embed_suffix == ".pt":
                logits = model.extract_features(**{"source": feats, "padding_mask": torch.BoolTensor(feats.shape).fill_(False).to(device), "output_layer": 9 if version == "v1" else 12})
                feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
            elif embed_suffix == ".onnx": feats = extract_features(model, feats, version).to(device)
            elif embed_suffix == ".safetensors":
                logits = model(feats)["last_hidden_state"]
                feats = (model.final_proj(logits[0]).unsqueeze(0) if version == "v1" else logits)
            else: raise ValueError(translations["option_not_valid"])

        feats = feats.squeeze(0).float().cpu().numpy()
        if not np.isnan(feats).any(): np.save(out_file_path, feats, allow_pickle=False)
        else: logger.warning(f"{file} {translations['NaN']}")

    with tqdm.tqdm(total=len(files), ncols=100, unit="p", leave=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for _ in concurrent.futures.as_completed([executor.submit(worker, f) for f in files]):
                pbar.update(1)

def run_embedding_extraction(exp_dir, version, num_processes, devices, embedder_model, embedders_mode, is_half):
    wav_path, out_path = setup_paths(exp_dir, version)
    start_time = time.time()
    logger.info(translations["start_extract_hubert"])
    paths = sorted([(os.path.join(wav_path, file), out_path) for file in os.listdir(wav_path) if file.endswith(".wav")])

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        concurrent.futures.wait([executor.submit(process_file_embedding, paths[i::len(devices)], embedder_model, embedders_mode, devices[i], version, is_half, num_processes // len(devices)) for i in range(len(devices))])

    logger.info(translations["extract_hubert_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

def main():
    args = parse_arguments()
    exp_dir = os.path.join("assets", "logs", args.model_name)
    f0_method, hop_length, num_processes, gpus, version, pitch_guidance, sample_rate, embedder_model, f0_onnx, embedders_mode = args.f0_method, args.hop_length, args.cpu_cores, args.gpu, args.rvc_version, args.pitch_guidance, args.sample_rate, args.embedder_model, args.f0_onnx, args.embedders_mode
    devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" for idx in gpus.split("-")]
    check_predictors(f0_method, f0_onnx); check_embedders(embedder_model, embedders_mode)

    if logger.hasHandlers(): logger.handlers.clear()
    else:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        file_handler = logging.handlers.RotatingFileHandler(os.path.join(exp_dir, "extract.log"), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
        file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    log_data = {translations['modelname']: args.model_name, translations['export_process']: exp_dir, translations['f0_method']: f0_method, translations['pretrain_sr']: sample_rate, translations['cpu_core']: num_processes, "Gpu": gpus, "Hop length": hop_length, translations['training_version']: version, translations['extract_f0']: pitch_guidance, translations['hubert_model']: embedder_model, translations["f0_onnx_mode"]: f0_onnx, translations["embed_mode"]: embedders_mode}
    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    pid_path = os.path.join(exp_dir, "extract_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))
    
    try:
        run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, devices, f0_onnx, config.is_half)
        run_embedding_extraction(exp_dir, version, num_processes, devices, embedder_model, embedders_mode, config.is_half)
        generate_config(version, sample_rate, exp_dir)
        generate_filelist(pitch_guidance, exp_dir, version, sample_rate, embedders_mode)
    except Exception as e:
        logger.error(f"{translations['extract_error']}: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    if os.path.exists(pid_path): os.remove(pid_path)
    logger.info(f"{translations['extract_success']} {args.model_name}.")

if __name__ == "__main__": 
    mp.set_start_method("spawn", force=True)
    main()