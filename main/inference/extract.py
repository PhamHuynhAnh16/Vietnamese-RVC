import os
import sys
import time
import tqdm
import torch
import shutil
import codecs
import librosa
import logging
import argparse
import warnings
import logging.handlers
import numpy as np
import soundfile as sf
import torch.nn.functional as F

from random import shuffle
from distutils.util import strtobool

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.configs.config import Config

config = Config()
logger = logging.getLogger(__name__)
translations = config.translations
logger.propagate = False

warnings.filterwarnings("ignore")
for l in ["torch", "faiss", "httpx", "fairseq", "httpcore", "faiss.loader", "transformers"]:
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

    return parser.parse_args()

def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != sample_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    except Exception as e:
        raise RuntimeError(f"{translations['errors_loading_audio']}: {e}")

    return audio.flatten()

def check_predictors(method):
    def download(predictors):
        if not os.path.exists(os.path.join("assets", "models", "predictors", predictors)):
            logger.info(f"{translations['downloads']} {predictors}") 
            huggingface.HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13") + predictors, os.path.join("assets", "models", "predictors", predictors))
    
    logger.info(translations["download_predictors"])
    if method == "rmvpe": download("rmvpe.pt")
    elif method == "fcpe": download("fcpe.pt")
    elif method == "crepe": download("crepe_full.pth")
    elif method == "crepe-tiny": download("crepe_tiny.pth")
    logger.info(translations["download_predictors_success"])

def check_embedders(hubert):
    if hubert in ["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "Hidden_Rabbit_last", "portuguese_hubert_base"]:
        model_path = os.path.join("assets", "models", "embedders", hubert + '.pt')
        if not os.path.exists(model_path): 
            logger.info(translations["download_hubert"])
            huggingface.HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13") + f"{hubert}.pt", model_path)
            logger.info(translations["download_hubert_success"])

def generate_config(rvc_version, sample_rate, model_path):
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path): shutil.copy(os.path.join("main", "configs", rvc_version, f"{sample_rate}.json"), config_save_path)

def generate_filelist(pitch_guidance, model_path, rvc_version, sample_rate):
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"{rvc_version}_extracted")
    f0_dir, f0nsf_dir = None, None
    
    if pitch_guidance:
        f0_dir = os.path.join(model_path, "f0")
        f0nsf_dir = os.path.join(model_path, "f0_voiced")

    gt_wavs_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
    feature_files = set(name.split(".")[0] for name in os.listdir(feature_dir))
    names = gt_wavs_files & feature_files & set(name.split(".")[0] for name in os.listdir(f0_dir)) & set(name.split(".")[0] for name in os.listdir(f0nsf_dir)) if pitch_guidance else gt_wavs_files & feature_files

    options = []
    mute_base_path = os.path.join("assets", "logs", "mute")
    for name in names:
        options.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|0" if pitch_guidance else f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|0")

    mute_audio_path = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav")
    mute_feature_path = os.path.join(mute_base_path, f"{rvc_version}_extracted", "mute.npy")
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
        output_root1 = os.path.join(exp_dir, "f0")
        output_root2 = os.path.join(exp_dir, "f0_voiced")
        os.makedirs(output_root1, exist_ok=True)
        os.makedirs(output_root2, exist_ok=True)
        return wav_path, output_root1, output_root2

def read_wave(wav_path, normalize = False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000, translations["sr_not_16000"]
    feats = torch.from_numpy(wav).float()

    if config.is_half: feats = feats.half()
    if feats.dim() == 2: feats = feats.mean(-1)

    feats = feats.view(1, -1)
    if normalize: feats = F.layer_norm(feats, feats.shape)
    return feats

def get_device(gpu_index):
    if gpu_index == "cpu": return "cpu"
    try:
        index = int(gpu_index)
        if index < torch.cuda.device_count(): return f"cuda:{index}"
        else: logger.warning(translations["gpu_not_valid"])
    except ValueError:
        logger.warning(translations["gpu_not_valid"])
    return "cpu"

class FeatureInput:
    def __init__(self, sample_rate=16000, hop_size=160, device="cpu"):
        self.fs = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device

    def compute_f0(self, np_arr, f0_method, hop_length):
        if f0_method == "pm": return self.get_pm(np_arr)
        elif f0_method == 'dio': return self.get_pyworld(np_arr, "dio")
        elif f0_method == "crepe": return self.get_crepe(np_arr, int(hop_length))
        elif f0_method == "crepe-tiny": return self.get_crepe(np_arr, int(hop_length), "tiny")
        elif f0_method == "fcpe": return self.get_fcpe(np_arr, int(hop_length))
        elif f0_method == "rmvpe": return self.get_rmvpe(np_arr)
        elif f0_method == "harvest": return self.get_pyworld(np_arr, "harvest")
        else: raise ValueError(translations["method_not_valid"])

    def get_pm(self, x):
        import parselmouth
        f0 = (parselmouth.Sound(x, self.fs).to_pitch_ac(time_step=(160 / 16000 * 1000) / 1000, voicing_threshold=0.6, pitch_floor=50, pitch_ceiling=1100).selected_array["frequency"])
        pad_size = ((x.size // self.hop) - len(f0) + 1) // 2
        if pad_size > 0 or (x.size // self.hop) - len(f0) - pad_size > 0: f0 = np.pad(f0, [[pad_size, (x.size // self.hop) - len(f0) - pad_size]], mode="constant")
        return f0
    
    def get_crepe(self, x, hop_length, model="full"):
        audio = torch.from_numpy(x.astype(np.float32)).to(self.device)
        audio /= torch.quantile(torch.abs(audio), 0.999)
        audio = audio.unsqueeze(0)
        from main.library.predictors.CREPE import predict
        source = predict(audio, self.fs, hop_length, self.f0_min, self.f0_max, model=model, batch_size=hop_length * 2, device=self.device, pad=True).squeeze(0).cpu().float().numpy()
        source[source < 0.001] = np.nan
        return np.nan_to_num(np.interp(np.arange(0, len(source) * (x.size // self.hop), len(source)) / (x.size // self.hop), np.arange(0, len(source)), source))
    
    def get_fcpe(self, x, hop_length):
        import gc
        from main.library.predictors.FCPE import FCPE
        model_fcpe = FCPE(os.path.join("assets", "models", "predictors", "fcpe.pt"), hop_length=int(hop_length), f0_min=self.f0_min, f0_max=self.f0_max, dtype=torch.float32, device=self.device, sample_rate=self.fs, threshold=0.03)
        f0 = model_fcpe.compute_f0(x, p_len=(x.size // self.hop))
        del model_fcpe
        gc.collect()
        return f0

    def get_rmvpe(self, x):
        from main.library.predictors.RMVPE import RMVPE
        return RMVPE(os.path.join("assets", "models", "predictors", "rmvpe.pt"), is_half=False, device=self.device).infer_from_audio(x, thred=0.03)

    def get_pyworld(self, x, model="harvest"):
        import pyworld
        if model == "harvest": f0, t = pyworld.harvest(x.astype(np.double), fs=self.fs, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=1000 * self.hop / self.fs)
        elif model == "dio": f0, t = pyworld.dio(x.astype(np.double), fs=self.fs, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=1000 * self.hop / self.fs)
        else: raise ValueError(translations["method_not_valid"])
        return pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        
    def coarse_f0(self, f0):
        return np.rint(np.clip(((1127 * np.log(1 + f0 / 700)) - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1, 1, self.f0_bin - 1)).astype(int)

    def process_file(self, file_info, f0_method, hop_length):
        inp_path, opt_path1, opt_path2, np_arr = file_info
        if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"): return

        try:
            feature_pit = self.compute_f0(np_arr, f0_method, hop_length)
            np.save(opt_path2, feature_pit, allow_pickle=False)
            np.save(opt_path1, self.coarse_f0(feature_pit), allow_pickle=False)
        except Exception as e:
            raise RuntimeError(f"{translations['extract_file_error']} {inp_path}: {e}")

    def process_files(self, files, f0_method, hop_length, pbar):
        for file_info in files:
            self.process_file(file_info, f0_method, hop_length)
            pbar.update()

def run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, gpus):
    input_root, *output_roots = setup_paths(exp_dir)
    if len(output_roots) == 2: output_root1, output_root2 = output_roots
    else: output_root1, output_root2 = output_roots[0], None

    paths = [(os.path.join(input_root, name), os.path.join(output_root1, name) if output_root1 else None, os.path.join(output_root2, name) if output_root2 else None, load_audio(os.path.join(input_root, name), 16000)) for name in sorted(os.listdir(input_root)) if "spec" not in name]
    logger.info(translations["extract_f0_method"].format(num_processes=num_processes, f0_method=f0_method))
    start_time = time.time()

    if gpus != "-":
        gpus = gpus.split("-")
        process_partials = []
        pbar = tqdm.tqdm(total=len(paths), desc=translations["extract_f0"], ncols=100, unit="p")

        for idx, gpu in enumerate(gpus):
            feature_input = FeatureInput(device=get_device(gpu))
            process_partials.append((feature_input, paths[idx::len(gpus)]))

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(FeatureInput.process_files, feature_input, part_paths, f0_method, hop_length, pbar) for feature_input, part_paths in process_partials]

            for future in as_completed(futures):
                pbar.update(1)
                logger.debug(pbar.format_meter(pbar.n, pbar.total, pbar.format_dict["elapsed"]))
                future.result()
        pbar.close()
    else:
        from functools import partial
        from multiprocessing import Pool
        
        with tqdm.tqdm(total=len(paths), desc=translations["extract_f0"], ncols=100, unit="p") as pbar:
            with Pool(processes=num_processes) as pool:
                for _ in pool.imap_unordered(partial(FeatureInput(device="cpu").process_file, f0_method=f0_method, hop_length=hop_length), paths):
                    pbar.update(1)
                    logger.debug(pbar.format_meter(pbar.n, pbar.total, pbar.format_dict["elapsed"]))
    elapsed_time = time.time() - start_time
    logger.info(translations["extract_f0_success"].format(elapsed_time=f"{elapsed_time:.2f}"))

def process_file_embedding(file, wav_path, out_path, model, device, version, saved_cfg):
    out_file_path = os.path.join(out_path, file.replace("wav", "npy"))
    if os.path.exists(out_file_path): return
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    feats = read_wave(os.path.join(wav_path, file), normalize=saved_cfg.task.normalize).to(dtype).to(device)
    inputs = {
        "source": feats,
        "padding_mask": torch.BoolTensor(feats.shape).fill_(False).to(dtype).to(device),
        "output_layer": 9 if version == "v1" else 12,
    }

    with torch.no_grad():
        model = model.to(device).to(dtype)
        logits = model.extract_features(**inputs)
        feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

    feats = feats.squeeze(0).float().cpu().numpy()
    if not np.isnan(feats).any(): np.save(out_file_path, feats, allow_pickle=False)
    else: logger.warning(f"{file} {translations['NaN']}")

def run_embedding_extraction(exp_dir, version, gpus, embedder_model):
    wav_path, out_path = setup_paths(exp_dir, version)
    logger.info(translations["start_extract_hubert"])
    start_time = time.time()

    try:
        from fairseq import checkpoint_utils
        models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task([os.path.join("assets", "models", "embedders", embedder_model + '.pt')], suffix="")
    except Exception as e:
        raise ImportError(translations["read_model_error"].format(e=e))

    devices = [get_device(gpu) for gpu in (gpus.split("-") if gpus != "-" else ["cpu"])]
    paths = sorted([file for file in os.listdir(wav_path) if file.endswith(".wav")])

    if not paths:
        logger.warning(translations["not_found_audio_file"])
        sys.exit(1)

    pbar = tqdm.tqdm(total=len(paths) * len(devices), desc=translations["extract_hubert"], ncols=100, unit="p")
    for task in [(file, wav_path, out_path, models[0], device, version, saved_cfg) for file in paths for device in devices]:
        try:
            process_file_embedding(*task)
        except Exception as e:
            raise RuntimeError(f"{translations['process_error']} {task[0]}: {e}")
        pbar.update(1)
        logger.debug(pbar.format_meter(pbar.n, pbar.total, pbar.format_dict["elapsed"]))
    pbar.close()
    elapsed_time = time.time() - start_time
    logger.info(translations["extract_hubert_success"].format(elapsed_time=f"{elapsed_time:.2f}"))

if __name__ == "__main__":
    args = parse_arguments()
    exp_dir = os.path.join("assets", "logs", args.model_name)
    f0_method, hop_length, num_processes, gpus, version, pitch_guidance, sample_rate, embedder_model = args.f0_method, args.hop_length, args.cpu_cores, args.gpu, args.rvc_version, args.pitch_guidance, args.sample_rate, args.embedder_model

    check_predictors(f0_method)
    check_embedders(embedder_model)

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

    logger.debug(f"{translations['modelname']}: {args.model_name}")
    logger.debug(f"{translations['export_process']}: {exp_dir}")
    logger.debug(f"{translations['f0_method']}: {f0_method}")
    logger.debug(f"{translations['pretrain_sr']}: {sample_rate}")
    logger.debug(f"{translations['cpu_core']}: {num_processes}")
    logger.debug(f"Gpu: {gpus}")
    if f0_method == "crepe" or f0_method == "crepe-tiny" or f0_method == "fcpe": logger.debug(f"Hop length: {hop_length}")
    logger.debug(f"{translations['training_version']}: {version}")
    logger.debug(f"{translations['extract_f0']}: {pitch_guidance}")
    logger.debug(f"{translations['hubert_model']}: {embedder_model}")

    pid_path = os.path.join(exp_dir, "extract_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    try:
        run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, gpus)
        run_embedding_extraction(exp_dir, version, gpus, embedder_model)
        generate_config(version, sample_rate, exp_dir)
        generate_filelist(pitch_guidance, exp_dir, version, sample_rate)
    except Exception as e:
        logger.error(f"{translations['extract_error']}: {e}")

        import traceback
        logger.debug(traceback.format_exc())

    if os.path.exists(pid_path): os.remove(pid_path)
    logger.info(f"{translations['extract_success']} {args.model_name}.")