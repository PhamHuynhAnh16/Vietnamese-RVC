import os
import gc
import sys
import time
import tqdm
import torch
import shutil
import codecs
import pyworld
import librosa
import logging
import argparse
import warnings
import subprocess
import torchcrepe
import parselmouth
import logging.handlers

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from random import shuffle
from functools import partial
from multiprocessing import Pool
from distutils.util import strtobool
from fairseq import checkpoint_utils
from concurrent.futures import ThreadPoolExecutor, as_completed

now_dir = os.getcwd()
sys.path.append(now_dir)

from main.configs.config import Config
from main.library.predictors.FCPE import FCPE
from main.library.predictors.RMVPE import RMVPE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logger.propagate = False

config = Config()

def parse_arguments() -> tuple:
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

    args = parser.parse_args()
    return args

def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != sample_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    except Exception as error:
        raise RuntimeError(f"Đã xảy ra lỗi khi tải âm thanh: {error}")

    return audio.flatten()

def check_rmvpe_fcpe(method):
    if method == "rmvpe" and not os.path.exists(os.path.join("assets", "model", "predictors", "rmvpe.pt")): subprocess.run(["wget", "--no-check-certificate", codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Pbyno_EIP_Cebwrpg_2/erfbyir/znva/", "rot13") + "rmvpe.pt", "-P", os.path.join("assets", "model", "predictors")], check=True)
    elif method == "fcpe" and not os.path.exists(os.path.join("assets", "model", "predictors", "fcpe.pt")): subprocess.run(["wget", "--no-check-certificate", codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Pbyno_EIP_Cebwrpg_2/erfbyir/znva/", "rot13") + "fcpe.pt", "-P", os.path.join("assets", "model", "predictors")], check=True)

def check_hubert(hubert):
    if hubert == "contentvec_base" or hubert == "hubert_base" or hubert == "japanese_hubert_base" or hubert == "korean_hubert_base" or hubert == "chinese_hubert_base":
        model_path = os.path.join(now_dir, "assets", "model", "embedders", hubert + '.pt')

        if not os.path.exists(model_path): subprocess.run(["wget", "--no-check-certificate", codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Pbyno_EIP_Cebwrpg_2/erfbyir/znva/", "rot13") + f"{hubert}.pt", "-P", os.path.join("assets", "model", "embedders")], check=True)

def generate_config(rvc_version, sample_rate, model_path):
    config_path = os.path.join("main", "configs", rvc_version, f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path): shutil.copy(config_path, config_save_path)


def generate_filelist(pitch_guidance, model_path, rvc_version, sample_rate):
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"{rvc_version}_extracted")

    f0_dir, f0nsf_dir = None, None
    
    if pitch_guidance:
        f0_dir = os.path.join(model_path, "f0")
        f0nsf_dir = os.path.join(model_path, "f0_voiced")

    gt_wavs_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
    feature_files = set(name.split(".")[0] for name in os.listdir(feature_dir))

    if pitch_guidance:
        f0_files = set(name.split(".")[0] for name in os.listdir(f0_dir))
        f0nsf_files = set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
        names = gt_wavs_files & feature_files & f0_files & f0nsf_files
    else: names = gt_wavs_files & feature_files

    options = []
    mute_base_path = os.path.join(now_dir, "assets", "logs", "mute")

    for name in names:
        if pitch_guidance: options.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|0")
        else: options.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|0")

    mute_audio_path = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav")
    mute_feature_path = os.path.join(mute_base_path, f"{rvc_version}_extracted", "mute.npy")

    for _ in range(2):
        if pitch_guidance:
            mute_f0_path = os.path.join(mute_base_path, "f0", "mute.wav.npy")
            mute_f0nsf_path = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")
            options.append(f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|0")
        else: options.append(f"{mute_audio_path}|{mute_feature_path}|0")

    shuffle(options)

    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))

def setup_paths(exp_dir, version = None):
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    if version:
        out_path = os.path.join(exp_dir, "v1_extracted" if version == "v1" else "v2_extracted")
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
    assert sr == 16000, "Tỉ lệ mẫu phải là 16000"

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
        else: logger.warning("Chỉ số GPU không hợp lệ. Chuyển sang CPU.")
    except ValueError:
        logger.warning("Định dạng chỉ mục GPU không hợp lệ. Chuyển sang CPU.")

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
        elif f0_method == 'dio': return self.get_dio(np_arr)
        elif f0_method == "crepe": return self.get_crepe(np_arr, hop_length)
        elif f0_method == "crepe-tiny": return self.get_crepe(np_arr, hop_length, "tiny")
        elif f0_method == "fcpe": return self.get_fcpe(np_arr)
        elif f0_method == "rmvpe": return self.get_rmvpe(np_arr)
        elif f0_method == "harvest": return self.get_harvest(np_arr)
        else: raise ValueError(f"Phương pháp trích xuất F0 không xác định: {f0_method}")


    def get_pm(self, x):
        time_step = 160 / 16000 * 1000
        f0 = (parselmouth.Sound(x, self.fs).to_pitch_ac(time_step=time_step / 1000, voicing_threshold=0.6, pitch_floor=50, pitch_ceiling=1100).selected_array["frequency"])
        pad_size = ((x.size // self.hop) - len(f0) + 1) // 2
        if pad_size > 0 or (x.size // self.hop) - len(f0) - pad_size > 0: f0 = np.pad(f0, [[pad_size, (x.size // self.hop) - len(f0) - pad_size]], mode="constant")

        return f0
    

    def get_dio(self, x):
        f0, t = pyworld.dio(x.astype(np.double), fs=self.fs, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=1000 * self.hop / self.fs)
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        
        return f0
    

    def get_crepe(self, x, hop_length, model="full"):
        audio = torch.from_numpy(x.astype(np.float32)).to(self.device)
        audio /= torch.quantile(torch.abs(audio), 0.999)
        audio = audio.unsqueeze(0)

        pitch = torchcrepe.predict(audio, self.fs, hop_length, self.f0_min, self.f0_max, model=model, batch_size=hop_length * 2, device=self.device, pad=True)

        source = pitch.squeeze(0).cpu().float().numpy()
        source[source < 0.001] = np.nan
        target = np.interp(np.arange(0, len(source) * (x.size // self.hop), len(source)) / (x.size // self.hop), np.arange(0, len(source)), source)

        return np.nan_to_num(target)
    

    def get_fcpe(self, x):
        self.model_fcpe = FCPE(os.path.join("assets", "model", "predictors", "fcpe.pt"), f0_min=self.f0_min, f0_max=self.f0_max, dtype=torch.float32, device=self.device, sample_rate=self.fs, threshold=0.03)
        f0 = self.model_fcpe.compute_f0(x, p_len=(x.size // self.hop))
        del self.model_fcpe
        gc.collect()
        return f0
    

    def get_rmvpe(self, x):
        self.model_rmvpe = RMVPE(os.path.join("assets", "model", "predictors", "rmvpe.pt"), is_half=False, device=self.device)
        return self.model_rmvpe.infer_from_audio(x, thred=0.03)


    def get_harvest(self, x):
        f0, t = pyworld.harvest(x.astype(np.double), fs=self.fs, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=1000 * self.hop / self.fs)
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        return f0
        

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel = np.clip((f0_mel - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1, 1, self.f0_bin - 1)
        return np.rint(f0_mel).astype(int)


    def process_file(self, file_info, f0_method, hop_length):
        inp_path, opt_path1, opt_path2, np_arr = file_info

        if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"): return

        try:
            feature_pit = self.compute_f0(np_arr, f0_method, hop_length)
            np.save(opt_path2, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path1, coarse_pit, allow_pickle=False)
        except Exception as error:
            raise RuntimeError(f"Đã xảy ra lỗi khi giải nén tập tin {inp_path}: {error}")


    def process_files(self, files, f0_method, hop_length, pbar):
        for file_info in files:
            self.process_file(file_info, f0_method, hop_length)
            pbar.update()

def run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, gpus):
    input_root, *output_roots = setup_paths(exp_dir)

    if len(output_roots) == 2: output_root1, output_root2 = output_roots
    else:
        output_root1 = output_roots[0]
        output_root2 = None

    paths = [
        (
            os.path.join(input_root, name),
            os.path.join(output_root1, name) if output_root1 else None,
            os.path.join(output_root2, name) if output_root2 else None,
            load_audio(os.path.join(input_root, name), 16000),
        )
        for name in sorted(os.listdir(input_root))
        if "spec" not in name
    ]

    logger.info(f"Bắt đầu trích xuất cao độ với {num_processes} lõi với phương pháp trích xuất {f0_method}...")
    start_time = time.time()

    if gpus != "-":
        gpus = gpus.split("-")
        num_gpus = len(gpus)

        process_partials = []

        pbar = tqdm.tqdm(total=len(paths), desc="Trích Xuất Cao Độ")

        for idx, gpu in enumerate(gpus):
            device = get_device(gpu)
            feature_input = FeatureInput(device=device)

            part_paths = paths[idx::num_gpus]
            process_partials.append((feature_input, part_paths))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(FeatureInput.process_files, feature_input, part_paths, f0_method, hop_length, pbar) for feature_input, part_paths in process_partials]

            for future in as_completed(futures):
                pbar.update(1)
                future.result()

        pbar.close()
    else:
        feature_input = FeatureInput(device="cpu")
        
        with tqdm.tqdm(total=len(paths), desc="Trích Xuất Cao Độ") as pbar:
            with Pool(processes=num_processes) as pool:
                process_file_partial = partial(feature_input.process_file, f0_method=f0_method, hop_length=hop_length)

                for _ in pool.imap_unordered(process_file_partial, paths):
                    pbar.update(1)


    elapsed_time = time.time() - start_time
    logger.info(f"Quá trình trích xuất cao độ đã hoàn tất vào {elapsed_time:.2f} giây.")

def process_file_embedding(file, wav_path, out_path, model, device, version, saved_cfg):
    wav_file_path = os.path.join(wav_path, file)
    out_file_path = os.path.join(out_path, file.replace("wav", "npy"))

    if os.path.exists(out_file_path): return

    feats = read_wave(wav_file_path, normalize=saved_cfg.task.normalize)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    feats = feats.to(dtype).to(device)

    padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(dtype).to(device)

    inputs = {
        "source": feats,
        "padding_mask": padding_mask,
        "output_layer": 9 if version == "v1" else 12,
    }

    with torch.no_grad():
        model = model.to(device).to(dtype)
        logits = model.extract_features(**inputs)
        feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

    feats = feats.squeeze(0).float().cpu().numpy()

    if not np.isnan(feats).any(): np.save(out_file_path, feats, allow_pickle=False)
    else: logger.warning(f"{file} chứa giá trị NaN và sẽ bị bỏ qua.")

def run_embedding_extraction(exp_dir, version, gpus, embedder_model):
    wav_path, out_path = setup_paths(exp_dir, version)

    logger.info("Đang bắt đầu nhúng trích xuất...")
    start_time = time.time()

    try:
        models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task([os.path.join(now_dir, "assets", "model", "embedders", embedder_model + '.pt')], suffix="")
    except Exception as e:
        raise ImportError(f"Thất bại khi tải mô hình: {e}")

    model = models[0]
    devices = [get_device(gpu) for gpu in (gpus.split("-") if gpus != "-" else ["cpu"])]

    paths = sorted([file for file in os.listdir(wav_path) if file.endswith(".wav")])

    if not paths:
        logger.warning("Không tìm thấy tập tin âm thanh. Hãy chắc chắn rằng bạn đã cung cấp âm thanh chính xác.")
        sys.exit(1)

    pbar = tqdm.tqdm(total=len(paths) * len(devices), desc="Trích xuất nhúng")

    tasks = [(file, wav_path, out_path, model, device, version, saved_cfg) for file in paths for device in devices]

    for task in tasks:
        try:
            process_file_embedding(*task)
        except Exception as error:
            raise RuntimeError(f"Đã xảy ra lỗi khi xử lý {task[0]}: {error}")

        pbar.update(1)

    pbar.close()

    elapsed_time = time.time() - start_time
    logger.info(f"Quá trình trích xuất nhúng đã hoàn tất trong {elapsed_time:.2f} giây.")

if __name__ == "__main__":
    args = parse_arguments()

    exp_dir = os.path.join("assets", "logs", args.model_name)
    f0_method = args.f0_method
    hop_length = args.hop_length
    num_processes = args.cpu_cores
    gpus = args.gpu
    version = args.rvc_version
    pitch_guidance = args.pitch_guidance
    sample_rate = args.sample_rate
    embedder_model = args.embedder_model

    check_rmvpe_fcpe(f0_method)
    check_hubert(embedder_model)

    if len([f for f in os.listdir(os.path.join(exp_dir, "sliced_audios")) if os.path.isfile(os.path.join(exp_dir, "sliced_audios", f))]) < 1 or len([f for f in os.listdir(os.path.join(exp_dir, "sliced_audios_16k")) if os.path.isfile(os.path.join(exp_dir, "sliced_audios_16k", f))]) < 1: raise FileNotFoundError("Không tìm thấy dữ liệu được xử lý, vui lòng xử lý lại âm thanh")

    log_file = os.path.join(exp_dir, "extract.log")

    if logger.hasHandlers(): logger.handlers.clear()
    else:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
        file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    logger.debug(f"Tên mô hình: {args.model_name}")
    logger.debug(f"Đường dẫn trích xuất của mô hình: {exp_dir}")
    logger.debug(f"Phương pháp trích xuất {f0_method}")
    logger.debug(f"Tốc độ lấy mẫu của mô hình: {sample_rate}")
    logger.debug(f"Số lượng lõi trích xuất: {num_processes}")
    logger.debug(f"Gpu: {gpus}")
    if f0_method == "crepe" or f0_method == "crepe-tiny": logger.debug(f"Hop length: {hop_length}")
    logger.debug(f"Phiên bản của mô hình: {version}")
    logger.debug(f"Trích xuất cao độ: {pitch_guidance}")
    logger.debug(f"Mô hình học cách nói: {embedder_model}")

    try:
        run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, gpus)
        run_embedding_extraction(exp_dir, version, gpus, embedder_model)
        
        generate_config(version, sample_rate, exp_dir)
        generate_filelist(pitch_guidance, exp_dir, version, sample_rate)
    except Exception as e:
        logger.error(f"Đã xảy ra lỗi khi trích xuất dữ liệu: {e}")

    logger.info(f"Đã trích xuất thành công mô hình {args.model_name}.")