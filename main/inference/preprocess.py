import os
import sys
import time
import torch
import logging
import librosa
import argparse
import logging.handlers

import numpy as np
import soundfile as sf
import multiprocessing
import noisereduce as nr

from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
from distutils.util import strtobool
from concurrent.futures import ProcessPoolExecutor, as_completed

now_directory = os.getcwd()
sys.path.append(now_directory)

from main.configs.config import Config

logger = logging.getLogger(__name__)


logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

OVERLAP = 0.3
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000


def parse_arguments() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="./dataset")
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--cpu_cores", type=int, default=2)
    parser.add_argument("--cut_preprocess", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--process_effects", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_dataset", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)

    args = parser.parse_args()
    return args


config = Config()
per = 3.0 if config.is_half else 3.7


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != sample_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    except Exception as error:
        raise RuntimeError(f"Đã xảy ra lỗi khi tải âm thanh: {error}")

    return audio.flatten()

class Slicer:
    def __init__(self, sr, threshold = -40.0, min_length = 5000, min_interval = 300, hop_size = 20, max_sil_kept = 5000):
        if not min_length >= min_interval >= hop_size: raise ValueError("min_length lớn hơn hoặc bằng min_interval lớn hơn hoặc bằng hop_size là bắt buộc")
        if not max_sil_kept >= hop_size: raise ValueError("max_sil_kept lớn hơn hoặc bằng hop_size là bắt buộc")

        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        start_idx = begin * self.hop_size

        if len(waveform.shape) > 1:
            end_idx = min(waveform.shape[1], end * self.hop_size)
            return waveform[:, start_idx:end_idx]
        else:
            end_idx = min(waveform.shape[0], end * self.hop_size)
            return waveform[start_idx:end_idx]

    def slice(self, waveform):
        samples = waveform.mean(axis=0) if len(waveform.shape) > 1 else waveform
        if samples.shape[0] <= self.min_length: return [waveform]

        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)

        sil_tags = []
        silence_start, clip_start = None, 0

        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None: silence_start = i
                continue

            if silence_start is None: continue

            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (i - silence_start >= self.min_interval and i - clip_start >= self.min_length)

            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0: sil_tags.append((0, pos))
                else: sil_tags.append((pos, pos))

                clip_start = pos

            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept : silence_start + self.max_sil_kept + 1].argmin()

                pos += i - self.max_sil_kept

                pos_l = (rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start)
                pos_r = (rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept)

                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start)
                pos_r = (rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept)

                if silence_start == 0: sil_tags.append((0, pos_r))
                else: sil_tags.append((pos_l, pos_r))

                clip_start = pos_r
            silence_start = None

        total_frames = rms_list.shape[0]

        if (silence_start is not None and total_frames - silence_start >= self.min_interval):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        if not sil_tags: return [waveform]
        else:
            chunks = []

            if sil_tags[0][0] > 0: chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))

            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))

            if sil_tags[-1][1] < total_frames: chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))

            return chunks


def get_rms(y, frame_length=2048, hop_length=512, pad_mode="constant"):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)

    target_axis = axis - 1 if axis < 0 else axis + 1


    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)


class PreProcess:
    def __init__(self, sr, exp_dir, per):
        self.slicer = Slicer(sr=sr, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
        self.sr = sr
        self.b_high, self.a_high = signal.butter(N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr)
        self.per = per
        self.exp_dir = exp_dir
        self.device = "cpu"
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")

        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def _normalize_audio(self, audio: torch.Tensor):
        tmp_max = torch.abs(audio).max()
        if tmp_max > 2.5: return None
        
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    def process_audio_segment(self, normalized_audio: np.ndarray, sid, idx0, idx1):
        if normalized_audio is None:
            logs(f"{sid}-{idx0}-{idx1}-filtered")
            return
        
        wavfile.write(os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{idx1}.wav"), self.sr, normalized_audio.astype(np.float32))
        audio_16k = librosa.resample(normalized_audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K)
        wavfile.write(os.path.join(self.wavs16k_dir, f"{sid}_{idx0}_{idx1}.wav"), SAMPLE_RATE_16K, audio_16k.astype(np.float32))

    def process_audio(self, path, idx0, sid, cut_preprocess, process_effects, clean_dataset, clean_strength):
        try:
            audio = load_audio(path, self.sr)

            if process_effects: 
                audio = signal.lfilter(self.b_high, self.a_high, audio)
                audio = self._normalize_audio(audio)

            if clean_dataset: audio = nr.reduce_noise(y=audio, sr=self.sr, prop_decrease=clean_strength)

            idx1 = 0

            if cut_preprocess:
                for audio_segment in self.slicer.slice(audio):
                    i = 0

                    while 1:
                        start = int(self.sr * (self.per - OVERLAP) * i)
                        i += 1

                        if len(audio_segment[start:]) > (self.per + OVERLAP) * self.sr:
                            tmp_audio = audio_segment[start : start + int(self.per * self.sr)]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1)
                            idx1 += 1
                        else:
                            tmp_audio = audio_segment[start:]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1)
                            idx1 += 1
                            break
                        
            else: self.process_audio_segment(audio, sid, idx0, idx1)
        except Exception as error:
            raise RuntimeError(f"Đã xảy ra lỗi khi xử lý âm thanh: {error}")

def process_file(args):
    pp, file, cut_preprocess, process_effects, clean_dataset, clean_strength = (args)
    file_path, idx0, sid = file

    pp.process_audio(file_path, idx0, sid, cut_preprocess, process_effects, clean_dataset, clean_strength)

def preprocess_training_set(input_root, sr, num_processes, exp_dir, per, cut_preprocess, process_effects, clean_dataset, clean_strength):
    start_time = time.time()

    pp = PreProcess(sr, exp_dir, per)
    logger.info(f"Đang bắt đầu xử lý dữ liệu với {num_processes} lõi xử lý...")

    files = []
    idx = 0

    for root, _, filenames in os.walk(input_root):
        try:
            sid = 0 if root == input_root else int(os.path.basename(root))
            for f in filenames:
                if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    files.append((os.path.join(root, f), idx, sid))
                    idx += 1
        except ValueError:
            raise ValueError(f'Thư mục ID giọng nói phải là số nguyên, thay vào đó có "{os.path.basename(root)}".')

    with tqdm(total=len(files), desc="Xử lý dữ liệu", unit="iB", unit_scale=True) as pbar:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(
                    process_file,
                    (
                        pp,
                        file,
                        cut_preprocess,
                        process_effects,
                        clean_dataset,
                        clean_strength,
                    ),
                )
                for file in files
            ]
            for future in as_completed(futures):
                try:
                    future.result() 
                except Exception as e:
                    raise RuntimeError(f"Đã xảy ra lỗi khi xử lý: {e}")

                pbar.update(1)

    elapsed_time = time.time() - start_time
    logger.info(f"Quá trình xử lý hoàn tất sau {elapsed_time:.2f} giây.")

if __name__ == "__main__":
    args = parse_arguments()

    experiment_directory = os.path.join("assets", "logs", args.model_name)
    num_processes = args.cpu_cores
    num_processes = multiprocessing.cpu_count() if num_processes is None else int(num_processes)
    dataset = args.dataset_path
    sample_rate = args.sample_rate
    cut_preprocess = args.cut_preprocess
    preprocess_effects = args.process_effects
    clean_dataset = args.clean_dataset
    clean_strength = args.clean_strength

    os.makedirs(experiment_directory, exist_ok=True)

    if len([f for f in os.listdir(os.path.join(dataset)) if os.path.isfile(os.path.join(dataset, f)) and f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]) < 1: raise FileNotFoundError("Không tìm thấy dữ liệu")

    log_file = os.path.join(experiment_directory, "preprocess.log")

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
    logger.debug(f"Đường dẫn xử lý của mô hình: {experiment_directory}")
    logger.debug(f"Đường dẫn chứa dữ liệu huấn luyện {dataset}")
    logger.debug(f"Tốc độ lấy mẫu của mô hình: {sample_rate}")
    logger.debug(f"Số lượng lõi xử lý: {num_processes}")
    logger.debug(f"Cắt xử lý trước: {cut_preprocess}")
    logger.debug(f"Hiệu ứng xử lý trước: {preprocess_effects}")
    logger.debug(f"Làm sạch dữ liệu xử lý: {clean_dataset}")

    if clean_dataset: logger.debug(f"Mức độ làm sạch dữ liệu xử lý: {clean_strength}")
    
    try:
        preprocess_training_set(dataset, sample_rate, num_processes, experiment_directory, per, cut_preprocess, preprocess_effects, clean_dataset, clean_strength)
    except Exception as e:
        logger.error(f"Đã xảy ra lỗi khi xử lý âm thanh {e}")

    logger.info(f"Đã hoàn thành xử lý trước dữ liệu cho mô hình {args.model_name}")