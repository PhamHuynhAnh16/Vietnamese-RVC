import gc
import re
import os
import sys
import time
import torch
import faiss
import shutil
import codecs
import pyworld
import librosa
import logging
import argparse
import warnings
import traceback
import torchcrepe
import subprocess
import parselmouth
import logging.handlers

import numpy as np
import soundfile as sf
import noisereduce as nr
import torch.nn.functional as F
import torch.multiprocessing as mp

from tqdm import tqdm
from scipy import signal
from torch import Tensor
from scipy.io import wavfile
from audio_upscaler import upscale
from distutils.util import strtobool
from fairseq import checkpoint_utils
from pydub import AudioSegment, silence

now_dir = os.getcwd()
sys.path.append(now_dir)

from main.configs.config import Config
from main.library.predictors.FCPE import FCPE
from main.library.predictors.RMVPE import RMVPE
from main.library.algorithm.synthesizers import Synthesizer


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("wget").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)


FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48  
SAMPLE_RATE = 16000  

bh, ah = signal.butter(N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE)
input_audio_path2wav = {}

log_file = os.path.join("assets", "logs", "convert.log")

logger = logging.getLogger(__name__)
logger.propagate = False


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


def parse_arguments() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--filter_radius", type=int, default=3)
    parser.add_argument("--index_rate", type=float, default=0.5)
    parser.add_argument("--volume_envelope", type=float, default=1)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--hop_length", type=int, default=64)
    parser.add_argument( "--f0_method", type=str, default="rmvpe")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./audios/output.wav")
    parser.add_argument("--pth_path",  type=str,  required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_autotune_strength", type=float, default=1)
    parser.add_argument("--clean_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)
    parser.add_argument("--export_format", type=str, default="wav")
    parser.add_argument("--embedder_model", type=str, default="contentvec_base")
    parser.add_argument("--upscale_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--resample_sr", type=int, default=0)
    parser.add_argument("--batch_process", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--split_audio", type=lambda x: bool(strtobool(x)), default=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    pitch = args.pitch
    filter_radius = args.filter_radius
    index_rate = args.index_rate
    volume_envelope = args.volume_envelope 
    protect = args.protect
    hop_length = args.hop_length 
    f0_method = args.f0_method 
    input_path = args.input_path 
    output_path = args.output_path 
    pth_path = args.pth_path 
    index_path = args.index_path 
    f0_autotune = args.f0_autotune 
    f0_autotune_strength = args.f0_autotune_strength 
    clean_audio = args.clean_audio 
    clean_strength = args.clean_strength 
    export_format = args.export_format 
    embedder_model = args.embedder_model 
    upscale_audio = args.upscale_audio 
    resample_sr = args.resample_sr 
    batch_process = args.batch_process 
    batch_size = args.batch_size 
    split_audio = args.split_audio

    logger.debug(f"Cao độ giọng nói: {pitch}")
    logger.debug(f"Lọc trung vị: {filter_radius}")
    logger.debug(f"Ảnh hưởng của chỉ mục: {index_rate}")
    logger.debug(f"Đường bao âm thanh: {volume_envelope}")
    logger.debug(f"Bảo vệ phụ âm: {protect}")
    if f0_method == "crepe" or f0_method == "crepe-tiny": logger.debug(f"Hop length: {hop_length}")
    logger.debug(f"Phương pháp trích xuất âm thanh: {f0_method}")
    logger.debug(f"Đường dẫn đầu vào âm thanh: {input_path}")
    logger.debug(f"Đường dẫn đầu ra ân thanh: {output_path.replace('.wav', f'.{export_format}')}")
    logger.debug(f"Đường dẫn tệp tin mô hình: {pth_path}")
    logger.debug(f"Đường dẫn tệp tin chỉ mục: {index_path}")
    logger.debug(f"Tự động điều chỉnh trích xuất: {f0_autotune}")
    logger.debug(f"Làm sạch âm thanh: {clean_audio}")
    if clean_audio: logger.debug(f"Mức độ làm sạch âm thanh: {clean_strength}")
    logger.debug(f"Định dạng âm thanh đầu ra: {export_format}")
    logger.debug(f"Mô hình học cách nói: {embedder_model}")
    logger.debug(f"Tăng chất lượng âm thanh: {upscale_audio}")
    if resample_sr != 0: logger.debug(f"Tỷ lệ lấy mẫu lại: {resample_sr}")
    if split_audio: logger.debug(f"Sử dụng xử lý đa luồng: {batch_process}")
    if batch_process and split_audio: logger.debug(f"Số luồng xử lý cùng lúc: {batch_size}")
    logger.debug(f"Cắt nhỏ âm thanh: {split_audio}")
    if f0_autotune: logger.debug(f"Mức độ điều chỉnh trích xuất: {f0_autotune_strength}")


    check_rmvpe_fcpe(f0_method)
    check_hubert(embedder_model)

    run_convert_script(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, input_path=input_path, output_path=output_path, pth_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, upscale_audio=upscale_audio, resample_sr=resample_sr, batch_process=batch_process, batch_size=batch_size, split_audio=split_audio)


def check_rmvpe_fcpe(method):
    def download_rmvpe():
        if not os.path.exists(os.path.join("assets", "model", "predictors", "rmvpe.pt")): subprocess.run(["wget", "-q", "--show-progress", "--no-check-certificate", codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Pbyno_EIP_Cebwrpg_2/erfbyir/znva/", "rot13") + "rmvpe.pt", "-P", os.path.join("assets", "model", "predictors")], check=True)

    def download_fcpe():
        if not os.path.exists(os.path.join("assets", "model", "predictors", "fcpe.pt")): subprocess.run(["wget", "-q", "--show-progress", "--no-check-certificate", codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Pbyno_EIP_Cebwrpg_2/erfbyir/znva/", "rot13") + "fcpe.pt", "-P", os.path.join("assets", "model", "predictors")], check=True)

    if method == "rmvpe": download_rmvpe()
    elif method == "fcpe": download_fcpe()
    elif "hybrid" in method:
        methods_str = re.search("hybrid\[(.+)\]", method)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        for method in methods:
            if method == "rmvpe": download_rmvpe()
            elif method == "fcpe": download_fcpe()


def check_hubert(hubert):
    if hubert == "contentvec_base" or hubert == "hubert_base" or hubert == "japanese_hubert_base" or hubert == "korean_hubert_base" or hubert == "chinese_hubert_base":
        model_path = os.path.join(now_dir, "assets", "model", "embedders", hubert + '.pt')

        if not os.path.exists(model_path): subprocess.run(["wget", "-q", "--show-progress", "--no-check-certificate", codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Pbyno_EIP_Cebwrpg_2/erfbyir/znva/", "rot13") + f"{hubert}.pt", "-P", os.path.join("assets", "model", "embedders")], check=True)


def load_audio_infer(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file): raise FileNotFoundError(f"Không tìm thấy tệp: {file}")

        audio, sr = sf.read(file)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != sample_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    except Exception as e:
        raise RuntimeError(f"Đã xảy ra lỗi khi tải âm thanh: {e}") 
     
    return audio.flatten()


def process_audio(file_path, output_path):
    try:
        song = AudioSegment.from_file(file_path)
        nonsilent_parts = silence.detect_nonsilent(song, min_silence_len=750, silence_thresh=-70)

        cut_files = []
        time_stamps = []

        min_chunk_duration = 30

        for i, (start_i, end_i) in enumerate(nonsilent_parts):
            chunk = song[start_i:end_i]

            if len(chunk) >= min_chunk_duration:
                chunk_file_path = os.path.join(output_path, f"chunk{i}.wav")

                if os.path.exists(chunk_file_path): os.remove(chunk_file_path)
                chunk.export(chunk_file_path, format="wav")

                cut_files.append(chunk_file_path)
                time_stamps.append((start_i, end_i))
            else: logger.debug(f"Phần {i} được bỏ qua vì quá ngắn: {len(chunk)}ms")

        logger.info(f"Tổng số phần đã cắt: {len(cut_files)}")
        return cut_files, time_stamps
    except Exception as e:
        raise RuntimeError(f"Đã xảy ra lỗi khi cắt âm thanh: {e}")


def merge_audio(files_list, time_stamps, original_file_path, output_path, format):
    try:
        def extract_number(filename):
            match = re.search(r'_(\d+)', filename)
            return int(match.group(1)) if match else 0

        files_list = sorted(files_list, key=extract_number)
        total_duration = len(AudioSegment.from_file(original_file_path))

        combined = AudioSegment.empty() 
        current_position = 0 

        for file, (start_i, end_i) in zip(files_list, time_stamps):
            if start_i > current_position:
                silence_duration = start_i - current_position
                combined += AudioSegment.silent(duration=silence_duration)  

            combined += AudioSegment.from_file(file)  
            current_position = end_i

        if current_position < total_duration: combined += AudioSegment.silent(duration=total_duration - current_position)

        combined.export(output_path, format=format)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Đã xảy ra lỗi khi ghép âm thanh: {e}")


def run_batch_convert(params):
    cvt = VoiceConverter()

    path = params["path"]
    audio_temp = params["audio_temp"]
    export_format = params["export_format"]
    cut_files = params["cut_files"]
    pitch = params["pitch"]
    filter_radius = params["filter_radius"]
    index_rate = params["index_rate"]
    volume_envelope = params["volume_envelope"]
    protect = params["protect"]
    hop_length = params["hop_length"]
    f0_method = params["f0_method"]
    pth_path = params["pth_path"]
    index_path = params["index_path"]
    f0_autotune = params["f0_autotune"]
    f0_autotune_strength = params["f0_autotune_strength"]
    clean_audio = params["clean_audio"]
    clean_strength = params["clean_strength"]
    upscale_audio = params["upscale_audio"]
    embedder_model = params["embedder_model"]
    resample_sr = params["resample_sr"]
    processed_segments = params["processed_segments"]
    
    segment_output_path = os.path.join(audio_temp, f"output_{cut_files.index(path)}.{export_format}")
    if os.path.exists(segment_output_path): os.remove(segment_output_path)

    cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=path, audio_output_path=segment_output_path, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, upscale_audio=upscale_audio, embedder_model=embedder_model, resample_sr=resample_sr)
    os.remove(path)

    if os.path.exists(segment_output_path): processed_segments.append(segment_output_path)
    else: 
        logger.warning(f"Không tìm thấy tệp đã xử lý: {segment_output_path}")
        sys.exit(1)
 

def run_convert_script(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, f0_autotune_strength, clean_audio, clean_strength, export_format, upscale_audio, embedder_model, resample_sr, batch_process, batch_size, split_audio):
    cvt = VoiceConverter()
    start_time = time.time()

    if not pth_path or not os.path.exists(pth_path) or os.path.isdir(pth_path) or not pth_path.endswith(".pth"):
        logger.warning("Mô hình không hợp lệ!")
        sys.exit(1)
    
    if not index_path or not os.path.exists(index_path) or os.path.isdir(index_path) or not index_path.endswith(".index"):
        logger.warning("Chỉ mục không hợp lệ!")
        sys.exit(1)

    output_dir = os.path.dirname(output_path)
    output_dir = output_path if not output_dir else output_dir

    if output_dir is None: output_dir = "audios"

    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    mp.set_start_method("spawn", force=True)

    audio_temp = os.path.join("audios_temp")
    if not os.path.exists(audio_temp) and split_audio: os.makedirs(audio_temp, exist_ok=True)
    
    if os.path.isdir(input_path):
        try:
            logger.info(f"Chuyển đổi hàng loạt...")

            audio_files = [f for f in os.listdir(input_path) if f.endswith(("wav","mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]
            if not audio_files: 
                logger.warning("Không tìm thấy tệp âm thanh!")
                sys.exit(1)

            logger.info(f"Tìm thấy {len(audio_files)} tệp âm thanh cho việc chuyển đổi.")

            for audio in audio_files:
                audio_path = os.path.join(input_path, audio)
                output_audio = os.path.join(input_path, os.path.splitext(audio)[0] + f"_output.{export_format}")

                if split_audio:
                    try:
                        cut_files, time_stamps = process_audio(audio_path, audio_temp)
                        processed_segments = []

                        params_list = [
                            {
                                "path": path,
                                "audio_temp": audio_temp,
                                "export_format": export_format,
                                "cut_files": cut_files,
                                "pitch": pitch,
                                "filter_radius": filter_radius,
                                "index_rate": index_rate,
                                "volume_envelope": volume_envelope,
                                "protect": protect,
                                "hop_length": hop_length,
                                "f0_method": f0_method,
                                "pth_path": pth_path,
                                "index_path": index_path,
                                "f0_autotune": f0_autotune,
                                "f0_autotune_strength": f0_autotune_strength,
                                "clean_audio": clean_audio,
                                "clean_strength": clean_strength,
                                "upscale_audio": upscale_audio,
                                "embedder_model": embedder_model,
                                "resample_sr": resample_sr,
                                "processed_segments": processed_segments,
                            }
                            for path in cut_files
                        ]


                        if batch_process:
                            num_threads = min(batch_size, len(cut_files))

                            with mp.Pool(processes=num_threads) as pool:
                                with tqdm(total=len(params_list), desc="Chuyển Đổi Âm Thanh", unit="iB", unit_scale=True) as pbar:
                                    for _ in pool.imap_unordered(run_batch_convert, params_list):
                                        pbar.update(1)
                        else: 
                            for params in tqdm(params_list, desc="Chuyển Đổi Âm Thanh", unit="iB", unit_scale=True):
                                run_batch_convert(params)

                        merge_audio(processed_segments, time_stamps, audio_path, output_audio, export_format)
                    except Exception as e:
                        logger.error(f"Đã xảy ra lỗi khi chuyển đổi các đoạn âm thanh cắt: {e}")
                    finally:
                        if os.path.exists(audio_temp): shutil.rmtree(audio_temp, ignore_errors=True)
                else:
                    try:
                        logger.info(f"Chuyển đổi âm thanh '{audio_path}'...")

                        if os.path.exists(output_audio): os.remove(output_audio)

                        with tqdm(total=1, desc="Chuyển Đổi Âm Thanh", unit="iB", unit_scale=True) as pbar:
                            cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=audio_path, audio_output_path=output_audio, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, upscale_audio=upscale_audio, embedder_model=embedder_model, resample_sr=resample_sr)
                            pbar.update(1)
                    except Exception as e:
                        logger.error(f"Đã xảy ra lỗi khi chuyển đổi âm thanh: {e}")

            elapsed_time = time.time() - start_time
            logger.info(f"Đã chuyển đổi hàng loạt thành công sau {elapsed_time:.2f} giây. {output_path.replace('.wav', f'.{export_format}')}")
        except Exception as e:
            logger.error(f"Đã xảy ra lỗi khi chuyển đổi âm thanh hàng loạt: {e}")
    else:
        logger.info(f"Chuyển đổi âm thanh '{input_path}'...")

        if not os.path.exists(input_path):
            logger.warning("Không tìm thấy tệp âm thanh!")
            sys.exit(1)

        if split_audio:
            try:              
                cut_files, time_stamps = process_audio(input_path, audio_temp)
                processed_segments = []

                params_list = [
                    {
                        "path": path,
                        "audio_temp": audio_temp,
                        "export_format": export_format,
                        "cut_files": cut_files,
                        "pitch": pitch,
                        "filter_radius": filter_radius,
                        "index_rate": index_rate,
                        "volume_envelope": volume_envelope,
                        "protect": protect,
                        "hop_length": hop_length,
                        "f0_method": f0_method,
                        "pth_path": pth_path,
                        "index_path": index_path,
                        "f0_autotune": f0_autotune,
                        "f0_autotune_strength": f0_autotune_strength,
                        "clean_audio": clean_audio,
                        "clean_strength": clean_strength,
                        "upscale_audio": upscale_audio,
                        "embedder_model": embedder_model,
                        "resample_sr": resample_sr,
                        "processed_segments": processed_segments,
                    }
                    for path in cut_files
                ]

                if batch_process:
                    num_threads = min(batch_size, len(cut_files))

                    with mp.Pool(processes=num_threads) as pool:
                        with tqdm(total=len(params_list), desc="Chuyển Đổi Âm Thanh", unit="iB", unit_scale=True) as pbar:
                            for _ in pool.imap_unordered(run_batch_convert, params_list):
                                pbar.update(1)
                else: 
                    for params in tqdm(params_list, desc="Chuyển Đổi Âm Thanh", unit="iB", unit_scale=True):
                        run_batch_convert(params)

                merge_audio(processed_segments, time_stamps, input_path, output_path.replace(".wav", f".{export_format}"), export_format)
            except Exception as e:
                logger.error(f"Đã xảy ra lỗi khi chuyển đổi các đoạn âm thanh cắt: {e}")
            finally:
                if os.path.exists(audio_temp): shutil.rmtree(audio_temp, ignore_errors=True)
        else:
            try:
                if os.path.exists(output_path): os.remove(output_path)

                with tqdm(total=1, desc="Chuyển Đổi Âm Thanh", unit="iB", unit_scale=True) as pbar:
                    cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=input_path, audio_output_path=output_path, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, upscale_audio=upscale_audio, embedder_model=embedder_model, resample_sr=resample_sr)
                    pbar.update(1)
            except Exception as e:
                logger.error(f"Đã xảy ra lỗi khi chuyển đổi âm thanh: {e}")

        elapsed_time = time.time() - start_time
        logger.info(f"Tệp {input_path} được chuyển đổi thành công sau {elapsed_time:.2f} giây. {output_path.replace('.wav', f'.{export_format}')}")


def change_rms(source_audio: np.ndarray, source_rate: int, target_audio: np.ndarray, target_rate: int, rate: float) -> np.ndarray:
    rms1 = librosa.feature.rms(
        y=source_audio,
        frame_length=source_rate // 2 * 2,
        hop_length=source_rate // 2,
    )
    
    rms2 = librosa.feature.rms(
        y=target_audio,
        frame_length=target_rate // 2 * 2,
        hop_length=target_rate // 2,
    )

    rms1 = F.interpolate(
        torch.from_numpy(rms1).float().unsqueeze(0),
        size=target_audio.shape[0],
        mode="linear",
    ).squeeze()

    rms2 = F.interpolate(
        torch.from_numpy(rms2).float().unsqueeze(0),
        size=target_audio.shape[0],
        mode="linear",
    ).squeeze()

    rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

    adjusted_audio = (target_audio * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy())
    return adjusted_audio


class Autotune:
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs


    def autotune_f0(self, f0, f0_autotune_strength):
        autotuned_f0 = np.zeros_like(f0)

        for i, freq in enumerate(f0):
            closest_note = min(self.note_dict, key=lambda x: abs(x - freq))
            autotuned_f0[i] = freq + (closest_note - freq) * f0_autotune_strength

        return autotuned_f0


class VC:
    def __init__(self, tgt_sr, config):
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.is_half = config.is_half
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = config.device
        self.ref_freqs = [
            49.00,  
            51.91,  
            55.00, 
            58.27,  
            61.74,  
            65.41, 
            69.30, 
            73.42, 
            77.78,  
            82.41, 
            87.31, 
            92.50, 
            98.00,  
            103.83,  
            110.00, 
            116.54, 
            123.47, 
            130.81, 
            138.59,  
            146.83,  
            155.56,  
            164.81, 
            174.61, 
            185.00,  
            196.00,  
            207.65, 
            220.00,  
            233.08,  
            246.94, 
            261.63, 
            277.18,  
            293.66, 
            311.13, 
            329.63,  
            349.23, 
            369.99, 
            392.00, 
            415.30,  
            440.00,  
            466.16,  
            493.88, 
            523.25,  
            554.37, 
            587.33,  
            622.25, 
            659.25, 
            698.46, 
            739.99,  
            783.99,  
            830.61, 
            880.00, 
            932.33,  
            987.77, 
            1046.50
        ]
        self.autotune = Autotune(self.ref_freqs)
        self.note_dict = self.autotune.note_dict


    def get_f0_crepe(self, x, f0_min, f0_max, p_len, hop_length, model="full"):
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)

        audio = torch.from_numpy(x).to(self.device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)

        if audio.ndim == 2 and audio.shape[0] > 1: audio = torch.mean(audio, dim=0, keepdim=True).detach()

        audio = audio.detach()
        pitch: Tensor = torchcrepe.predict(audio, self.sample_rate, hop_length, f0_min, f0_max, model, batch_size=hop_length * 2, device=self.device, pad=True)

        p_len = p_len or x.shape[0] // hop_length
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )

        f0 = np.nan_to_num(target)
        return f0


    def get_f0_hybrid(self, methods_str, x, f0_min, f0_max, p_len, hop_length, filter_radius):
        methods_str = re.search("hybrid\[(.+)\]", methods_str)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        f0_computation_stack = []
        logger.debug(f"Tính toán ước lượng cao độ f0 cho các phương pháp {str(methods)}")

        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)


        for method in methods:
            f0 = None

            if method == "pm":
                f0 = (parselmouth.Sound(x, self.sample_rate).to_pitch_ac(time_step=self.window / self.sample_rate * 1000 / 1000, voicing_threshold=0.6, pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array["frequency"])
                pad_size = (p_len - len(f0) + 1) // 2

                if pad_size > 0 or p_len - len(f0) - pad_size > 0: f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
            elif method == 'dio':
                f0, t = pyworld.dio(x.astype(np.double), fs=self.sample_rate, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=10)
                f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sample_rate)

                f0 = signal.medfilt(f0, 3)
            elif method == "crepe-tiny":
                f0 = self.get_f0_crepe(x, self.f0_min, self.f0_max, p_len, int(hop_length), "tiny")
            elif method == "crepe": 
                f0 = self.get_f0_crepe(x, f0_min, f0_max, p_len, int(hop_length))
            elif method == "fcpe":
                self.model_fcpe = FCPE(os.path.join("assets", "model", "predictors", "fcpe.pt"), hop_length=int(hop_length), f0_min=int(f0_min), f0_max=int(f0_max), dtype=torch.float32, device=self.device, sample_rate=self.sample_rate, threshold=0.03)
                f0 = self.model_fcpe.compute_f0(x, p_len=p_len)

                del self.model_fcpe
                gc.collect() 
            elif method == "rmvpe":
                f0 = RMVPE(os.path.join("assets", "model", "predictors", "rmvpe.pt"), is_half=self.is_half, device=self.device).infer_from_audio(x, thred=0.03)
                f0 = f0[1:]
            elif method == "harvest":
                f0, t = pyworld.harvest(x.astype(np.double), fs=self.sample_rate, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=10)
                f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sample_rate)

                if filter_radius > 2: f0 = signal.medfilt(f0, 3)
            else: raise ValueError("Phương pháp không hợp lệ")
   
            f0_computation_stack.append(f0)
            
        resampled_stack = []

        for f0 in f0_computation_stack:
            resampled_f0 = np.interp(np.linspace(0, len(f0), p_len), np.arange(len(f0)), f0)
            resampled_stack.append(resampled_f0)

        f0_median_hybrid = resampled_stack[0] if len(resampled_stack) == 1 else np.nanmedian(np.vstack(resampled_stack), axis=0)
        return f0_median_hybrid


    def get_f0(self, input_audio_path, x, p_len, pitch, f0_method, filter_radius, hop_length, f0_autotune, f0_autotune_strength):
        global input_audio_path2wav
        

        if f0_method == "pm":
            f0 = (parselmouth.Sound(x, self.sample_rate).to_pitch_ac(time_step=self.window / self.sample_rate * 1000 / 1000, voicing_threshold=0.6, pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array["frequency"])
            pad_size = (p_len - len(f0) + 1) // 2

            if pad_size > 0 or p_len - len(f0) - pad_size > 0: f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        elif f0_method == "dio":
            f0, t = pyworld.dio(x.astype(np.double), fs=self.sample_rate, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=10)
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sample_rate)

            f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe-tiny":
            f0 = self.get_f0_crepe(x, self.f0_min, self.f0_max, p_len, int(hop_length), "tiny")
        elif f0_method == "crepe":
            f0 = self.get_f0_crepe(x, self.f0_min, self.f0_max, p_len, int(hop_length))
        elif f0_method == "fcpe":
            self.model_fcpe = FCPE(os.path.join("assets", "model", "predictors", "fcpe.pt"), hop_length=int(hop_length), f0_min=int(self.f0_min), f0_max=int(self.f0_max), dtype=torch.float32, device=self.device, sample_rate=self.sample_rate, threshold=0.03)
            f0 = self.model_fcpe.compute_f0(x, p_len=p_len)

            del self.model_fcpe
            gc.collect()
        elif f0_method == "rmvpe":
            f0 = RMVPE(os.path.join("assets", "model", "predictors", "rmvpe.pt"), is_half=self.is_half, device=self.device).infer_from_audio(x, thred=0.03)
        elif f0_method == "harvest":
            f0, t = pyworld.harvest(x.astype(np.double), fs=self.sample_rate, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=10)
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sample_rate)

            if filter_radius > 2: f0 = signal.medfilt(f0, 3)
        elif "hybrid" in f0_method:
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = self.get_f0_hybrid(f0_method, x, self.f0_min, self.f0_max, p_len, hop_length, filter_radius)
        else: raise ValueError("Method không hợp lệ")

        if f0_autotune: f0 = Autotune.autotune_f0(self, f0, f0_autotune_strength)

        f0 *= pow(2, pitch / 12)

        f0bak = f0.copy()

        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255

        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak


    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, index_rate, version, protect):
        pitch_guidance = pitch != None and pitchf != None

        feats = (torch.from_numpy(audio0).half() if self.is_half else torch.from_numpy(audio0).float())

        if feats.dim() == 2: feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()

        feats = feats.view(1, -1)

        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }

        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

        if protect < 0.5 and pitch_guidance: feats0 = feats.clone()

        if (not isinstance(index, type(None)) and not isinstance(big_npy, type(None)) and index_rate != 0):
            npy = feats[0].cpu().numpy()

            if self.is_half: npy = npy.astype("float32")

            score, ix = index.search(npy, k=8)

            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)

            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half: npy = npy.astype("float16")

            feats = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats)

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        if protect < 0.5 and pitch_guidance: feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        p_len = audio0.shape[0] // self.window

        if feats.shape[1] < p_len:
            p_len = feats.shape[1]

            if pitch_guidance:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch_guidance:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        p_len = torch.tensor([p_len], device=self.device).long()

        with torch.no_grad():
            audio1 = ((net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]).data.cpu().float().numpy()) if pitch_guidance else ((net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy())

        del feats, p_len, padding_mask

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return audio1
    

    def pipeline(self, model, net_g, sid, audio, input_audio_path, pitch, f0_method, file_index, index_rate, pitch_guidance, filter_radius, tgt_sr, resample_sr, volume_envelope, version, protect, hop_length, f0_autotune, f0_autotune_strength):
        if file_index != "" and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                logger.error(f"Đã xảy ra lỗi khi đọc chỉ mục FAISS: {e}")
                index = big_npy = None
        else: index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []

        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)

            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]

            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min())[0][0])

        s = 0
        audio_opt = []
        t = None

        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        if pitch_guidance:
            pitch, pitchf = self.get_f0(input_audio_path, audio_pad, p_len, pitch, f0_method, filter_radius, hop_length, f0_autotune, f0_autotune_strength)
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]

            if self.device == "mps": pitchf = pitchf.astype(np.float32)

            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        for t in opt_ts:
            t = t // self.window * self.window

            if pitch_guidance: audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[s : t + self.t_pad2 + self.window], pitch[:, s // self.window : (t + self.t_pad2) // self.window], pitchf[:, s // self.window : (t + self.t_pad2) // self.window], index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
            else: audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[s : t + self.t_pad2 + self.window], None, None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])

            s = t
            
        if pitch_guidance: audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[t:], pitch[:, t // self.window :] if t is not None else pitch, pitchf[:, t // self.window :] if t is not None else pitchf, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
        else: audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[t:], None, None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
            
        audio_opt = np.concatenate(audio_opt)

        if volume_envelope != 1: audio_opt = change_rms(audio, self.sample_rate, audio_opt, tgt_sr, volume_envelope)
        if resample_sr >= self.sample_rate and tgt_sr != resample_sr: audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr)

        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768

        if audio_max > 1: max_int16 /= audio_max

        audio_opt = (audio_opt * max_int16).astype(np.int16)

        if pitch_guidance: del pitch, pitchf
        del sid

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return audio_opt


class VoiceConverter:
    def __init__(self):
        self.config = Config()  
        self.hubert_model = (None)
        self.tgt_sr = None 
        self.net_g = None 
        self.vc = None
        self.cpt = None  
        self.version = None 
        self.n_spk = None  
        self.use_f0 = None  
        self.loaded_model = None
    

    def load_hubert(self, embedder_model):
        try:
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task([os.path.join(now_dir, "assets", "model", "embedders", embedder_model + '.pt')], suffix="")
        except Exception as e:
            raise ImportError(f"Thất bại khi tải mô hình: {e}")
        
        self.hubert_model = models[0].to(self.config.device)
        self.hubert_model = (self.hubert_model.half() if self.config.is_half else self.hubert_model.float())
        self.hubert_model.eval()


    @staticmethod
    def remove_audio_noise(input_audio_path, reduction_strength=0.7):
        try:
            rate, data = wavfile.read(input_audio_path)
            reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=reduction_strength)

            return reduced_noise
        except Exception as e:
            logger.error(f"Đã xảy ra lỗi khi loại bỏ tiếng ồn: {e}")
            return None


    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        try:
            if output_format != "wav":
                logger.info(f"Đang chuyển đổi âm thanh sang định dạng {output_format}...")
                audio, sample_rate = sf.read(input_path)

                common_sample_rates = [
                    8000, 
                    11025, 
                    12000, 
                    16000, 
                    22050, 
                    24000, 
                    32000, 
                    44100, 
                    48000
                ]

                target_sr = min(common_sample_rates, key=lambda x: abs(x - sample_rate))
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)

                sf.write(output_path, audio, target_sr, format=output_format)

            return output_path
        except Exception as e:
            raise RuntimeError(f"Đã xảy ra lỗi khi chuyển đổi định dạng âm thanh: {e}")


    def convert_audio(self, audio_input_path, audio_output_path, model_path, index_path, embedder_model, pitch, f0_method, index_rate, volume_envelope, protect, hop_length, f0_autotune, f0_autotune_strength, filter_radius, clean_audio, clean_strength, export_format, upscale_audio, resample_sr = 0, sid = 0):
        self.get_vc(model_path, sid)

        try:
            if upscale_audio: upscale(audio_input_path, audio_input_path)

            audio = load_audio_infer(audio_input_path, 16000)

            audio_max = np.abs(audio).max() / 0.95

            if audio_max > 1: audio /= audio_max

            if not self.hubert_model: 
                if not os.path.exists(os.path.join(now_dir, "assets", "model", "embedders", embedder_model + '.pt')): raise FileNotFoundError(f"Không tìm thấy mô hình: {embedder_model}")
                
                self.load_hubert(embedder_model)

            if self.tgt_sr != resample_sr >= 16000: self.tgt_sr = resample_sr

            file_index = (index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added"))

            audio_opt = self.vc.pipeline(model=self.hubert_model, net_g=self.net_g, sid=sid, audio=audio, input_audio_path=audio_input_path, pitch=pitch, f0_method=f0_method, file_index=file_index, index_rate=index_rate, pitch_guidance=self.use_f0, filter_radius=filter_radius, tgt_sr=self.tgt_sr, resample_sr=resample_sr, volume_envelope=volume_envelope, version=self.version, protect=protect, hop_length=hop_length, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength)

            if audio_output_path: sf.write(audio_output_path, audio_opt, self.tgt_sr, format="wav")

            if clean_audio:
                cleaned_audio = self.remove_audio_noise(audio_output_path, clean_strength)
                if cleaned_audio is not None: sf.write(audio_output_path, cleaned_audio, self.tgt_sr, format="wav")

            output_path_format = audio_output_path.replace(".wav", f".{export_format}")
            audio_output_path = self.convert_audio_format(audio_output_path, output_path_format, export_format)
        except Exception as e:
            logger.error(f"Đã xảy ra lỗi trong quá trình chuyển đổi âm thanh: {e}")
            logger.error(traceback.format_exc())


    def get_vc(self, weight_root, sid):
        if sid == "" or sid == []:
            self.cleanup_model()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if not self.loaded_model or self.loaded_model != weight_root:
          self.load_model(weight_root)

          if self.cpt is not None:
              self.setup_network()
              self.setup_vc_instance()

          self.loaded_model = weight_root


    def cleanup_model(self):
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr

            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None

            if torch.cuda.is_available(): torch.cuda.empty_cache()

        del self.net_g, self.cpt

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self.cpt = None


    def load_model(self, weight_root):
        self.cpt = (torch.load(weight_root, map_location="cpu") if os.path.isfile(weight_root) else None)


    def setup_network(self):
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = self.cpt.get("f0", 1)

            self.version = self.cpt.get("version", "v1")
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256

            self.net_g = Synthesizer(*self.cpt["config"], use_f0=self.use_f0, text_enc_hidden_dim=self.text_enc_hidden_dim, is_half=self.config.is_half)

            del self.net_g.enc_q

            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g.eval().to(self.config.device)
            self.net_g = (self.net_g.half() if self.config.is_half else self.net_g.float())


    def setup_vc_instance(self):
        if self.cpt is not None:
            self.vc = VC(self.tgt_sr, self.config)
            self.n_spk = self.cpt["config"][-3]

if __name__ == "__main__": main()