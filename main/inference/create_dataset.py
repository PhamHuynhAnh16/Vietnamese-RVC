import os
import re
import sys
import time
import yt_dlp
import shutil
import librosa
import logging
import argparse
import warnings
import logging.handlers

import soundfile as sf
import noisereduce as nr

from distutils.util import strtobool
from pydub import AudioSegment, silence


now_dir = os.getcwd()
sys.path.append(now_dir)

from main.library.algorithm.separator import Separator


log_file = os.path.join("assets", "logs", "create_dataset.log")
logger = logging.getLogger(__name__)


if not logger.hasHandlers():  
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def parse_arguments() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", type=str, required=True)
    parser.add_argument("--output_dataset", type=str, required=True)
    parser.add_argument("--resample", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--resample_sr", type=int, default=44100)
    parser.add_argument("--clean_dataset", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)
    parser.add_argument("--separator_music", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--separator_reverb", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--kim_vocal_version", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--segments_size", type=int, default=256)
    parser.add_argument("--mdx_hop_length", type=int, default=1024)
    parser.add_argument("--mdx_batch_size", type=int, default=1)
    parser.add_argument("--denoise_mdx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--skip", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--skip_start_audios", type=str, default="0")
    parser.add_argument("--skip_end_audios", type=str, default="0")
    
    args = parser.parse_args()
    return args


dataset_temp = os.path.join("dataset_temp")


def main():
    args = parse_arguments()
    input_audio = args.input_audio
    output_dataset = args.output_dataset
    resample = args.resample
    resample_sr = args.resample_sr
    clean_dataset = args.clean_dataset
    clean_strength = args.clean_strength
    separator_music = args.separator_music
    separator_reverb = args.separator_reverb
    kim_vocal_version = args.kim_vocal_version
    overlap = args.overlap
    segments_size = args.segments_size
    hop_length = args.mdx_hop_length
    batch_size = args.mdx_batch_size
    denoise_mdx = args.denoise_mdx
    skip = args.skip
    skip_start_audios = args.skip_start_audios
    skip_end_audios = args.skip_end_audios

    logger.debug(f"Đầu vào: {input_audio}")
    logger.debug(f"Đầu ra: {output_dataset}")
    logger.debug(f"Lấy mẫu lại: {resample}")
    if resample: logger.debug(f"Tỷ lệ lấy mẫu lại: {resample_sr}")
    logger.debug(f"Làm sạch dữ liệu: {clean_dataset}")
    if clean_dataset: logger.debug(f"Mức độ làm sạch dữ liệu: {clean_strength}")
    logger.debug(f"Tách bỏ nhạc: {separator_music}")
    logger.debug(f"Tách bỏ vang: {separator_reverb}")
    if separator_music: logger.debug(f"Phiên bản mô hình tách giọng: {kim_vocal_version}")
    logger.debug(f"Kích thước phân đoạn: {segments_size}")
    logger.debug(f"Mức chồng chéo: {overlap}")
    logger.debug(f"Hop length: {hop_length}")
    logger.debug(f"Kích thước lô: {batch_size}")
    logger.debug(f"Khữ nhiễu MDX: {denoise_mdx}")
    logger.debug(f"Bỏ qua âm thanh: {skip}")
    if skip: logger.debug(f"Bỏ qua âm thanh đầu: {skip_start_audios}")
    if skip: logger.debug(f"Bỏ qua âm thanh cuối: {skip_end_audios}")


    if kim_vocal_version != 1 and kim_vocal_version != 2: raise ValueError("Phiên bản tách giọng không hợp lệ")
    if separator_reverb and not separator_music: raise ValueError("Bật tùy chọn tách nhạc để có thể sử dụng tùy chọn tách vang")

    start_time = time.time()

    try:
        paths = []

        if not os.path.exists(dataset_temp): os.makedirs(dataset_temp, exist_ok=True)

        urls = input_audio.replace(", ", ",").split(",")

        for url in urls:
            path = downloader(url, urls.index(url))
            paths.append(path)

        if skip:
            skip_start_audios = skip_start_audios.replace(", ", ",").split(",")
            skip_end_audios = skip_end_audios.replace(", ", ",").split(",")

            if len(skip_start_audios) < len(paths) or len(skip_end_audios) < len(paths): logger.warning("Không thể bỏ qua vì số lượng thời gian bỏ qua thấp hơn số lượng tệp âm thanh")
            elif len(skip_start_audios) > len(paths) or len(skip_end_audios) > len(paths): logger.warning("Không thể bỏ qua vì số lượng thời gian bỏ qua cao hơn số lượng tệp âm thanh")
            else:
                for audio, skip_start_audio, skip_end_audio in zip(paths, skip_start_audios, skip_end_audios):
                    skip_start(audio, skip_start_audio)
                    skip_end(audio, skip_end_audio)

        if separator_music:
            separator_paths = []

            for audio in paths:
                vocals = separatormusic(audio, dataset_temp, segments_size, overlap, denoise_mdx, kim_vocal_version, hop_length, batch_size)

                if separator_reverb: vocals = separator_reverb_audio(vocals, dataset_temp, segments_size, overlap, denoise_mdx, hop_length, batch_size)
                separator_paths.append(vocals)
            
            paths = separator_paths

        processed_paths = []

        for audio in paths:
            output = process_audio(audio)
            processed_paths.append(output)

        paths = processed_paths
                
        for audio_path in paths:
            data, sample_rate = sf.read(audio_path)

            if resample_sr != sample_rate and resample_sr > 0 and resample: 
                data = librosa.resample(data, orig_sr=sample_rate, target_sr=resample_sr)
                sample_rate = resample_sr

            if clean_dataset: data = nr.reduce_noise(y=data, prop_decrease=clean_strength)


            sf.write(audio_path, data, sample_rate)
    except Exception as e:
        raise RuntimeError(f"Đã xảy ra lỗi khi tạo dữ liệu huấn luyện: {e}")
    finally:
        for audio in paths:
            shutil.copy(audio, output_dataset)

        if os.path.exists(dataset_temp): shutil.rmtree(dataset_temp, ignore_errors=True)

    elapsed_time = time.time() - start_time
    logger.info(f"Quá trình tạo dữ liệu huấn huyện đã hoàn thành sau: {elapsed_time:.2f} giây")


def downloader(url, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(dataset_temp, f"{name}"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'noplaylist': True,
            'verbose': False, 
        }

        logger.info(f"Bắt đầu tải xuống: {url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url)  
            logger.info(f"Đã tải xuống xong: {url}")
        
    return os.path.join(dataset_temp, f"{name}" + ".wav")


def skip_start(input_file, seconds):
    data, sr = sf.read(input_file)
    
    total_duration = len(data) / sr
    
    if seconds <= 0: logger.warning(f"Thời gian bỏ qua bằng 0 nên bỏ qua")
    elif seconds >= total_duration: logger.warning(f"Thời gian bỏ qua ({seconds} giây) vượt quá độ dài âm thanh ({total_duration:.2f} giây). Bỏ qua.")
    else: 
        logger.info(f"Bỏ qua âm thanh đầu: {input_file}...")

        sf.write(input_file, data[int(seconds * sr):], sr)

        logger.info(f"Bỏ qua âm thanh đầu thành công: {input_file}")


def skip_end(input_file, seconds):
    data, sr = sf.read(input_file)
    
    total_duration = len(data) / sr

    if seconds <= 0: logger.warning(f"Thời gian bỏ qua bằng 0 nên bỏ qua")
    elif seconds > total_duration: logger.warning(f"Số giây cần bỏ qua ({seconds} giây) vượt quá thời lượng âm thanh ({total_duration:.2f} giây). Bỏ qua.")
    else: 
        logger.info(f"Bỏ qua âm thanh cuối: {input_file}...")

        sf.write(input_file, data[:-int(seconds * sr)], sr)

        logger.info(f"Bỏ qua âm thanh cuối thành công: {input_file}")


def process_audio(file_path):
    try:
        song = AudioSegment.from_file(file_path)
        nonsilent_parts = silence.detect_nonsilent(song, min_silence_len=750, silence_thresh=-70)

        cut_files = []

        for i, (start_i, end_i) in enumerate(nonsilent_parts):
            chunk = song[start_i:end_i]

            if len(chunk) >= 30:
                chunk_file_path = os.path.join(os.path.dirname(file_path), f"chunk{i}.wav")
                if os.path.exists(chunk_file_path): os.remove(chunk_file_path)
                
                chunk.export(chunk_file_path, format="wav")

                cut_files.append(chunk_file_path)
            else: logger.warning(f"Phần {i} được bỏ qua vì quá ngắn: {len(chunk)}ms")

        logger.info(f"Tổng số phần chứa âm thanh của {file_path} là: {len(cut_files)}")

        def extract_number(filename):
            match = re.search(r'_(\d+)', filename)

            return int(match.group(1)) if match else 0

        cut_files = sorted(cut_files, key=extract_number)

        combined = AudioSegment.empty()

        for file in cut_files:
            combined += AudioSegment.from_file(file)

        output_path = os.path.splitext(file_path)[0] + "_processed" + ".wav"

        logger.info("Đã ghép các phần chứa âm thanh lại")

        combined.export(output_path, format="wav")

        return output_path
    except Exception as error:
        raise RuntimeError(f"Đã xảy ra lỗi khi xử lý và ghép âm thanh: {error}")


def separatormusic(input, output, segments_size, overlap, denoise, version, hop_length, batch_size):
    if not os.path.exists(input): 
        logger.warning("Không tìm thấy đầu vào")
        return None
    
    if not os.path.exists(output): 
        logger.warning("Không tìm thấy đầu ra")
        return None

    model = f"Kim_Vocal_{version}.onnx"

    logger.info(f"Đang tách giọng: {input}...")
    output_separator = separator_main(audio_file=input, model_filename=model, output_format="wav", output_dir=output, mdx_segment_size=segments_size, mdx_overlap=overlap, mdx_batch_size=batch_size, mdx_hop_length=hop_length, mdx_enable_denoise=denoise)

    for f in output_separator:
        path = os.path.join(output, f)

        if not os.path.exists(path): logger.error(f"Không tìm thấy: {f}")

        if '_(Instrumental)_' in f: os.rename(path, os.path.splitext(path)[0].replace("(", "").replace(")", "") + ".wav")
        elif '_(Vocals)_' in f:
            rename_file = os.path.splitext(path)[0].replace("(", "").replace(")", "") + ".wav"
            os.rename(path, rename_file)

    logger.info(f"Đã tách giọng giọng thành công: {rename_file}")
    return rename_file


def separator_reverb_audio(input, output, segments_size, overlap, denoise, hop_length, batch_size):
    reverb_models = "Reverb_HQ_By_FoxJoy.onnx"
    
    if not os.path.exists(input): 
        logger.warning("Không tìm thấy đầu vào")
        return None
    
    if not os.path.exists(output): 
        logger.warning("Không tìm thấy đầu ra")
        return None

    logger.info(f"Đang tách âm vang: {input}...")
    output_dereverb = separator_main(audio_file=input, model_filename=reverb_models, output_format="wav", output_dir=output, mdx_segment_size=segments_size, mdx_overlap=overlap, mdx_batch_size=hop_length, mdx_hop_length=batch_size, mdx_enable_denoise=denoise)

    for f in output_dereverb:
        path = os.path.join(output, f)

        if not os.path.exists(path): logger.error(f"Không tìm thấy: {f}")

        if '_(Reverb)_' in f: os.rename(path, os.path.splitext(path)[0].replace("(", "").replace(")", "") + ".wav")
        elif '_(No Reverb)_' in f:
            rename_file = os.path.splitext(path)[0].replace("(", "").replace(")", "") + ".wav"
            os.rename(path, rename_file)    

    logger.info(f"Đã tách âm vang thành công: {rename_file}")
    return rename_file


def separator_main(audio_file=None, model_filename="Kim_Vocal_1.onnx", output_format="wav", output_dir=".", mdx_segment_size=256, mdx_overlap=0.25, mdx_batch_size=1, mdx_hop_length=1024, mdx_enable_denoise=True):
    separator = Separator(
        log_formatter=file_formatter,
        log_level=logging.INFO,
        output_dir=output_dir,
        output_format=output_format,
        output_bitrate=None,
        normalization_threshold=0.9,
        output_single_stem=None,
        invert_using_spec=False,
        sample_rate=44100,
        mdx_params={
            "hop_length": mdx_hop_length,
            "segment_size": mdx_segment_size,
            "overlap": mdx_overlap,
            "batch_size": mdx_batch_size,
            "enable_denoise": mdx_enable_denoise,
        },
    )

    separator.load_model(model_filename=model_filename)
    return separator.separate(audio_file)

if __name__ == "__main__": main()