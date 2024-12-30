import os
import sys
import time
import yt_dlp
import shutil
import librosa
import logging
import argparse
import warnings
import logging.handlers

from soundfile import read, write
from distutils.util import strtobool

sys.path.append(os.getcwd())
from main.configs.config import Config

translations = Config().translations
dataset_temp = os.path.join("dataset_temp")
log_file = os.path.join("assets", "logs", "create_dataset.log")
logger = logging.getLogger(__name__)

if logger.hasHandlers(): logger.handlers.clear()
else: 
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", type=str, required=True)
    parser.add_argument("--output_dataset", type=str, default="./dataset")
    parser.add_argument("--sample_rate", type=int, default=44100)
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

    return parser.parse_args()

def main():
    pid_path = os.path.join("assets", "create_dataset_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    args = parse_arguments()
    input_audio, output_dataset, sample_rate, clean_dataset, clean_strength, separator_music, separator_reverb, kim_vocal_version, overlap, segments_size, hop_length, batch_size, denoise_mdx, skip, skip_start_audios, skip_end_audios = args.input_audio, args.output_dataset, args.sample_rate, args.clean_dataset, args.clean_strength, args.separator_music, args.separator_reverb, args.kim_vocal_version, args.overlap, args.segments_size, args.mdx_hop_length, args.mdx_batch_size, args.denoise_mdx, args.skip, args.skip_start_audios, args.skip_end_audios

    logger.debug(f"{translations['audio_path']}: {input_audio}")
    logger.debug(f"{translations['output_path']}: {output_dataset}")
    logger.debug(f"{translations['sr']}: {sample_rate}")
    logger.debug(f"{translations['clear_dataset']}: {clean_dataset}")
    if clean_dataset: logger.debug(f"{translations['clean_strength']}: {clean_strength}")
    logger.debug(f"{translations['separator_music']}: {separator_music}")
    logger.debug(f"{translations['dereveb_audio']}: {separator_reverb}")
    if separator_music: logger.debug(f"{translations['training_version']}: {kim_vocal_version}")
    logger.debug(f"{translations['segments_size']}: {segments_size}")
    logger.debug(f"{translations['overlap']}: {overlap}")
    logger.debug(f"Hop length: {hop_length}")
    logger.debug(f"{translations['batch_size']}: {batch_size}")
    logger.debug(f"{translations['denoise_mdx']}: {denoise_mdx}")
    logger.debug(f"{translations['skip']}: {skip}")
    if skip: logger.debug(f"{translations['skip_start']}: {skip_start_audios}")
    if skip: logger.debug(f"{translations['skip_end']}: {skip_end_audios}")
    if kim_vocal_version not in [1, 2]: raise ValueError(translations["version_not_valid"])
    if separator_reverb and not separator_music: raise ValueError(translations["create_dataset_value_not_valid"])

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

            if len(skip_start_audios) < len(paths) or len(skip_end_audios) < len(paths): 
                logger.warning(translations["skip<audio"])
                sys.exit(1)
            elif len(skip_start_audios) > len(paths) or len(skip_end_audios) > len(paths): 
                logger.warning(translations["skip>audio"])
                sys.exit(1)
            else:
                for audio, skip_start_audio, skip_end_audio in zip(paths, skip_start_audios, skip_end_audios):
                    skip_start(audio, skip_start_audio)
                    skip_end(audio, skip_end_audio)

        if separator_music:
            separator_paths = []

            for audio in paths:
                vocals = separator_music_main(audio, dataset_temp, segments_size, overlap, denoise_mdx, kim_vocal_version, hop_length, batch_size, sample_rate)
                if separator_reverb: vocals = separator_reverb_audio(vocals, dataset_temp, segments_size, overlap, denoise_mdx, hop_length, batch_size, sample_rate)
                separator_paths.append(vocals)
            
            paths = separator_paths

        processed_paths = []

        for audio in paths:
            output = process_audio(audio)
            processed_paths.append(output)

        paths = processed_paths
                
        for audio_path in paths:
            data, sample_rate = read(audio_path)
            data = librosa.to_mono(data.T)
            
            if clean_dataset: 
                from main.tools.noisereduce import reduce_noise
                data = reduce_noise(y=data, prop_decrease=clean_strength)

            write(audio_path, data, sample_rate)
    except Exception as e:
        logger.error(f"{translations['create_dataset_error']}: {e}")

        import traceback
        logger.error(traceback.format_exc())
    finally:
        for audio in paths:
            shutil.move(audio, output_dataset)

        if os.path.exists(dataset_temp): shutil.rmtree(dataset_temp, ignore_errors=True)

    if os.path.exists(pid_path): os.remove(pid_path)

    elapsed_time = time.time() - start_time
    logger.info(translations["create_dataset_success"].format(elapsed_time=f"{elapsed_time:.2f}"))

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

        logger.info(f"{translations['starting_download']}: {url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url)  
            logger.info(f"{translations['download_success']}: {url}")
    return os.path.join(dataset_temp, f"{name}" + ".wav")

def skip_start(input_file, seconds):
    data, sr = read(input_file)
    total_duration = len(data) / sr
    
    if seconds <= 0: logger.warning(translations["=<0"])
    elif seconds >= total_duration: logger.warning(translations["skip_warning"].format(seconds=seconds, total_duration=f"{total_duration:.2f}"))
    else: 
        logger.info(f"{translations['skip_start']}: {input_file}...")
        write(input_file, data[int(seconds * sr):], sr)
        logger.info(translations["skip_start_audio"].format(input_file=input_file))

def skip_end(input_file, seconds):
    data, sr = read(input_file)
    total_duration = len(data) / sr

    if seconds <= 0: logger.warning(translations["=<0"])
    elif seconds > total_duration: logger.warning(translations["skip_warning"].format(seconds=seconds, total_duration=f"{total_duration:.2f}"))
    else: 
        logger.info(f"{translations['skip_end']}: {input_file}...")
        write(input_file, data[:-int(seconds * sr)], sr)
        logger.info(translations["skip_end_audio"].format(input_file=input_file))

def process_audio(file_path):
    try:
        from pydub import AudioSegment, silence
        song = AudioSegment.from_file(file_path)
        nonsilent_parts = silence.detect_nonsilent(song, min_silence_len=750, silence_thresh=-70)
        cut_files = []

        for i, (start_i, end_i) in enumerate(nonsilent_parts):
            chunk = song[start_i:end_i]

            if len(chunk) > 10:
                chunk_file_path = os.path.join(os.path.dirname(file_path), f"chunk{i}.wav")
                if os.path.exists(chunk_file_path): os.remove(chunk_file_path)
                
                chunk.export(chunk_file_path, format="wav")
                cut_files.append(chunk_file_path)
            else: logger.warning(translations["skip_file"].format(i=i, chunk=len(chunk)))

        logger.info(f"{translations['split_total']}: {len(cut_files)}")

        def extract_number(filename):
            import re

            match = re.search(r'_(\d+)', filename)
            return int(match.group(1)) if match else 0

        cut_files = sorted(cut_files, key=extract_number)
        combined = AudioSegment.empty()

        for file in cut_files:
            combined += AudioSegment.from_file(file)

        output_path = os.path.splitext(file_path)[0] + "_processed" + ".wav"
        logger.info(translations["merge_audio"])
        combined.export(output_path, format="wav")
        return output_path
    except Exception as e:
        raise RuntimeError(f"{translations['process_audio_error']}: {e}")

def separator_music_main(input, output, segments_size, overlap, denoise, version, hop_length, batch_size, sample_rate):
    if not os.path.exists(input): 
        logger.warning(translations["input_not_valid"])
        return None
    
    if not os.path.exists(output): 
        logger.warning(translations["output_not_valid"])
        return None

    model = f"Kim_Vocal_{version}.onnx"
    logger.info(translations["separator_process"].format(input=input))
    output_separator = separator_main(audio_file=input, model_filename=model, output_format="wav", output_dir=output, mdx_segment_size=segments_size, mdx_overlap=overlap, mdx_batch_size=batch_size, mdx_hop_length=hop_length, mdx_enable_denoise=denoise, sample_rate=sample_rate)

    for f in output_separator:
        path = os.path.join(output, f)
        if not os.path.exists(path): logger.error(translations["not_found"].format(name=path))
        if '_(Instrumental)_' in f: os.rename(path, os.path.splitext(path)[0].replace("(", "").replace(")", "") + ".wav")
        elif '_(Vocals)_' in f:
            rename_file = os.path.splitext(path)[0].replace("(", "").replace(")", "") + ".wav"
            os.rename(path, rename_file)

    logger.info(f": {rename_file}")
    return rename_file

def separator_reverb_audio(input, output, segments_size, overlap, denoise, hop_length, batch_size, sample_rate):
    reverb_models = "Reverb_HQ_By_FoxJoy.onnx"
    
    if not os.path.exists(input): 
        logger.warning(translations["input_not_valid"])
        return None
    
    if not os.path.exists(output): 
        logger.warning(translations["output_not_valid"])
        return None

    logger.info(f"{translations['dereverb']}: {input}...")
    output_dereverb = separator_main(audio_file=input, model_filename=reverb_models, output_format="wav", output_dir=output, mdx_segment_size=segments_size, mdx_overlap=overlap, mdx_batch_size=hop_length, mdx_hop_length=batch_size, mdx_enable_denoise=denoise, sample_rate=sample_rate)

    for f in output_dereverb:
        path = os.path.join(output, f)
        if not os.path.exists(path): logger.error(translations["not_found"].format(name=path))
        if '_(Reverb)_' in f: os.rename(path, os.path.splitext(path)[0].replace("(", "").replace(")", "") + ".wav")
        elif '_(No Reverb)_' in f:
            rename_file = os.path.splitext(path)[0].replace("(", "").replace(")", "") + ".wav"
            os.rename(path, rename_file)    

    logger.info(f"{translations['dereverb_success']}: {rename_file}")
    return rename_file

def separator_main(audio_file=None, model_filename="Kim_Vocal_1.onnx", output_format="wav", output_dir=".", mdx_segment_size=256, mdx_overlap=0.25, mdx_batch_size=1, mdx_hop_length=1024, mdx_enable_denoise=True, sample_rate=44100):
    from main.library.algorithm.separator import Separator

    try:
        separator = Separator(logger=logger, log_formatter=file_formatter, log_level=logging.INFO, output_dir=output_dir, output_format=output_format, output_bitrate=None, normalization_threshold=0.9, output_single_stem=None, invert_using_spec=False, sample_rate=sample_rate, mdx_params={"hop_length": mdx_hop_length, "segment_size": mdx_segment_size, "overlap": mdx_overlap, "batch_size": mdx_batch_size, "enable_denoise": mdx_enable_denoise})
        separator.load_model(model_filename=model_filename)
        return separator.separate(audio_file)
    except:
        logger.debug(translations["default_setting"])

        separator = Separator(logger=logger, log_formatter=file_formatter, log_level=logging.INFO, output_dir=output_dir, output_format=output_format, output_bitrate=None, normalization_threshold=0.9, output_single_stem=None, invert_using_spec=False, sample_rate=44100, mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": mdx_enable_denoise})
        separator.load_model(model_filename=model_filename)
        return separator.separate(audio_file)

if __name__ == "__main__": main()