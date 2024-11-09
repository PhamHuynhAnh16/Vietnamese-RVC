import os
import sys
import time
import logging
import argparse
import logging.handlers

import soundfile as sf
import noisereduce as nr

from pydub import AudioSegment
from distutils.util import strtobool

now_dir = os.getcwd()
sys.path.append(now_dir)

from main.library.algorithm.separator import Separator

log_file = os.path.join("assets", "logs", "separator.log")
logger = logging.getLogger(__name__)

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

demucs_models = {
    "HT-Tuned": "htdemucs_ft.yaml",
    "HT-Normal": "htdemucs.yaml",
    "HD_MMI": "hdemucs_mmi.yaml",
    "HT_6S": "htdemucs_6s.yaml"
}

mdx_models = {
    "Main_340": "UVR-MDX-NET_Main_340.onnx",
    "Main_390": "UVR-MDX-NET_Main_390.onnx",
    "Main_406": "UVR-MDX-NET_Main_406.onnx",
    "Main_427": "UVR-MDX-NET_Main_427.onnx",
    "Main_438": "UVR-MDX-NET_Main_438.onnx",
    "Inst_full_292": "UVR-MDX-NET-Inst_full_292.onnx",
    "Inst_HQ_1": "UVR-MDX-NET_Inst_HQ_1.onnx",
    "Inst_HQ_2": "UVR-MDX-NET_Inst_HQ_2.onnx",
    "Inst_HQ_3": "UVR-MDX-NET_Inst_HQ_3.onnx",
    "Inst_HQ_4": "UVR-MDX-NET-Inst_HQ_4.onnx",
    "Kim_Vocal_1": "Kim_Vocal_1.onnx",
    "Kim_Vocal_2": "Kim_Vocal_2.onnx",
    "Kim_Inst": "Kim_Inst.onnx",
    "Inst_187_beta": "UVR-MDX-NET_Inst_187_beta.onnx",
    "Inst_82_beta": "UVR-MDX-NET_Inst_82_beta.onnx",
    "Inst_90_beta": "UVR-MDX-NET_Inst_90_beta.onnx",
    "Voc_FT": "UVR-MDX-NET-Voc_FT.onnx",
    "Crowd_HQ": "UVR-MDX-NET_Crowd_HQ_1.onnx",
    "MDXNET_9482": "UVR_MDXNET_9482.onnx",
    "Inst_1": "UVR-MDX-NET-Inst_1.onnx",
    "Inst_2": "UVR-MDX-NET-Inst_2.onnx",
    "Inst_3": "UVR-MDX-NET-Inst_3.onnx",
    "MDXNET_1_9703": "UVR_MDXNET_1_9703.onnx",
    "MDXNET_2_9682": "UVR_MDXNET_2_9682.onnx",
    "MDXNET_3_9662": "UVR_MDXNET_3_9662.onnx",
    "Inst_Main": "UVR-MDX-NET-Inst_Main.onnx",
    "MDXNET_Main": "UVR_MDXNET_Main.onnx"
}

kara_models = {
    "Version-1": "UVR_MDXNET_KARA.onnx",
    "Version-2": "UVR_MDXNET_KARA_2.onnx"
}


def parse_arguments() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./audios")
    parser.add_argument("--format", type=str, default="wav")
    parser.add_argument("--shifts", type=int, default=10)
    parser.add_argument("--segments_size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--mdx_hop_length", type=int, default=1024)
    parser.add_argument("--mdx_batch_size", type=int, default=1)
    parser.add_argument("--clean_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)
    parser.add_argument("--backing_denoise", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--demucs_model", type=str, default="HT-Normal")
    parser.add_argument("--kara_model", type=str, default="Version-1")
    parser.add_argument("--mdx_model", type=str, default="Main_340")
    parser.add_argument("--backing", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--mdx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--mdx_denoise", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--reverb", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--reverb_denoise", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--backing_reverb", type=lambda x: bool(strtobool(x)), default=False)

    args = parser.parse_args()
    return args

def main():
    try:
        args = parse_arguments()

        input_path = args.input_path
        output_path = args.output_path
        export_format = args.format
        shifts = args.shifts
        segments_size = args.segments_size
        overlap = args.overlap
        hop_length = args.mdx_hop_length
        batch_size = args.mdx_batch_size
        clean_audio = args.clean_audio
        clean_strength = args.clean_strength
        backing_denoise = args.backing_denoise
        demucs_model = args.demucs_model
        kara_model = args.kara_model
        backing = args.backing
        mdx = args.mdx
        mdx_model = args.mdx_model
        mdx_denoise = args.mdx_denoise
        reverb = args.reverb
        reverb_denoise = args.reverb_denoise
        backing_reverb = args.backing_reverb

        if backing_reverb and not reverb: 
            logger.warning("Điều kiện cần để sử dụng tách vang giọng bè là phải bật tách vang")
            return None, None, None, None

        if backing_reverb and not backing: 
            logger.warning("Điều kiện cần để sử dụng tách vang giọng bè là phải bật tách bè")
            return None, None, None, None

        logger.debug(f"Đường dẫn đầu vào: {input_path}")
        logger.debug(f"Đường dẫn đầu ra: {output_path}")
        logger.debug(f"Định dạng đầu ra: {export_format}")
        if not mdx: logger.debug(f"Số lượng dự đoán: {shifts}") 
        logger.debug(f"Kích thước phân đoạn: {segments_size}")
        logger.debug(f"Mức chồng chéo: {overlap}")
        if clean_audio: logger.debug(f"Lọc Tạp Âm: {clean_audio}")
        if clean_audio: logger.debug(f"Sức mạnh làm sạch: {clean_strength}")
        if not mdx: logger.debug(f"Mô hình của Demucs: {demucs_model}")
        if backing: logger.debug(f"Khữ nhiễu của tách bè: {backing_denoise}")
        if backing: logger.debug(f"Phiên bản mô hình của tách bè: {kara_model}")
        if backing: logger.debug(f"Có tách bè hay không: {backing}")
        if mdx: logger.debug(f"Sử dụng mô hình mdx: {mdx}")
        if mdx: logger.debug(f"Mô hình mdx: {mdx_model}")
        if mdx: logger.debug(f"Khữ nhiễu của mô hình mdx: {mdx_denoise}")
        if mdx or backing or reverb: logger.debug(f"Hop length: {hop_length}")
        if mdx or backing or reverb: logger.debug(f"Kích thước lô: {batch_size}")
        if reverb: logger.debug(f"Tách âm vang: {reverb}")
        if reverb: logger.debug(f"Khữ nhiễu âm vang: {reverb_denoise}")
        if reverb: logger.debug(f"Có tách vang giọng bè: {backing_reverb}")

        start_time = time.time()

        if not mdx: vocals, instruments = separator_music_demucs(input_path, output_path, export_format, shifts, overlap, segments_size, demucs_model)
        else: vocals, instruments = separator_music_mdx(input_path, output_path, export_format, segments_size, overlap, mdx_denoise, mdx_model, hop_length, batch_size)

        if backing: main_vocals, backing_vocals = separator_backing(vocals, output_path, export_format, segments_size, overlap, backing_denoise, kara_model, hop_length, batch_size)
        if reverb: vocals_no_reverb, main_vocals_no_reverb, backing_vocals_no_reverb = separator_reverb(output_path, export_format, segments_size, overlap, reverb_denoise, reverb, backing, backing_reverb, hop_length, batch_size)

        original_output = os.path.join(output_path, f"Original_Vocals_No_Reverb.{export_format}") if reverb else os.path.join(output_path, f"Original_Vocals.{export_format}")
        main_output = os.path.join(output_path, f"Main_Vocals_No_Reverb.{export_format}") if reverb and backing else os.path.join(output_path, f"Main_Vocals.{export_format}")
        backing_output = os.path.join(output_path, f"Backing_Vocals_No_Reverb.{export_format}") if reverb and backing_reverb else os.path.join(output_path, f"Backing_Vocals.{export_format}")
        
        if clean_audio:
            logger.info(f"Đang thực hiện lọc tạp âm...")
            vocal_data, vocal_sr = sf.read(vocals_no_reverb if reverb else vocals)
            main_data, main_sr = sf.read(main_vocals_no_reverb if reverb and backing else main_vocals)
            backing_data, backing_sr = sf.read(backing_vocals_no_reverb if reverb and backing_reverb else backing_vocals)

            vocals_clean = nr.reduce_noise(y=vocal_data, prop_decrease=clean_strength)
            
            sf.write(original_output, vocals_clean, vocal_sr, format=export_format)

            if backing:
                mains_clean = nr.reduce_noise(y=main_data, sr=main_sr, prop_decrease=clean_strength)
                backing_clean = nr.reduce_noise(y=backing_data, sr=backing_sr, prop_decrease=clean_strength)
                sf.write(main_output, mains_clean, main_sr, format=export_format)
                sf.write(backing_output, backing_clean, backing_sr, format=export_format)          

            logger.info(f"Đã lọc tạp âm thành công!")
    except Exception as e:
        logger.error(f"Đã xảy ra lỗi khi tách nhạc: {e}")
        
    elapsed_time = time.time() - start_time
    logger.info(f"Quá trình tách nhạc đã hoàn thành sau: {elapsed_time:.2f} giây")
    
    return original_output, instruments, main_output, backing_output

def separator_music_demucs(input, output, format, shifts, overlap, segments_size, demucs_model):
    if not os.path.exists(input): 
        logger.warning("Không tìm thấy đầu vào")
        return None, None
    
    if not os.path.exists(output): 
        logger.warning("Không tìm thấy đầu ra")
        return None, None
    
    for i in [f"Original_Vocals.{format}", f"Instruments.{format}"]:
        if os.path.exists(os.path.join(output, i)): os.remove(os.path.join(output, i))

    model = demucs_models.get(demucs_model)

    segment_size = segments_size / 2

    logger.info(f"Đang xử lý tách nhạc...")

    demucs_output = separator_main(audio_file=input, model_filename=model, output_format=format, output_dir=output, demucs_segment_size=segment_size, demucs_shifts=shifts, demucs_overlap=overlap)
    
    for f in demucs_output:
        path = os.path.join(output, f)

        if not os.path.exists(path): logger.error(f"Không tìm thấy: {path}")

        if '_(Drums)_' in f: drums = path
        elif '_(Bass)_' in f: bass = path
        elif '_(Other)_' in f: other = path
        elif '_(Vocals)_' in f: os.rename(path, os.path.join(output, f"Original_Vocals.{format}"))

    AudioSegment.from_file(drums).overlay(AudioSegment.from_file(bass)).overlay(AudioSegment.from_file(other)).export(os.path.join(output, f"Instruments.{format}"), format=format)

    for f in [drums, bass, other]:
        if os.path.exists(f): os.remove(f)
    
    logger.info("Đã tách nhạc thành công!")
    return os.path.join(output, f"Original_Vocals.{format}"), os.path.join(output, f"Instruments.{format}")

def separator_backing(input, output, format, segments_size, overlap, denoise, kara_model, hop_length, batch_size):
    if not os.path.exists(input): 
        logger.warning("Không tìm thấy đầu vào")
        return None, None
    
    if not os.path.exists(output): 
        logger.warning("Không tìm thấy đầu ra")
        return None, None
    
    for f in [f"Main_Vocals.{format}", f"Backing_Vocals.{format}"]:
        if os.path.exists(os.path.join(output, f)): os.remove(os.path.join(output, f))

    model_2 = kara_models.get(kara_model)

    logger.info(f"Đang xử lý tách giọng bè...")
    backing_outputs = separator_main(audio_file=input, model_filename=model_2, output_format=format, output_dir=output, mdx_segment_size=segments_size, mdx_overlap=overlap, mdx_batch_size=batch_size, mdx_hop_length=hop_length, mdx_enable_denoise=denoise)

    main_output = os.path.join(output, f"Main_Vocals.{format}")
    backing_output = os.path.join(output, f"Backing_Vocals.{format}")

    for f in backing_outputs:
        path = os.path.join(output, f)

        if not os.path.exists(path): logger.error(f"Không tìm thấy: {path}")

        if '_(Instrumental)_' in f: os.rename(path, backing_output)
        elif '_(Vocals)_' in f: os.rename(path, main_output)

    logger.info(f"Đã tách giọng bè thành công!")
    return main_output, backing_output

def separator_music_mdx(input, output, format, segments_size, overlap, denoise, mdx_model, hop_length, batch_size):
    if not os.path.exists(input): 
        logger.warning("Không tìm thấy đầu vào")
        return None, None
    
    if not os.path.exists(output): 
        logger.warning("Không tìm thấy đầu ra")
        return None, None

    for i in [f"Original_Vocals.{format}", f"Instruments.{format}"]:
        if os.path.exists(os.path.join(output, i)): os.remove(os.path.join(output, i))
    
    model_3 = mdx_models.get(mdx_model)

    logger.info("Đang xử lý tách nhạc...")
    output_music = separator_main(audio_file=input, model_filename=model_3, output_format=format, output_dir=output, mdx_segment_size=segments_size, mdx_overlap=overlap, mdx_batch_size=batch_size, mdx_hop_length=hop_length, mdx_enable_denoise=denoise)
    
    original_output = os.path.join(output, f"Original_Vocals.{format}")
    instruments_output = os.path.join(output, f"Instruments.{format}")

    for f in output_music:
        path = os.path.join(output, f)

        if not os.path.exists(path): logger.error(f"Không tìm thấy: {path}")

        if '_(Instrumental)_' in f: os.rename(path, instruments_output)
        elif '_(Vocals)_' in f: os.rename(path, original_output)

    logger.info(f"Đã tách giọng nhạc thành công!")
    return original_output, instruments_output

def separator_reverb(output, format, segments_size, overlap, denoise, original, main, backing_reverb, hop_length, batch_size):
    if not os.path.exists(input): 
        logger.warning("Không tìm thấy đầu vào")
        return None, None, None
    
    if not os.path.exists(output): 
        logger.warning("Không tìm thấy đầu ra")
        return None, None, None
    
    for i in [f"Original_Vocals_Reverb.{format}", f"Main_Vocals_Reverb.{format}", f"Original_Vocals_No_Reverb.{format}", f"Main_Vocals_No_Reverb.{format}"]:
        if os.path.exists(os.path.join(output, i)): os.remove(os.path.join(output, i))

    dereveb_path = []

    if original: 
        try:
            dereveb_path.append(os.path.join(output, [f for f in os.listdir(output) if 'Original_Vocals' in f][0]))
        except IndexError:
            logger.warning("Không tìm thấy giọng gốc")
            return None, None, None
        
    if main:
        try:
            dereveb_path.append(os.path.join(output, [f for f in os.listdir(output) if 'Main_Vocals' in f][0]))
        except IndexError:
            logger.warning("Không tìm thấy giọng chính")
            return None, None, None
    
    if backing_reverb:
        try:
            dereveb_path.append(os.path.join(output, [f for f in os.listdir(output) if 'Backing_Vocals' in f][0]))
        except IndexError:
            logger.warning("Không tìm thấy giọng bè")
            return None, None, None
        
    for path in dereveb_path:
        if not os.path.exists(path): 
            logger.warning(f"Không tìm thấy: {path}")
            return None, None, None
        
        if "Original_Vocals" in path: 
            reverb_path = os.path.join(output, f"Original_Vocals_Reverb.{format}")
            no_reverb_path = os.path.join(output, f"Original_Vocals_No_Reverb.{format}")
            start_title = "Đang xử lý tách âm vang giọng gốc..."
            end_title = "Đã tách âm vang giọng gốc thành công!"
        elif "Main_Vocals" in path:
            reverb_path = os.path.join(output, f"Main_Vocals_Reverb.{format}")
            no_reverb_path = os.path.join(output, f"Main_Vocals_No_Reverb.{format}")
            start_title = "Đang xử lý tách âm vang giọng chính..."
            end_title = "Đã tách âm vang giọng chính thành công!"
        elif "Backing_Vocals" in path:
            reverb_path = os.path.join(output, f"Backing_Vocals_Reverb.{format}")
            no_reverb_path = os.path.join(output, f"Backing_Vocals_No_Reverb.{format}")
            start_title = "Đang xử lý tách âm vang giọng bè..."
            end_title = "Đã tách âm vang giọng bè thành công!"

        print(start_title)
        output_dereveb = separator_main(audio_file=path, model_filename="Reverb_HQ_By_FoxJoy.onnx", output_format=format, output_dir=output, mdx_segment_size=segments_size, mdx_overlap=overlap, mdx_batch_size=batch_size, mdx_hop_length=hop_length, mdx_enable_denoise=denoise)

        for f in output_dereveb:
            path = os.path.join(output, f)

            if not os.path.exists(path): logger.error(f"Không tìm thấy: {path}")

            if '_(Reverb)_' in f: os.rename(path, reverb_path)
            elif '_(No Reverb)_' in f: os.rename(path, no_reverb_path)

        print(end_title)

    return (os.path.join(output, f"Original_Vocals_No_Reverb.{format}") if original else None), (os.path.join(output, f"Main_Vocals_No_Reverb.{format}") if main else None), (os.path.join(output, f"Backing_Vocals_No_Reverb.{format}") if backing_reverb else None)

def separator_main(audio_file=None, model_filename="UVR-MDX-NET_Main_340.onnx", output_format="wav", output_dir=".", mdx_segment_size=256, mdx_overlap=0.25, mdx_batch_size=1, mdx_hop_length=1024, mdx_enable_denoise=True, demucs_segment_size=256, demucs_shifts=2, demucs_overlap=0.25):
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
        demucs_params={
            "segment_size": demucs_segment_size,
            "shifts": demucs_shifts,
            "overlap": demucs_overlap,
            "segments_enabled": True,
        }
    )

    separator.load_model(model_filename=model_filename)
    return separator.separate(audio_file)

if __name__ == "__main__": main()