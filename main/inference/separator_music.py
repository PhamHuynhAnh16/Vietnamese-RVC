import os
import sys
import time
import argparse

import numpy as np

from distutils.util import strtobool

sys.path.append(os.getcwd())

from main.library.utils import pydub_load
from main.library.uvr5_separator.separator import Separator
from main.app.variables import config, logger, translations

demucs_models = {"HT-Tuned": "htdemucs_ft.yaml", "HT-Normal": "htdemucs.yaml", "HD_MMI": "hdemucs_mmi.yaml", "HT_6S": "htdemucs_6s.yaml"}
mdx_models = {"Main_340": "UVR-MDX-NET_Main_340.onnx", "Main_390": "UVR-MDX-NET_Main_390.onnx", "Main_406": "UVR-MDX-NET_Main_406.onnx", "Main_427": "UVR-MDX-NET_Main_427.onnx","Main_438": "UVR-MDX-NET_Main_438.onnx", "Inst_full_292": "UVR-MDX-NET-Inst_full_292.onnx", "Inst_HQ_1": "UVR-MDX-NET-Inst_HQ_1.onnx", "Inst_HQ_2": "UVR-MDX-NET-Inst_HQ_2.onnx", "Inst_HQ_3": "UVR-MDX-NET-Inst_HQ_3.onnx", "Inst_HQ_4": "UVR-MDX-NET-Inst_HQ_4.onnx", "Inst_HQ_5": "UVR-MDX-NET-Inst_HQ_5.onnx", "Kim_Vocal_1": "Kim_Vocal_1.onnx", "Kim_Vocal_2": "Kim_Vocal_2.onnx", "Kim_Inst": "Kim_Inst.onnx", "Inst_187_beta": "UVR-MDX-NET_Inst_187_beta.onnx", "Inst_82_beta": "UVR-MDX-NET_Inst_82_beta.onnx", "Inst_90_beta": "UVR-MDX-NET_Inst_90_beta.onnx", "Voc_FT": "UVR-MDX-NET-Voc_FT.onnx", "Crowd_HQ": "UVR-MDX-NET_Crowd_HQ_1.onnx", "MDXNET_9482": "UVR_MDXNET_9482.onnx", "Inst_1": "UVR-MDX-NET-Inst_1.onnx", "Inst_2": "UVR-MDX-NET-Inst_2.onnx", "Inst_3": "UVR-MDX-NET-Inst_3.onnx", "MDXNET_1_9703": "UVR_MDXNET_1_9703.onnx", "MDXNET_2_9682": "UVR_MDXNET_2_9682.onnx", "MDXNET_3_9662": "UVR_MDXNET_3_9662.onnx", "Inst_Main": "UVR-MDX-NET-Inst_Main.onnx", "MDXNET_Main": "UVR_MDXNET_Main.onnx"}
kara_models = {"Version-1": "UVR_MDXNET_KARA.onnx", "Version-2": "UVR_MDXNET_KARA_2.onnx"}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--separator_music", action='store_true')
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./audios")
    parser.add_argument("--format", type=str, default="wav")
    parser.add_argument("--shifts", type=int, default=2)
    parser.add_argument("--segments_size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--mdx_hop_length", type=int, default=1024)
    parser.add_argument("--mdx_batch_size", type=int, default=1)
    parser.add_argument("--clean_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)
    parser.add_argument("--model_name", type=str, default="HT-Normal")
    parser.add_argument("--kara_model", type=str, default="Version-1")
    parser.add_argument("--backing", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--mdx_denoise", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--reverb", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--backing_reverb", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--sample_rate", type=int, default=44100)

    return parser.parse_args()

def main():
    start_time = time.time()
    pid_path = os.path.join("assets", "separate_pid.txt")

    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    try:
        args = parse_arguments()
        input_path, output_path, export_format, shifts, segments_size, overlap, hop_length, batch_size, clean_audio, clean_strength, model_name, kara_model, backing, mdx_denoise, reverb, backing_reverb, sample_rate = args.input_path, args.output_path, args.format, args.shifts, args.segments_size, args.overlap, args.mdx_hop_length, args.mdx_batch_size, args.clean_audio, args.clean_strength, args.model_name, args.kara_model, args.backing, args.mdx_denoise, args.reverb, args.backing_reverb, args.sample_rate

        if backing_reverb and not reverb: 
            logger.warning(translations["turn_on_dereverb"])
            sys.exit(1)

        if backing_reverb and not backing: 
            logger.warning(translations["turn_on_separator_backing"])
            sys.exit(1)

        input_path = input_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        output_path = os.path.dirname(output_path) or output_path

        log_data = {translations['audio_path']: input_path, translations['output_path']: output_path, translations['export_format']: export_format, translations['shift']: shifts, translations['segments_size']: segments_size, translations['overlap']: overlap, translations['modelname']: model_name, translations['denoise_mdx']: mdx_denoise, "Hop length": hop_length, translations['batch_size']: batch_size, translations['sr']: sample_rate}

        if clean_audio:
            log_data[translations['clear_audio']] = clean_audio
            log_data[translations['clean_strength']] = clean_strength

        if backing:
            log_data[translations['backing_model_ver']] = kara_model
            log_data[translations['separator_backing']] = backing

        if reverb:
            log_data[translations['dereveb_audio']] = reverb
            log_data[translations['dereveb_backing']] = backing_reverb

        for key, value in log_data.items():
            logger.debug(f"{key}: {value}")

        if os.path.isdir(input_path):
            for f in os.listdir(input_path):
                f_path = os.path.join(input_path, f)
                if os.path.isfile(f_path): separation(f_path, output_path, export_format, shifts, overlap, segments_size, model_name, sample_rate, mdx_denoise, hop_length, batch_size, backing, reverb, kara_model, backing_reverb, clean_audio, clean_strength)
        else: separation(input_path, output_path, export_format, shifts, overlap, segments_size, model_name, sample_rate, mdx_denoise, hop_length, batch_size, backing, reverb, kara_model, backing_reverb, clean_audio, clean_strength)

    except Exception as e:
        logger.error(f"{translations['separator_error']}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    if os.path.exists(pid_path): os.remove(pid_path)
    elapsed_time = time.time() - start_time
    logger.info(translations["separator_success"].format(elapsed_time=f"{elapsed_time:.2f}"))

def separation(input_path, output_path, export_format, shifts, overlap, segments_size, model_name, sample_rate, mdx_denoise, hop_length, batch_size, backing, reverb, kara_model, backing_reverb, clean_audio, clean_strength):
    filename, _ = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(output_path, filename)
    os.makedirs(output_path, exist_ok=True)

    if model_name in ["HT-Tuned", "HT-Normal", "HD_MMI", "HT_6S"]: vocals, _ = separator_music_demucs(input_path, output_path, export_format, shifts, overlap, segments_size, model_name, sample_rate)
    else: vocals, _ = separator_music_mdx(input_path, output_path, export_format, segments_size, overlap, mdx_denoise, model_name, hop_length, batch_size, sample_rate)

    if backing: main_vocals, backing_vocals = separator_backing(vocals, output_path, export_format, segments_size, overlap, mdx_denoise, kara_model, hop_length, batch_size, sample_rate)
    if reverb: vocals_no_reverb, main_vocals_no_reverb, backing_vocals_no_reverb = separator_reverb(output_path, export_format, segments_size, overlap, mdx_denoise, reverb, backing_reverb, hop_length, batch_size, sample_rate)

    original_output = os.path.join(output_path, f"Original_Vocals_No_Reverb.{export_format}") if reverb else os.path.join(output_path, f"Original_Vocals.{export_format}")
    main_output = os.path.join(output_path, f"Main_Vocals_No_Reverb.{export_format}") if reverb and backing_reverb else os.path.join(output_path, f"Main_Vocals.{export_format}")
    backing_output = os.path.join(output_path, f"Backing_Vocals_No_Reverb.{export_format}") if reverb and backing_reverb else os.path.join(output_path, f"Backing_Vocals.{export_format}")
    
    if clean_audio:
        import soundfile as sf
        
        logger.info(f"{translations['clear_audio']}...")
        vocal_data, vocal_sr = sf.read(vocals_no_reverb if reverb else vocals, dtype=np.float32)

        from main.tools.noisereduce import reduce_noise
        sf.write(original_output, reduce_noise(y=vocal_data, sr=vocal_sr, prop_decrease=clean_strength), vocal_sr, format=export_format, device=config.device)

        if backing:
            main_data, main_sr = sf.read(main_vocals_no_reverb if reverb and backing else main_vocals, dtype=np.float32)
            backing_data, backing_sr = sf.read(backing_vocals_no_reverb if reverb and backing_reverb else backing_vocals, dtype=np.float32)

            sf.write(main_output, reduce_noise(y=main_data, sr=main_sr, prop_decrease=clean_strength), main_sr, format=export_format, device=config.device)
            sf.write(backing_output, reduce_noise(y=backing_data, sr=backing_sr, prop_decrease=clean_strength), backing_sr, format=export_format, device=config.device)  

        logger.info(translations["clean_audio_success"])
        
def separator_music_demucs(input, output, format, shifts, overlap, segments_size, demucs_model, sample_rate):
    if not os.path.exists(input): 
        logger.warning(translations["input_not_valid"])
        sys.exit(1)
    
    if not os.path.exists(output): 
        logger.warning(translations["output_not_valid"])
        sys.exit(1)
    
    for i in [f"Original_Vocals.{format}", f"Instruments.{format}"]:
        if os.path.exists(os.path.join(output, i)): os.remove(os.path.join(output, i))

    logger.info(f"{translations['separator_process_2']}...")
    demucs_output = separator_main(audio_file=input, model_filename=demucs_models.get(demucs_model), output_format=format, output_dir=output, demucs_segment_size=(segments_size / 2), demucs_shifts=shifts, demucs_overlap=overlap, sample_rate=sample_rate)
    
    for f in demucs_output:
        path = os.path.join(output, f)
        if not os.path.exists(path): logger.error(translations["not_found"].format(name=path))

        if '_(Drums)_' in f: drums = path
        elif '_(Bass)_' in f: bass = path
        elif '_(Other)_' in f: other = path
        elif '_(Vocals)_' in f: os.rename(path, os.path.join(output, f"Original_Vocals.{format}"))

    pydub_load(drums).overlay(pydub_load(bass)).overlay(pydub_load(other)).export(os.path.join(output, f"Instruments.{format}"), format=format)

    for f in [drums, bass, other]:
        if os.path.exists(f): os.remove(f)
    
    logger.info(translations["separator_success_2"])
    return os.path.join(output, f"Original_Vocals.{format}"), os.path.join(output, f"Instruments.{format}")

def separator_backing(input, output, format, segments_size, overlap, denoise, kara_model, hop_length, batch_size, sample_rate):
    if not os.path.exists(input): 
        logger.warning(translations["input_not_valid"])
        sys.exit(1)
    
    if not os.path.exists(output): 
        logger.warning(translations["output_not_valid"])
        sys.exit(1)
    
    for f in [f"Main_Vocals.{format}", f"Backing_Vocals.{format}"]:
        if os.path.exists(os.path.join(output, f)): os.remove(os.path.join(output, f))

    model_2 = kara_models.get(kara_model)
    logger.info(f"{translations['separator_process_backing']}...")

    backing_outputs = separator_main(audio_file=input, model_filename=model_2, output_format=format, output_dir=output, mdx_segment_size=segments_size, mdx_overlap=overlap, mdx_batch_size=batch_size, mdx_hop_length=hop_length, mdx_enable_denoise=denoise, sample_rate=sample_rate)
    main_output = os.path.join(output, f"Main_Vocals.{format}")
    backing_output = os.path.join(output, f"Backing_Vocals.{format}")

    for f in backing_outputs:
        path = os.path.join(output, f)
        if not os.path.exists(path): logger.error(translations["not_found"].format(name=path))

        if '_(Instrumental)_' in f: os.rename(path, backing_output)
        elif '_(Vocals)_' in f: os.rename(path, main_output)

    logger.info(translations["separator_process_backing_success"])
    return main_output, backing_output

def separator_music_mdx(input, output, format, segments_size, overlap, denoise, mdx_model, hop_length, batch_size, sample_rate):
    if not os.path.exists(input): 
        logger.warning(translations["input_not_valid"])
        sys.exit(1)
    
    if not os.path.exists(output): 
        logger.warning(translations["output_not_valid"])
        sys.exit(1)

    for i in [f"Original_Vocals.{format}", f"Instruments.{format}"]:
        if os.path.exists(os.path.join(output, i)): os.remove(os.path.join(output, i))
    
    model_3 = mdx_models.get(mdx_model)
    logger.info(f"{translations['separator_process_2']}...")

    output_music = separator_main(audio_file=input, model_filename=model_3, output_format=format, output_dir=output, mdx_segment_size=segments_size, mdx_overlap=overlap, mdx_batch_size=batch_size, mdx_hop_length=hop_length, mdx_enable_denoise=denoise, sample_rate=sample_rate)
    original_output, instruments_output = os.path.join(output, f"Original_Vocals.{format}"), os.path.join(output, f"Instruments.{format}")

    for f in output_music:
        path = os.path.join(output, f)
        if not os.path.exists(path): logger.error(translations["not_found"].format(name=path))

        if '_(Instrumental)_' in f: os.rename(path, instruments_output)
        elif '_(Vocals)_' in f: os.rename(path, original_output)

    logger.info(translations["separator_success_2"])
    return original_output, instruments_output

def separator_reverb(output, format, segments_size, overlap, denoise, original, backing_reverb, hop_length, batch_size, sample_rate):
    if not os.path.exists(output): 
        logger.warning(translations["output_not_valid"])
        sys.exit(1)
    
    for i in [f"Original_Vocals_Reverb.{format}", f"Main_Vocals_Reverb.{format}", f"Original_Vocals_No_Reverb.{format}", f"Main_Vocals_No_Reverb.{format}"]:
        if os.path.exists(os.path.join(output, i)): os.remove(os.path.join(output, i))

    dereveb_path = []

    if original: 
        try:
            dereveb_path.append(os.path.join(output, [f for f in os.listdir(output) if 'Original_Vocals' in f][0]))
        except IndexError:
            logger.warning(translations["not_found_original_vocal"])
            sys.exit(1)
        
    if backing_reverb:
        try:
            dereveb_path.append(os.path.join(output, [f for f in os.listdir(output) if 'Main_Vocals' in f][0]))
        except IndexError:
            logger.warning(translations["not_found_main_vocal"])
            sys.exit(1)
    
    if backing_reverb:
        try:
            dereveb_path.append(os.path.join(output, [f for f in os.listdir(output) if 'Backing_Vocals' in f][0]))
        except IndexError:
            logger.warning(translations["not_found_backing_vocal"])
            sys.exit(1)
    
    for path in dereveb_path:
        if not os.path.exists(path): 
            logger.warning(translations["not_found"].format(name=path))
            sys.exit(1)
        
        if "Original_Vocals" in path: 
            reverb_path, no_reverb_path = os.path.join(output, f"Original_Vocals_Reverb.{format}"), os.path.join(output, f"Original_Vocals_No_Reverb.{format}")
            start_title, end_title = translations["process_original"], translations["process_original_success"]
        elif "Main_Vocals" in path:
            reverb_path, no_reverb_path = os.path.join(output, f"Main_Vocals_Reverb.{format}"), os.path.join(output, f"Main_Vocals_No_Reverb.{format}")
            start_title, end_title = translations["process_main"], translations["process_main_success"]
        elif "Backing_Vocals" in path:
            reverb_path, no_reverb_path = os.path.join(output, f"Backing_Vocals_Reverb.{format}"), os.path.join(output, f"Backing_Vocals_No_Reverb.{format}")
            start_title, end_title = translations["process_backing"], translations["process_backing_success"]

        logger.info(start_title)
        output_dereveb = separator_main(audio_file=path, model_filename="Reverb_HQ_By_FoxJoy.onnx", output_format=format, output_dir=output, mdx_segment_size=segments_size, mdx_overlap=overlap, mdx_batch_size=batch_size, mdx_hop_length=hop_length, mdx_enable_denoise=denoise, sample_rate=sample_rate)

        for f in output_dereveb:
            path = os.path.join(output, f)
            if not os.path.exists(path): logger.error(translations["not_found"].format(name=path))

            if '_(Reverb)_' in f: os.rename(path, reverb_path)
            elif '_(No Reverb)_' in f: os.rename(path, no_reverb_path)

        logger.info(end_title)

    return (os.path.join(output, f"Original_Vocals_No_Reverb.{format}") if original else None), (os.path.join(output, f"Main_Vocals_No_Reverb.{format}") if backing_reverb else None), (os.path.join(output, f"Backing_Vocals_No_Reverb.{format}") if backing_reverb else None)

def separator_main(audio_file=None, model_filename="UVR-MDX-NET_Main_340.onnx", output_format="wav", output_dir=".", mdx_segment_size=256, mdx_overlap=0.25, mdx_batch_size=1, mdx_hop_length=1024, mdx_enable_denoise=True, demucs_segment_size=256, demucs_shifts=2, demucs_overlap=0.25, sample_rate=44100):
    try:
        separator = Separator(logger=logger, output_dir=output_dir, output_format=output_format, output_bitrate=None, normalization_threshold=0.9, output_single_stem=None, invert_using_spec=False, sample_rate=sample_rate, mdx_params={"hop_length": mdx_hop_length, "segment_size": mdx_segment_size, "overlap": mdx_overlap, "batch_size": mdx_batch_size, "enable_denoise": mdx_enable_denoise}, demucs_params={"segment_size": demucs_segment_size, "shifts": demucs_shifts, "overlap": demucs_overlap, "segments_enabled": config.configs.get("demucs_segments_enable", True)})
        separator.load_model(model_filename=model_filename)

        return separator.separate(audio_file)
    except:
        logger.debug(translations["default_setting"])
        separator = Separator(logger=logger, output_dir=output_dir, output_format=output_format, output_bitrate=None, normalization_threshold=0.9, output_single_stem=None, invert_using_spec=False, sample_rate=44100, mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": mdx_enable_denoise}, demucs_params={"segment_size": 128, "shifts": 2, "overlap": 0.25, "segments_enabled": config.configs.get("demucs_segments_enable", True)})
        separator.load_model(model_filename=model_filename)

        return separator.separate(audio_file)

if __name__ == "__main__": main()