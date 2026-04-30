import os
import re
import gc
import sys
import shutil
import datetime
import subprocess

import numpy as np

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning, gr_error, process_output, replace_export_format
from main.app.variables import logger, config, configs, translations, python, file_types, gradio_version

def convert(
    pitch, 
    filter_radius, 
    index_rate, 
    rms_mix_rate, 
    protect, 
    hop_length, 
    f0_method, 
    input_path, 
    output_path, 
    pth_path, 
    index_path, 
    f0_autotune, 
    clean_audio, 
    clean_strength, 
    export_format, 
    embedder_model, 
    resample_sr, 
    split_audio, 
    f0_autotune_strength, 
    checkpointing, 
    predictor_onnx, 
    embedders_mode, 
    formant_shifting, 
    formant_qfrency, 
    formant_timbre, 
    f0_file, 
    proposal_pitch, 
    proposal_pitch_threshold, 
    audio_processing=False, 
    alpha=0.5,
    sid=0,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5,
    noise_scale = 0.35
):    
    subprocess.run([
        python, 
        configs["convert_path"], 
        "--pitch", str(pitch), 
        "--filter_radius", str(filter_radius), 
        "--index_rate", str(index_rate), 
        "--rms_mix_rate", str(rms_mix_rate), 
        "--protect", str(protect), 
        "--hop_length", str(hop_length), 
        "--f0_method", f0_method, 
        "--input_path", input_path, 
        "--output_path", output_path, 
        "--pth_path", pth_path, 
        "--index_path", index_path, 
        "--f0_autotune", str(f0_autotune), 
        "--clean_audio", str(clean_audio), 
        "--clean_strength", str(clean_strength), 
        "--export_format", export_format, 
        "--embedder_model", embedder_model, 
        "--resample_sr", str(resample_sr), 
        "--split_audio", str(split_audio), 
        "--f0_autotune_strength", str(f0_autotune_strength), 
        "--checkpointing", str(checkpointing), 
        "--predictor_onnx", str(predictor_onnx), 
        "--embedders_mode", embedders_mode, 
        "--formant_shifting", str(formant_shifting), 
        "--formant_qfrency", str(formant_qfrency), 
        "--formant_timbre", str(formant_timbre), 
        "--f0_file", f0_file, 
        "--proposal_pitch", str(proposal_pitch), 
        "--proposal_pitch_threshold", str(proposal_pitch_threshold),
        "--audio_processing", str(audio_processing),
        "--alpha", str(alpha),
        "--sid", str(sid),
        "--embedders_mix", str(embedders_mix),
        "--embedders_mix_layers", str(embedders_mix_layers),
        "--embedders_mix_ratio", str(embedders_mix_ratio),
        "--noise_scale", str(noise_scale)
    ])

def convert_audio(
    clean_audio, 
    autotune, 
    use_separate_audio, 
    use_original, 
    convert_backing, 
    not_merge_backing, 
    merge_instrument, 
    pitch, 
    clean_strength, 
    model, 
    index, 
    index_rate, 
    input_path, 
    output_path, 
    export_format, 
    f0_method, 
    hybrid_method, 
    hop_length, 
    embedders, 
    custom_embedders, 
    resample_sr, 
    filter_radius, 
    rms_mix_rate, 
    protect, 
    split_audio, 
    f0_autotune_strength, 
    input_audio_name, 
    checkpointing, 
    predictor_onnx, 
    formant_shifting, 
    formant_qfrency, 
    formant_timbre, 
    f0_file, 
    embedders_mode, 
    proposal_pitch, 
    proposal_pitch_threshold, 
    audio_processing=False, 
    alpha=0.5,
    sid=0,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5,
    noise_scale = 0.35
):
    model_path = os.path.join(configs["weights_path"], model) if not os.path.exists(model) else model

    return_none = [None]*6
    return_none[5] = {"visible": True, "__type__": "update"}

    if (
        not use_separate_audio and (
            merge_instrument or 
            not_merge_backing or 
            convert_backing or 
            use_original
        )
    ):
        gr_warning(translations["turn_on_use_audio"])
        return return_none

    if use_original:
        if convert_backing:
            gr_warning(translations["turn_off_convert_backup"])
            return return_none
        elif not_merge_backing:
            gr_warning(translations["turn_off_merge_backup"])
            return return_none

    if (
        not model or 
        not os.path.exists(model_path) or 
        os.path.isdir(model_path) or 
        not model.endswith((".pth", ".onnx"))
    ):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return return_none

    f0method, embedder_model = (
        f0_method if f0_method != "hybrid" else hybrid_method, 
        embedders if embedders != "custom" else custom_embedders
    )

    if use_separate_audio:
        output_audio = os.path.join(configs["audios_path"], input_audio_name)

        from main.library.utils import pydub_load
        
        def get_audio_file(label):
            matching_files = [f for f in os.listdir(output_audio) if label in f]

            if not matching_files: return translations["notfound"]   
            return os.path.join(output_audio, matching_files[0])

        output_path = os.path.join(output_audio, f"Convert_Vocals.{export_format}")
        output_backing = os.path.join(output_audio, f"Convert_Backing.{export_format}")
        output_merge_backup = os.path.join(output_audio, f"Vocals+Backing.{export_format}")
        output_merge_instrument = os.path.join(output_audio, f"Vocals+Instruments.{export_format}")

        if os.path.exists(output_audio): os.makedirs(output_audio, exist_ok=True)
        output_path = process_output(output_path)

        if use_original:
            original_vocal = get_audio_file('Original_Vocals_No_Reverb.')
            if original_vocal == translations["notfound"]: 
                original_vocal = get_audio_file('Original_Vocals.')

            if original_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_original_vocal"])
                return return_none
            
            input_path = original_vocal
        else:
            main_vocal = get_audio_file('Main_Vocals_No_Reverb.')
            backing_vocal = get_audio_file('Backing_Vocals.')

            if main_vocal == translations["notfound"]: 
                main_vocal = get_audio_file('Main_Vocals.')

            if main_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_main_vocal"])
                return return_none
            
            if (
                not not_merge_backing and 
                backing_vocal == translations["notfound"]
            ): 
                gr_warning(translations["not_found_backing_vocal"])
                return return_none
            
            input_path = main_vocal
            backing_path = backing_vocal

        gr_info(translations["convert_vocal"])

        convert(
            pitch, 
            filter_radius, 
            index_rate, 
            rms_mix_rate, 
            protect, 
            hop_length, 
            f0method, 
            input_path, 
            output_path, 
            model_path, 
            index, 
            autotune, 
            clean_audio, 
            clean_strength, 
            export_format, 
            embedder_model, 
            resample_sr, 
            split_audio, 
            f0_autotune_strength, 
            checkpointing, 
            predictor_onnx, 
            embedders_mode, 
            formant_shifting, 
            formant_qfrency, 
            formant_timbre, 
            f0_file, 
            proposal_pitch, 
            proposal_pitch_threshold, 
            audio_processing, 
            alpha,
            sid,
            embedders_mix,
            embedders_mix_layers,
            embedders_mix_ratio,
            noise_scale
        )

        gr_info(translations["convert_success"])

        if convert_backing:
            output_backing = process_output(output_backing)
            gr_info(translations["convert_backup"])

            convert(
                pitch, 
                filter_radius, 
                index_rate, 
                rms_mix_rate, 
                protect, 
                hop_length, 
                f0method, 
                backing_path, 
                output_backing, 
                model_path, 
                index, 
                autotune, 
                clean_audio, 
                clean_strength, 
                export_format, 
                embedder_model, 
                resample_sr, 
                split_audio, 
                f0_autotune_strength, 
                checkpointing, 
                predictor_onnx, 
                embedders_mode, 
                formant_shifting, 
                formant_qfrency, 
                formant_timbre, 
                f0_file, 
                proposal_pitch, 
                proposal_pitch_threshold, 
                audio_processing, 
                alpha,
                sid,
                embedders_mix,
                embedders_mix_layers,
                embedders_mix_ratio,
                noise_scale
            )

            gr_info(translations["convert_backup_success"])

        try:
            if not not_merge_backing and not use_original:
                backing_source = (
                    output_backing 
                    if convert_backing else 
                    backing_vocal
                )

                output_merge_backup = process_output(output_merge_backup)

                gr_info(translations["merge_backup"])

                pydub_load(
                    output_path, 
                    volume=-4
                ).overlay(
                    pydub_load(
                        backing_source, 
                        volume=-6
                    )
                ).export(
                    output_merge_backup, 
                    format=export_format
                )

                gr_info(translations["merge_success"])

            if merge_instrument:    
                vocals = (
                    output_merge_backup 
                    if not not_merge_backing and not use_original else 
                    output_path
                )

                output_merge_instrument = process_output(output_merge_instrument)
                gr_info(translations["merge_instruments_process"])

                instruments = get_audio_file('Instruments.')
                if instruments == translations["notfound"]: 
                    gr_warning(translations["not_found_instruments"])

                    output_merge_instrument = None
                else: 
                    pydub_load(
                        instruments, 
                        volume=-7
                    ).overlay(
                        pydub_load(
                            vocals, 
                            volume=-4 if use_original else None
                        )
                    ).export(
                        output_merge_instrument, 
                        format=export_format
                    )
                
                gr_info(translations["merge_success"])
        except:
            return return_none

        return [
            (None if use_original else output_path), 
            output_backing, 
            (None if not_merge_backing and use_original else output_merge_backup), 
            (output_path if use_original else None), 
            (output_merge_instrument if merge_instrument else None), 
            {"visible": True, "__type__": "update"}
        ]
    else:
        if not input_path or not os.path.exists(input_path): 
            gr_warning(translations["input_not_valid"])
            return return_none
        
        if not output_path:
            gr_warning(translations["output_not_valid"])
            return return_none
        
        output_path = replace_export_format(output_path, export_format)

        if os.path.isdir(input_path):
            gr_info(translations["is_folder"])

            if not [
                f for f in os.listdir(input_path) 
                if f.lower().endswith(tuple(file_types))
            ]:
                gr_warning(translations["not_found_in_folder"])
                return return_none
            
            gr_info(translations["batch_convert"])
            output_dir = os.path.dirname(output_path) or output_path

            convert(
                pitch, 
                filter_radius, 
                index_rate, 
                rms_mix_rate, 
                protect, 
                hop_length, 
                f0method, 
                input_path, 
                output_dir, 
                model_path, 
                index, 
                autotune, 
                clean_audio, 
                clean_strength, 
                export_format, 
                embedder_model, 
                resample_sr, 
                split_audio, 
                f0_autotune_strength, 
                checkpointing, 
                predictor_onnx, 
                embedders_mode, 
                formant_shifting, 
                formant_qfrency, 
                formant_timbre, 
                f0_file, 
                proposal_pitch, 
                proposal_pitch_threshold, 
                audio_processing, 
                alpha,
                sid,
                embedders_mix,
                embedders_mix_layers,
                embedders_mix_ratio,
                noise_scale
            )

            gr_info(translations["batch_convert_success"])

            return return_none
        else:
            output_dir = os.path.dirname(output_path) or output_path

            if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
            output_path = process_output(output_path)

            gr_info(translations["convert_vocal"])

            convert(
                pitch, 
                filter_radius, 
                index_rate, 
                rms_mix_rate, 
                protect, 
                hop_length, 
                f0method, 
                input_path, 
                output_path, 
                model_path, 
                index, 
                autotune, 
                clean_audio, 
                clean_strength, 
                export_format, 
                embedder_model, 
                resample_sr, 
                split_audio, 
                f0_autotune_strength, 
                checkpointing, 
                predictor_onnx, 
                embedders_mode, 
                formant_shifting, 
                formant_qfrency, 
                formant_timbre, 
                f0_file, 
                proposal_pitch, 
                proposal_pitch_threshold, 
                audio_processing, 
                alpha,
                sid,
                embedders_mix,
                embedders_mix_layers,
                embedders_mix_ratio,
                noise_scale
            )

            gr_info(translations["convert_success"])

            return_none[0] = output_path
            return return_none

def convert_selection(
    clean_audio,
    autotune,
    use_separate_audio,
    use_original,
    convert_backing,
    not_merge_backing,
    merge_instrument,
    pitch,
    clean_strength,
    model,
    index,
    index_rate,
    input_path,
    output_path,
    export_format,
    f0_method,
    hybrid_method,
    hop_length,
    embedders,
    custom_embedders,
    resample_sr,
    filter_radius,
    rms_mix_rate,
    protect,
    split_audio,
    f0_autotune_strength,
    checkpointing,
    predictor_onnx,
    formant_shifting,
    formant_qfrency,
    formant_timbre,
    f0_file,
    embedders_mode,
    proposal_pitch,
    proposal_pitch_threshold,
    audio_processing=False,
    alpha=0.5,
    sid=0,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5,
    noise_scale = 0.35
):
    if use_separate_audio:
        gr_info(translations["search_separate"])
        choice = [
            f for f in os.listdir(configs["audios_path"]) 
            if os.path.isdir(os.path.join(configs["audios_path"], f))
        ] if config.debug_mode else [
            f for f in os.listdir(configs["audios_path"]) 
            if (
                os.path.isdir(os.path.join(configs["audios_path"], f)) and any(
                    file.lower().endswith(tuple(file_types)) 
                    for file in os.listdir(os.path.join(configs["audios_path"], f))
                )
            )
        ]

        gr_info(translations["found_choice"].format(choice=len(choice)))

        if len(choice) == 0: 
            gr_warning(translations["separator==0"])

            return [
                {
                    "choices": [], 
                    "value": "", 
                    "interactive": False, 
                    "visible": False if gradio_version else "hidden", 
                    "__type__": "update"
                }, 
                None, 
                None, 
                None, 
                None, 
                None, 
                {
                    "visible": True, 
                    "__type__": "update"
                }, 
                {
                    "visible": False if gradio_version else "hidden", 
                    "__type__": "update"
                }
            ]
        elif len(choice) == 1:
            convert_output = convert_audio(
                clean_audio, 
                autotune, 
                use_separate_audio, 
                use_original, 
                convert_backing, 
                not_merge_backing, 
                merge_instrument, 
                pitch, 
                clean_strength, 
                model, 
                index, 
                index_rate, 
                None, 
                None, 
                export_format, 
                f0_method, 
                hybrid_method, 
                hop_length, 
                embedders, 
                custom_embedders, 
                resample_sr, 
                filter_radius, 
                rms_mix_rate, 
                protect, 
                split_audio, 
                f0_autotune_strength, 
                choice[0], 
                checkpointing, 
                predictor_onnx, 
                formant_shifting, 
                formant_qfrency, 
                formant_timbre, 
                f0_file, 
                embedders_mode, 
                proposal_pitch, 
                proposal_pitch_threshold, 
                audio_processing, 
                alpha,
                sid,
                embedders_mix,
                embedders_mix_layers,
                embedders_mix_ratio,
                noise_scale
            )

            return [
                {
                    "choices": [], 
                    "value": "", 
                    "interactive": False, 
                    "visible": False if gradio_version else "hidden", 
                    "__type__": "update"
                }, 
                convert_output[0], 
                convert_output[1], 
                convert_output[2], 
                convert_output[3], 
                convert_output[4], 
                {
                    "visible": True, 
                    "__type__": "update"
                }, 
                {
                    "visible": False if gradio_version else "hidden", 
                    "__type__": "update"
                }
            ]
        else: 
            return [
                {
                    "choices": choice, 
                    "value": choice[0], 
                    "interactive": True, 
                    "visible": True, 
                    "__type__": "update"
                }, 
                None, 
                None, 
                None, 
                None, 
                None, 
                {
                    "visible": False if gradio_version else "hidden", 
                    "__type__": "update"
                }, 
                {
                    "visible": True, 
                    "__type__": "update"
                }
            ]
    else:
        main_convert = convert_audio(
            clean_audio, 
            autotune, 
            use_separate_audio, 
            use_original, 
            convert_backing, 
            not_merge_backing, 
            merge_instrument, 
            pitch, 
            clean_strength, 
            model, 
            index, 
            index_rate, 
            input_path, 
            output_path, 
            export_format, 
            f0_method, 
            hybrid_method, 
            hop_length, 
            embedders, 
            custom_embedders, 
            resample_sr, 
            filter_radius, 
            rms_mix_rate, 
            protect, 
            split_audio, 
            f0_autotune_strength, 
            None, 
            checkpointing, 
            predictor_onnx, 
            formant_shifting, 
            formant_qfrency, 
            formant_timbre, 
            f0_file, 
            embedders_mode, 
            proposal_pitch, 
            proposal_pitch_threshold, 
            audio_processing, 
            alpha,
            sid,
            embedders_mix,
            embedders_mix_layers,
            embedders_mix_ratio,
            noise_scale
        )

        return [
            {
                "choices": [], 
                "value": "", 
                "interactive": False, 
                "visible": False if gradio_version else "hidden", 
                "__type__": "update"
            }, 
            main_convert[0], 
            None, 
            None, 
            None, 
            None, 
            {
                "visible": True, 
                "__type__": "update"
            }, 
            {
                "visible": False if gradio_version else "hidden", 
                "__type__": "update"
            }
        ]

def convert_with_vad(
    num_spk, 
    cleaner, 
    clean_strength, 
    autotune, 
    f0_autotune_strength, 
    checkpointing, 
    model_1, 
    model_2, 
    model_index_1, 
    model_index_2, 
    pitch_1, 
    pitch_2, 
    index_strength_1, 
    index_strength_2, 
    export_format, 
    input_audio, 
    output_audio, 
    predictor_onnx, 
    f0_method, 
    hybrid_method, 
    hop_length, 
    embed_mode, 
    embedders, 
    custom_embedders, 
    resample_sr, 
    filter_radius, 
    rms_mix_rate, 
    protect, 
    formant_shifting, 
    formant_qfrency_1, 
    formant_timbre_1, 
    formant_qfrency_2, 
    formant_timbre_2, 
    proposal_pitch, 
    proposal_pitch_threshold, 
    audio_processing=False, 
    alpha=0.5,
    sid_1=0,
    sid_2=0,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5,
    noise_scale_1 = 0.35,
    noise_scale_2 = 0.35,
    vad_sensitivity = 3,
    vad_frame_ms = 30
):
    import librosa

    from pydub import AudioSegment
    from sklearn.cluster import AgglomerativeClustering

    from main.library.utils import clear_gpu_cache
    from main.library.speaker_diarization.audio import Audio
    from main.library.speaker_diarization.segment import Segment
    from main.library.utils import check_spk_diarization, pydub_load
    from main.library.speaker_diarization.embedding import SpeechBrainPretrainedSpeakerEmbedding
    from main.inference.realtime.vad_utils import VADProcessor
    
    check_spk_diarization(model_size=None)
    model_pth_1, model_pth_2 = (
        os.path.join(configs["weights_path"], model_1) if not os.path.exists(model_1) else model_1, 
        os.path.join(configs["weights_path"], model_2) if not os.path.exists(model_2) else model_2
    )

    if (
        not model_1 or 
        not os.path.exists(model_pth_1) or 
        os.path.isdir(model_pth_1) or 
        not model_pth_1.endswith((".pth", ".onnx"))
    ) and (
        not model_2 or 
        not os.path.exists(model_pth_2) or 
        os.path.isdir(model_pth_2) or 
        not model_pth_2.endswith((".pth", ".onnx"))
    ):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return None
    
    if not model_1: model_pth_1 = model_pth_2
    if not model_2: model_pth_2 = model_pth_1

    if not input_audio or not os.path.exists(input_audio) or os.path.isdir(input_audio): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_audio:
        gr_warning(translations["output_not_valid"])
        return None
    
    output_audio = process_output(output_audio)
    gr_info(translations["start_vad"])
    
    try:
        y, sr = librosa.load(input_audio, sr=48000)

        vad_processor = VADProcessor(sensitivity_mode=vad_sensitivity, frame_duration_ms=vad_frame_ms, sample_rate=sr)
        segments = vad_processor.get_speech(librosa.util.normalize(y))

        if not segments:
            gr_warning(translations["speech_not_in_segments"])
            return None
        
        audio = Audio()

        embedding_model = SpeechBrainPretrainedSpeakerEmbedding(
            embedding=os.path.join(configs["speaker_diarization_path"], "models", "speechbrain"), 
            device=config.device
        )
            
        def segment_embedding(segment):
            waveform, _ = audio.crop(
                input_audio, 
                Segment(
                    segment["start"], 
                    segment["end"]
                )
            )

            return embedding_model(
                waveform.mean(dim=0, keepdim=True)[None] if waveform.shape[0] == 2 else waveform[None]
            )  

        def get_formatted_stats(merged_segments):
            rows, lines = [], []
            speaker_stats = {}
            headers = translations["headers"].split(" ")

            for seg in merged_segments:
                spk = seg["speaker"]
                start_t = seg["start"]
                end_t = seg["end"]
                dur = end_t - start_t

                time_range = f"{str(datetime.timedelta(seconds=int(start_t)))} -> {str(datetime.timedelta(seconds=int(end_t)))}"
                dur_str = f"{dur:.2f}s"

                rows.append([spk, time_range, dur_str])
                speaker_stats[spk] = speaker_stats.get(spk, 0.0) + dur
            col_widths = []
            for i in range(3):
                max_len = max(len(headers[i].replace("-", " ")), *(len(row[i]) for row in rows))
                col_widths.append(max_len)

            def make_line(left, mid, right):
                return left + mid.join("─" * (w + 2) for w in col_widths) + right

            def format_row(row):
                return "│ " + " │ ".join(f"{cell:<{col_widths[i]}}" for i, cell in enumerate(row)) + " │"
            
            lines.append("\n")
            lines.append(make_line("┌", "┬", "┐"))
            lines.append(format_row(headers))
            lines.append(make_line("├", "┼", "┤"))

            for row in rows:
                lines.append(format_row(row))

            lines.append(make_line("└", "┴", "┘"))
            return "\n".join(lines)
        
        def merge_audio(
            files_list, 
            time_stamps, 
            original_file_path, 
            output_path, 
            export_format
        ):
            def extract_number(filename):
                match = re.search(r'_(\d+)', filename)
                return int(match.group(1)) if match else 0

            total_duration = len(pydub_load(original_file_path))
            combined = AudioSegment.empty() 
            current_position = 0 

            for file, (start_i, end_i) in zip(sorted(files_list, key=extract_number), time_stamps):
                if start_i > current_position: 
                    combined += AudioSegment.silent(
                        duration=start_i - current_position
                    )  
                
                combined += pydub_load(file)  
                current_position = end_i

            if current_position < total_duration: 
                combined += AudioSegment.silent(
                    duration=total_duration - current_position
                )

            combined.export(output_path, format=export_format)
            return output_path

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)

        labels = AgglomerativeClustering(num_spk).fit(np.nan_to_num(embeddings)).labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = f"SPEAKER {labels[i] + 1}"
        
        merged_segments = []

        if segments:
            curr = segments[0].copy()

            for i in range(1, len(segments)):
                if segments[i]["speaker"] == curr["speaker"]: curr["end"] = segments[i]["end"]
                else:
                    merged_segments.append(curr)
                    curr = segments[i].copy()

            merged_segments.append(curr)

        gr_info(translations["analysis_completed"].format(length=len(merged_segments)))
        
        del audio, embedding_model, segments, labels
        clear_gpu_cache()
        gc.collect()

        gr_info(translations["process_audio"])

        pydub_audio = pydub_load(input_audio)
        output_folder = "audios_temp"

        if os.path.exists(output_folder): shutil.rmtree(output_folder, ignore_errors=True)

        for f in [
            output_folder, 
            os.path.join(output_folder, "1"), 
            os.path.join(output_folder, "2")
        ]:
            os.makedirs(f, exist_ok=True)

        time_stamps, processed_segments = [], []
        for i, seg in enumerate(merged_segments):
            model_id = 1 if int(seg["speaker"].split()[-1]) % 2 != 0 else 2
            start_ms, end_ms = int(seg["start"] * 1000), int(seg["end"] * 1000)

            target_path = os.path.join(output_folder, str(model_id), f"segment_{i+1}.wav")
            pydub_audio[start_ms:end_ms].export(target_path, format="wav")
            
            processed_segments.append(os.path.join(output_folder, str(model_id), f"segment_{i+1}_output.wav"))
            time_stamps.append((start_ms, end_ms))

        logger.info(get_formatted_stats(merged_segments))

        f0method, embedder_model = (
            f0_method if f0_method != "hybrid" else hybrid_method, 
            embedders if embedders != "custom" else custom_embedders
        )

        gr_info(translations["process_done_start_convert"])

        convert(
            pitch_1, 
            filter_radius, 
            index_strength_1, 
            rms_mix_rate, 
            protect, 
            hop_length, 
            f0method, 
            os.path.join(output_folder, "1"), 
            output_folder, 
            model_pth_1, 
            model_index_1, 
            autotune, 
            cleaner, 
            clean_strength, 
            "wav", 
            embedder_model, 
            48000 if resample_sr == 0 else resample_sr, 
            False, 
            f0_autotune_strength, 
            checkpointing, 
            predictor_onnx, 
            embed_mode, 
            formant_shifting, 
            formant_qfrency_1, 
            formant_timbre_1, 
            "", 
            proposal_pitch, 
            proposal_pitch_threshold, 
            audio_processing, 
            alpha,
            sid_1,
            embedders_mix,
            embedders_mix_layers,
            embedders_mix_ratio,
            noise_scale_1
        )

        convert(
            pitch_2, 
            filter_radius, 
            index_strength_2, 
            rms_mix_rate, 
            protect, 
            hop_length, 
            f0method, 
            os.path.join(output_folder, "2"), 
            output_folder, 
            model_pth_2, 
            model_index_2, 
            autotune, 
            cleaner, 
            clean_strength, 
            "wav", 
            embedder_model, 
            48000 if resample_sr == 0 else resample_sr, 
            False, 
            f0_autotune_strength, 
            checkpointing, 
            predictor_onnx, 
            embed_mode, 
            formant_shifting, 
            formant_qfrency_2, 
            formant_timbre_2, 
            "", 
            proposal_pitch, 
            proposal_pitch_threshold, 
            audio_processing, 
            alpha,
            sid_2,
            embedders_mix,
            embedders_mix_layers,
            embedders_mix_ratio,
            noise_scale_2
        )

        gr_info(translations["convert_success"])

        return merge_audio(
            processed_segments, 
            time_stamps, 
            input_audio, 
            replace_export_format(output_audio, export_format), 
            export_format
        )
    except Exception as e:
        import traceback

        gr_error(translations["error_occurred"].format(e=e))
        logger.debug(traceback.format_exc())

        return None
    finally:
        if os.path.exists("audios_temp"): 
            shutil.rmtree("audios_temp", ignore_errors=True)

def convert_tts(
    clean_audio, 
    autotune, 
    pitch, 
    clean_strength, 
    model, 
    index, 
    index_rate, 
    input_path, 
    output_path, 
    export_format, 
    method, 
    hybrid_method, 
    hop_length, 
    embedders, 
    custom_embedders, 
    resample_sr, 
    filter_radius, 
    rms_mix_rate, 
    protect, 
    split_audio, 
    f0_autotune_strength, 
    checkpointing, 
    predictor_onnx, 
    formant_shifting, 
    formant_qfrency, 
    formant_timbre, 
    f0_file, 
    embedders_mode, 
    proposal_pitch, 
    proposal_pitch_threshold, 
    audio_processing=False, 
    alpha=0.5,
    sid=0,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5,
    noise_scale = 0.35
):
    model_path = os.path.join(configs["weights_path"], model) if not os.path.exists(model) else model

    if (
        not model_path or 
        not os.path.exists(model_path) or 
        os.path.isdir(model_path) or 
        not model.endswith((".pth", ".onnx"))
    ):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return None

    if not input_path or not os.path.exists(input_path): 
        gr_warning(translations["input_not_valid"])
        return None
    
    if os.path.isdir(input_path): 
        input_audio = [
            f for f in os.listdir(input_path) 
            if "tts" in f and f.lower().endswith(tuple(file_types))
        ]
        
        if not input_audio:
            gr_warning(translations["not_found_in_folder"])
            return None
        
        input_path = os.path.join(input_path, input_audio[0])
    
    if not output_path:
        gr_warning(translations["output_not_valid"])
        return None
    
    output_path = replace_export_format(output_path, export_format)
    if os.path.isdir(output_path): output_path = os.path.join(output_path, f"tts.{export_format}")

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    output_path = process_output(output_path)

    f0method, embedder_model = (
        method if method != "hybrid" else hybrid_method, 
        embedders if embedders != "custom" else custom_embedders
    )

    gr_info(translations["convert_vocal"])

    convert(
        pitch, 
        filter_radius, 
        index_rate, 
        rms_mix_rate, 
        protect, 
        hop_length, 
        f0method, 
        input_path, 
        output_path, 
        model_path, 
        index, 
        autotune, 
        clean_audio, 
        clean_strength, 
        export_format, 
        embedder_model, 
        resample_sr, 
        split_audio, 
        f0_autotune_strength, 
        checkpointing, 
        predictor_onnx, 
        embedders_mode, 
        formant_shifting, 
        formant_qfrency, 
        formant_timbre, 
        f0_file, 
        proposal_pitch, 
        proposal_pitch_threshold, 
        audio_processing, 
        alpha,
        sid,
        embedders_mix,
        embedders_mix_layers,
        embedders_mix_ratio,
        noise_scale
    )

    gr_info(translations["convert_success"])
    return output_path