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
    pitch=0, 
    filter_radius=3, 
    index_rate=0.5, 
    rms_mix_rate=1.0, 
    protect=0.33, 
    hop_length=160, 
    f0_method="rmvpe", 
    input_path=None, 
    output_path="audios/output.wav", 
    pth_path=None, 
    index_path=None, 
    f0_autotune=1.0, 
    clean_audio=False, 
    clean_strength=0.5, 
    export_format="wav", 
    embedder_model="hubert_base", 
    resample_sr=0, 
    split_audio=False, 
    f0_autotune_strength=1.0, 
    checkpointing=False, 
    predictor_onnx=False, 
    embedders_mode="fairseq", 
    formant_shifting=False, 
    formant_qfrency=1.0, 
    formant_timbre=1.0, 
    f0_file=None, 
    proposal_pitch=False, 
    proposal_pitch_threshold=255, 
    audio_processing=False, 
    alpha=0.5,
    sid=0,
    embedders_mix=False,
    embedders_mix_layers=9,
    embedders_mix_ratio=0.5,
    noise_scale=0.35,
    nprobe=1,
    audio_upscaler=False
):    
    """
    Executes the core voice conversion script as a separate subprocess.

    This function wraps the command-line interface execution for the RVC/SVC inference 
    process, passing all hyperparameters as string arguments to a Python subprocess.
    """

    # Invoke the underlying RVC inference script via a command line subprocess execution
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
        "--noise_scale", str(noise_scale),
        "--nprobe", str(nprobe),
        "--audio_upscaler", str(audio_upscaler)
    ])

def convert_audio(
    clean_audio=False, 
    autotune=False, 
    use_separate_audio=False, 
    use_original=False, 
    convert_backing=False, 
    not_merge_backing=False, 
    merge_instrument=False, 
    pitch=0, 
    clean_strength=0.5, 
    model=None, 
    index=None, 
    index_rate=0.5, 
    input_path=None, 
    output_path=None, 
    export_format="wav", 
    f0_method="rmvpe", 
    hybrid_method="hybrid[rmvpe+harvest]", 
    hop_length=160, 
    embedders="hubert_base", 
    custom_embedders=None, 
    resample_sr=0, 
    filter_radius=3, 
    rms_mix_rate=1.0, 
    protect=0.33, 
    split_audio=False, 
    f0_autotune_strength=1.0, 
    input_audio_name=None, 
    checkpointing=False, 
    predictor_onnx=False, 
    formant_shifting=False, 
    formant_qfrency=1.0, 
    formant_timbre=1.0, 
    f0_file=None, 
    embedders_mode="fairseq", 
    proposal_pitch=False, 
    proposal_pitch_threshold=255.0, 
    audio_processing=False, 
    alpha=0.5,
    sid=0,
    embedders_mix=False,
    embedders_mix_layers=9,
    embedders_mix_ratio=0.5,
    noise_scale=0.35,
    nprobe=1,
    audio_upscaler=False
):
    """
    Manages full audio conversion pipelines including single file, batch, and separate stems workflow.

    This function serves as the central manager for UI-triggered conversions (Gradio).
    It handles input parameter validations, extracts target file paths, toggles stem 
    processing (Vocals vs. Backing Track vs. Instruments), overlays backing components, 
    and handles file system structures.

    Returns:
        list: A list containing updated states and file paths intended for Gradio outputs: [converted_vocals_path, backing_path, merged_backing_path, original_vocals_path, merged_instrument_path, ui_visibility_update]
    """

    # Determine the model path, checking inside global weights path if a relative name is given
    model_path = os.path.join(configs["weights_path"], model) if not os.path.exists(model) else model
    # Initialize standard error/empty return payload structure matching UI outputs
    return_none = [None]*6
    return_none[5] = {"visible": True, "__type__": "update"}

    # Validation Guard: Prevent stem manipulation flags if separate audio workflow is turned off
    if (not use_separate_audio and (merge_instrument or not_merge_backing or convert_backing or use_original)):
        gr_warning(translations["turn_on_use_audio"])
        return return_none

    # Validation Guard: Enforce parameter logic safety rules when 'use_original' is active
    if use_original:
        if convert_backing:
            gr_warning(translations["turn_off_convert_backup"])
            return return_none
        elif not_merge_backing:
            gr_warning(translations["turn_off_merge_backup"])
            return return_none

    # Validation Guard: Ensure valid RVC model file path and supported formats (.pth or .onnx)
    if (
        not model or 
        not os.path.exists(model_path) or 
        os.path.isdir(model_path) or 
        not model.endswith((".pth", ".onnx"))
    ):
        gr_warning(translations["provide_model"])
        return return_none

    # Compute actual extraction configurations based on hybrid/custom selection overrides
    f0method, embedder_model = (
        f0_method if f0_method != "hybrid" else hybrid_method, 
        embedders if embedders != "custom" else custom_embedders
    )

    # Local helper closure to streamline passing configurations down to the backend convert routine
    def _convert_audio(input_path, output_path, f0_file = ""):
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
            noise_scale,
            nprobe,
            audio_upscaler
        )
    
    # Internal utility to find pre-isolated stems (e.g. from UVR separation output) by substring label
    def get_audio_file(label):
        matching_files = [f for f in os.listdir(output_audio) if label in f]

        if not matching_files: return translations["notfound"]   
        return os.path.join(output_audio, matching_files[0])

    if use_separate_audio: # Pipeline A: Separated stems workflow
        output_audio = os.path.join(configs["audios_path"], input_audio_name)

        # Lazy load Pydub binding module for audio mixing/overlay tasks
        from main.library.audio.audio import pydub_load

        # Define explicit absolute file paths for targeted stem modifications
        output_path = os.path.join(output_audio, f"Convert_Vocals.{export_format}")
        output_backing = os.path.join(output_audio, f"Convert_Backing.{export_format}")
        output_merge_backup = os.path.join(output_audio, f"Vocals+Backing.{export_format}")
        output_merge_instrument = os.path.join(output_audio, f"Vocals+Instruments.{export_format}")

        # Construct specific working directories safely if missing
        if os.path.exists(output_audio): os.makedirs(output_audio, exist_ok=True)
        output_path = process_output(output_path)

        # Handle path assignments for original vocal inputs if requested
        if use_original:
            original_vocal = get_audio_file('Original_Vocals_No_Reverb.')
            if original_vocal == translations["notfound"]: 
                original_vocal = get_audio_file('Original_Vocals.')

            if original_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_original_vocal"])
                return return_none
            
            input_path = original_vocal
        else:
            # Handle path assignments for main isolated vocals and backing stem inputs
            main_vocal = get_audio_file('Main_Vocals_No_Reverb.')
            if main_vocal == translations["notfound"]: main_vocal = get_audio_file('Main_Vocals.')

            if main_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_main_vocal"])
                return return_none

            backing_vocal = get_audio_file('Backing_Vocals.')
            
            if (not not_merge_backing and backing_vocal == translations["notfound"]): 
                gr_warning(translations["not_found_backing_vocal"])
                return return_none
            
            input_path = main_vocal
            backing_path = backing_vocal

        # Step 1: Process Voice Conversion on the primary vocal track
        gr_info(translations["convert_vocal"])
        _convert_audio(input_path, output_path)
        gr_info(translations["convert_success"])

        # Step 2: Optionally process Voice Conversion on the background vocal tracks
        if convert_backing:
            output_backing = process_output(output_backing)
            gr_info(translations["convert_backup"])

            _convert_audio(backing_path, output_backing)
            gr_info(translations["convert_backup_success"])

        # Step 3: Handle structural merging/overlay processes via Pydub
        try:
            # Combine converted vocals with backings if not explicitly blocked
            if not not_merge_backing and not use_original:
                backing_source = (
                    output_backing 
                    if convert_backing else 
                    backing_vocal
                )

                output_merge_backup = process_output(output_merge_backup)
                gr_info(translations["merge_backup"])

                # Load components, shift specific relative DB volumes and export combined audio
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

            # Combine output vocals/backings with instrument stems if requested
            if merge_instrument:    
                vocals = (output_merge_backup if not not_merge_backing and not use_original else output_path)
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
            # Return early to UI if any operational audio processing errors break execution
            return return_none

        # Assemble and pass output arrays successfully to UI tracks
        return [
            (None if use_original else output_path), 
            output_backing, 
            (None if not_merge_backing and use_original else output_merge_backup), 
            (output_path if use_original else None), 
            (output_merge_instrument if merge_instrument else None), 
            {"visible": True, "__type__": "update"}
        ]
    else: # Pipeline B: Single/Batch direct workflow
        # Check if basic input track parameters match existing target paths
        if not input_path or not os.path.exists(input_path): 
            gr_warning(translations["input_not_valid"])
            return return_none
        
        if not output_path:
            gr_warning(translations["output_not_valid"])
            return return_none
        
        # Format filename to target selected export extension
        output_path = replace_export_format(output_path, export_format)

        if os.path.isdir(input_path): # Scenario B1: Input target is an entire Directory -> Execute Batch Mode
            gr_info(translations["is_folder"])

            # Verify that the directory contains at least one supported audio file format
            if not [
                f for f in os.listdir(input_path) 
                if f.lower().endswith(tuple(file_types))
            ]:
                gr_warning(translations["not_found_in_folder"])
                return return_none
            
            gr_info(translations["batch_convert"])
            output_dir = os.path.dirname(output_path) or output_path
            # Direct directory references down to trigger internal conversion iterations
            _convert_audio(input_path, output_dir)
            gr_info(translations["batch_convert_success"])

            return return_none
        else: # Scenario B2: Input target is a standard single file path
            output_dir = os.path.dirname(output_path) or output_path

            if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
            output_path = process_output(output_path)

            gr_info(translations["convert_vocal"])
            # Direct voice conversion run
            _convert_audio(input_path, output_path, f0_file)
            gr_info(translations["convert_success"])

            return_none[0] = output_path
            return return_none

def convert_selection(
    clean_audio=False,
    autotune=False,
    use_separate_audio=False,
    use_original=False,
    convert_backing=False,
    not_merge_backing=False,
    merge_instrument=False,
    pitch=0,
    clean_strength=0.5,
    model=None,
    index=None,
    index_rate=0.5,
    input_path=None,
    output_path=None,
    export_format="wav",
    f0_method="rmvpe",
    hybrid_method="hybrid[rmvpe+harvest]",
    hop_length=160,
    embedders="hubert_base",
    custom_embedders=None,
    resample_sr=0,
    filter_radius=3,
    rms_mix_rate=1.0,
    protect=0.33,
    split_audio=False,
    f0_autotune_strength=1.0,
    checkpointing=False,
    predictor_onnx=False,
    formant_shifting=False,
    formant_qfrency=1.0,
    formant_timbre=1.0,
    f0_file=None,
    embedders_mode="fairseq",
    proposal_pitch=False,
    proposal_pitch_threshold=255.0,
    audio_processing=False,
    alpha=0.5,
    sid=0,
    embedders_mix=False,
    embedders_mix_layers=9,
    embedders_mix_ratio=0.5,
    noise_scale=0.35,
    nprobe=1,
    audio_upscaler=False
):
    """
    Pre-evaluates separated folder choices before routing arguments to convert_audio.

    This function coordinates Gradio UI components behavior based on isolated directory 
    searches. If multiple separate folders exist, it builds selections interactive elements; 
    if exactly 1 exists, it automatically executes the conversions instantly.

    Returns:
        list: A series of Gradio target component `.update()` dictionaries and execution payloads.
    """

    # Evaluate separate audio directories context paths if flag condition is met
    if use_separate_audio:
        gr_info(translations["search_separate"])

        # Scan and filter out separate stems output folders inside the target directory path
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

        if len(choice) == 0: # Case 1: No valid separated audio source stems are available
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
        elif len(choice) == 1: # Case 2: Exactly 1 isolated source folder matches -> Trigger auto execution immediately
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
                choice[0], # Automatically binds the single located target directory folder name
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
                noise_scale,
                nprobe,
                audio_upscaler
            )

            # Return structural updates showing output file variables and updating layouts UI elements
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
        else: # Case 3: Multiple choice directories found -> Pop up the dropdown selector list in UI
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
    else: # If separate audio stems processing option is unchecked -> Run default single standard pipeline path
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
            noise_scale,
            nprobe,
            audio_upscaler
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

def vad_diarization(input_audio=None, num_spk=2, vad_sensitivity=3, vad_frame_ms=30, queue=None):
    """
    Performs Voice Activity Detection (VAD) and Speaker Diarization on an audio file.

    This function detects speech segments using a VAD processor, extracts speaker 
    embeddings for each detected speech segment using a pretrained SpeechBrain model, 
    and clusters those embeddings to distinguish between different speakers (who spoke when).
    Finally, it merges consecutive segments belonging to the same speaker and sends 
    the result back via a multiprocessing Queue.

    Args:
        input_audio (str, optional): Path to the input audio file. Defaults to None.
        num_spk (int, optional): Number of expected speakers to cluster. Defaults to 2.
        vad_sensitivity (int, optional): VAD sensitivity mode (typically 0-3, 3 being the most sensitive). Defaults to 3.
        vad_frame_ms (int, optional): Duration of a VAD frame in milliseconds. Defaults to 30.
        queue (multiprocessing.Queue, optional): Queue used to safely send results or errors back to the parent process. Defaults to None.

    Returns:
        None: Results are put directly into the provided `queue`.
    """

    try:
        # Importing heavy machine learning libraries inside the function to save 
        # memory if this function is spawned in a separate multiprocessing context.
        from sklearn.cluster import AgglomerativeClustering
    
        from main.library.audio.audio import load_audio
        from main.library.speaker_diarization.audio import Audio
        from main.inference.realtime.vad_utils import VADProcessor
        from main.library.speaker_diarization.segment import Segment
        from main.library.utils import clear_gpu_cache, check_spk_diarization
        from main.library.speaker_diarization.embedding import SpeechBrainPretrainedSpeakerEmbedding

        # Load the input audio file, forcing a high-quality sample rate of 48kHz
        y, sr = load_audio(input_audio, sample_rate=48000, return_sr=True)

        # Initialize the VAD processor with specified sensitivity, frame duration, and sample rate
        vad_processor = VADProcessor(sensitivity_mode=vad_sensitivity, frame_duration_ms=vad_frame_ms, sample_rate=sr)
        # Extract timestamp segments where actual speech is detected
        segments = vad_processor.get_speech(y)

        # If no speech is found in the entire audio, send a warning and exit early
        if not segments:
            queue.put({"status": "warning", "message": "speech_not_in_segments"})
            return

        audio = Audio()
        merged_segments = []
        # Ensure speaker diarization assets are valid/downloaded
        check_spk_diarization(model_size=None)

        # Initialize the pretrained SpeechBrain embedding model on the designated hardware (CPU/GPU)
        embedding_model = SpeechBrainPretrainedSpeakerEmbedding(
            embedding=os.path.join(configs["speaker_diarization_path"], "models", "speechbrain"), 
            device=config.device
        )
        
        # Allocate a numpy array to store a 192-dimensional vector embedding for each speech segment
        embeddings = np.zeros(shape=(len(segments), 192))
        # Iterate over each VAD-detected segment to extract its speaker signature
        for i, segment in enumerate(segments):
            # Crop the audio to match the current segment's start and end times
            waveform, _ = audio.crop(input_audio, Segment(segment["start"], segment["end"]))
            # Format the waveform to fit the SpeechBrain model expectation (handle stereo vs mono channels)
            # and extract the speaker identity embedding
            embeddings[i] = embedding_model(waveform.mean(dim=0, keepdim=True)[None] if waveform.shape[0] == 2 else waveform[None])

        # Cluster the speaker embeddings using Agglomerative Clustering based on the target speaker count
        # np.nan_to_num safely handles any NaN/Infinite values in the embeddings before clustering
        labels = AgglomerativeClustering(num_spk).fit(np.nan_to_num(embeddings)).labels_

        # Assign the clustered speaker labels back to their corresponding speech segments
        for i in range(len(segments)):
            segments[i]["speaker"] = f"SPEAKER {labels[i] + 1}"

        # Merge contiguous timeline blocks if the same speaker continues talking
        if segments:
            curr = segments[0].copy()

            for i in range(1, len(segments)):
                # If the next segment belongs to the current speaker, just extend the 'end' timestamp
                if segments[i]["speaker"] == curr["speaker"]: curr["end"] = segments[i]["end"]
                else:
                    # If a new speaker starts, save the finished segment and switch tracking to the new one
                    merged_segments.append(curr)
                    curr = segments[i].copy()

            # Append the very last processed segment to the final list
            merged_segments.append(curr)

        # Manually delete large objects, clear GPU cache, and force garbage collection 
        # to prevent severe VRAM leaks or out-of-memory crashes during multi-turn processing
        del audio, embedding_model, segments, labels, embeddings
        clear_gpu_cache()
        gc.collect()

        # Send the successful diarization data back to the main thread/process
        queue.put({"status": "success", "data": merged_segments})
    except Exception as e:
        # Catch any unexpected runtime errors, extract the complete traceback log, 
        # and send it safely via the queue to avoid freezing the system execution

        import traceback
        queue.put({"status": "error", "message": str(e), "traceback": traceback.format_exc()})

def get_formatted_stats(merged_segments):
    """
    Generates a beautifully formatted ASCII table representing speaker diarization statistics.

    This utility processes individual speaker segments, calculates the duration of each speech
    block, formats timestamps into human-readable hours:minutes:seconds formats, and dynamically
    calculates column widths to draw an aligned text-based table grid.

    Args:
        merged_segments (list of dict): A list of consolidated speech segments, where each 
            dictionary contains "speaker", "start", and "end" keys.

    Returns:
        str: A multi-line string containing the drawn ASCII statistics table.
    """

    rows, lines = [], []
    speaker_stats = {}

    # Extract translated table column headers (expected format: "Speaker Time_Range Duration")
    headers = translations["headers"].split(" ")

    # Data Extraction and Preprocessing
    for seg in merged_segments:
        spk = seg["speaker"]
        start_t = seg["start"]
        end_t = seg["end"]
        dur = end_t - start_t

        # Convert raw float seconds into a standard HH:MM:SS timedelta string format
        time_range = f"{str(datetime.timedelta(seconds=int(start_t)))} -> {str(datetime.timedelta(seconds=int(end_t)))}"
        dur_str = f"{dur:.2f}s"

        # Group data into rows for tabular rendering
        rows.append([spk, time_range, dur_str])
        # Keep track of cumulative speaking duration per unique speaker
        speaker_stats[spk] = speaker_stats.get(spk, 0.0) + dur
    
    # Dynamic Column Width Calculation
    col_widths = []
    for i in range(3):
        # Determine the maximum text length for each column (comparing header and all values)
        # Replacing hyphens with spaces ensures correct visual padding boundary logic
        max_len = max(len(headers[i].replace("-", " ")), *(len(row[i]) for row in rows))
        col_widths.append(max_len)

    # Grid Drawing Helper Functions
    def make_line(left, mid, right):
        """Constructs a horizontal border line using standard box-drawing characters."""

        return left + mid.join("─" * (w + 2) for w in col_widths) + right

    def format_row(row):
        """Formats a single row of data, left-aligning content based on pre-calculated widths."""

        return "│ " + " │ ".join(f"{cell:<{col_widths[i]}}" for i, cell in enumerate(row)) + " │"
    
    lines.append("\n")
    lines.append(make_line("┌", "┬", "┐")) # Top border
    lines.append(format_row(headers)) # Column headers
    lines.append(make_line("├", "┼", "┤")) # Header separator

    for row in rows:
        lines.append(format_row(row)) # Data rows

    lines.append(make_line("└", "┴", "┘")) # Bottom border
    return "\n".join(lines)

def merge_audio(
    files_list, 
    time_stamps, 
    original_file_path, 
    output_path, 
    export_format
):
    """
    Recombines processed individual audio segments into a single unified audio file.

    This function pieces back together separate speaker audio segments while carefully inserting 
    precise durations of synthetic silence in gaps where no speech was detected. This architecture
    preserves the absolute temporal alignment and total duration relative to the original file.

    Args:
        files_list (list of str): Paths to the fragmented, modified audio segment files.
        time_stamps (list of tuple): Expected millisecond coordinates `(start_ms, end_ms)` corresponding chronologically to each file in the `files_list`.
        original_file_path (str): Path to the source raw audio file used to determine total duration bounds.
        output_path (str): Target filesystem destination path for the final exported audio.
        export_format (str): Target container codec format extension (e.g., "wav", "mp3", "flac").

    Returns:
        str: The filesystem path pointing to the successfully exported composite audio file.
    """

    from pydub import AudioSegment

    from main.library.audio.audio import pydub_load

    def extract_number(filename):
        """Helper to extract trailing indices from filenames for exact numerical sorting order."""

        match = re.search(r'_(\d+)', filename)
        return int(match.group(1)) if match else 0

    # Load original track to capture exact global duration boundary constraints
    total_duration = len(pydub_load(original_file_path))
    combined = AudioSegment.empty() 
    current_position = 0 

    # Zip and iterate using sorted filenames to guarantee strict temporal execution order
    for file, (start_i, end_i) in zip(sorted(files_list, key=extract_number), time_stamps):
        # If there is a temporal gap between current tracker and next segment start, insert silence
        if start_i > current_position: 
            combined += AudioSegment.silent(
                duration=start_i - current_position
            )  
        
        # Append the processed speaker segment audio
        combined += pydub_load(file)  
        current_position = end_i

    # Handle edge case where the final speaker stopped talking before the absolute physical track end
    if current_position < total_duration: 
        combined += AudioSegment.silent(
            duration=total_duration - current_position
        )

    # Export output file onto the disk storage and wipe container from volatile memory
    combined.export(output_path, format=export_format)
    del combined

    return output_path

def convert_with_vad(
    num_spk=2, 
    cleaner=False, 
    clean_strength=0.5, 
    autotune=False, 
    f0_autotune_strength=1.0, 
    checkpointing=False, 
    model_1=None, 
    model_2=None, 
    model_index_1=None, 
    model_index_2=None, 
    pitch_1=0, 
    pitch_2=0, 
    index_strength_1=0.5, 
    index_strength_2=0.5, 
    export_format="wav", 
    input_audio=None, 
    output_audio=None, 
    predictor_onnx=False, 
    f0_method="rmvpe", 
    hybrid_method="hybrid[rmvpe+harvest]", 
    hop_length=160, 
    embed_mode="fairseq", 
    embedders="hubert_base", 
    custom_embedders=None, 
    resample_sr=0, 
    filter_radius=3, 
    rms_mix_rate=1.0, 
    protect=0.33, 
    formant_shifting=False, 
    formant_qfrency_1=1.0, 
    formant_timbre_1=1.0, 
    formant_qfrency_2=1.0, 
    formant_timbre_2=1.0, 
    proposal_pitch=False, 
    proposal_pitch_threshold=255.0, 
    audio_processing=False, 
    alpha=0.5,
    sid_1=0,
    sid_2=0,
    embedders_mix=False,
    embedders_mix_layers=9,
    embedders_mix_ratio=0.5,
    noise_scale_1=0.35,
    noise_scale_2=0.35,
    vad_sensitivity=3,
    vad_frame_ms=30,
    nprobe_1=1,
    nprobe_2=1
):
    """
    Executes a complete end-to-end multi-speaker (RVC/SVC) pipeline.

    The execution chain operates as follows:
      1. Validates presence and structural patterns of inputs/RVC/SVC model checkpoints.
      2. Spawns an isolated Subprocess to execute VAD and Speaker Diarization safely without CUDA race conditions.
      3. Splits the source file into physical segment chunks sorted into distinct folder structures per speaker.
      4. Invokes RVC/SVC inference engine (`convert()`) targeting each specific speaker folder with unique settings.
      5. Re-assembles the converted speaker voices back into a singular composite file using `merge_audio()`.

    Returns:
        str or None: Path to the finalized multi-speaker converted audio track, or None if validation fails.
    """

    import multiprocessing as mp

    from main.library.audio.audio import pydub_load

    model_pth_1, model_pth_2 = (
        os.path.join(configs["weights_path"], model_1) if not os.path.exists(model_1) else model_1, 
        os.path.join(configs["weights_path"], model_2) if not os.path.exists(model_2) else model_2
    )

    # Halt operation if neither model path 1 nor model path 2 represents a valid existing RVC model file
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
        gr_warning(translations["provide_model"])
        return None
    
    # Fallback assignment: if single speaker model is provided, duplicate it to fill both channels
    if not model_1: model_pth_1 = model_pth_2
    if not model_2: model_pth_2 = model_pth_1

    # Verify input audio path presence on disk storage
    if not input_audio or not os.path.exists(input_audio) or os.path.isdir(input_audio): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_audio:
        gr_warning(translations["output_not_valid"])
        return None
    
    output_audio = process_output(output_audio)
    gr_info(translations["start_vad"])
    
    try:
        # Utilizing 'spawn' context ensures a clean memory layout, avoiding parent CUDA state cloning bugs
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        # Allocate worker process specifically bound to target diarization operation
        process = ctx.Process(
            target=vad_diarization,
            args=(input_audio, num_spk, vad_sensitivity, vad_frame_ms, queue)
        )

        process.start()
        result = queue.get() # Blocking wait call until child pipeline delivers structural dictionary response
        process.join()

        # Parse worker state signals
        if result["status"] == "warning":
            gr_warning(translations[result["message"]])
            return None
        elif result["status"] == "error":
            logger.debug(result["traceback"])
            gr_error(result["message"])
            return None

        merged_segments = result["data"]

        gr_info(translations["analysis_completed"].format(length=len(merged_segments)))
        gr_info(translations["process_audio"])

        pydub_audio = pydub_load(input_audio)
        output_folder = "audios_temp"

        # Wipe pre-existing cache directories to avoid crosstalk artifact leakage across execution runs
        if os.path.exists(output_folder): shutil.rmtree(output_folder, ignore_errors=True)

        # Build clean directory maps partitioning Speaker 1 and Speaker 2 distinct batches
        for f in [
            output_folder, 
            os.path.join(output_folder, "1"), 
            os.path.join(output_folder, "2")
        ]:
            os.makedirs(f, exist_ok=True)

        time_stamps, processed_segments = [], []
        for i, seg in enumerate(merged_segments):
            # Parse numeric speaker ID tag to map segment distribution pattern alternately (Odd IDs -> Folder 1, Even IDs -> Folder 2)
            model_id = 1 if int(seg["speaker"].split()[-1]) % 2 != 0 else 2
            start_ms, end_ms = int(seg["start"] * 1000), int(seg["end"] * 1000)

            # Export raw un-converted slice into speaker-specific storage space
            target_path = os.path.join(output_folder, str(model_id), f"segment_{i+1}.wav")
            pydub_audio[start_ms:end_ms].export(target_path, format="wav")
            
            # Formulate hypothetical path string pattern where the core convert engine will spit out outputs
            processed_segments.append(os.path.join(output_folder, str(model_id), f"segment_{i+1}_output.wav"))
            time_stamps.append((start_ms, end_ms))

        # Log formatted ASCII runtime status overview matrix to standard console logs
        logger.info(get_formatted_stats(merged_segments))

        # Re-map algorithm string flags to internal system keys
        f0method, embedder_model = (
            f0_method if f0_method != "hybrid" else hybrid_method, 
            embedders if embedders != "custom" else custom_embedders
        )

        gr_info(translations["process_done_start_convert"])
        # Run conversion processing for Speaker 1 Folder Group
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
            noise_scale_1,
            nprobe_1,
            False
        )
        # Run conversion processing for Speaker 2 Folder Group
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
            noise_scale_2,
            nprobe_2,
            False
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
        # Guarantee dynamic filesystem cleanup actions to preserve free storage blocks on server instances
        if os.path.exists("audios_temp"): 
            shutil.rmtree("audios_temp", ignore_errors=True)

def convert_tts(
    clean_audio=False, 
    autotune=False, 
    pitch=0, 
    clean_strength=0.5, 
    model=None, 
    index=None, 
    index_rate=0.5, 
    input_path=None, 
    output_path=None, 
    export_format="wav", 
    method="rmvpe", 
    hybrid_method="hybrid[rmvpe+harvest]", 
    hop_length=160, 
    embedders="hubert_base", 
    custom_embedders=None, 
    resample_sr=0, 
    filter_radius=3, 
    rms_mix_rate=1.0, 
    protect=0.33, 
    split_audio=False, 
    f0_autotune_strength=1.0, 
    checkpointing=False, 
    predictor_onnx=False, 
    formant_shifting=False, 
    formant_qfrency=1.0, 
    formant_timbre=1.0, 
    f0_file=None, 
    embedders_mode="fairseq", 
    proposal_pitch=False, 
    proposal_pitch_threshold=255.0, 
    audio_processing=False, 
    alpha=0.5,
    sid=0,
    embedders_mix=False,
    embedders_mix_layers=9,
    embedders_mix_ratio=0.5,
    noise_scale=0.35,
    nprobe=1,
    audio_upscaler=False
):
    """
    Executes (RVC/SVC) specifically tailored for Text-to-Speech (TTS) outputs.

    This function coordinates the post-processing pipeline for synthesized TTS audio. It handles 
    validation of voice weights (.pth/.onnx), scans folders or maps individual input/output audio files, 
    manages runtime target path formatting, and routes all analytical hyper-parameters cleanly down 
    to the main RVC/SVC `convert` inference core.

    Returns:
        str or None: Path to the finalized voice-converted TTS file, or None if runtime validation fails.
    """

    # Attempt to resolve the checkpoint path either relative to the weights storage workspace or as an absolute literal path
    model_path = os.path.join(configs["weights_path"], model) if not os.path.exists(model) else model

    # Validate model path structure, extensions, and asset integrity rules
    if (
        not model_path or 
        not os.path.exists(model_path) or 
        os.path.isdir(model_path) or 
        not model.endswith((".pth", ".onnx"))
    ):
        gr_warning(translations["provide_model"])
        return None

    if not input_path or not os.path.exists(input_path): 
        gr_warning(translations["input_not_valid"])
        return None
    
    # If the user passed a directory path instead of a file, scan it to automatically find the correct TTS file
    if os.path.isdir(input_path): 
        input_audio = [
            f for f in os.listdir(input_path) 
            if "tts" in f and f.lower().endswith(tuple(file_types))
        ]
        
        # Abort if no matching internal TTS file pattern matching system `file_types` exists
        if not input_audio:
            gr_warning(translations["not_found_in_folder"])
            return None
        
        # Pick the first discovered matching TTS track artifact as the active source input file
        input_path = os.path.join(input_path, input_audio[0])
    
    if not output_path:
        gr_warning(translations["output_not_valid"])
        return None
    
    # Ensure correct target suffix structure layout matches specified export format requirements
    output_path = replace_export_format(output_path, export_format)
    # If output points to a folder path template, append a generic fallback output filename
    if os.path.isdir(output_path): output_path = os.path.join(output_path, f"tts.{export_format}")

    # Build underlying directory structure maps dynamically if they are missing
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    # Post-process output path mapping layout structure (e.g., handling spaces/unicode sanitization)
    output_path = process_output(output_path)

    # Handle conditional fallbacks when 'hybrid' mode or 'custom' embedders are configured via UI controls
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
        noise_scale,
        nprobe,
        audio_upscaler
    )

    gr_info(translations["convert_success"])
    return output_path