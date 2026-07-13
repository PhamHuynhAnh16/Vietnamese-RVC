import os
import gc
import sys

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning, process_output
from main.app.variables import config, translations, configs, logger
from main.library.utils import check_spk_diarization, clear_gpu_cache

def whisper_process(
    model_size, 
    input_audio, 
    configs, 
    device, 
    out_queue, 
    language="vi"
):
    """
    Loads the Whisper model and transcribes the input audio within an isolated process.

    This function is designed to run inside a separate multiprocessing pool/process 
    to isolate heavy GPU usage and safely clear VRAM upon completion.

    Args:
        model_size (str): Size/name of the Whisper model to load (e.g., 'base', 'large-v3').
        input_audio (str): Path to the source audio file.
        configs (dict): Transcription configuration settings (e.g., fp16 options).
        device (str): Target device for computation ('cuda' or 'cpu').
        out_queue (multiprocessing.Queue): Inter-process queue to return results or exceptions.
        language (str, optional): ISO code for target transcription language. Defaults to "vi".
    """

    # Lazy import to prevent heavy model dependencies loading in the parent process
    from main.library.speaker_diarization.whisper import load_model

    try:
        # Initialize and load the target Whisper model onto the specified device
        model = load_model(model_size, device=device)

        # Optimize performance via torch.compile if enabled in global configuration
        if config.compile_all:
            import torch

            model.encoder = torch.compile(model.encoder, mode=config.compile_mode)
            model.decoder = torch.compile(model.decoder, mode=config.compile_mode)
        
        # Execute the transcription process with word-level timestamps
        segments = model.transcribe(
            input_audio, 
            fp16=configs.get("fp16", False), 
            word_timestamps=True,
            language=language
        )

        # Send the successfully extracted audio segments back to the main process
        out_queue.put(segments["segments"])
    except Exception as e:
        # Pass any runtime exception to the main process for proper handling
        out_queue.put(e)
    finally:
        # Strict memory management: Flush GPU cache and delete references to free VRAM
        clear_gpu_cache()
        del segments
        gc.collect()

def create_srt(
    model_size, 
    input_audio, 
    output_file, 
    language
):
    """
    Validates paths, runs Whisper transcription in a separate process, and generates an SRT subtitle file.

    Args:
        model_size (str): Size of the Whisper model to be deployed.
        input_audio (str): File path of the audio to transcribe.
        output_file (str): Target file path where the SRT file will be saved.
        language (str): Target transcription language.

    Returns:
        list: A list containing a Gradio UI update dictionary object and the raw SRT text string.
              Returns [None, None] if any input or output path validation fails.
    """

    import multiprocessing as mp

    if not input_audio or not os.path.exists(input_audio) or os.path.isdir(input_audio): 
        gr_warning(translations["input_not_valid"])
        return [None]*2
    
    # Enforce standard SubRip Subtitle format extension
    if not output_file.endswith(".srt"): output_file += ".srt"
        
    if not output_file:
        gr_warning(translations["output_not_valid"])
        return [None]*2
    
    # Ensure target directory tree exists before writing
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    info = ""
    output_file = process_output(output_file)
    # Pre-execution environment/model checks
    check_spk_diarization(model_size, speechbrain=False)
    gr_info(translations["csrt"])

    # Using 'spawn' context to guarantee a clean memory state, critical for CUDA operations
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    process = ctx.Process(
        target=whisper_process, 
        args=(
            model_size, 
            input_audio, 
            configs, 
            config.device, 
            queue, 
            language
        )
    )

    process.start()
    segments = queue.get() # Halts execution until the child process pushes data into the queue
    process.join()

    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            # Formatting block structures matching standard SRT syntax specifications
            index = f"{i+1}\n"
            timestamp = f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
            text1 = f"{text}\n\n"
            # Write directly to disk
            f.write(index)
            f.write(timestamp)
            f.write(text1)
            # Keep a localized log string copy for display/logging
            info = info + index + timestamp + text1
        logger.info(info)
    
    gr_info(translations["success"])
    # Return a UI state update payload along with the log text raw string
    return [{"value": output_file, "visible": True, "__type__": "update"}, info]

def format_timestamp(seconds):
    """
    Converts a duration in raw floating-point seconds into a standardized SRT timecode string.

    Format follows: HH:MM:SS,mmm

    Args:
        seconds (float): Total time duration represented in seconds.

    Returns:
        str: Format matched timecode string (e.g., '01:23:45,678').
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)

    seconds = int(seconds % 60)
    miliseconds = int(round((seconds % 1) * 1000))

    return f"{hours:02}:{minutes:02}:{seconds:02},{miliseconds:03}"