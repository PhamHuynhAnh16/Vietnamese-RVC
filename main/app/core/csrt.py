import os
import sys

sys.path.append(os.getcwd())

from main.library.speaker_diarization.whisper import load_model
from main.app.core.ui import gr_info, gr_warning, process_output
from main.app.variables import config, translations, configs, logger

def create_srt(model_size, input_audio, output_file, word_timestamps):
    if not input_audio or not os.path.exists(input_audio) or os.path.isdir(input_audio): 
        gr_warning(translations["input_not_valid"])
        return [None]*2
    
    if not output_file.endswith(".srt"): output_file += ".srt"
        
    if not output_file:
        gr_warning(translations["output_not_valid"])
        return [None]*2
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    info = ""
    output_file = process_output(output_file)

    gr_info(translations["csrt"])

    model = load_model(model_size, device=config.device)
    result = model.transcribe(input_audio, fp16=configs.get("fp16", False), word_timestamps=word_timestamps)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"]):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            
            index = f"{i+1}\n"
            timestamp = f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
            text1 = f"{text}\n\n"

            f.write(index)
            f.write(timestamp)
            f.write(text1)

            info = info + index + timestamp + text1
            logger.info(info)
    
    gr_info(translations["success"])

    return [{"value": output_file, "visible": True, "__type__": "update"}, info]

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)

    seconds = int(seconds % 60)
    miliseconds = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{miliseconds:03}"