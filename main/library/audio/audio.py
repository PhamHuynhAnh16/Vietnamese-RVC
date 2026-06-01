import os
import sys
import librosa

import numpy as np
import soundfile as sf

from pydub import AudioSegment

sys.path.append(os.getcwd())

from main.app.variables import translations

def load_audio(
    file, 
    sample_rate=16000, 
    formant_shifting=False, 
    formant_qfrency=0.8, 
    formant_timbre=0.8, 
    flatten=True, 
    res_type="soxr_vhq",
    dtype = np.float64,
    return_sr = False,
    mode = "soundfile"
):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file): raise FileNotFoundError(translations["not_found"].format(name=file))

        try:
            if mode == "soundfile": audio, sr = sf.read(file, dtype=dtype)
            elif mode == "librosa": audio, sr = librosa.load(file, sr=None, dtype=dtype)
            elif mode == "ffmpeg":
                from main.library.audio import ffmpeg

                audio, _ = (
                    ffmpeg.input(file, threads=0)
                    .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
                    .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
                )
                audio = np.frombuffer(audio, dtype=np.float32).astype(dtype=dtype)
                sr = sample_rate
            else:
                raise ValueError
        except:
            modes = ["soundfile", "librosa", "ffmpeg"]
            if mode in modes: modes.remove(mode)

            for m in modes:
                try:
                    return load_audio(
                        file, 
                        sample_rate=sample_rate, 
                        formant_shifting=formant_shifting, 
                        formant_qfrency=formant_qfrency, 
                        formant_timbre=formant_timbre, 
                        flatten=flatten, 
                        res_type=res_type,
                        dtype=dtype,
                        return_sr=return_sr,
                        mode=m
                    )
                except:
                    pass
            raise RuntimeError

        if audio.ndim > 1: audio = audio.mean(axis=0, dtype=dtype) if audio.shape[0] < audio.shape[1] else audio.mean(axis=1, dtype=dtype)

        if sr is not None and sr != sample_rate: 
            audio = librosa.resample(
                audio, 
                orig_sr=sr, 
                target_sr=sample_rate, 
                res_type=res_type
            )
            sr = sample_rate

        if formant_shifting:
            from main.library.algorithm.stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(
                framesize=1024, 
                hopsize=32, 
                samplerate=sample_rate
            )

            audio = pitchshifter.shiftpitch(
                audio, 
                factors=1, 
                quefrency=formant_qfrency * 1e-3, 
                distortion=formant_timbre
            )
        
        audio = audio.flatten() if flatten else audio
    except Exception as e:
        raise RuntimeError(f"{translations['errors_loading_audio']}: {e}")
    
    if return_sr: return audio, sr
    return audio

def pydub_load(input_path, volume = None):
    try:
        if input_path.endswith(".wav"): audio = AudioSegment.from_wav(input_path)
        elif input_path.endswith(".mp3"): audio = AudioSegment.from_mp3(input_path)
        elif input_path.endswith(".ogg"): audio = AudioSegment.from_ogg(input_path)
        else: audio = AudioSegment.from_file(input_path)
    except:
        audio = AudioSegment.from_file(input_path)
        
    return audio if volume is None else (audio + volume)

def cut(audio, sr, db_thresh=-60, min_interval=250):
    from main.inference.preprocess.slicer2 import Slicer2

    slicer = Slicer2(
        sr=sr, 
        threshold=db_thresh, 
        min_interval=min_interval
    )

    return slicer.slice2(audio)

def restore(segments, total_len, dtype=np.float32):
    out = []
    last_end = 0

    for start, end, processed_seg in segments:
        if start > last_end: 
            out.append(
                np.zeros(start - last_end, dtype=dtype)
            )

        out.append(processed_seg)
        last_end = end

    if last_end < total_len: 
        out.append(
            np.zeros(total_len - last_end, dtype=dtype)
        )

    return np.concatenate(out, axis=-1)