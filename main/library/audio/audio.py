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
    """
    Load an audio file using one of several available backends.

    Supported backends:
        - soundfile : Fast and accurate for PCM-based formats.
        - librosa   : Broad codec support through audioread.
        - ffmpeg    : Maximum compatibility with compressed formats.

    If the selected backend fails, the remaining backends are tried
    automatically before reporting an error.

    Processing steps:
        1. Load audio.
        2. Convert stereo/multi-channel audio to mono.
        3. Resample to the requested sample rate.
        4. Optionally apply formant shifting.
        5. Optionally flatten the output array.

    Args:
        file: Path to the audio file.
        sample_rate: Desired output sample rate.
        formant_shifting: Enable formant shifting.
        formant_qfrency: Formant quefrency in milliseconds.
        formant_timbre: Formant distortion factor.
        flatten: Flatten the output array.
        res_type: librosa resampling algorithm.
        dtype: Output numpy dtype.
        return_sr: Return the sample rate together with the audio.
        mode: Preferred loading backend.

    Returns:
        np.ndarray or (np.ndarray, int)
    """

    try:
        # Remove accidental whitespace and quotes from the file path.
        file = file.strip().strip('"').strip("\n").strip('"').strip()
        if not os.path.isfile(file): raise FileNotFoundError(translations["not_found"].format(name=file))

        try:
            if mode == "soundfile": 
                # Native PCM decoder with excellent performance.
                audio, sr = sf.read(file, dtype=dtype)
            elif mode == "librosa": 
                # librosa keeps the original sample rate when sr=None.
                audio, sr = librosa.load(file, sr=None, dtype=dtype)
            elif mode == "ffmpeg":
                # FFmpeg provides the broadest codec compatibility.

                from main.library.audio import ffmpeg

                audio, _ = (
                    ffmpeg.input(file, threads=0)
                    .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
                    .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
                )
                audio = np.frombuffer(audio, dtype=np.float32).astype(dtype=dtype)
                sr = sample_rate
            else:
                raise ValueError("This backend is not supported; please choose from ['soundfile', 'librosa', 'ffmpeg'].")
        except:
            # Automatically fall back to the remaining loaders.
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

        # Convert multi-channel audio into mono.
        # Channels may be stored as (channels, samples)
        # or (samples, channels) depending on the backend.
        if audio.ndim > 1: audio = audio.mean(axis=0, dtype=dtype) if audio.shape[0] < audio.shape[1] else audio.mean(axis=1, dtype=dtype)

        # Resample only when necessary.
        # soxr_vhq provides very high-quality sinc resampling.
        if sr is not None and sr != sample_rate: 
            audio = librosa.resample(
                audio, 
                orig_sr=sr, 
                target_sr=sample_rate, 
                res_type=res_type
            )
            sr = sample_rate

        if formant_shifting:
            from main.library.audio.stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(
                framesize=1024, 
                hopsize=32, # Small hop size minimizes phase discontinuities.
                samplerate=sample_rate
            )
            # Preserve pitch while modifying vocal timbre.
            audio = pitchshifter.shiftpitch(
                audio, 
                factors=1, 
                quefrency=formant_qfrency * 1e-3, 
                distortion=formant_timbre
            )
        
        audio = audio.flatten() if flatten else audio # Ensure the output is a contiguous 1-D array.
    except Exception as e:
        raise RuntimeError(f"{translations['errors_loading_audio']}: {e}")
    
    if return_sr: return audio, sr
    return audio

def pydub_load(input_path, volume = None):
    """
    Load audio using pydub.

    This loader is mainly intended for operations that rely on
    AudioSegment, such as volume adjustment or exporting.

    Args:
        input_path: Audio file path.
        volume: Gain in dB.

    Returns:
        AudioSegment
    """

    try:
        # Use dedicated decoders for common formats when available.
        if input_path.endswith(".wav"): audio = AudioSegment.from_wav(input_path)
        elif input_path.endswith(".mp3"): audio = AudioSegment.from_mp3(input_path)
        elif input_path.endswith(".ogg"): audio = AudioSegment.from_ogg(input_path)
        else: audio = AudioSegment.from_file(input_path)
    except:
        # Fall back to FFmpeg's automatic format detection.
        audio = AudioSegment.from_file(input_path)
        
    return audio if volume is None else (audio + volume)

def cut(audio, sr, db_thresh=-60, min_interval=250):
    """
    Split audio into voiced segments.

    Silent regions below the threshold are removed while preserving
    the original timeline information.

    Args:
        audio: Input waveform.
        sr: Sample rate.
        db_thresh: Silence threshold in dBFS.
            Lower values detect quieter speech.
        min_interval: Minimum silence duration (ms) required to
            split the audio.

    Returns:
        list
    """

    from main.inference.preprocess.slicer2 import Slicer2

    slicer = Slicer2(
        sr=sr, 
        threshold=db_thresh, 
        min_interval=min_interval
    )

    return slicer.slice2(audio)

def restore(segments, total_len, dtype=np.float32):
    """
    Restore the original timeline after segmented processing.

    Any removed silent regions are replaced with zeros so the final
    waveform matches the original duration.

    Args:
        segments: List of (start, end, processed_audio).
        total_len: Original waveform length.
        dtype: Output dtype.

    Returns:
        np.ndarray
    """

    out = []
    last_end = 0

    for start, end, processed_seg in segments:
        if start > last_end: # Restore skipped silent regions.
            out.append(
                np.zeros(start - last_end, dtype=dtype)
            )

        out.append(processed_seg) # Insert the processed segment.
        last_end = end

    # Restore trailing silence.
    if last_end < total_len: 
        out.append(
            np.zeros(total_len - last_end, dtype=dtype)
        )

    return np.concatenate(out, axis=-1)