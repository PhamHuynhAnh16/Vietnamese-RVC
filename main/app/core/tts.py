import os
import sys
import pysrt
import codecs
import asyncio
import requests
import tempfile

sys.path.append(os.getcwd())

from main.app.variables import translations
from main.app.core.ui import gr_info, gr_warning, gr_error

def synthesize_tts(
    prompt, 
    voice = "vi-VN-NamMinhNeural", 
    speed = 0, 
    output = "audios/tts.wav", 
    pitch = 0, 
    google = False
):
    """
    Synthesizes text into speech using either Edge-TTS or Google Translate TTS.
    """

    if not google: 
        # Lazy import to avoid loading edge_tts unnecessarily if Google is selected
        from edge_tts import Communicate

        # Run the asynchronous edge_tts saving process synchronously
        asyncio.run(
            Communicate(
                text=prompt, 
                voice=voice, 
                # Format rate and pitch strings according to edge_tts requirements (e.g., "+0%", "-5Hz")
                rate=f"+{speed}%" if speed >= 0 else f"{speed}%", 
                pitch=f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"
            ).save(output)
        )
    else: 
        # Request speech from Google Translate TTS API (Obfuscated via ROT13)
        response = requests.get(
            codecs.decode(
                "uggcf://genafyngr.tbbtyr.pbz/genafyngr_ggf", 
                "rot13"
            ), 
            params={
                "ie": "UTF-8", 
                "q": prompt, 
                "tl": voice, 
                # Map speed percentage to a factor range between 0.24 and 4.0
                "ttsspeed": max(0.24, min(4.0, 1 + speed / 100)), 
                "client": "tw-ob"
            }, 
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
            },
            timeout=15
        )

        if response.status_code == 200:
            # Save the raw audio binary data
            with open(output, "wb") as f:
                f.write(response.content)

            # Apply post-processing modifications since Google API doesn't support pitch adjustments natively
            if pitch != 0 or speed != 0:
                import librosa
                import soundfile as sf

                # Load the newly saved audio file while preserving its original sample rate
                y, sr = librosa.load(output, sr=None)

                # Shift audio pitch if specified
                if pitch != 0: 
                    y = librosa.effects.pitch_shift(
                        y, 
                        sr=sr, 
                        n_steps=pitch
                    )

                # Stretch or compress time if speed deviates from normal
                rate = max(0.01, 1 + speed / 100)
                if rate != 1.0: 
                    y = librosa.effects.time_stretch(
                        y, 
                        rate=rate
                    )

                sf.write(
                    file=output, 
                    data=y, 
                    samplerate=sr, 
                    # Determine file format from extension and overwrite the file with adjusted audio
                    format=os.path.splitext(
                        os.path.basename(output)
                    )[-1].lower().replace('.', '')
                )
        else: gr_error(f"{response.status_code}, {response.text}") # Display UI error if API request fails

def srt_tts(
    srt_file, 
    out_file = "audios/tts.wav", 
    voice = "vi-VN-NamMinhNeural", 
    rate = 0, 
    sr = 24000, 
    google = False
):
    """
    Converts a SubRip Subtitle (SRT) file into a synchronized full-length audio track.
    """

    import librosa
    import numpy as np
    import soundfile as sf

    def time_stretch(y, sr, target_duration):
        """Stretches or compresses audio to fit perfectly into a target duration slot."""
    
        rate = (len(y) / sr) / target_duration

        if rate != 1.0: 
            y = librosa.effects.time_stretch(
                y=y.astype(np.float32), 
                rate=rate
            )

        # Pad with silence or truncate to guarantee precise sample alignment
        n_target = int(round(target_duration * sr))
        return np.pad(y, (0, n_target - len(y))) if len(y) < n_target else y[:n_target]

    def pysrttime_to_seconds(t):
        """Converts a pysrt SubRipTime timestamp object into total seconds."""

        return (t.hours * 60 + t.minutes) * 60 + t.seconds + t.milliseconds / 1000

    # Parse subtitle file
    subs = pysrt.open(srt_file)
    if not subs: raise ValueError(translations["srt"])

    # Allocate the entire timeline buffer based on the end timestamp of the final segment
    final_audio = np.zeros(
        int(round(pysrttime_to_seconds(subs[-1].end) * sr)), 
        dtype=np.float32
    )

    # Use a secure temporary directory to cache individual segment audios
    with tempfile.TemporaryDirectory() as tempdir:
        for idx, seg in enumerate(subs):
            wav_path = os.path.join(tempdir, f"seg_{idx}.wav")
            # Generate individual audio clip for the current subtitle block
            synthesize_tts(
                # Flatten multi-line segment text into a single cohesive string line
                " ".join(seg.text.splitlines()), 
                voice, 
                0, 
                wav_path, 
                rate, 
                google
            )

            # Read back generated segment clip
            audio, file_sr = sf.read(wav_path, dtype=np.float32)
            # Resample audio if it doesn't match the target master sample rate
            if file_sr != sr: audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr, res_type="soxr_vhq")

            # Stretch segment audio to precisely match the timestamp duration allocated in the SRT
            adjusted = time_stretch(
                audio, 
                sr, 
                pysrttime_to_seconds(seg.duration)
            )

            # Calculate precise insertion index boundaries relative to the master timeline
            start_sample = int(round(pysrttime_to_seconds(seg.start) * sr))
            end_sample = start_sample + adjusted.shape[0]
            # Prevent buffer overflow bounds issues near the timeline tail end
            if end_sample > final_audio.shape[0]:
                adjusted = adjusted[: final_audio.shape[0] - start_sample]
                end_sample = final_audio.shape[0]

            # Overlay segment audio onto the master track canvas
            final_audio[start_sample:end_sample] += adjusted

    # Write out the finalized synchronized track
    sf.write(out_file, final_audio, sr)

def TTS(
    prompt, 
    voice = "vi-VN-NamMinhNeural", 
    speed = 0, 
    output = "audios/tts.wav", 
    pitch = 0, 
    google = False, 
    srt_input = ""
):
    """
    High-level orchestration entry point managing validations and pipeline distribution.
    """

    # Standardize string inputs
    if not srt_input: srt_input = ""

    # Validate that at least one viable text target or script file is loaded
    if not prompt and not srt_input.endswith(".srt"):
        gr_warning(translations["enter_the_text"])
        return None
    
    # Validate voice model selection parameter
    if not voice:
        gr_warning(translations["choose_voice"])
        return None
    
    # Validate destination target configurations
    if not output: 
        gr_warning(translations["output_not_valid"])
        return None
    
    # If the user passes a folder instead of a filename, append a default filename
    if os.path.isdir(output): output = os.path.join(output, f"tts.wav")
    gr_info(translations["converttext"])

    # Ensure container directories exist safely before attempting file writes
    output_dir = os.path.dirname(output) or output
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    # Route logic workflows depending on whether file configuration is an SRT or standard prompt text
    if srt_input.endswith(".srt"): 
        srt_tts(
            srt_input, 
            output, 
            voice, 
            0, 
            24000, 
            google
        )
    else: 
        synthesize_tts(
            prompt, 
            voice, 
            speed, 
            output, 
            pitch, 
            google
        )

    gr_info(translations["success"])
    return output