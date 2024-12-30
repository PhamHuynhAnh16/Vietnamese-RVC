import os
import codecs
import librosa
import requests

import soundfile as sf

def google_tts(text, lang="vi", speed=1, pitch=0, output_file="output.mp3"):
    try:
        response = requests.get(codecs.decode("uggcf://genafyngr.tbbtyr.pbz/genafyngr_ggf", "rot13"), params={"ie": "UTF-8", "q": text, "tl": lang, "ttsspeed": speed, "client": "tw-ob"}, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"})

        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)

            format = os.path.splitext(os.path.basename(output_file))[-1].lower().replace('.', '')

            if pitch != 0: pitch_shift(input_file=output_file, output_file=output_file, pitch=pitch, export_format=format)
            if speed != 1: change_speed(input_file=output_file, output_file=output_file, speed=speed, export_format=format)
        else: raise ValueError(f"{response.status_code}, {response.text}")
    except Exception as e:
        raise RuntimeError(e)

def pitch_shift(input_file, output_file, pitch, export_format):
    y, sr = librosa.load(input_file, sr=None)
    sf.write(file=output_file, data=librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch), samplerate=sr, format=export_format)

def change_speed(input_file, output_file, speed, export_format):
    y, sr = librosa.load(input_file, sr=None)   
    sf.write(file=output_file, data=librosa.effects.time_stretch(y, rate=speed), samplerate=sr, format=export_format)