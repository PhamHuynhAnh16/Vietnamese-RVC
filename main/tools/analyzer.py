import os
import sys
import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from main.app.variables import translations

def calculate_features(y, sr):
    stft = np.abs(librosa.stft(y))
    return stft, librosa.get_duration(y=y, sr=sr), librosa.feature.spectral_centroid(S=stft, sr=sr)[0], librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0], librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]

def plot_title(title):
    plt.suptitle(title, fontsize=16, fontweight="bold")

def plot_spectrogram(sr, stft, duration, cmap="inferno"):
    plt.subplot(3, 1, 1)
    plt.imshow(librosa.amplitude_to_db(stft, ref=np.max), origin="lower", extent=[0, duration, 0, sr / 1000], aspect="auto", cmap=cmap)
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel(translations["times"])
    plt.ylabel(translations["frequencykhz"])
    plt.title(translations["spectrogram"])

def plot_waveform(y, sr):
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel(translations["times"])
    plt.ylabel(translations["amplitude"])
    plt.title(translations["waveform"])

def plot_features(times, cent, bw, rolloff):
    plt.subplot(3, 1, 3)
    plt.plot(times, cent, label=translations["centroid"], color="b")
    plt.plot(times, bw, label=translations["bandwidth"], color="g")
    plt.plot(times, rolloff, label=translations["rolloff"], color="r")
    plt.xlabel(translations["times"])
    plt.title(translations["features"])
    plt.legend()

def analyze_audio(audio_file):
    save_plot_path = "assets/audio_analysis.png"

    y, sr = librosa.load(audio_file, sr=None)
    stft, duration, cent, bw, rolloff = calculate_features(y, sr)

    plt.figure(figsize=(12, 10))

    plot_title(translations["analysis"])
    plot_spectrogram(sr, stft, duration)
    plot_waveform(y, sr)
    plot_features(librosa.times_like(cent), cent, bw, rolloff)

    plt.tight_layout()

    if save_plot_path: plt.savefig(save_plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    audio_info = translations["analysis_info"].format(
        sr=sr, 
        duration=(str(round(duration, 2)) + translations["seconds"] if duration < 60 else str(round(duration / 60, 2)) + translations["minutes"]), 
        length=len(y), 
        bit=librosa.get_samplerate(audio_file), 
        channel=translations["mono"] if y.ndim == 1 else translations["stereo"]
    )
    
    return audio_info, save_plot_path