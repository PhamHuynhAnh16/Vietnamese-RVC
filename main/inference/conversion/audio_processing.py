import os
import sys
import torch
import librosa

import numpy as np
import scipy.signal as signal

sys.path.append(os.getcwd())

def rms(x, eps=1e-9):
    return np.sqrt(np.mean(x**2) + eps)

def dc_remove(x):
    return x - np.mean(x)

def normalize_rms(x, target=0.08):
    r = rms(x)
    return x if r < 1e-6 else x * (target / r)

def soft_limiter(x, th=0.98):
    return np.tanh(x / th) * th

def preprocess(
    audio,
    target_sr=16000
):
    x = dc_remove(audio)
    x = signal.filtfilt(*signal.butter(2, 80 / (target_sr / 2), btype="high"), x)

    x = normalize_rms(x, 0.06)
    x = spectral_denoise_np(x, alpha=1.0)

    x /= (np.max(np.abs(x)) + 1e-9) * 0.99
    return x.astype(np.float32)

def spectral_denoise_np(
    audio,
    n_fft=2048,
    hop=480,
    alpha=2.0
):
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop,
        window="hann"
    )

    mag = np.abs(stft)
    phase = np.angle(stft)

    energy = np.mean(mag, axis=0)
    ref = np.median(energy)
    silent = energy < 0.6 * ref

    noise = (
        np.mean(mag[:, silent], axis=1, keepdims=True)
        if np.any(silent)
        else np.mean(mag, axis=1, keepdims=True)
    )

    mask = mag / (mag + alpha * noise + 1e-9)
    clean = mask * mag

    out = librosa.istft(
        clean * np.exp(1j * phase),
        hop_length=hop,
        length=len(audio)
    )

    return out.astype(np.float32)

def output_eq(audio, sr):
    out = audio.copy()

    bands = [
        (0,   80,  -6.0),
        (80,  200, -2.0),
        (200, 3500, 0.0),
        (3500, 9000, +2.0),
    ]

    for low, high, g in bands:
        gain = 10 ** (g / 20)

        if low == 0:
            b, a = signal.butter(2, high / (sr / 2), "low")
        else:
            b, a = signal.butter(2, [low / (sr / 2), high / (sr / 2)], "band")

        band = signal.filtfilt(b, a, audio)
        out += (gain - 1.0) * band

    return out

def presence_exciter(audio, sr, strength=0.03):
    b, a = signal.butter(2, [4000 / (sr / 2), 9000 / (sr / 2)], "band")
    band = signal.filtfilt(b, a, audio)

    exc = np.tanh(band * 3.0)
    return audio + strength * exc

def postprocess(
    audio,
    sr=48000
):
    x = dc_remove(audio)
    x = signal.filtfilt(*signal.butter(2, 60 / (sr / 2), "high"), x)

    x = spectral_denoise_np(x)
    x = output_eq(x, sr)

    x = presence_exciter(x, sr)
    x = soft_limiter(x, 0.98)

    x /= (np.max(np.abs(x)) + 1e-9) * 0.99
    return x.astype(np.float32)