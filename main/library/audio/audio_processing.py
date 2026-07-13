import os
import sys
import librosa

import numpy as np
import scipy.signal as signal

sys.path.append(os.getcwd())

def rms(x, eps=1e-9):
    """
    Compute the root mean square (RMS) level of an audio signal.

    Args:
        x: Input audio signal.
        eps: Small constant for numerical stability.

    Returns:
        float: RMS value.
    """

    return np.sqrt(np.mean(x**2) + eps)

def dc_remove(x):
    """
    Remove the DC offset from an audio signal.

    Args:
        x: Input audio signal.

    Returns:
        np.ndarray: Zero-centered audio.
    """

    return x - np.mean(x)

def normalize_rms(x, target=0.08):
    """
    Normalize the signal to a target RMS level.

    Args:
        x: Input audio signal.
        target: Desired RMS level.

    Returns:
        np.ndarray: RMS-normalized audio.
    """

    r = rms(x)
    return x if r < 1e-6 else x * (target / r)

def soft_limiter(x, th=0.98):
    """
    Apply a soft limiter to prevent clipping.

    Samples below the threshold remain unchanged, while peaks are
    compressed smoothly instead of hard-clipped.

    Args:
        x: Input audio signal.
        th: Limiter threshold.

    Returns:
        np.ndarray: Limited audio.
    """

    return np.where(
        np.abs(x) < th,
        x,
        th * np.sign(x) + (x - th * np.sign(x)) / (1 + ((x - th) / (1 - th))**2)
    )

def preprocess(
    audio,
    target_sr=16000
):
    """
    Preprocess audio before inference.

    Processing steps:
        1. Remove DC offset.
        2. Apply a high-pass filter.
        3. Normalize RMS.
        4. Reduce stationary noise.
        5. Normalize peak amplitude.

    Args:
        audio: Input waveform.
        target_sr: Audio sample rate.

    Returns:
        np.ndarray: Processed audio.
    """

    # Remove DC offset and low-frequency rumble.
    x = dc_remove(audio)

    # High-pass filter at 80 Hz.
    # Removes DC drift, microphone rumble, wind noise and handling vibrations
    # while preserving the fundamental frequencies of most speech.
    x = signal.filtfilt(*signal.butter(2, 80 / (target_sr / 2), btype="high"), x)

    # Normalize loudness before denoising.
    x = normalize_rms(x, 0.06)

    # Suppress stationary background noise.
    x = spectral_denoise_np(x, alpha=1.0)

    # Normalize peak amplitude.
    x /= (np.max(np.abs(x)) + 1e-9) * 0.99
    return x.astype(np.float32)

def spectral_denoise_np(
    audio,
    n_fft=4096,
    hop=1024,
    alpha=2.5
):
    """
    Reduce stationary noise using spectral Wiener filtering.

    Noise statistics are estimated from relatively quiet frames,
    then used to attenuate the noise spectrum.

    Args:
        audio: Input waveform.
        n_fft: FFT size.
        hop: Hop size.
        alpha: Noise suppression strength.

    Returns:
        np.ndarray: Denoised audio.
    """

    if hop is None: hop = n_fft // 4

    # Compute STFT.
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop,
        window="hann"
    )

    mag = np.abs(stft)
    phase = np.angle(stft)

    # Detect relatively silent frames.
    energy = np.mean(mag**2, axis=0)
    silent = energy < np.percentile(energy, 30)

    # Estimate the average spectrum from relatively quiet frames.
    # These frames are assumed to contain mostly stationary background noise.
    noise = (
        np.median(mag[:, silent], axis=1, keepdims=True)
        if np.any(silent)
        else np.median(mag, axis=1, keepdims=True)
    )

    # Wiener-like spectral gain.
    # alpha controls the amount of noise suppression:
    #   alpha < 1.0 : weaker denoising
    #   alpha = 1.0 : balanced
    #   alpha > 2.0 : stronger suppression but may introduce artifacts
    mask = (
        mag**2 /
        (mag**2 + alpha * (noise**2) + 1e-9)
    )

    clean = mask * mag

    # Reconstruct waveform.
    out = librosa.istft(
        clean * np.exp(1j * phase),
        hop_length=hop,
        length=len(audio)
    )

    return out.astype(np.float32)

def output_eq(audio, sr):
    """
    Apply a fixed multi-band equalizer.

    The EQ slightly attenuates low frequencies while enhancing
    speech presence and high-frequency clarity.

    Args:
        audio: Input waveform.
        sr: Sample rate.

    Returns:
        np.ndarray: Equalized audio.
    """

    out = audio.copy()

    bands = [
        (0, 60, -8.0), # Sub-bass. Strong attenuation removes HVAC hum, microphone vibration and unnecessary low-frequency energy.
        (60, 200, -3.0), # Bass. Slight cut reduces muddiness and boominess in speech.
        (200, 1000, 0.0), # Midrange. Keep natural vocal body unchanged.
        (1000, 3500, +2.5), # Presence. Boost intelligibility by emphasizing consonants.
        (3500, 8000, +3.0), # High presence. Improve clarity and articulation.
        (8000, 16000, +1.5) # Air. Add brightness without excessive harshness.
    ]

    for low, high, g in bands:
        gain = 10 ** (g / 20)

        if low == 0:
            b, a = signal.butter(2, high / (sr / 2), "low")
        else:
            b, a = signal.butter(2, [low / (sr / 2), high / (sr / 2)], "band")

        band = signal.filtfilt(b, a, audio)
        out += (gain - 1.0) * band # Blend the adjusted frequency band back into the signal.

    return out

def presence_exciter(audio, sr, strength=0.03):
    """
    Add harmonic excitation to improve vocal presence.

    Args:
        audio: Input waveform.
        sr: Sample rate.
        strength: Exciter intensity.

    Returns:
        np.ndarray: Enhanced audio.
    """

    # Isolate the upper vocal harmonics.
    # Frequencies around 4–9 kHz contribute to speech clarity,
    # articulation and perceived detail.
    b, a = signal.butter(2, [4000 / (sr / 2), 9000 / (sr / 2)], "band")
    band = signal.filtfilt(b, a, audio)

    # Generate harmonics using soft saturation.
    exc = np.sign(band) * (1 - np.exp(-np.abs(band) * 4.0))
    return audio + strength * exc

def postprocess(
    audio,
    sr=48000
):
    """
    Postprocess audio after inference.

    Processing steps:
        1. Remove DC offset.
        2. High-pass filtering.
        3. Spectral denoising.
        4. Multi-band EQ.
        5. Harmonic excitation.
        6. Soft limiting.
        7. Peak normalization.

    Args:
        audio: Input waveform.
        sr: Sample rate.

    Returns:
        np.ndarray: Final processed audio.
    """

    # Remove DC and low-frequency rumble.
    x = dc_remove(audio)
    x = signal.filtfilt(*signal.butter(2, 60 / (sr / 2), "high"), x)

    # Reduce residual background noise.
    x = spectral_denoise_np(x)
    # Shape the frequency response.
    x = output_eq(x, sr)
    # Restore brightness and articulation.
    x = presence_exciter(x, sr)
    # Prevent clipping.
    x = soft_limiter(x, 0.98)

    # Normalize output level.
    x /= (np.max(np.abs(x)) + 1e-9) * 0.99
    return x.astype(np.float32)