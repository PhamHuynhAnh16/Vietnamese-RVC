import os
import gc
import sys
import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from main.app.variables import translations

def calculate_features(y, sr):
    """
    Calculates various audio features including STFT, duration, and spectral features.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.

    Returns:
        Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]: A tuple
        containing:
            - stft (np.ndarray): Absolute Short-Time Fourier Transform.
            - duration (float): Duration of the audio in seconds.
            - centroid (np.ndarray): Spectral centroid.
            - bandwidth (np.ndarray): Spectral bandwidth.
            - rolloff (np.ndarray): Spectral rolloff.
    """

    # Compute the Short-Time Fourier Transform (STFT) and get its magnitude
    stft = np.abs(librosa.stft(y))
    return (
        stft, 
        librosa.get_duration(y=y, sr=sr), 
        # Extract spectral features from the STFT magnitude
        librosa.feature.spectral_centroid(S=stft, sr=sr)[0], 
        librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0], 
        librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
    )

def plot_title(title):
    """
    Sets a global bold title for the current matplotlib figure.

    Args:
        title (str): The text to display as the main title.
    """

    plt.suptitle(title, fontsize=16, fontweight="bold")

def plot_spectrogram(sr, stft, duration, cmap="inferno"):
    """
    Plots the spectrogram of the audio signal.

    Args:
        sr (int): Sampling rate of the audio.
        stft (np.ndarray): Magnitude of the STFT.
        duration (float): Duration of the audio in seconds.
        cmap (str, optional): Colormap for the spectrogram plot. Defaults to "inferno".
    """

    # Define position in a 3x1 grid subplot
    plt.subplot(3, 1, 1)
    # Convert amplitude to decibels and display as an image
    # Frequency axis is converted to kHz (sr / 1000)
    plt.imshow(librosa.amplitude_to_db(stft, ref=np.max), origin="lower", extent=[0, duration, 0, sr / 1000], aspect="auto", cmap=cmap)
    # Add a colorbar indicating dB levels
    plt.colorbar(format="%+2.0f dB")
    # Set localized labels and title from translations dictionary
    plt.xlabel(translations["times"])
    plt.ylabel(translations["frequencykhz"])
    plt.title(translations["spectrogram"])

def plot_waveform(y, sr):
    """
    Plots the time-domain waveform of the audio signal.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.
    """

    # Define position in a 3x1 grid subplot
    plt.subplot(3, 1, 2)
    # Use librosa's optimized waveform display helper
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel(translations["times"])
    plt.ylabel(translations["amplitude"])
    plt.title(translations["waveform"])

def plot_features(times, cent, bw, rolloff):
    """
    Plots spectral features (centroid, bandwidth, rolloff) over time.

    Args:
        times (np.ndarray): Array of time boundaries for each frame.
        cent (np.ndarray): Spectral centroid values.
        bw (np.ndarray): Spectral bandwidth values.
        rolloff (np.ndarray): Spectral rolloff values.
    """

    # Define position in a 3x1 grid subplot
    plt.subplot(3, 1, 3)
    plt.plot(times, cent, label=translations["centroid"], color="b")
    plt.plot(times, bw, label=translations["bandwidth"], color="g")
    plt.plot(times, rolloff, label=translations["rolloff"], color="r")
    # Set localized labels, title, and display the legend
    plt.xlabel(translations["times"])
    plt.title(translations["features"])
    plt.legend()

def analyze_audio(audio_file):
    """Performs comprehensive audio analysis, generates visualization plots, and saves the output image.

    Args:
        audio_file (str): Path to the target audio file.

    Returns:
        Tuple[str, str]: A tuple containing:
            - audio_info (str): A formatted summary of the audio properties.
            - save_plot_path (str): The file path where the analysis plot was
              saved.
    """

    # Define the output directory and path for the analysis plot
    save_plot_path = os.path.join("assets", "audio_analysis.png")

    # Load audio file (sr=None preserves native sampling rate)
    y, sr = librosa.load(audio_file, sr=None)
    # Extract all necessary audio features
    stft, duration, cent, bw, rolloff = calculate_features(y, sr)

    # Initialize a clean matplotlib figure size
    plt.figure(figsize=(12, 10))

    # Generate all subplots
    plot_title(translations["analysis"])
    plot_spectrogram(sr, stft, duration)
    plot_waveform(y, sr)
    plot_features(librosa.times_like(cent), cent, bw, rolloff)

    # Adjust subplot parameters to give specified padding
    plt.tight_layout()

    # Save the visualization to disk if path is provided
    if save_plot_path: plt.savefig(save_plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Construct the localized summary string
    audio_info = translations["analysis_info"].format(
        sr=sr, 
        # Format the duration string dynamically based on length (seconds vs minutes)
        duration=(str(round(duration, 2)) + translations["seconds"] if duration < 60 else str(round(duration / 60, 2)) + translations["minutes"]), 
        length=len(y), 
        bit=librosa.get_samplerate(audio_file), 
        channel=translations["mono"] if y.ndim == 1 else translations["stereo"]
    )

    # Clean up large memory-consuming objects explicitly
    del y, sr, stft, duration, cent, bw, rolloff
    gc.collect()

    return audio_info, save_plot_path