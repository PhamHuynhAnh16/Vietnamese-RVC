import os
import sys
import librosa
import platform

import numpy as np
import soundfile as sf

sys.path.append(os.getcwd())

from main.app.variables import translations, logger

wav_resolution = ("polyphase" if platform.processor() == "arm" or "arm" in platform.platform() else "sinc_fastest") if platform.system() == "Darwin" else "sinc_fastest"

def crop_center(h1, h2):
    """
    Crops the time dimension (width) of tensor h1 to precisely match the size of tensor h2.

    Args:
        h1 (torch.Tensor): Source tensor to be cropped, typically with a larger time dimension.
        h2 (torch.Tensor): Target reference tensor providing the destination shape bounds.

    Returns:
        torch.Tensor: Center-cropped slice of the h1 tensor.

    Raises:
        ValueError: If the time dimension of h1 is smaller than h2.
    """

    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]: return h1
    elif h1_shape[3] < h2_shape[3]: raise ValueError("Source tensor dimensions cannot be smaller than target reference bounds.")

    # Calculate symmetrical slicing boundary markers
    s_time = (h1_shape[3] - h2_shape[3]) // 2

    h1 = h1[:, :, :, s_time:s_time + h2_shape[3]]
    return h1

def preprocess(X_spec):
    """
    Decomposes a complex-valued spectrogram matrix into its magnitude and phase components.

    Args:
        X_spec (np.ndarray): Complex spectrogram array.

    Returns:
        tuple: (magnitude, phase) as numpy float arrays.
    """

    return np.abs(X_spec), np.angle(X_spec)

def make_padding(width, cropsize, offset):
    """
    Calculates symmetrical window border padding layouts for block-based inference overlapping loops.

    Args:
        width (int): Total length of the audio track in frames.
        cropsize (int): Standard block execution segment width window size.
        offset (int): Trim overhead factor to strip edge window fading artifacts.

    Returns:
        tuple: (left_offset, total_padding_needed, region_of_interest_size)
    """

    roi_size = cropsize - offset * 2

    if roi_size == 0: roi_size = cropsize
    return offset, roi_size - (width % roi_size) + offset, roi_size

def normalize(wave, max_peak=1.0):
    """
    Rescales a waveform linearly so that its absolute peak matches the target maximum ceiling.

    Args:
        wave (np.ndarray): Input audio waveform array.
        max_peak (float): Absolute ceiling limit for amplitude normalization. Defaults to 1.0.

    Returns:
        np.ndarray: Amplitude-normalized audio wave matrix.
    """

    maxv = np.abs(wave).max()

    if maxv > max_peak: wave *= max_peak / maxv
    return wave

def write_array_to_mem(audio_data, subtype):
    """
    Encodes raw numpy audio signal waveforms directly into an in-memory WAV byte buffer stream.

    Args:
        audio_data (np.ndarray): Target raw numpy audio signal data.
        subtype (str): Soundfile encoding bit-depth subtype specification (e.g., 'PCM_16', 'FLOAT').

    Returns:
        io.BytesIO or object: Seeked in-memory byte buffer containing the encoded file content, or the original object if not a numpy array.
    """

    if isinstance(audio_data, np.ndarray):
        import io

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, 44100, subtype=subtype, format="WAV")

        audio_buffer.seek(0) # Rewind pointer stream to start index position
        return audio_buffer
    else: return audio_data

def reduce_vocal_aggressively(X, y, softmask):
    """
    Suppresses remaining vocal bleed or transient leakage from an isolated accompaniment spectrogram.

    Args:
        X (np.ndarray): Original full mixture track spectrogram matrix.
        y (np.ndarray): Separated instrumental/accompaniment track spectrogram.
        softmask (float): Masking weight ratio modifier scaling factor.

    Returns:
        np.ndarray: Filtered spectrogram matrix with reduced bleed artifacts.
    """

    y_mag_tmp = np.abs(y)
    v_mag_tmp = np.abs(X - y)

    # Apply weighted error compensation mask to cancel phase overlaps
    return np.clip(y_mag_tmp - v_mag_tmp * (v_mag_tmp > y_mag_tmp) * softmask, 0, np.inf) * np.exp(1.0j * np.angle(y))

def merge_artifacts(y_mask, thres=0.01, min_range=64, fade_size=32):
    """
    Smoothes out localized mask disruptions to eliminate isolated clicking/chirping audio artifacts.

    Args:
        y_mask (np.ndarray): Original execution binary or soft tracking matrix.
        thres (float): Energy threshold filter level to detect active frame areas. Defaults to 0.01.
        min_range (int): Minimum contiguous frame duration to qualify as a valid mask area. Defaults to 64.
        fade_size (int): Symmetrical cross-fade sample length step sizes. Defaults to 32.

    Returns:
        np.ndarray: Cleaned artifact-free separation mask tracking matrix.
    """

    mask = y_mask

    try:
        if min_range < fade_size * 2: raise ValueError("Minimum range parameter must comfortably wrap double the cross-fade step lengths.")

        # Identify frames where the isolation performance exceeds mask tolerance thresholds
        idx = np.where(y_mask.min(axis=(0, 1)) > thres)[0]
        start_idx = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
        end_idx = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
        artifact_idx = np.where(end_idx - start_idx > min_range)[0]
        weight = np.zeros_like(y_mask)

        if len(artifact_idx) > 0:
            start_idx = start_idx[artifact_idx]
            end_idx = end_idx[artifact_idx]
            old_e = None

            # Calculate window slopes to patch transient spikes smoothly
            for s, e in zip(start_idx, end_idx):
                if old_e is not None and s - old_e < fade_size: s = old_e - fade_size * 2

                if s != 0: weight[:, :, s : s + fade_size] = np.linspace(0, 1, fade_size)
                else: s -= fade_size

                if e != y_mask.shape[2]: weight[:, :, e - fade_size : e] = np.linspace(1, 0, fade_size)
                else: e += fade_size

                weight[:, :, s + fade_size : e - fade_size] = 1
                old_e = e

        v_mask = 1 - y_mask
        y_mask += weight * v_mask
        mask = y_mask
    except Exception as e:
        import traceback
        logger.error(f'{translations["not_success"]} {type(e).__name__}: {e}\n{traceback.format_exc()}')

    return mask

def convert_channels(spec, mp, band):
    """
    Converts spatial audio channels (Stereo vs Mid-Side matrix formulations) based on model band rules.

    Args:
        spec (np.ndarray): Multi-channel input spectrogram array.
        mp (object): Model parameter storage abstraction configuration instance.
        band (int): Identifier tag mapping specific target frequency bands.

    Returns:
        np.ndarray: Spatial channel-reconfigured Fortran-contiguous array layout.
    """

    cc = mp.param["band"][str(band)].get("convert_channels")

    if "mid_side_c" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25)
        spec_right = np.subtract(spec[1], spec[0] * 0.25)
    elif "mid_side" == cc:
        spec_left = np.add(spec[0], spec[1]) / 2
        spec_right = np.subtract(spec[0], spec[1])
    elif "stereo_n" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25) / 0.9375
        spec_right = np.add(spec[1], spec[0] * 0.25) / 0.9375
    else: return spec

    return np.asfortranarray([spec_left, spec_right])

def combine_spectrograms(specs, mp, is_v51_model=False):
    """
    Merges diverse multi-band spectrogram chunks into a single unified master full-band tracking array.

    Args:
        specs (dict): Dictionary mapping band index tags to their isolated partial spectrogram matrices.
        mp (object): Model parameters class instance containing structural frequency definitions.
        is_v51_model (bool): Flag defining whether processing rules use newer UVR v5.1 architecture schemas.

    Returns:
        np.ndarray: Recombined master full-band spectrogram matrix.
    """

    l = min([specs[i].shape[2] for i in specs])
    spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, l), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param["band"])

    # Stitch target bands across designated frequency bin ranges
    for d in range(1, bands_n + 1):
        h = mp.param["band"][str(d)]["crop_stop"] - mp.param["band"][str(d)]["crop_start"]
        spec_c[:, offset : offset + h, :l] = specs[d][:, mp.param["band"][str(d)]["crop_start"] : mp.param["band"][str(d)]["crop_stop"], :l]
        offset += h

    if offset > mp.param["bins"]: raise ValueError("Assembled frequency bin height exceeds system maximum definition targets.")
    # Apply low-pass anti-aliasing filters across matching edge joints
    if mp.param["pre_filter_start"] > 0:
        if is_v51_model: spec_c *= get_lp_filter_mask(spec_c.shape[1], mp.param["pre_filter_start"], mp.param["pre_filter_stop"])
        else:
            if bands_n == 1: spec_c = fft_lp_filter(spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"])
            else:
                import math
                gp = 1

                # Apply a gradual logarithmic filter slope across crossover regions
                for b in range(mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]):
                    g = math.pow(10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0)
                    gp = g
                    spec_c[:, b, :] *= g

    return np.asfortranarray(spec_c)

def wave_to_spectrogram(wave, hop_length, n_fft, mp, band, is_v51_model=False):
    """
    Converts time-domain wave signals into multi-channel complex short-time Fourier transform frames.

    Args:
        wave (np.ndarray): Target input audio waveform array.
        hop_length (int): Distance stepping parameters between processing context frames.
        n_fft (int): Total size length window of Fourier transform bins.
        mp (object): Model metadata parameters configuration map.
        band (int): Identifier index mapping active target processing tracks.
        is_v51_model (bool): Flag defining whether processing rules use UVR v5.1 architecture schemas.

    Returns:
        np.ndarray: Evaluated complex spectrogram matrix array.
    """

    if wave.ndim == 1: wave = np.asfortranarray([wave, wave])
    # Apply spatial pre-processing channel conversions depending on version standards
    if not is_v51_model:
        if mp.param["reverse"]:
            wave_left = np.flip(np.asfortranarray(wave[0]))
            wave_right = np.flip(np.asfortranarray(wave[1]))
        elif mp.param["mid_side"]:
            wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
        elif mp.param["mid_side_b2"]:
            wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
        else:
            wave_left = np.asfortranarray(wave[0])
            wave_right = np.asfortranarray(wave[1])
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft=n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)

    spec = np.asfortranarray([spec_left, spec_right])

    if is_v51_model: spec = convert_channels(spec, mp, band)
    return spec

def spectrogram_to_wave(spec, hop_length=1024, mp={}, band=0, is_v51_model=True):
    """
    Transforms multi-channel complex spectrogram matrices back into standard time-domain waveform tracks.

    Args:
        spec (np.ndarray): Processed complex target spectrogram arrays.
        hop_length (int): Distance stepping parameters between processing context frames. Defaults to 1024.
        mp (dict): Configuration parameter storage dictionary.
        band (int): Active band index tracker identifier. Defaults to 0.
        is_v51_model (bool): Flag defining whether processing rules use UVR v5.1 architecture schemas.

    Returns:
        np.ndarray: Reconstructed time-domain audio track waveform matrix.
    """

    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)

    # Perform channel decoding matrix expansions based on architectural configurations
    if is_v51_model:
        cc = mp.param["band"][str(band)].get("convert_channels")

        if "mid_side_c" == cc: 
            return np.asfortranarray([
                np.subtract(wave_left / 1.0625, wave_right / 4.25), 
                np.add(wave_right / 1.0625, wave_left / 4.25)
            ])
        elif "mid_side" == cc: 
            return np.asfortranarray([
                np.add(wave_left, wave_right / 2), 
                np.subtract(wave_left, wave_right / 2)
            ])
        elif "stereo_n" == cc: 
            return np.asfortranarray([
                np.subtract(wave_left, wave_right * 0.25), 
                np.subtract(wave_right, wave_left * 0.25)
            ])
    else:
        if mp.param["reverse"]: 
            return np.asfortranarray([
                np.flip(wave_left), 
                np.flip(wave_right)
            ])
        elif mp.param["mid_side"]: 
            return np.asfortranarray([
                np.add(wave_left, wave_right / 2), 
                np.subtract(wave_left, wave_right / 2)
            ])
        elif mp.param["mid_side_b2"]: 
            return np.asfortranarray([
                np.add(wave_right / 1.25, 0.4 * wave_left), 
                np.subtract(wave_left / 1.25, 0.4 * wave_right)
            ])

    return np.asfortranarray([wave_left, wave_right])

def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None, is_v51_model=False):
    """
    Decomposes an integrated full master spectrogram tracking array back into its multi-band 
    sub-components and executes individual inverse STFT operations with appropriate cross-resampling.

    Args:
        spec_m (np.ndarray): Assembled master full-band spectrogram matrix.
        mp (object): Model parameter storage abstraction instance.
        extra_bins_h (int, optional): Spatial allocation height limits for ultra-high frequency padding.
        extra_bins (np.ndarray, optional): Reference matrices containing ultra-high frequency spectrogram padding.
        is_v51_model (bool): Flag defining whether processing rules use UVR v5.1 architecture schemas.

    Returns:
        np.ndarray: Recombined time-domain audio waveform track array.
    """

    bands_n = len(mp.param["band"])
    offset = 0

    for d in range(1, bands_n + 1):
        bp = mp.param["band"][str(d)]
        spec_s = np.zeros(shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex)
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[:, offset : offset + h, :]
        offset += h

        if d == bands_n:
            # Inject ultra-high frequency details from the original mix to preserve high-end clarity
            if extra_bins_h:  
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[:, :extra_bins_h, :]

            # Apply high-pass crossover isolation filtering configurations
            if bp["hpf_start"] > 0:
                if is_v51_model: spec_s *= get_hp_filter_mask(spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1)
                else: spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)

            # Reconstruct the waveform, accumulation-blending multi-band layers together
            wave = (
                spectrogram_to_wave(
                    spec_s, 
                    bp["hl"], 
                    mp, 
                    d, 
                    is_v51_model
                )
            ) if bands_n == 1 else (
                np.add(
                    wave, 
                    spectrogram_to_wave(
                        spec_s, 
                        bp["hl"], 
                        mp, 
                        d, 
                        is_v51_model
                    )
                )
            )
        else:
            sr = mp.param["band"][str(d + 1)]["sr"]

            if d == 1: 
                if is_v51_model: 
                    spec_s *= get_lp_filter_mask(
                        spec_s.shape[1], 
                        bp["lpf_start"], 
                        bp["lpf_stop"]
                    )
                else: 
                    spec_s = fft_lp_filter(
                        spec_s, 
                        bp["lpf_start"], 
                        bp["lpf_stop"]
                    )

                try:
                    # Execute high-quality resampling matching subsequent band processing sample rates
                    wave = librosa.resample(
                        spectrogram_to_wave(
                            spec_s, 
                            bp["hl"], 
                            mp, 
                            d, 
                            is_v51_model
                        ), 
                        orig_sr=bp["sr"], 
                        target_sr=sr, 
                        res_type="soxr_vhq"
                    )
                except ValueError as e:
                    logger.error(f"{translations['resample_error']}: {e}")
                    logger.error(f"{translations['shapes']} Spec_s: {spec_s.shape}, SR: {sr}, {translations['wav_resolution']}: {wav_resolution}")
            else:  
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(
                        spec_s.shape[1], 
                        bp["hpf_start"], 
                        bp["hpf_stop"] - 1
                    )

                    spec_s *= get_lp_filter_mask(
                        spec_s.shape[1], 
                        bp["lpf_start"], 
                        bp["lpf_stop"]
                    )
                else:
                    spec_s = fft_hp_filter(
                        spec_s, 
                        bp["hpf_start"], 
                        bp["hpf_stop"] - 1
                    )

                    spec_s = fft_lp_filter(
                        spec_s, 
                        bp["lpf_start"], 
                        bp["lpf_stop"]
                    )

                try:
                    wave = librosa.resample(
                        np.add(wave, spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)), 
                        orig_sr=bp["sr"], 
                        target_sr=sr, 
                        res_type="soxr_vhq"
                    )
                except ValueError as e:
                    logger.error(f"{translations['resample_error']}: {e}")
                    logger.error(f"{translations['shapes']} Spec_s: {spec_s.shape}, SR: {sr}, {translations['wav_resolution']}: {wav_resolution}")

    return wave

def get_lp_filter_mask(n_bins, bin_start, bin_stop):
    """
    Generates a low-pass linear-slope frequency masking filter array.
    """

    return np.concatenate([
        np.ones((bin_start - 1, 1)), 
        np.linspace(1, 0, bin_stop - bin_start + 1)[:, None], 
        np.zeros((n_bins - bin_stop, 1))
    ], axis=0)

def get_hp_filter_mask(n_bins, bin_start, bin_stop):
    """
    Generates a high-pass linear-slope frequency masking filter array.
    """

    return np.concatenate([
        np.zeros((bin_stop + 1, 1)), 
        np.linspace(0, 1, 1 + bin_start - bin_stop)[:, None], 
        np.ones((n_bins - bin_start - 2, 1))
    ], axis=0)

def fft_lp_filter(spec, bin_start, bin_stop):
    """
    Applies an in-place low-pass attenuation filter across designated spectrogram frequency ranges.
    """

    g = 1.0

    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, bin_stop:, :] *= 0
    return spec

def fft_hp_filter(spec, bin_start, bin_stop):
    """
    Applies an in-place high-pass attenuation filter across designated spectrogram frequency ranges.
    """

    g = 1.0

    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0 : bin_stop + 1, :] *= 0
    return spec

def mirroring(a, spec_m, input_high_end, mp):
    """
    Reconstructs missing ultra-high frequencies by mirroring lower frequency bands 
    to prevent low-pass muffled sound artifacts.

    Args:
        a (str): Mirroring algorithm style keyword strategy selector ('mirroring' or 'mirroring2').
        spec_m (np.ndarray): Assembled master full-band spectrogram matrix.
        input_high_end (np.ndarray): Target high-frequency spectrogram matrix layer to reconstruct.
        mp (object): Model parameter storage abstraction metadata configurations.

    Returns:
        np.ndarray: Spectrogram matrix block augmented with mirrored high-end frequency components.
    """

    if "mirroring" == a:
        mirror = np.flip(
            np.abs(spec_m[:, mp.param["pre_filter_start"] - 10 - input_high_end.shape[1] : mp.param["pre_filter_start"] - 10, :]), 1
        ) * np.exp(1.0j * np.angle(input_high_end))

        return np.where(np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror)

    if "mirroring2" == a:
        mi = np.multiply(
            np.flip(
                np.abs(spec_m[:, mp.param["pre_filter_start"] - 10 - input_high_end.shape[1] : mp.param["pre_filter_start"] - 10, :]), 1
            ), input_high_end * 1.7
        )

        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)

def adjust_aggr(mask, is_non_accom_stem, aggressiveness):
    """
    Adjusts the separation mask dynamically according to non-linear model isolation aggressiveness constraints.

    Args:
        mask (np.ndarray): Target mask tracking matrix array.
        is_non_accom_stem (bool): Identifies if processing targets vocal/isolated foreground stems.
        aggressiveness (dict): Parameter configuration properties dictating amplification factor thresholds.

    Returns:
        np.ndarray: Modified power-scaled separation mask array.
    """

    aggr = aggressiveness["value"] * 2

    if aggr != 0:
        if is_non_accom_stem: aggr = 1 - aggr
        if np.any(aggr > 10) or np.any(aggr < -10): logger.warning(f"{translations['warnings']}: {aggr}")

        aggr = [aggr, aggr]

        if aggressiveness["aggr_correction"] is not None:
            aggr[0] += aggressiveness["aggr_correction"]["left"]
            aggr[1] += aggressiveness["aggr_correction"]["right"]

        # Apply exponential mathematical warping across defined frequency cutoff bins
        for ch in range(2):
            mask[ch, : aggressiveness["split_bin"]] = np.power(mask[ch, : aggressiveness["split_bin"]], 1 + aggr[ch] / 3)
            mask[ch, aggressiveness["split_bin"] :] = np.power(mask[ch, aggressiveness["split_bin"] :], 1 + aggr[ch])

    return mask

def spectrogram_to_wave_no_mp(spec, n_fft=2048, hop_length=1024):
    """
    Standard fallback function executing standalone Inverse STFT wave generation without model parameter maps.
    """

    wave = librosa.istft(spec, n_fft=n_fft, hop_length=hop_length)
    if wave.ndim == 1: wave = np.asfortranarray([wave, wave])

    return wave

def wave_to_spectrogram_no_mp(wave):
    """
    Standard fallback function executing standalone Forward STFT tracking without model parameter maps.
    """

    spec = librosa.stft(wave, n_fft=2048, hop_length=1024)

    if spec.ndim == 1: spec = np.asfortranarray([spec, spec])
    return spec

def invert_audio(specs, invert_p=True):
    """
    Computes a phase-aware spectral subtraction mask between a mix and a target stem to isolate the remaining tracks.

    Args:
        specs (list): A list containing exactly two elements: `[mixture_spectrogram, isolated_stem_spectrogram]`.
        invert_p (bool, optional): Activates precise phase-aware matrix inversion math. Defaults to True.

    Returns:
        np.ndarray: Complex spectrogram matrix representing the isolated, inverted signal.
    """

    ln = min([specs[0].shape[2], specs[1].shape[2]])
    specs[0] = specs[0][:, :, :ln]
    specs[1] = specs[1][:, :, :ln]

    if invert_p:
        X_mag, y_mag = np.abs(specs[0]), np.abs(specs[1])
        # Execute absolute destructive phase negation cancellation equations
        v_spec = specs[1] - np.where(X_mag >= y_mag, X_mag, y_mag) * np.exp(1.0j * np.angle(specs[0]))
    else:
        specs[1] = reduce_vocal_aggressively(specs[0], specs[1], 0.2)
        v_spec = specs[0] - specs[1]

    return v_spec

def invert_stem(mixture, stem):
    """
    Performs standard time-domain phase inversion subtraction to derive an operational secondary track.

    Args:
        mixture (np.ndarray): Original raw master mixed audio wave track.
        stem (np.ndarray): Isolated primary target audio wave track (e.g., vocal stem).

    Returns:
        np.ndarray: Reconstructed secondary audio wave track matrix (e.g., background instrumental track).
    """

    return -spectrogram_to_wave_no_mp(invert_audio([wave_to_spectrogram_no_mp(mixture), wave_to_spectrogram_no_mp(stem)])).T