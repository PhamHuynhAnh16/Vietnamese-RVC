import os
import re
import sys
import math
import torch
import parselmouth

import numba as nb
import numpy as np

from scipy.signal import medfilt
from librosa import yin, pyin, piptrack

sys.path.append(os.getcwd())

from main.library.utils import circular_write
from main.library.predictors.CREPE.filter import mean, median
from main.library.predictors.WORLD.SWIPE import swipe, stonemask
from main.app.variables import config, configs, logger, translations

NOTE = [
    49.00, # G1
    51.91, # G#1 / Ab1
    55.00, # A1
    58.27, # A#1 / Bb1
    61.74, # B1
    65.41, # C2
    69.30, # C#2 / Db2
    73.42, # D2
    77.78, # D#2 / Eb2
    82.41, # E2
    87.31, # F2
    92.50, # F#2 / Gb2
    98.00, # G2
    103.83, # G#2 / Ab2
    110.00, # A2
    116.54, # A#2 / Bb2
    123.47, # B2
    130.81, # C3
    138.59, # C#3 / Db3
    146.83, # D3
    155.56, # D#3 / Eb3
    164.81, # E3
    174.61, # F3
    185.00, # F#3 / Gb3
    196.00, # G3
    207.65, # G#3 / Ab3
    220.00, # A3
    233.08, # A#3 / Bb3
    246.94, # B3
    261.63, # C4
    277.18, # C#4 / Db4
    293.66, # D4
    311.13, # D#4 / Eb4
    329.63, # E4
    349.23, # F4
    369.99, # F#4 / Gb4
    392.00, # G4
    415.30, # G#4 / Ab4
    440.00, # A4
    466.16, # A#4 / Bb4
    493.88, # B4
    523.25, # C5
    554.37, # C#5 / Db5
    587.33, # D5
    622.25, # D#5 / Eb5
    659.25, # E5
    698.46, # F5
    739.99, # F#5 / Gb5
    783.99, # G5
    830.61, # G#5 / Ab5
    880.00, # A5
    932.33, # A#5 / Bb5
    987.77, # B5
    1046.50, # C6
]

def medfilts(x, kernel_size = 3):
    """
    Applies a 1D median filter to the input signal.
    Supports both NumPy arrays and PyTorch tensors (including specialized OpenCL/PrivateUse1 hardware).

    Args:
        x: The input 1D signal (tensor or array).
        kernel_size: The size of the window. Must be an odd integer.

    Returns:
        The filtered signal matching the input type and device.

    Raises:
        ValueError: If kernel_size is even or x is not 1D.
    """

    # Validation checks
    if kernel_size % 2 == 0: raise ValueError("Kernel size must be an odd number.")
    if x.ndim != 1: raise ValueError("Input signal must be 1-dimensional.")

    if torch.is_tensor(x):
        # Fallback for specific hardware backends that lack native unfold/median implementations
        if x.device.type.startswith(("ocl", "privateuseone")): 
            # Convert to CPU numpy, run standard medfilt, convert back to target device
            return torch.from_numpy(medfilt(x.float().cpu().numpy(), kernel_size)).float().to(x.device)

        # Native PyTorch implementation using tensor sliding window (unfold)
        # Pad symmetrically on both sides with constant zeros
        return torch.nn.functional.pad(
            x.unsqueeze(0).unsqueeze(0), 
            (kernel_size // 2, kernel_size // 2), 
            mode="constant", 
            value=0.0
        ).squeeze(0).squeeze(0).unfold(0, kernel_size, 1).median(dim=-1).values # Extract sliding windows, find the median along the last dimension, and return the values
    else:
        # Standard SciPy median filter for NumPy arrays
        return medfilt(x, kernel_size)

def autotune_f0(note_dict, f0, f0_autotune_strength):
    """
    Snaps the pitch (F0) to the nearest musical notes in the dictionary based on tuning strength.

    Args:
        note_dict: List or array containing target musical note frequencies.
        f0: The original fundamental frequency sequence.
        f0_autotune_strength: Blend factor (0.0 = no change, 1.0 = fully snapped to note).

    Returns:
        The autotuned pitch array/tensor.
    """

    if torch.is_tensor(f0):
        # Convert note dictionary to matching tensor
        notes = torch.as_tensor(note_dict, dtype=f0.dtype, device=f0.device)
        # Calculate pairwise distance between each f0 point and all target notes, pick the index of the closest note
        nearest = notes[torch.cdist(f0[:, None], notes[:, None]).argmin(dim=1)]
    else:
        # NumPy counterpart utilizing absolute distance mapping
        notes = np.asarray(note_dict, dtype=f0.dtype)
        nearest = notes[np.abs(f0[:, None] - notes).argmin(axis=1)]

    # Linearly interpolate between raw pitch and nearest note frequencies using the strength factor
    return f0 + (nearest - f0) * f0_autotune_strength

def extract_median_f0(f0):
    """
    Calculates the global median frequency of an F0 sequence, excluding unvoiced frames (0 Hz).
    Unvoiced areas are filled in using linear interpolation before taking the median.

    Args:
        f0: The fundamental frequency sequence.

    Returns:
        The median pitch value as a float.
    """

    # Replace unvoiced 0 frames with NaN to separate them from valid pitch values
    f0 = np.where(f0 == 0, np.nan, f0)

    return float(
        np.median(
            # Linearly interpolate missing NaN boundaries to keep sequence continuous
            np.interp(
                np.arange(len(f0)), 
                np.where(~np.isnan(f0))[0], # Target indices containing true numerical values
                f0[~np.isnan(f0)] # True numerical values
            )
        )
    )

def proposal_f0_up_key(f0, target_f0 = 155.0, limit = 12):
    """
    Proposes a semi-tone shift key to align the median of the input F0 to a target pitch.

    Args:
        f0: Input pitch sequence.
        target_f0: Desired target baseline frequency (default 155.0 Hz).
        limit: Max semi-tone ceiling and floor boundaries allowed for shifting.

    Returns:
        The computed integer value representing the pitch key change.
    """

    if torch.is_tensor(f0): f0 = f0.cpu().numpy()

    try:
        # Clamp shift amount to within range [-limit, limit]
        return max(
            -limit, 
            min(
                limit, int(np.round(12 * np.log2(target_f0 / extract_median_f0(f0)))) # Calculate required semi-tone offset: 12 * log2(f2 / f1)
            )
        )
    except ValueError:
        # Fallback if log calculation encounters math exceptions or all values are unvoiced
        return 0

@nb.jit(nopython=True)
def post_process(
    f0, 
    f0_up_key, 
    manual_x_pad, 
    f0_mel_min, 
    f0_mel_max, 
    manual_f0 = None
):
    """
    Processes F0 arrays by applying key shifts, overrides, and quantizing to standard Mel scales.
    Optimized via Numba JIT compilation.

    Args:
        f0: Original continuous pitch sequence array.
        f0_up_key: Number of semi-tones to shift pitch up/down.
        manual_x_pad: Padding configuration adjustment timestamp value.
        f0_mel_min: Mel range boundary floor.
        f0_mel_max: Mel range boundary ceiling.
        manual_f0: Optional 2D array of manually specified override pitch frames [[frame_idx, hz], ...].

    Returns:
        A tuple containing:
            - np.ndarray: Quantized coarse Mel pitch array mapped between integers [1, 255].
            - np.ndarray: Adjusted continuous real frequency F0 sequence.
    """

    # Scale pitch sequence based on semi-tone shift factor
    f0 *= pow(2, f0_up_key / 12)

    if manual_f0 is not None: # Apply manual manual override boundaries if provided
        start_frame = int(manual_x_pad * 100)
        # Interpolate custom values across override window length
        replace_f0 = np.interp(
            np.linspace(
                manual_f0[:, 0][0], 
                manual_f0[:, 0][-1], 
                len(manual_f0[:, 0])
            ), 
            manual_f0[:, 0], 
            manual_f0[:, 1]
        )

        end_frame = start_frame + len(replace_f0)
        # Prevent manual sequence from extending past original array limits
        if end_frame > len(f0):
            end_frame = len(f0)
            replace_f0 = replace_f0[:end_frame - start_frame]

        # Splice manual values over specified target segment window
        f0[start_frame:end_frame] = replace_f0

    # Convert Hz frequencies into logarithmic Mel scale representation
    f0_mel = 1127 * np.log(1 + f0 / 700)
    # Standardize scale distribution mapping range onto token index values between 1 and 255
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255

    return np.rint(f0_mel).astype(np.int32), f0

def realtime_post_process(
    f0, 
    pitch, 
    pitchf, 
    f0_up_key = 0, 
    f0_min = 50.0, 
    f0_max = 1100.0
):
    """
    Applies pitch shifts and Mel mapping optimized for streaming or real-time pipelines.

    Args:
        f0: Incoming real-time pitch tensor frame block.
        pitch: Optional tracking target buffer destination for coarse tokens.
        pitchf: Optional tracking target buffer destination for continuous floats.
        f0_up_key: Number of semi-tones to shift pitch.
        f0_min: Minimum tracking boundary cutoff (Hz).
        f0_max: Maximum tracking boundary cutoff (Hz).

    Returns:
        A tuple of tensors containing batched unsqueezed pitch outputs: (pitch, pitchf).
    """

    # Shift pitch based on semi-tone factor
    f0 *= 2 ** (f0_up_key / 12)

    # Convert frequency to Mel scale tensor layout
    f0_mel = 1127.0 * (1.0 + f0 / 700.0).log()
    # Normalize, range-bound [1, 255] out-of-place or in-place safely using torch clip
    f0_mel = torch.clip((f0_mel - f0_min) * 254 / (f0_max - f0_min) + 1, 1, 255, out=f0_mel)
    f0_coarse = torch.round(f0_mel, out=f0_mel).long()

    # If stream pipeline tracking targets are assigned, pipe frame block chunks to circular storage hooks
    if pitch is not None and pitchf is not None:
        # Slice padding artifacts away if chunk length is larger than base boundary sizing threshold
        f0_coarse = f0_coarse[3:-1] if f0_coarse.shape[0] > 4 else f0_coarse
        f0 = f0[3:-1] if f0.shape[0] > 4 else f0

        circular_write(f0_coarse, pitch)
        circular_write(f0, pitchf)
    else:
        # Fallback allocation targets if no stream buffer contexts were provided
        pitch = f0_coarse
        pitchf = f0

    return pitch.unsqueeze(0), pitchf.unsqueeze(0)

class Generator:
    """Pitch conversion controller managing models, feature estimation routing, and inference hooks."""

    def __init__(
        self, 
        sample_rate = 16000, 
        hop_length = 160, 
        f0_min = 50, 
        f0_max = 1100, 
        alpha = 0.5, 
        is_half = False, 
        device = "cpu", 
        predictor_onnx = False, 
        return_tensor = False
    ):
        """Initializes configuration properties, caching structures, and model prediction backends."""

        self.alpha = alpha
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        # Pre-calculate constant baseline Mel boundaries
        self.f0_mel_min = 1127 * math.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * math.log(1 + self.f0_max / 700)

        self.window = 160
        self.batch_size = 512
        self.ref_freqs = NOTE
        self.frame_period = self.window / self.sample_rate * 1000
        self.time_step = self.frame_period / 1000

        self.device = device
        self.is_half = is_half
        self.providers = config.providers # Expected globally scoped configuration access
        # Lazily loaded predictors initialized dynamically on-demand during pipeline calls
        self.pw = None
        self.fcpe = None
        self.djcm = None
        self.penn = None
        self.pesto = None
        self.swift = None
        self.rmvpe = None
        self.crepe = None
        self.mangio_penn = None
        self.mangio_crepe = None
        # Method cache trackers to optimize parsing lookups
        self.cache_args = None
        self.cache_flags = None
        self.cache_f0_method = None

        self.int8 = config.int8
        self.return_tensor = return_tensor
        self.compile_model = config.compile_all
        self.compile_mode = config.compile_mode
        self.predictor_onnx = predictor_onnx or config.int8
        # Dynamic strategy selection based on operational mode configurations
        self.onnx_mode = '.onnx' if self.predictor_onnx else '.pt'
        self.resize = configs.get("turn_on_resize_f0_for_all", False)
        self.int8_mode = "-int8" if config.int8 and self.predictor_onnx else ""
        self.resize_f0 = self._resize_tensor_f0 if self.return_tensor else self._resize_array_f0
        self.resize_dtype = torch.float64 if self.device.startswith(("cpu", "cuda")) else torch.float32
        self.quantile_audio = self._easy_quantile_audio if self.device.startswith(("ocl", "privateuseone")) else self._quantile_audio

    def calculator(
        self, 
        x_pad, 
        f0_method, 
        x, 
        f0_up_key = 0, 
        p_len = None, 
        filter_radius = 3, 
        f0_autotune = False, 
        f0_autotune_strength = 1, 
        manual_f0 = None, 
        proposal_pitch = False, 
        proposal_pitch_threshold = 255.0
    ):
        """
        Calculates F0 track profiles from standard offline audio vectors.

        Handles model routing, automatic shift recommendations, key modification, and quantization formats.
        """

        if p_len is None: p_len = x.shape[0] // self.window
        # Route processing through standard routines or hybrid combinations depending on naming strings
        compute_fn = self.get_f0_hybrid if "hybrid" in f0_method else self.compute_f0

        # Extract pitch values (enforces odd radius value limits for filters)
        f0 = compute_fn(
            f0_method, 
            x, 
            p_len, 
            filter_radius if filter_radius % 2 != 0 else filter_radius + 1
        )
        
        # Automatically recommend key signature transformations if requested
        if proposal_pitch: 
            up_key = proposal_f0_up_key(
                f0, 
                proposal_pitch_threshold, 
                configs["limit_f0"]
            )

            logger.debug(translations["proposal_f0"].format(up_key=up_key))
            f0_up_key += up_key

        # Quantize results to strict discrete musical notes via auto-tuning algorithms
        if f0_autotune: 
            logger.debug(translations["startautotune"])

            f0 = autotune_f0(
                self.ref_freqs, 
                f0, 
                f0_autotune_strength
            )

        return post_process(f0, f0_up_key, x_pad, self.f0_mel_min, self.f0_mel_max, manual_f0)

    def realtime_calculator(
        self, 
        audio, 
        f0_method, 
        pitch, 
        pitchf, 
        f0_up_key = 0, 
        filter_radius = 3, 
        f0_autotune = False, 
        f0_autotune_strength = 1, 
        proposal_pitch = False, 
        proposal_pitch_threshold = 255.0
    ):
        """Calculates and streams F0 frames optimized for low-latency pipelines."""

        p_len = audio.shape[0] // self.window

        f0 = self.compute_f0(
            f0_method,
            audio,
            p_len,
            filter_radius if filter_radius % 2 != 0 else filter_radius + 1
        )

        if f0_autotune: 
            f0 = autotune_f0(
                self.ref_freqs, 
                f0, 
                f0_autotune_strength
            )

        if proposal_pitch: 
            up_key = proposal_f0_up_key(
                f0, 
                proposal_pitch_threshold, 
                configs["limit_f0"]
            )

            f0_up_key += up_key

        return realtime_post_process(f0, pitch, pitchf, f0_up_key, self.f0_min, self.f0_max)

    def _resize_array_f0(self, x, target_len, resize = False):
        """Resizes standard continuous numpy tracking vectors using linear data interpolation."""

        if torch.is_tensor(x): x = x.cpu().numpy().astype(np.float32)
        if not resize or len(x) == target_len: return x

        source = np.array(x)
        source[source < 0.001] = np.nan # Isolate unvoiced parts to avoid interpolation artifacts

        return np.nan_to_num(
            np.interp(
                np.arange(0, len(source) * target_len, len(source)) / target_len, 
                np.arange(0, len(source)), 
                source
            )
        )

    def _resize_tensor_f0(self, x, target_len, resize = False):
        """Resizes GPU tracking data directly inside PyTorch memory maps via linear mapping logic."""

        if not torch.is_tensor(x): x = torch.from_numpy(x).to(self.device)
        if not resize or len(x) == target_len: return x

        source = torch.as_tensor(x, dtype=self.resize_dtype, device=self.device).clone()
        source[source < 0.001] = torch.nan
        n = source.shape[0]

        # Calculate exact linear sample positions across targeted frame indices
        xp = torch.arange(target_len, dtype=self.resize_dtype, device=self.device) * (n / target_len)
        x0 = xp.floor().long()

        y0 = source[x0]
        y1 = source[(x0 + 1).clamp(max=n - 1)]
        w = xp - x0.to(self.resize_dtype) # Distance delta weights

        # Perform manual interpolation calculation step
        return torch.nan_to_num(
            torch.where(
                w == 0, 
                y0, 
                y0 + (y1 - y0) * w
            )
        ) 
    
    def _quantile_audio(self, audio):
        """Normalizes audio dynamics by dividing amplitudes by the 99.9th percentile value."""

        x = audio.float()
        x /= x.abs().quantile(0.999)
        return x.unsqueeze(dim=0)

    def _easy_quantile_audio(self, audio):
        """Normalizes audio dynamics by dividing amplitudes by the 99.9th percentile value."""
    
        x = audio.float()
        y = x.abs().flatten()

        if y.numel() == 0: raise ValueError("Input tensor must contain at least one element.")

        values = y.sort().values
        x /= values[min(int(0.999 * (values.numel() - 1)), values.numel() - 1)] + 1e-8

        return x.unsqueeze(dim=0)
    
    def compute_f0(self, f0_method, x, p_len, filter_radius):
        """
        Routes processing flags to designated pitch estimator components.

        Caches string configurations to bypass string processing overhead on subsequent iterations.
        """

        # Parse and cache string settings upon format transitions
        if f0_method != self.cache_f0_method:
            self.cache_f0_method = f0_method
            self.cache_args = f0_method.split("-")
            self.cache_flags = set(self.cache_args)

        # Execution Model Routing tree
        if "pm" in self.cache_flags:
            f0 = self.get_f0_pm(
                x, 
                p_len, 
                filter_radius=filter_radius, 
                mode=self.cache_args[1]
            )
        elif self.cache_args[0] in {"harvest", "dio"}:
            f0 = self.get_f0_pyworld(
                x, 
                p_len, 
                filter_radius, 
                self.cache_args[0], 
                use_stonemask="stonemask" in self.cache_flags
            )
        elif "crepe" in self.cache_flags:
            f0 = (
                self.get_f0_mangio_crepe(
                    x, 
                    p_len, 
                    self.cache_args[2]
                )
            ) if "mangio" in self.cache_flags else (
                self.get_f0_crepe(
                    x, 
                    p_len, 
                    self.cache_args[1], 
                    filter_radius=filter_radius
                )
            )
        elif "fcpe" in self.cache_flags:
            f0 = self.get_f0_fcpe(
                x, 
                p_len, 
                legacy="legacy" in self.cache_flags and "previous" not in self.cache_flags, 
                previous="previous" in self.cache_flags, 
                filter_radius=filter_radius
            )
        elif "rmvpe" in self.cache_flags:
            f0 = self.get_f0_rmvpe(
                x, 
                p_len, 
                clipping="clipping" in self.cache_flags, 
                filter_radius=filter_radius, 
                hpa="hpa" in self.cache_flags,
                previous="previous" in self.cache_flags,
                mix="mix" in self.cache_flags,
                v4="v4" in self.cache_flags
            )
        elif self.cache_args[0] in {"yin", "pyin", "piptrack"}:
            f0 = self.get_f0_librosa(
                x, 
                p_len, 
                mode=self.cache_args[0],
                filter_radius=filter_radius
            )
        elif "swipe" in self.cache_flags:
            f0 = self.get_f0_swipe(
                x, 
                p_len, 
                filter_radius=filter_radius, 
                use_stonemask="stonemask" in self.cache_flags
            )
        elif "penn" in self.cache_flags:
            f0 = (
                self.get_f0_mangio_penn(
                    x, 
                    p_len
                )
            ) if self.cache_args[0] == "mangio" else (
                self.get_f0_penn(
                    x, 
                    p_len, 
                    filter_radius=filter_radius
                )
            )
        elif "djcm" in self.cache_flags:
            f0 = self.get_f0_djcm(
                x, 
                p_len, 
                clipping="clipping" in self.cache_flags, 
                svs="svs" in self.cache_flags, 
                filter_radius=filter_radius
            )
        elif "pesto" in self.cache_flags:
            f0 = self.get_f0_pesto(
                x, 
                p_len
            )
        elif "swift" in self.cache_flags:
            f0 = self.get_f0_swift(
                x, 
                p_len, 
                filter_radius=filter_radius
            )
        else:
            raise ValueError(translations["option_not_valid"])
        
        if isinstance(f0, tuple): f0 = f0[0]
        if "medfilt" in self.cache_flags or "svs" in self.cache_flags: f0 = medfilts(f0, kernel_size=5)

        return f0
    
    def get_f0_hybrid(self, methods_str, x, p_len, filter_radius):
        """Combines multiple pitch estimation methods by computing a weighted geometric mean of their results."""

        logger.debug(translations["hybrid_calc"].format(f0_method=methods_str))
        # Parse inside bracket strings: hybrid[rmvpe+fcpe] -> ['rmvpe', 'fcpe']
        methods_str = re.search(r"hybrid\[(.+)\]", methods_str)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        n = len(methods)
        f0_stack = []

        for method in methods: # Gather estimation tracks from all designated sub-algorithms
            f0_stack.append(
                self.resize_f0(
                    self.compute_f0(
                        method, 
                        x, 
                        p_len, 
                        filter_radius
                    ),
                    p_len,
                    True
                )
            )
        
        f0_mix = np.zeros(p_len)

        if not f0_stack: return f0_mix
        if len(f0_stack) == 1: return f0_stack[0]

        # Calculate geometric weight allocations influenced by interpolation parameters (alpha)
        weights = (1 - np.abs(np.arange(n) / (n - 1) - (1 - self.alpha))) ** 2
        weights /= weights.sum()

        stacked = np.vstack(f0_stack)
        voiced_mask = np.any(stacked > 0, axis=0) # Process active tracking regions only

        # Compute the weighted geometric mean in log-space to ensure smooth transitions
        f0_mix[voiced_mask] = np.exp(
            np.nansum(
                np.log(stacked + 1e-6) * weights[:, None], axis=0
            )[voiced_mask]
        )

        return f0_mix

    def get_f0_pm(self, x, p_len, filter_radius=3, mode="ac"):
        """Extracts pitch tracks utilizing Praat Parselmouth estimators (AC, CC, or SHS algorithms)."""

        if torch.is_tensor(x): x = x.cpu().numpy()

        pm = parselmouth.Sound(x, self.sample_rate)
        # Map functional calls to corresponding target backends
        pm_fn = {
            "ac": pm.to_pitch_ac, 
            "cc": pm.to_pitch_cc, 
            "shs": pm.to_pitch_shs
        }.get(mode, pm.to_pitch_ac)

        # Execute estimation with specific target property arguments
        pitch = (
            pm_fn(
                time_step=self.time_step, 
                voicing_threshold=filter_radius / 10 * 2, 
                pitch_floor=self.f0_min, 
                pitch_ceiling=self.f0_max
            )
        ) if mode != "shs" else (
            pm_fn(
                time_step=self.time_step,
                minimum_pitch=self.f0_min,
                maximum_frequency_component=self.f0_max
            )
        )

        f0 = pitch.selected_array["frequency"]
        # Center-pad or truncate arrays to maintain alignment matching the expected sequence length
        pad_size = (p_len - len(f0) + 1) // 2

        if pad_size > 0 or p_len - len(f0) - pad_size > 0:  f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return self.resize_f0(f0, p_len, self.resize)
    
    def get_f0_mangio_crepe(self, x, p_len, model="full"):
        """Runs inference via Mangio-CREPE neural network architectures."""

        if self.mangio_crepe is None:
            from main.library.predictors.CREPE.CREPE import CREPE

            self.mangio_crepe = CREPE(
                os.path.join(
                    configs["predictors_path"], 
                    f"crepe_{model}{self.int8_mode}{self.onnx_mode}" + ("" if self.predictor_onnx else "h")
                ), 
                model_size=model, 
                hop_length=self.hop_length, 
                batch_size=self.hop_length * 2, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                device=self.device, 
                sample_rate=self.sample_rate, 
                providers=self.providers, 
                onnx=self.predictor_onnx, 
                is_half=False, # Don't use Is Half; it's terrible.
                return_periodicity=False,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode
            )
        
        if not torch.is_tensor(x): x = torch.from_numpy(x).to(device=self.device)
        audio = self.quantile_audio(x)

        # Average multi-channel audio tracks down to 1D mono signals
        if audio.ndim == 2 and audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True).detach()
        f0 = self.mangio_crepe.compute_f0(audio.detach(), pad=True).squeeze(0)

        return self.resize_f0(f0, p_len, True)
    
    def get_f0_crepe(self, x, p_len, model="full", filter_radius=3):
        """Runs estimation via standard CREPE model distributions incorporating confidence masks."""

        if self.crepe is None:
            from main.library.predictors.CREPE.CREPE import CREPE

            self.crepe = CREPE(
                os.path.join(
                    configs["predictors_path"], 
                    f"crepe_{model}{self.int8_mode}{self.onnx_mode}" + ("" if self.predictor_onnx else "h")
                ), 
                model_size=model, 
                hop_length=self.window, 
                batch_size=self.batch_size, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                device=self.device, 
                sample_rate=self.sample_rate, 
                providers=self.providers, 
                onnx=self.predictor_onnx, 
                is_half=False, # Don't use Is Half; it's terrible.
                return_periodicity=True,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode
            )
        
        if not torch.is_tensor(x): x = torch.from_numpy(x).to(device=self.device)

        f0, pd = self.crepe.compute_f0(x.float().unsqueeze(0), pad=True)
        # Smooth pitch tracking curves using mean and median filter runs
        f0, pd = mean(f0, filter_radius), median(pd, filter_radius)
        # Clear noise artifacts by zeroing out tracking frames with low periodicity confidence
        f0[pd < 0.1] = 0

        return self.resize_f0(f0[0], p_len, self.resize)
    
    def get_f0_fcpe(self, x, p_len, legacy=False, previous=False, filter_radius=3):
        """Extracts tracking pitch via Fast Context-aware Pitch Estimation (FCPE) pipelines."""

        if self.fcpe is None:
            from main.library.predictors.FCPE.FCPE import FCPE

            self.fcpe = FCPE(
                os.path.join(
                    configs["predictors_path"], 
                    ( # Pick target model checkpoint names based on version options
                        "fcpe_legacy" 
                        if legacy else 
                        ("fcpe" if previous else "ddsp_200k")
                    ) + self.int8_mode + self.onnx_mode
                ),  
                device=self.device, 
                threshold=(filter_radius / 100) if legacy else (filter_radius / 1000 * 2), 
                providers=self.providers, 
                onnx=self.predictor_onnx, 
                legacy=legacy,
                is_half=self.is_half,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode
            )
        
        f0 = self.fcpe.compute_f0(x)
        return self.resize_f0(f0, p_len, self.resize)
    
    def get_f0_rmvpe(self, x, p_len, clipping=False, filter_radius=3, hpa=False, previous=False, mix=False, v4=False):
        """Extracts pitch sequences via Robust Minimum Variance Pitch Estimation (RMVPE) networks."""

        if self.rmvpe is None:
            from main.library.predictors.RMVPE.RMVPE import RMVPE

            self.rmvpe = RMVPE(
                os.path.join(
                    configs["predictors_path"], 
                    ( # Multi-tier selection logic determining model checkpoint targeting names
                        (
                            "hpa-v4" if v4 else (
                                "hpa-rmvpe-76000"
                                if previous else
                                "hpa-rmvpe-112000"
                            )
                        ) if hpa else (
                            "rmvpe-mix"
                            if mix else
                            "rmvpe"
                        )
                    ) + self.int8_mode + self.onnx_mode
                ), 
                is_half=self.is_half, 
                device=self.device, 
                onnx=self.predictor_onnx, 
                providers=self.providers,
                hpa=hpa,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode,
                enable_chunk=not self.return_tensor and (configs.get("enable_rmvpe_chunk", False) or self.device.startswith(("privateuseone", "ocl"))),
                chunk_size=configs.get("rmvpe_chunk_size", 8000),
                return_tensor=self.return_tensor,
                f0_min=self.f0_min, 
                f0_max=self.f0_max
            )
            # Route method call hooks to custom alternative routines if clipping flags are set
            if clipping: self.rmvpe.infer_from_audio = self.rmvpe.infer_from_audio_with_pitch

        f0 = self.rmvpe.infer_from_audio(x, thred=filter_radius / 100)
        return self.resize_f0(f0, p_len, self.resize)
    
    def get_f0_pyworld(self, x, p_len, filter_radius, model="harvest", use_stonemask=True):
        """Extracts pitch using PyWorld vocoder analysis frameworks (Harvest or Dio models)."""

        if self.pw is None:
            from main.library.predictors.WORLD.WORLD import PYWORLD

            self.pw = PYWORLD(
                os.path.join(configs["predictors_path"], "world"), 
                os.path.join(configs["binary_path"], "world.bin"),
                harvest=model == "harvest"
            )
        
        if torch.is_tensor(x): x = x.cpu().numpy()
        x = x.astype(np.double)

        f0, t = self.pw.infer(
            x, 
            fs=self.sample_rate, 
            f0_ceil=self.f0_max, 
            f0_floor=self.f0_min, 
            frame_period=self.frame_period
        )

        if use_stonemask: # Refine coarse pitch estimations using StoneMask refinements
            f0 = self.pw.stonemask(
                x, 
                self.sample_rate, 
                t, 
                f0
            )

        # Apply noise-spike filters to harvest tracks, or precision rounding rules on dio tracks
        if filter_radius > 2 and model == "harvest": f0 = medfilts(f0, filter_radius)
        elif model == "dio":
            for index, pitch in enumerate(f0):
                f0[index] = round(pitch, 1)

        return self.resize_f0(f0, p_len, self.resize)
    
    def get_f0_swipe(self, x, p_len, filter_radius=3, use_stonemask=True):
        """Extracts F0 estimates utilizing SWIPE algorithms."""

        if torch.is_tensor(x): x = x.cpu().numpy()

        f0, t = swipe(
            x.astype(np.float32), 
            self.sample_rate, 
            f0_floor=self.f0_min, 
            f0_ceil=self.f0_max, 
            frame_period=self.frame_period,
            sTHR=filter_radius / 10
        )

        if use_stonemask:
            f0 = stonemask(
                x, 
                self.sample_rate, 
                t, 
                f0
            )

        return self.resize_f0(f0, p_len, self.resize)
    
    def get_f0_librosa(self, x, p_len, mode="yin", filter_radius=3):
        """Extracts pitch using classical DSP analysis routines provided via Librosa (YIN, pYIN, or PipTrack)."""

        if torch.is_tensor(x): x = x.cpu().numpy()

        if mode != "piptrack":
            if_yin = mode == "yin"
            yin_fn = yin if if_yin else pyin

            f0 = yin_fn(
                x.astype(np.float32), 
                sr=self.sample_rate, 
                fmin=self.f0_min, 
                fmax=self.f0_max, 
                hop_length=self.hop_length
            )

            # Extracted structures from pyin contain lists of alternative items; extract index zero tracking values
            if not if_yin: f0 = f0[0]
        else:
            pitches, magnitudes = piptrack(
                y=x.astype(np.float32),
                sr=self.sample_rate,
                fmin=self.f0_min,
                fmax=self.f0_max,
                hop_length=self.hop_length,
                threshold=filter_radius / 10
            )
            # Find candidate bins with the highest energy response across frames
            f0 = pitches[np.argmax(magnitudes, axis=0), range(magnitudes.shape[1])]

        return self.resize_f0(f0, p_len, True)

    def get_f0_penn(self, x, p_len, filter_radius=3):
        """Runs pitch inference with standard PENN models."""

        if self.penn is None:
            from main.library.predictors.PENN.PENN import PENN

            self.penn = PENN(
                os.path.join(
                    configs["predictors_path"], 
                    f"fcn{self.int8_mode}{self.onnx_mode}"
                ), 
                hop_length=self.window // 2, 
                batch_size=self.batch_size, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                sample_rate=self.sample_rate,
                device=self.device, 
                providers=self.providers, 
                onnx=self.predictor_onnx, 
                is_half=self.is_half,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode
            )

        if not torch.is_tensor(x): x = torch.from_numpy(x).to(device=self.device)

        f0, pd = self.penn.compute_f0(x.float().unsqueeze(0))
        f0, pd = mean(f0, filter_radius), median(pd, filter_radius)
        f0[pd < 0.1] = 0 # Zero out frames under confidence thresholds

        f0 = medfilts(f0[0], kernel_size=5)
        return self.resize_f0(f0, p_len, False)

    def get_f0_mangio_penn(self, x, p_len):
        """Runs pitch inference using Mangio-variant optimized configurations of the PENN architecture."""

        if self.mangio_penn is None:
            from main.library.predictors.PENN.PENN import PENN

            self.mangio_penn = PENN(
                os.path.join(
                    configs["predictors_path"], 
                    f"fcn{self.int8_mode}{self.onnx_mode}"
                ), 
                hop_length=self.hop_length, 
                batch_size=self.hop_length * 2, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                sample_rate=self.sample_rate,
                device=self.device, 
                providers=self.providers, 
                onnx=self.predictor_onnx, 
                is_half=self.is_half,
                interp_unvoiced_at=0.1,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode
            )

        if not torch.is_tensor(x): x = torch.from_numpy(x).to(device=self.device)
        audio = self.quantile_audio(x)

        if audio.ndim == 2 and audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True).detach()
        f0 = self.mangio_penn.compute_f0(audio.detach()).squeeze(0)
        f0 = medfilts(f0, kernel_size=5)

        return self.resize_f0(f0, p_len, True)

    def get_f0_djcm(self, x, p_len, clipping=False, svs=False, filter_radius=3):
        """Extracts tracking sequences using Deep Joint Creak and Melody (DJCM) models."""

        if self.djcm is None:
            from main.library.predictors.DJCM.DJCM import DJCM
            
            self.djcm = DJCM(
                os.path.join(
                    configs["predictors_path"], 
                    (
                        "djcm-svs" 
                        if svs else 
                        "djcm"
                    ) + self.int8_mode + self.onnx_mode
                ), 
                is_half=self.is_half, 
                device=self.device, 
                onnx=self.predictor_onnx, 
                svs=svs,
                providers=self.providers,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode,
                return_tensor=self.return_tensor,
                f0_min=self.f0_min, 
                f0_max=self.f0_max
            )
            if clipping: self.djcm.infer_from_audio = self.djcm.infer_from_audio_with_pitch
            
        f0 = self.djcm.infer_from_audio(x, thred=filter_radius / 10)
        return self.resize_f0(f0, p_len, self.resize)
    
    def get_f0_swift(self, x, p_len, filter_radius=3):
        """Runs pitch estimation with high-efficiency SWIFT ONNX runtimes."""

        if self.swift is None:
            from main.library.predictors.SWIFT.SWIFT import SWIFT

            self.swift = SWIFT(
                os.path.join(
                    configs["predictors_path"], 
                    "swift-int8.onnx" if self.int8_mode else "swift.onnx"
                ), 
                fmin=self.f0_min, 
                fmax=self.f0_max, 
                confidence_threshold=filter_radius / 4 + 0.137,
                providers=self.providers
            )
        
        if torch.is_tensor(x): x = x.cpu().numpy()

        pitch_hz, _, _ = self.swift.detect_from_array(x)
        return self.resize_f0(pitch_hz, p_len, True)

    def get_f0_pesto(self, x, p_len):
        """Infers pitch using PESTO neural estimators."""

        if self.pesto is None:
            from main.library.predictors.PESTO.PESTO import PESTO

            self.pesto = PESTO(
                os.path.join(
                    configs["predictors_path"], 
                    f"pesto{self.int8_mode}{self.onnx_mode}"
                ), 
                step_size=self.frame_period, 
                reduction="alwa", 
                sample_rate=self.sample_rate, 
                device=self.device, 
                providers=self.providers, 
                onnx=self.predictor_onnx,
                is_half=self.is_half,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode,
                chunk_size=(30 * self.sample_rate) if not self.return_tensor else None
            )

        if not torch.is_tensor(x): x = torch.from_numpy(x).to(device=self.device)
        audio = self.quantile_audio(x)

        if audio.ndim == 2 and audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True).detach()
        f0, _ = self.pesto.compute_f0(audio.detach())

        return self.resize_f0(f0.squeeze(0), p_len, self.resize)