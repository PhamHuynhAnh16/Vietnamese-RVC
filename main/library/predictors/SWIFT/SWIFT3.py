import math

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
HOP_LENGTH = 256
FRAME_PERIOD = HOP_LENGTH / SAMPLE_RATE
FMIN = 46.875
FMAX = 2093.75
# The model scores 95 log-spaced pitch steps; a band narrower than one step holds no candidate.
BIN_RATIO = (FMAX / FMIN) ** (1 / 94)
# The model's receptive field reaches 2815 samples each way: half of the 2048 window plus seven
# frames. A frame's own hop covers 256 of those samples on the future side, so a streamed frame
# is final once 10 later frames exist, and 11 earlier frames reproduce the batch result.
LOOKAHEAD_FRAMES = 10
LEFT_FRAMES = 11
WINDOW_FRAMES = 1875

# Digital silence sits on the spectrogram's log floor in every bin, a texture the model never saw,
# and it answers with random voiced frames. Frames whose audio peaks below this get confidence 0.
SILENCE_PEAK = 1e-3

def _mono(audio):
    array = np.asarray(audio)
    dtype = array.dtype

    if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)): raise TypeError(f"audio must be a real integer or float array, got {dtype}")

    if array.ndim == 2:
        if array.shape[1] == 0: raise ValueError("audio has no channels")
        if array.shape[1] > max(array.shape[0], 32): raise ValueError(f"audio must be channels last, got shape {array.shape}")
    elif array.ndim != 1: raise ValueError("audio must be a 1-D (mono) or 2-D (channels last) array")

    integer = np.issubdtype(dtype, np.integer)
    array = array.astype(np.float64 if integer else np.float32, copy=False)

    if array.ndim == 2:
        # Adding the channel columns is far faster than a numpy reduction along a length-2 axis.
        channels = array.shape[1]
        array = array[:, 0] if channels == 1 else sum(array[:, c] for c in range(channels)) / channels

    if integer:
        # Integer audio is mapped to -1..1 as soundfile does: signed types are divided by
        # 2**(bits-1), unsigned types first have their midpoint removed.
        full_scale = 2.0 ** (np.iinfo(dtype).bits - 1)
        offset = 0.0 if np.issubdtype(dtype, np.signedinteger) else full_scale
        array = (array - offset) / full_scale

    # A copy, so the result never aliases the caller's array.
    signal = np.array(array, dtype=np.float32)

    if not np.isfinite(signal).all(): raise ValueError("audio must be finite and within the float32 range")
    return signal

def _check_number(name, value, *, allow_inf = False, nonnegative = False):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)): raise TypeError(f"{name} must be a number")
    number = float(value)

    if math.isnan(number) or (math.isinf(number) and not allow_inf) or (nonnegative and number < 0): raise ValueError(f"{name} must be a {'' if allow_inf else 'finite '}number{' >= 0' if nonnegative else ''}")
    return number

def _check_rate(sample_rate):
    rate = _check_number("sample_rate", sample_rate)

    if not rate.is_integer() or rate <= 0: raise ValueError("sample_rate must be a positive integer")
    return int(rate)

def _range(fmin, fmax):
    fmin = FMIN if fmin is None else max(FMIN, _check_number("fmin", fmin, allow_inf=True))
    fmax = FMAX if fmax is None else min(FMAX, _check_number("fmax", fmax, allow_inf=True))

    if not fmin < fmax: raise ValueError(f"require fmin < fmax within the model range {FMIN} to {FMAX} Hz, got fmin={fmin}, fmax={fmax}")
    if fmax < fmin * BIN_RATIO: raise ValueError(f"fmax must be at least {BIN_RATIO:.5f} times fmin so the band holds a pitch candidate")

    return fmin, fmax

def _timestamps(first_frame, n_frames):
    return (np.arange(n_frames) + first_frame) * FRAME_PERIOD

class SWIFT:
    """
    SWIFT - A fast and accurate fundamental frequency (F0) detector using an ONNX model.

    The model takes mono audio at 16 kHz and returns a pitch and a confidence
    for every 256-sample frame (16 ms). A frame is voiced when its confidence
    is at least 0.5. Pitches lie between 46.875 Hz and 2093.75 Hz.

    Construction takes about 20 ms and allocates the thread pool, so build one
    detector and reuse it. `detect` may be called from several threads at
    once; a `PitchStream` must be driven from one thread.

    `threads` sets the size of the ONNX Runtime thread pool; the default is
    the number of physical cores, and more than about six threads do not help
    this model. `spin=False` lets the pool sleep between calls instead of
    busy-waiting, which costs about 1 ms per call and frees the CPU
    between calls: the right setting for streaming and for shared servers.
    """

    def __init__(
        self, 
        model_path, 
        threads = None, 
        spin = True,
        providers = ["CPUExecutionProvider"]
    ):
        if threads is not None and (isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or threads < 1):
            raise ValueError("threads must be a positive integer")

        # 1. Parameterize runtime execution threads to enforce deterministic serialization benchmarks
        session_options = ort.SessionOptions()
        if threads is not None: session_options.intra_op_num_threads = threads
        if not spin: session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_options.log_severity_level = 3 # Suppress non-critical warning alerts
        if providers[0][0].startswith("Tensorrt"): session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 2. Track device context variables depending on target hardware strings
        self._device = "cuda" if providers[0][0].startswith(("Tensorrt", "CUDA")) else "cpu"
        # 3. Compile backend runtime graph engine structures
        self.session = ort.InferenceSession(model_path, session_options, providers=providers)
        self._run = self._run_io if providers[0][0].startswith(("Tensorrt", "CUDA", "CPU")) else self._run_non_io


    def _run_non_io(self, audio, fmin, fmax):
        pitch, confidence = self.session.run(
            ["pitch", "confidence"], {"audio": audio[None, :], "fmin": np.asarray(fmin, dtype=np.float32), "fmax": np.asarray(fmax, dtype=np.float32)},
        )

        pitch, confidence = np.asarray(pitch[0], dtype=np.float64), np.asarray(confidence[0], dtype=np.float64)
        n = len(confidence)

        hops = audio[: n * HOP_LENGTH].reshape(n, HOP_LENGTH) if len(audio) >= HOP_LENGTH else audio[None, :]
        confidence[np.abs(hops).max(axis=1) < SILENCE_PEAK] = 0.0
        power = np.zeros(n)

        for column in hops.T:
            power += np.square(column, dtype=np.float64)

        loudness = 20 * np.log10(np.maximum(np.sqrt((np.concatenate(([0.0], power[:-1])) + power) / 512), 1e-7))
        return pitch, confidence, loudness

    def _run_io(self, audio, fmin, fmax):
        io_binding = self.session.io_binding()
        io_binding.bind_cpu_input("audio", audio[None, :].astype(np.float32))
        io_binding.bind_cpu_input("fmin", np.asarray(fmin, dtype=np.float32))
        io_binding.bind_cpu_input("fmax", np.asarray(fmax, dtype=np.float32))
        # Pre-allocate output buffers explicitly on the target execution device contexts
        io_binding.bind_output(name="pitch", device_type=self._device)
        io_binding.bind_output(name="confidence", device_type=self._device)
        # Execute model path optimization passes
        self.session.run_with_iobinding(io_binding)

        pitch, confidence = np.asarray(io_binding.get_outputs()[0].numpy()[0], dtype=np.float64), np.asarray(io_binding.get_outputs()[1].numpy()[0], dtype=np.float64)
        n = len(confidence)

        hops = audio[: n * HOP_LENGTH].reshape(n, HOP_LENGTH) if len(audio) >= HOP_LENGTH else audio[None, :]
        confidence[np.abs(hops).max(axis=1) < SILENCE_PEAK] = 0.0
        power = np.zeros(n)

        for column in hops.T:
            power += np.square(column, dtype=np.float64)

        loudness = 20 * np.log10(np.maximum(np.sqrt((np.concatenate(([0.0], power[:-1])) + power) / 512), 1e-7))
        return pitch, confidence, loudness

    def detect(self, audio, sample_rate, fmin = None, fmax = None):
        sample_rate = _check_rate(sample_rate)
        fmin, fmax = _range(fmin, fmax)

        signal = _mono(audio)
        if signal.size == 0: raise ValueError("audio must not be empty")

        n = max(1, len(signal) // HOP_LENGTH)
        parts = []

        for start in range(0, n, WINDOW_FRAMES):
            end = min(start + WINDOW_FRAMES, n)
            left = max(0, start - LEFT_FRAMES)

            window = signal[left * HOP_LENGTH : (end + LOOKAHEAD_FRAMES) * HOP_LENGTH] if end < n else signal[left * HOP_LENGTH :]
            parts.append([values[start - left : end - left] for values in self._run(window, fmin, fmax)])

        pitch, confidence, loudness = (np.concatenate([part[i] for part in parts]) for i in range(3))
        return _timestamps(0, len(pitch)), pitch, confidence, loudness

    def _repair_subharmonics(self, pitch, confidence, frame_period=0.016):
        pitch = np.asarray(pitch, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        corrected = pitch.copy()
        repaired = np.zeros(len(pitch), dtype=bool)

        i = 1
        while i < len(pitch) - 1:
            if confidence[i - 1] < 0.5 or not 0.3 <= confidence[i] < 0.95:
                i += 1
                continue

            ratio = pitch[i - 1] / pitch[i]
            factor = min((2, 3), key=lambda value: abs(np.log2(ratio / value)))

            if abs(1200 * np.log2(ratio / factor)) > 100:
                i += 1
                continue

            j = i
            while j < len(pitch) and (j - i) * frame_period < 1.0:
                if not 0.3 <= confidence[j] < 0.95: break
                previous = pitch[i - 1] if j == i else pitch[j - 1] * factor

                if abs(1200 * np.log2(pitch[j] * factor / previous)) > 100: break
                j += 1

            if j > i and j < len(pitch) and confidence[j] >= 0.5:
                return_error = abs(1200 * np.log2(pitch[j] / (pitch[j - 1] * factor)))
                anchor_error = abs(1200 * np.log2(pitch[j] / pitch[i - 1]))

                if return_error <= 100 and anchor_error <= 150:
                    corrected[i:j] *= factor
                    repaired[i:j] = True

                    i = j + 1
                    continue

            i += 1

        return corrected, repaired