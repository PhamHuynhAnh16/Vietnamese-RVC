import os
import sys
import gzip
import zlib
import tqdm
import torch
import base64
import string
import tiktoken
import itertools

import numba as nb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import replace
from torch.distributions import Categorical
from functools import cached_property, lru_cache

sys.path.append(os.getcwd())

from main.app.variables import config, configs, logger
from main.library.algorithm.normalization import Fp32LayerNorm

LANGUAGES = {
    "en": "english", 
    "zh": "chinese", 
    "de": "german", 
    "es": "spanish", 
    "ru": "russian", 
    "ko": "korean", 
    "fr": "french", 
    "ja": "japanese", 
    "pt": "portuguese", 
    "tr": "turkish", 
    "pl": "polish", 
    "ca": "catalan", 
    "nl": "dutch", 
    "ar": "arabic", 
    "sv": "swedish", 
    "it": "italian", 
    "id": "indonesian", 
    "hi": "hindi", 
    "fi": "finnish", 
    "vi": "vietnamese", 
    "he": "hebrew", 
    "uk": "ukrainian", 
    "el": "greek", 
    "ms": "malay", 
    "cs": "czech", 
    "ro": "romanian", 
    "da": "danish", 
    "hu": "hungarian", 
    "ta": "tamil", 
    "no": "norwegian", 
    "th": "thai", 
    "ur": "urdu", 
    "hr": "croatian", 
    "bg": "bulgarian", 
    "lt": "lithuanian", 
    "la": "latin", 
    "mi": "maori", 
    "ml": "malayalam", 
    "cy": "welsh", 
    "sk": "slovak", 
    "te": "telugu", 
    "fa": "persian", 
    "lv": "latvian", 
    "bn": "bengali", 
    "sr": "serbian", 
    "az": "azerbaijani", 
    "sl": "slovenian", 
    "kn": "kannada", 
    "et": "estonian", 
    "mk": "macedonian", 
    "br": "breton", 
    "eu": "basque", 
    "is": "icelandic", 
    "hy": "armenian", 
    "ne": "nepali", 
    "mn": "mongolian", 
    "bs": "bosnian", 
    "kk": "kazakh", 
    "sq": "albanian", 
    "sw": "swahili", 
    "gl": "galician", 
    "mr": "marathi", 
    "pa": "punjabi", 
    "si": "sinhala", 
    "km": "khmer", 
    "sn": "shona", 
    "yo": "yoruba", 
    "so": "somali", 
    "af": "afrikaans", 
    "oc": "occitan", 
    "ka": "georgian", 
    "be": "belarusian", 
    "tg": "tajik", 
    "sd": "sindhi", 
    "gu": "gujarati", 
    "am": "amharic", 
    "yi": "yiddish", 
    "lo": "lao", 
    "uz": "uzbek", 
    "fo": "faroese", 
    "ht": "haitian creole", 
    "ps": "pashto", 
    "tk": "turkmen", 
    "nn": "nynorsk", 
    "mt": "maltese", 
    "sa": "sanskrit", 
    "lb": "luxembourgish", 
    "my": "myanmar", 
    "bo": "tibetan", 
    "tl": "tagalog", 
    "mg": "malagasy", 
    "as": "assamese", 
    "tt": "tatar", 
    "haw": "hawaiian", 
    "ln": "lingala", 
    "ha": "hausa", 
    "ba": "bashkir", 
    "jw": "javanese", 
    "su": "sundanese", 
    "yue": "cantonese"
}

TO_LANGUAGE_CODE = {
    **{
        language: code for code, language in LANGUAGES.items()
    }, 
    "burmese": "my", 
    "valencian": "ca", 
    "flemish": "nl", 
    "haitian": "ht", 
    "letzeburgesch": "lb", 
    "pushto": "ps", 
    "panjabi": "pa", 
    "moldavian": "ro", 
    "moldovan": "ro", 
    "sinhalese": "si", 
    "castilian": "es", 
    "mandarin": "zh"
}

_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00", 
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO", 
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00", 
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m", 
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00", 
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000", 
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00", 
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9", 
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj", 
    "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj", 
    "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00", 
    "large": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00", 
    "large-v3-turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`"
}

SAMPLE_RATE, N_FFT, HOP_LENGTH, CHUNK_LENGTH = 16000, 400, 160, 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE 
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  
stft = None

def exact_div(x, y):
    """
    Perform exact integer division and ensure there is no remainder.

    Args:
        x (int): The dividend.
        y (int): The divisor.

    Returns:
        int: The quotient of x // y.
    """

    assert x % y == 0
    return x // y

# Time-related and token-related scaling constants calculated via exact division
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH) 
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN) 

def load_model(name = "base", device = "cpu"):
    """
    Load a Whisper model checkpoint from disk and configure its alignment heads.

    Args:
        name (str): Name of the model architecture configuration (e.g., "base").
        device (str): Target hardware device to load the model onto (e.g., "cpu", "cuda").

    Returns:
        torch.nn.Module: The initialized and loaded Whisper model moved to the target device.
    """

    # Build the absolute path to the weights file
    checkpoint_file = os.path.join(configs["speaker_diarization_path"], "models", name + ".pt")
    alignment_heads = _ALIGNMENT_HEADS[name]

    # Safely load the model weights using CPU mapping first to avoid GPU OOM
    with open(checkpoint_file, "rb") as fp:
        checkpoint = torch.load(fp, map_location="cpu", weights_only=True)

    del checkpoint_file # Free up string reference

    # Initialize model dimensions and state dict
    model = Whisper(ModelDimensions(**checkpoint["dims"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_alignment_heads(alignment_heads)

    return model.to(device)

def merge_punctuations(alignment, prepended, appended):
    """
    Merge hanging punctuations into their adjacent words based on prefix/suffix rules.

    Args:
        alignment (list[WordTiming]): List of WordTiming objects representing word boundaries.
        prepended (str): String containing punctuation marks that should stick to the following word.
        appended (str): String containing punctuation marks that should stick to the previous word.
    """

    # Phase 1: Backward pass to merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1

    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        
        # If previous item is a standalone punctuation matching prepended rules
        if previous.word.startswith(" ") and previous.word.strip() in prepended:
            following.word = previous.word + following.word
            following.tokens = previous.tokens + following.tokens

            # Clear out the merged element
            previous.word = ""
            previous.tokens = []
        else: j = i # Move anchor pointer

        i -= 1

    # Phase 2: Forward pass to merge appended punctuations (e.g., "hello ." -> "hello.")
    i = 0
    j = 1

    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]

        # If following item is an isolated punctuation matching appended rules
        if not previous.word.endswith(" ") and following.word in appended:
            previous.word = previous.word + following.word
            previous.tokens = previous.tokens + following.tokens

            # Clear out the merged element
            following.word = ""
            following.tokens = []
        else: i = j # Move anchor pointer

        j += 1

class WordTiming:
    """Data structure holding timestamp and probability metadata for an individual word."""

    def __init__(
        self, 
        word, 
        tokens, 
        start, 
        end, 
        probability
    ):
        """Initializes WordTiming with temporal boundary metrics and token metadata."""

        self.word = word
        self.tokens = tokens
        self.start = start
        self.end = end
        self.probability = probability

def median_filter(x, filter_width):
    """
    Apply a 1D median filter along the last dimension of a PyTorch tensor.

    Args:
        x (torch.Tensor): Input tensor to be filtered.
        filter_width (int): Odd integer representing the size of the sliding window.

    Returns:
        torch.Tensor: Median filtered tensor maintaining structural dimensions.
    """

    pad_width = filter_width // 2
    # Return early if data is shorter than padding bounds
    if x.shape[-1] <= pad_width: return x
    # Temporarily unsqueeze to match expected 3D format for processing if needed
    if (ndim := x.ndim) <= 2: x = x[None, None, :]

    assert (filter_width > 0 and filter_width % 2 == 1)
    result = None
    # Pad tensor symmetrically using reflection on boundaries
    x = F.pad(
        x, 
        (filter_width // 2, filter_width // 2, 0, 0), 
        mode="reflect"
    )

    # Unfold along the last dimension to create windows, sort them, and select the median value
    if result is None: result = x.unfold(-1, filter_width, 1).sort()[0][..., filter_width // 2]
    # Revert dimension expansion if applied earlier
    if ndim <= 2: result = result[0, 0]

    return result

@nb.jit(nopython=True)
def backtrace(trace):
    """
    Trace back the optimal alignment path from a DTW direction matrix.

    Args:
        trace (np.ndarray): 2D matrix containing transition step history tags (0: diag, 1: up, 2: left).

    Returns:
        np.ndarray: A 2xN array containing the sequence indices mapping path alignment.
    """

    i = trace.shape[0] - 1
    j = trace.shape[1] - 1

    # Force boundary defaults to prevent running out of matrix edge bounds
    trace[0, :] = 2
    trace[:, 0] = 1

    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))
        # Backtrack steps based on encoded direction numbers
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1: i -= 1
        elif trace[i, j] == 2: j -= 1
        else: raise ValueError("Invalid trace matrix element encountered.")

    # Reverse the list path to make it read chronologically from beginning to end
    return np.array(result)[::-1, :].T


@nb.jit(nopython=True, parallel=True)
def dtw_cpu(x):
    """
    Compute Dynamic Time Warping cross-alignment path matrix on CPU using Numba acceleration.

    Args:
        x (np.ndarray): Cost matrix grid representing computed cross-entropy distance weights.

    Returns:
        np.ndarray: Reconstructed index map tracking the minimum cost alignment trajectory path.
    """

    N, M = x.shape
    # Allocate DP matrices initialized to infinity
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)
    cost[0, 0] = 0

    # Iteratively evaluate alignment costs cell by cell
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1] # Diagonal (Match/Substitution)
            c1 = cost[i - 1, j] # Vertical (Insertion)
            c2 = cost[i, j - 1] # Horizontal (Deletion)

            # Determine minimum local transition routing paths
            if c0 < c1 and c0 < c2: c, t = c0, 0
            elif c1 < c0 and c1 < c2: c, t = c1, 1
            else: c, t = c2, 2

            # Accumulate cumulative cost steps
            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return backtrace(trace)

def dtw(x):
    """
    Bridge interface converting standard PyTorch tensors to NumPy arrays for DTW tracking execution.

    Args:
        x (torch.Tensor): PyTorch tensor distance grid matrix.

    Returns:
        np.ndarray: Reconstructed alignment track path mapping.
    """

    return dtw_cpu(x.double().cpu().numpy())

def find_alignment(model, tokenizer, text_tokens, mel, num_frames, *, medfilt_width = 7, qk_scale = 1.0):
    """
    Map explicit token items directly down to corresponding audio frames utilizing cross-attention weights.

    Args:
        model: Whisper network configuration reference.
        tokenizer: Active processing token encoder instance.
        text_tokens (list[int]): Text sequence items to align.
        mel (torch.Tensor): Audio Mel spectrogram matrix.
        num_frames (int): Raw quantity number of audio context frames available.
        medfilt_width (int): Sliding frame size used in filtering operations.
        qk_scale (float): Multiplier factor for scaling Query-Key cross attention weights.

    Returns:
        list[WordTiming]: Array list of evaluated timestamps mapping per isolated word item.
    """

    if len(text_tokens) == 0: return []
    # Reconstruct whole contextual block format with mandatory structural prompt framing tags
    tokens = torch.tensor([
        *tokenizer.sot_sequence, 
        tokenizer.no_timestamps, 
        *text_tokens, 
        tokenizer.eot
    ]).to(model.device)

    # Setup forward hook collectors to intercept cross-attention layer values seamlessly
    QKs = [None] * model.dims.n_text_layer
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1][0])
        ) 
        for i, block in enumerate(model.decoder.blocks)
    ]

    with torch.no_grad():
        # Execute forward pass to collect generation probabilities
        token_probs = model(
            mel.unsqueeze(0), 
            tokens.unsqueeze(0)
        )[0][len(tokenizer.sot_sequence) :, : tokenizer.eot].softmax(dim=-1)

        # Retrieve confidence probabilities matching our active target text tokens
        text_token_probs = token_probs[np.arange(len(text_tokens)), text_tokens].tolist()

    # Clear hook references immediately to prevent memory leakage
    for hook in hooks:
        hook.remove()

    # Extract indices tracking designated alignment heads across the model layers
    if not config.device.startswith(("privateuseone", "ocl")):
        alignment_indices = model.alignment_heads.indices().T
    else:
        # Fallback loop configuration handling for specialized virtual backends
        alignment_indices = [
            (l, h) 
            for l in range(model.alignment_heads.size(0)) 
            for h in range(model.alignment_heads.size(1)) 
            if model.alignment_heads[l, h]
        ]

    # Stack cross-attention weights from chosen layers and slice off excess frames  
    weights = (
        torch.stack([
            QKs[_l][_h] 
            for _l, _h in alignment_indices
        ])[:, :, : num_frames // 2] * qk_scale
    ).softmax(dim=-1)

    # Normalize alignment activations across token sequence dimensions
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)

    if config.device.startswith("privateuseone"):
        weights = median_filter(
            ((weights - mean) / std).cpu(), 
            medfilt_width
        ).to(weights.device)
    else:
        weights = median_filter(
            (weights - mean) / std, 
            medfilt_width
        )

    # Apply DTW tracking logic over averaged attention maps
    text_indices, time_indices = dtw(-weights.mean(axis=0)[len(tokenizer.sot_sequence) : -1])
    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    if len(word_tokens) <= 1: return []

    # Reconstruct token boundary offsets mapping to explicit text words
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    jump_times = time_indices[np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)] / TOKENS_PER_SECOND

    # Format output items array into systematic structural WordTiming records
    return [
        WordTiming(
            word, 
            tokens, 
            start, 
            end, 
            probability
        ) 
        for word, tokens, start, end, probability in zip(
            words, 
            word_tokens, 
            jump_times[word_boundaries[:-1]], 
            jump_times[word_boundaries[1:]], 
            [
                np.mean(text_token_probs[i:j]) 
                for i, j in zip(
                    word_boundaries[:-1], 
                    word_boundaries[1:]
                )
            ]
        )
    ]

def add_word_timestamps(
    *, 
    segments, 
    model, 
    tokenizer, 
    mel, 
    num_frames, 
    prepend_punctuations = "\"'“¿([{-", 
    append_punctuations = "\"'.。,，!！?？:：”)]}、", 
    last_speech_timestamp, 
    **kwargs
):
    """
    Refine text segment ranges down to fine-grained, word-by-word timestamps using DTW alignment maps.

    Args:
        segments (list[dict]): Extracted transcription chunk items tracking active window clips.
        model: Active Whisper neural network instance.
        tokenizer: Model token sequence manager reference.
        mel (torch.Tensor): Normalized log-Mel input spectrum tracking frame contents.
        num_frames (int): Current count of valid processing frames.
        prepend_punctuations (str): Prefix punctuation filter layout.
        append_punctuations (str): Suffix punctuation filter layout.
        last_speech_timestamp (float): The timestamp when the last speech segment ended.
        **kwargs: Forward tracking argument configs passed directly to find_alignment.
    """

    if len(segments) == 0: return
    # Strip out any End-Of-Text or control sequence tokens from tracking maps
    text_tokens_per_segment = [
        [token for token in segment["tokens"] if token < tokenizer.eot] 
        for segment in segments
    ]

    # Combine tokens to compute massive matrix alignment tracks in one pass
    text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
    alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames, **kwargs)

    # Collect statistical metadata about word lengths to filter out outlier anomalies
    word_durations = np.array([t.end - t.start for t in alignment])
    word_durations = word_durations[word_durations.nonzero()]

    median_duration = min(0.7, float(np.median(word_durations) if len(word_durations) > 0 else 0.0))
    max_duration = median_duration * 2

    # Smooth out unrealistic jump spikes across punctuation boundaries
    if len(word_durations) > 0:
        sentence_end_marks = ".。!！?？"
        for i in range(1, len(alignment)):
            if alignment[i].end - alignment[i].start > max_duration:
                if alignment[i].word in sentence_end_marks: alignment[i].end = alignment[i].start + max_duration
                elif alignment[i - 1].word in sentence_end_marks: alignment[i].start = alignment[i].end - max_duration

    # Merge isolated punctuation tokens back into their target words
    merge_punctuations(alignment, prepend_punctuations, append_punctuations)
    time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE
    word_index = 0

    # Distribute global alignment tracking information back onto individual segments
    for segment, text_tokens in zip(segments, text_tokens_per_segment):
        saved_tokens = 0
        words = []

        while word_index < len(alignment) and saved_tokens < len(text_tokens):
            timing = alignment[word_index]

            if timing.word: 
                words.append(
                    dict(
                        word=timing.word, 
                        start=round(time_offset + timing.start, 2), 
                        end=round(time_offset + timing.end, 2), 
                        probability=timing.probability
                    )
                )

            saved_tokens += len(timing.tokens)
            word_index += 1

        # Heuristic checks to identify and correct alignment edge-case anomalies
        if len(words) > 0:
            if (
                words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                    words[0]["end"] - words[0]["start"] > max_duration or 
                    (len(words) > 1 and words[1]["end"] - words[0]["start"] > max_duration * 2)
                )
            ):
                if (
                    len(words) > 1 and 
                    words[1]["end"] - words[1]["start"] > max_duration
                ): 
                    words[0]["end"] = words[1]["start"] = max(words[1]["end"] / 2, words[1]["end"] - max_duration)

                words[0]["start"] = max(0, words[0]["end"] - max_duration)

            # Re-align segment bounds if they conflict with word boundaries
            if (
                segment["start"] < words[0]["end"] and 
                segment["start"] - 0.5 > words[0]["start"]
            ): 
                words[0]["start"] = max(0, min(words[0]["end"] - median_duration, segment["start"]))
            else: 
                segment["start"] = words[0]["start"]

            if (
                segment["end"] > words[-1]["start"] and 
                segment["end"] + 0.5 < words[-1]["end"]
            ): 
                words[-1]["end"] = max(words[-1]["start"] + median_duration, segment["end"])
            else: 
                segment["end"] = words[-1]["end"]

            last_speech_timestamp = segment["end"]

        segment["words"] = words

@lru_cache(maxsize=None)
def mel_filters(device, n_mels):
    """
    Load standard Mel-scale frequency filterbanks and cache them in memory.

    Args:
        device (torch.device): Device destination where weights should reside.
        n_mels (int): Target frequency band count size (Must be 80 or 128).

    Returns:
        torch.Tensor: Cached constant filter matrix mapping.
    """

    assert n_mels in {80, 128}

    with np.load(
        os.path.join(configs["speaker_diarization_path"], "assets", "mel_filters.npz"), 
        allow_pickle=False
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def log_mel_spectrogram(audio, n_mels = 80, padding = 0, device = None):
    """
    Compute the log-Mel spectrogram of a given audio signal.

    Args:
        audio (str | np.ndarray | torch.Tensor): Input audio source file path or raw data array.
        n_mels (int): Frequency scaling band resolution size.
        padding (int): Length of zero padding to apply to the end of the audio.
        device: Target execution hardware environment.

    Returns:
        torch.Tensor: Log-frequency spectral power map normalized matrix.
    """

    global stft

    # If input is a file path string, load it using external utility wrappers
    if not torch.is_tensor(audio):
        if isinstance(audio, str): 
            from main.library.audio.audio import load_audio
            audio = load_audio(audio, sample_rate=SAMPLE_RATE)

        audio = torch.from_numpy(audio).float()

    if device is not None: audio = audio.to(device)
    if padding > 0: audio = F.pad(audio, (0, padding))

    # Calculate Short-Time Fourier Transform (STFT) based on backend device capability
    if str(audio.device).startswith(("ocl", "privateuseone")):
        if stft is None: 
            from main.library.backends.utils import STFT

            stft = STFT(
                N_FFT, 
                HOP_LENGTH, 
                N_FFT
            ).to(audio.device)

        fft = stft.transform(
            audio.unsqueeze(0), 
            eps=1e-9
        ).squeeze(0)
    else:
        fft = torch.stft(
            audio, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH, 
            window=torch.hann_window(N_FFT).to(audio.device), 
            return_complex=True
        )

    # Map powers across Mel scales and bound lower logs to prevent numeric underflows
    log_spec = (
        mel_filters(audio.device, n_mels) @ fft[..., :-1].abs() ** 2
    ).clamp(min=1e-10).log10()

    # Apply scaling to conform to standard standardized amplitude ranges
    return (log_spec.maximum(log_spec.max() - 8.0) + 4.0) / 4.0

def pad_or_trim(array, length = N_SAMPLES, *, axis = -1):
    """
    Pad or trim a tensor/array along a specified axis to match a target length.

    Args:
        array (torch.Tensor | np.ndarray): Input sequence buffer layout.
        length (int): Final desired dimension size constraint.
        axis (int): Targeting dimension axis index.

    Returns:
        The uniformly resized data tensor or array structure.
    """

    if torch.is_tensor(array):
        # Slice off trailing ends if array exceeds targeted bounds
        if array.shape[axis] > length: 
            array = array.index_select(
                dim=axis, 
                index=torch.arange(length, device=array.device)
            )

        # Pad with zeros if array falls short of targeted bounds
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        # NumPy fallback operations path
        if array.shape[axis] > length: array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

def get_end(segments):
    """
    Retrieve the end timestamp of the last valid word found within active segment logs.

    Args:
        segments (list[dict]): Processed text segment history entries.

    Returns:
        float: Absolute timestamp indicating where the last vocal speech word item concluded.
    """

    return next((
        w["end"] 
        for s in reversed(segments) 
        for w in reversed(s["words"])
    ), segments[-1]["end"] if segments else None)

def transcribe_function(
    model, 
    audio, 
    *, 
    verbose = None, 
    temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), 
    compression_ratio_threshold = 2.4, 
    logprob_threshold = -1.0, 
    no_speech_threshold = 0.6, 
    condition_on_previous_text = True, 
    initial_prompt = None, 
    carry_initial_prompt = False, 
    word_timestamps = False, 
    prepend_punctuations = "\"'“¿([{-", 
    append_punctuations = "\"'.。,，!！?？:：”)]}、", 
    clip_timestamps = "0", 
    hallucination_silence_threshold = None, 
    fp16 = False, 
    **decode_options
):
    """
    Transcribe an audio source using a Whisper model with fallback mechanisms and timestamp adjustments.

    Args:
        model: Loaded instance configuration of the Whisper neural network.
        audio: Audio source to transcribe (file path or pre-loaded data array).
        verbose (bool): Set to true to print out runtime logging details.
        temperature (tuple): Sequence of decoding temperatures to attempt when fallbacks trigger.
        compression_ratio_threshold (float): Upper compression limit before rejecting bad text loops.
        logprob_threshold (float): Minimum log probability threshold to accept a generation.
        no_speech_threshold (float): Silence probability limit above which segments are skipped.
        condition_on_previous_text (bool): If True, feeds past generation back into model prompts.
        initial_prompt (str): Text phrase passed into initial context fields.
        carry_initial_prompt (bool): If True, retains initial prompts across all processing chunks.
        word_timestamps (bool): Extends metrics output generation down to word level granularities.
        prepend_punctuations (str): Prefix punctuation mapping string formatting definitions.
        append_punctuations (str): Suffix punctuation mapping string formatting definitions.
        clip_timestamps (str): Comma-separated time markers limiting execution processing windows.
        hallucination_silence_threshold (float): Maximum allowed gap width before assuming hallucinations.
        fp16 (bool): Switch encoding tensors execution path directly into float16 precision.

    Returns:
        dict: Complete structured dictionary containing text, segments, and detected language.
    """

    dtype = torch.float16 if fp16 else torch.float32
    decode_options["fp16"] = fp16

    # Convert audio into continuous normalized Log-Mel-Spectrogram spectrum frames
    mel = log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-1] - N_FRAMES
    content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)

    # Detect language if not specified explicitly in arguments
    if decode_options.get("language", None) is None:
        if not model.is_multilingual: decode_options["language"] = "vi"
        else:
            mel_segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)

            if verbose is not None: logger.info(f"{LANGUAGES[decode_options['language']].title()}")

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")

    tokenizer = get_tokenizer(
        model.is_multilingual, 
        num_languages=model.num_languages, 
        language=language, 
        task=task
    )

    # Parse timestamp clipping ranges to constrain decoding regions
    if isinstance(clip_timestamps, str): 
        clip_timestamps = [
            float(ts) 
            for ts in (
                clip_timestamps.split(",") if clip_timestamps else []
            )
        ]

    seek_points = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]

    if len(seek_points) == 0: seek_points.append(0)
    if len(seek_points) % 2 == 1: seek_points.append(content_frames)

    seek_clips = list(zip(seek_points[::2], seek_points[1::2]))
    punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

    def decode_with_fallback(segment):
        """Helper to retry decoding with higher temperatures if quality thresholds fail."""

        decode_result = None
        temperatures = (
            [temperature] if isinstance(temperature, (int, float)) else temperature
        )

        for t in temperatures:
            kwargs = {**decode_options}

            # Adjust options based on whether we are doing greedy decoding or sampling
            if t > 0:
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else: kwargs.pop("best_of", None)

            # Perform inference
            decode_result = model.decode(
                segment, 
                DecodingOptions(
                    **kwargs, 
                    temperature=t
                )
            )

            needs_fallback = False

            # Check if output is repetitive (high compression ratio)
            if (
                compression_ratio_threshold is not None and 
                decode_result.compression_ratio > compression_ratio_threshold
            ): 
                needs_fallback = True  

            # Check if overall confidence is too low
            if (
                logprob_threshold is not None and 
                decode_result.avg_logprob < logprob_threshold
            ): 
                needs_fallback = True  

            # Override fallback if it's likely just silence/no speech
            if (
                no_speech_threshold is not None and 
                decode_result.no_speech_prob > no_speech_threshold and 
                logprob_threshold is not None and 
                decode_result.avg_logprob < logprob_threshold
            ): 
                needs_fallback = False 

            if not needs_fallback: break

        return decode_result

    # Initialize tracking variables for the main loop
    clip_idx = 0
    seek = seek_clips[clip_idx][0]

    input_stride = exact_div(N_FRAMES, model.dims.n_audio_ctx)  
    time_precision = (input_stride * HOP_LENGTH / SAMPLE_RATE) 

    all_tokens, all_segments = [], []
    prompt_reset_since = 0

    remaining_prompt_length = model.dims.n_text_ctx // 2 - 1
    # Encode the initial prompt text if provided
    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
        remaining_prompt_length -= len(initial_prompt_tokens)
    else: initial_prompt_tokens = []

    def new_segment(*, start, end, tokens, result):
        """Construct a structured dictionary for a transcribed segment."""

        tokens = tokens.tolist()

        return {
            "seek": seek, 
            "start": start, 
            "end": end, 
            "text": tokenizer.decode([token for token in tokens if token < tokenizer.eot]), 
            "tokens": tokens, 
            "temperature": result.temperature, 
            "avg_logprob": result.avg_logprob, 
            "compression_ratio": result.compression_ratio, 
            "no_speech_prob": result.no_speech_prob
        }

    # Main chunk-by-chunk processing loop with a progress bar
    with tqdm.tqdm(total=content_frames, unit="frames", disable=verbose is not False) as pbar:
        last_speech_timestamp = 0.0
        while clip_idx < len(seek_clips):
            seek_clip_start, seek_clip_end = seek_clips[clip_idx]
            if seek < seek_clip_start: seek = seek_clip_start

            # Advance to the next time clip if the current one is finished
            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips): seek = seek_clips[clip_idx][0]
                continue

            # Calculate current window time offsets
            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            window_end_time = float((seek + N_FRAMES) * HOP_LENGTH / SAMPLE_RATE)

            segment_size = min(N_FRAMES, content_frames - seek, seek_clip_end - seek)
            mel_segment = mel[:, seek : seek + segment_size]

            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(model.device).to(dtype)

            # Prepare the prompt history for context conditioning
            if carry_initial_prompt: 
                decode_options["prompt"] = initial_prompt_tokens + all_tokens[max(len(initial_prompt_tokens), prompt_reset_since):][-remaining_prompt_length:]
            else: 
                decode_options["prompt"] = all_tokens[prompt_reset_since:]

            # Run transcription on the current chunk
            result = decode_with_fallback(mel_segment)
            tokens = torch.tensor(result.tokens)

            # Skip processing this segment if it is detected as silence
            if no_speech_threshold is not None:
                should_skip = result.no_speech_prob > no_speech_threshold
                if (logprob_threshold is not None and result.avg_logprob > logprob_threshold):
                    should_skip = False

                if should_skip:
                    seek += segment_size  
                    continue

            previous_seek = seek
            current_segments = []

            def word_anomaly_score(word):
                """Calculate an anomaly score based on unusual word lengths or low probability."""

                probability = word.get("probability", 0.0)
                duration = word["end"] - word["start"]
                score = 0.0

                if probability < 0.15: score += 1.0
                if duration < 0.133: score += (0.133 - duration) * 15
                if duration > 2.0: score += duration - 2.0

                return score

            def is_segment_anomaly(segment):
                """Check if a segment's initial words look like a model hallucination loop."""

                if segment is None or not segment["words"]: return False
                
                words = [w for w in segment["words"] if w["word"] not in punctuation]
                words = words[:8]

                score = sum(word_anomaly_score(w) for w in words)

                return score >= 3 or score + 0.01 >= len(words)

            def next_words_segment(segments):
                """Find the first segment in the list that contains word metadata."""
    
                return next((s for s in segments if s["words"]), None)

            # Parse timestamp tokens to slice the chunk into sub-segments
            timestamp_tokens = tokens.ge(tokenizer.timestamp_begin)
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
            consecutive.add_(1)

            if len(consecutive) > 0:
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    current_segments.append(
                        new_segment(
                            start=time_offset + (sliced_tokens[0].item() - tokenizer.timestamp_begin) * time_precision, 
                            end=time_offset + (sliced_tokens[-1].item() - tokenizer.timestamp_begin) * time_precision, 
                            tokens=sliced_tokens, 
                            result=result
                        )
                    )
                    last_slice = current_slice

                if single_timestamp_ending: seek += segment_size
                else: seek += (tokens[last_slice - 1].item() - tokenizer.timestamp_begin) * input_stride
            else:
                # Fallback path if no internal timestamp markers were generated
                duration = segment_duration

                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if (len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin): 
                    duration = (timestamps[-1].item() - tokenizer.timestamp_begin) * time_precision

                current_segments.append(
                    new_segment(
                        start=time_offset, 
                        end=time_offset + duration, 
                        tokens=tokens, 
                        result=result
                    )
                )

                seek += segment_size

            # Compute fine-grained word timestamps if requested
            if word_timestamps:
                add_word_timestamps(
                    segments=current_segments, 
                    model=model, 
                    tokenizer=tokenizer, 
                    mel=mel_segment, 
                    num_frames=segment_size, 
                    prepend_punctuations=prepend_punctuations, 
                    append_punctuations=append_punctuations, 
                    last_speech_timestamp=last_speech_timestamp
                )

                if not single_timestamp_ending:
                    last_word_end = get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset: 
                        seek = round(last_word_end * FRAMES_PER_SECOND)

                # Implement an anti-hallucination guardrail for prolonged periods of silence
                if hallucination_silence_threshold is not None:
                    threshold = hallucination_silence_threshold

                    if not single_timestamp_ending:
                        last_word_end = get_end(current_segments)
                        if last_word_end is not None and last_word_end > time_offset: 
                            seek = round(last_word_end * FRAMES_PER_SECOND) if (window_end_time - last_word_end) > threshold else (previous_seek + segment_size)

                    first_segment = next_words_segment(current_segments)
                    if first_segment is not None and is_segment_anomaly(first_segment):
                        gap = first_segment["start"] - time_offset

                        if gap > threshold:
                            seek = previous_seek + round(gap * FRAMES_PER_SECOND)
                            continue

                    hal_last_end = last_speech_timestamp

                    for si in range(len(current_segments)):
                        segment = current_segments[si]
                        if not segment["words"]: continue

                        # If a hallucination pattern is caught, cut the segment list short and reposition seek
                        if is_segment_anomaly(segment):
                            next_segment = next_words_segment(current_segments[si + 1 :])
                            hal_next_start = next_segment["words"][0]["start"] if next_segment is not None else (time_offset + segment_duration)

                            if (
                                segment["start"] - hal_last_end > threshold or segment["start"] < threshold or segment["start"] - time_offset < 2.0
                            ) and (
                                hal_next_start - segment["end"] > threshold or is_segment_anomaly(next_segment) or window_end_time - segment["end"] < 2.0
                            ):
                                seek = round(max(time_offset + 1, segment["start"]) * FRAMES_PER_SECOND)
                                if content_duration - segment["end"] < threshold: seek = content_frames

                                current_segments[si:] = []
                                break

                        hal_last_end = segment["end"]

                last_word_end = get_end(current_segments)
                if last_word_end is not None: last_speech_timestamp = last_word_end

            # Filter out and clean empty segments
            for _, segment in enumerate(current_segments):
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []

            # Append the calculated results to global arrays
            all_segments.extend([{"id": i, **segment} for i, segment in enumerate(current_segments, start=len(all_segments))])
            all_tokens.extend([token for segment in current_segments for token in segment["tokens"]])

            # Clear prompt conditioning history if temperature is high to prevent repetitive loops
            if not condition_on_previous_text or result.temperature > 0.5: prompt_reset_since = len(all_tokens)
            pbar.update(min(content_frames, seek) - previous_seek)

    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]), segments=all_segments, language=language)

def compression_ratio(text):
    """
    Calculate the compression ratio of a text string using zlib.
    
    Used as a heuristic metric to detect repetitive or low-quality generations.

    Args:
        text (str): The input string to evaluate.

    Returns:
        float: The ratio of raw byte length to compressed byte length.
    """
    text_bytes = text.encode("utf-8")
    # Ratio > 1.0 means compression successfully reduced the size
    return len(text_bytes) / len(zlib.compress(text_bytes))

def sinusoids(length, channels, max_timescale=10000):
    """
    Generate sinusoidal positional embeddings (used in the audio encoder).

    Args:
        length (int): Number of position steps (context length).
        channels (int): Dimensionality of the embedding space.
        max_timescale (float, optional): Maximum scale factor for frequency. Defaults to 10000.

    Returns:
        torch.Tensor: A tensor of shape (length, channels) containing sine/cosine waves.
    """

    assert channels % 2 == 0 # Dimensions must be even to split equally into sin and cos
    # Compute exponential frequency scaling factors
    scaled_time = torch.arange(length)[:, np.newaxis] * (-(np.log(max_timescale) / (channels // 2 - 1)) * torch.arange(channels // 2)).exp()[np.newaxis, :]
    # Concatenate sine and cosine waves along the feature dimension
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)

@torch.no_grad()
def detect_language_function(model, mel, tokenizer = None):
    """
    Detect the language of the given audio features (Mel Spectrogram).

    Args:
        model (nn.Module): The Whisper model instance.
        mel (torch.Tensor): Audio Mel Spectrogram tensor.
        tokenizer (Tokenizer, optional): A Whisper Tokenizer instance. Defaults to None.

    Returns:
        Tuple[torch.Tensor, List[Dict[str, float]]]: Detected language tokens and probability distributions.
    """

    # Initialize tokenizer if not provided
    if tokenizer is None: tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
    if (tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence): raise ValueError("Invalid tokenizer configuration for language detection.")

    single = mel.ndim == 2
    # Add batch dimension if single instance
    if single: mel = mel.unsqueeze(0)
    # Extract audio features if raw Mel spectrogram is passed
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state): mel = model.encoder(mel)

    n_audio = mel.shape[0]
    # Predict logits using the Start-Of-Transcript (SOT) token as prompt context
    logits = model.logits(torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device), mel)[:, 0]

    # Create a mask to suppress all tokens except explicit language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False

    logits[:, mask] = -np.inf # Set non-language tokens to negative infinity

    # Get the token ID of the highest probability language
    language_tokens = logits.argmax(dim=-1)
    # Compute full probability mapping distribution per batch item
    language_probs = [
        {
            c: logits.softmax(dim=-1).cpu()[i, j].item() 
            for j, c in zip(
                tokenizer.all_language_tokens, 
                tokenizer.all_language_codes
            )
        } 
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs

@lru_cache(maxsize=None)
def get_tokenizer(multilingual, *, num_languages = 99, language = None, task = None):
    """
    Fetch or create a cached Tokenizer instance based on language configurations.

    Args:
        multilingual (bool): Whether to load the multilingual vocabulary or English-only variant.
        num_languages (int, optional): Number of languages supported. Defaults to 99.
        language (str, optional): Target language name or code. Defaults to None.
        task (str, optional): Task type (e.g., 'transcribe', 'translate'). Defaults to None.

    Returns:
        Tokenizer: Initialized tokenizer wrapper instance.
    """

    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE: language = TO_LANGUAGE_CODE[language]
            else: raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    return Tokenizer(
        encoding_name=encoding_name, 
        num_languages=num_languages, 
        language=language, 
        task=task
    )

@lru_cache(maxsize=None)
def get_encoding(name = "gpt2", num_languages = 99):
    """
    Load and configure the Byte-Pair Encoding (BPE) structures utilizing tiktoken.

    Args:
        name (str, optional): Name of the tiktoken tokenizer mapping. Defaults to "gpt2".
        num_languages (int, optional): Number of language specific tokens to register. Defaults to 99.

    Returns:
        tiktoken.Encoding: The configured BPE byte encoder structure.
    """

    vocab_path = os.path.join(configs["speaker_diarization_path"], "assets", f"{name}.tiktoken")
    # Load core token merge ranks from file
    ranks = {
        base64.b64decode(token): int(rank) 
        for token, rank in (
            line.split() 
            for line in open(vocab_path) 
            if line
        )
    }

    n_vocab = len(ranks)
    special_tokens = {}
    # Define all structural and special control tokens for the transformer
    specials = [
        "<|endoftext|>", "<|startoftranscript|>", 
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]], 
        "<|translate|>", 
        "<|transcribe|>", 
        "<|startoflm|>", 
        "<|startofprev|>", 
        "<|nospeech|>", 
        "<|notimestamps|>", 
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)] # Special tokens capturing timestamp steps
    ]

    # Dynamically allocate indices for special control tokens
    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path), 
        explicit_n_vocab=n_vocab, 
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", 
        mergeable_ranks=ranks, 
        special_tokens=special_tokens
    )

class DecodingOptions:
    """Dataclass configuration containing adjustable options for custom auto-regressive generation."""

    def __init__(
        self, 
        task = "transcribe", 
        language = None, 
        temperature = 0.0, 
        sample_len = None, 
        best_of = None, 
        beam_size = None, 
        patience = None, 
        length_penalty = None, 
        prompt = None, 
        prefix = None, 
        suppress_tokens = "-1", 
        suppress_blank = True, 
        without_timestamps = False, 
        max_initial_timestamp = 1.0, 
        fp16 = False
    ):
        self.task = task
        self.language = language
        self.temperature = temperature
        self.sample_len = sample_len
        self.best_of = best_of
        self.beam_size = beam_size
        self.patience = patience
        self.length_penalty = length_penalty
        self.prompt = prompt
        self.prefix = prefix
        self.suppress_tokens = suppress_tokens
        self.suppress_blank = suppress_blank
        self.without_timestamps = without_timestamps
        self.max_initial_timestamp = max_initial_timestamp
        self.fp16 = fp16

@torch.no_grad()
def decode_function(model, mel, options = DecodingOptions(), **kwargs):
    """
    Main decoding utility wrapper to sequence tokens from given processed acoustic features.

    Args:
        model (nn.Module): The Whisper model block.
        mel (torch.Tensor): Audio features spectrogram context.
        options (DecodingOptions, optional): Decoding configuration settings.
        **kwargs: Arbitrary keyword arguments to override specified options dynamic values.

    Returns:
        Union[DecodingResult, List[DecodingResult]]: The result object(s) containing text and features.
    """

    if single := mel.ndim == 2: mel = mel.unsqueeze(0) # Ensure batched representation format
    if kwargs: options = replace(options, **kwargs) # Override default parameter options safely

    # Instantiate the dedicated core DecodingTask framework
    result = DecodingTask(model, options).run(mel)
    return result[0] if single else result

class ModelDimensions:
    """Structure wrapper explicitly specifying the model configuration sizes."""

    def __init__(
        self, 
        n_mels, 
        n_audio_ctx, 
        n_audio_state, 
        n_audio_head, 
        n_audio_layer, 
        n_vocab, 
        n_text_ctx, 
        n_text_state, 
        n_text_head, 
        n_text_layer
    ):
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.n_vocab = n_vocab
        self.n_text_ctx = n_text_ctx
        self.n_text_state = n_text_state
        self.n_text_head = n_text_head
        self.n_text_layer = n_text_layer
    
class Linear(nn.Linear):
    """Custom Linear layer wrapper executing safe runtime precision casting to match inputs dynamic precision."""

    def forward(
        self, 
        x
    ):
        return F.linear(
            x, 
            self.weight.to(x.dtype), 
            self.bias.to(x.dtype) if self.bias is not None else None
        )

class Conv1d(nn.Conv1d):
    """Custom 1D Convolution wrapper executing dynamic type casting to handle fp16 execution safely."""

    def _conv_forward(
        self, 
        x, 
        weight, 
        bias
    ):
        return super()._conv_forward(
            x, 
            weight.to(x.dtype), 
            bias.to(x.dtype) if bias is not None else None
        )

class TextDecoder(nn.Module):
    """The auto-regressive transformer decoder generating text output sequences based on acoustic contexts."""

    def __init__(
        self, 
        n_vocab, 
        n_ctx, 
        n_state, 
        n_head, 
        n_layer
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(
                n_state, 
                n_head, 
                cross_attention=True
            ) 
            for _ in range(n_layer)
        ])
        self.ln = Fp32LayerNorm(n_state)
        # Upper triangular causal matrix to avoid attention lookaheads on generation loops
        self.register_buffer("mask", torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1), persistent=False)

    def forward(self, x, xa, kv_cache = None):
        # Calculate generation sequence token offset index from current Key-Value cache
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]).to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        # Weight-tying projection back to raw token embedding logits
        return (x @ self.token_embedding.weight.to(x.dtype).transpose(0, 1)).float()

class AudioEncoder(nn.Module):
    """The convolutional-transformer block processing raw input audio Mel frames into continuous states."""

    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(
                n_state, 
                n_head
            ) 
            for _ in range(n_layer)
        ])
        self.ln_post = Fp32LayerNorm(n_state)

    def forward(self, x):
        # Pass through Conv1D feature extractors with GELU activations
        x = F.gelu(self.conv2(F.gelu(self.conv1(x)))).permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape
        x = (x + self.positional_embedding).to(x.dtype) # Inject static positional waves

        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)

class Whisper(nn.Module):
    """The overarching Sequence-to-Sequence multi-task Whisper pipeline model entity wrapper."""

    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels, 
            self.dims.n_audio_ctx, 
            self.dims.n_audio_state, 
            self.dims.n_audio_head, 
            self.dims.n_audio_layer
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab, 
            self.dims.n_text_ctx, 
            self.dims.n_text_state, 
            self.dims.n_text_head, 
            self.dims.n_text_layer
        )
        # Initialize cross-attention head alignments for timestamp calculation logic
        all_heads = torch.zeros(self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool)
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads if config.device.startswith(("privateuseone", "ocl")) else all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump):
        """Decompress and parse custom binary alignment head masks."""

        alignment = torch.from_numpy(
            np.frombuffer(
                gzip.decompress(base64.b85decode(dump)), dtype=bool
            ).copy()
        ).reshape(self.dims.n_text_layer, self.dims.n_text_head)

        if not config.device.startswith(("privateuseone", "ocl")): alignment = alignment.to_sparse()
        self.register_buffer("alignment_heads", alignment, persistent=False)

    def embed_audio(self, mel):
        return self.encoder(mel)

    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)

    def forward(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865 # Models with vocabulary larger than 51865 contain language tokens

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache = None):
        """
        Install native PyTorch forward hooks to continuously build and expand the generation KV cache.

        Args:
            cache (dict, optional): Existing cached tensor collection mappings. Defaults to None.

        Returns:
            Tuple[dict, list]: The active cache structure and a list of registered hook handles.
        """

        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            # Concat current output tensors directly into the key-value historical state dictionary
            cache[module] = output if module not in cache or output.shape[1] > self.dims.n_text_ctx else torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    # Bind internal core execution strategies explicitly
    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

class ResidualAttentionBlock(nn.Module):
    """A standard Transformer layer module possessing Self-Attention, Optional Cross-Attention, and an MLP block."""

    def __init__(
        self, 
        n_state, 
        n_head, 
        cross_attention = False
    ):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = Fp32LayerNorm(n_state)
        self.cross_attn = (MultiHeadAttention(n_state, n_head) if cross_attention else None)
        self.cross_attn_ln = Fp32LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = Fp32LayerNorm(n_state)

    def forward(self, x, xa = None, mask = None, kv_cache = None):
        # Self Attention over sequence history with residual connection
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        # Cross Attention using contextual visual/audio encoder activations (xa)
        if self.cross_attn: x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]

        # Feed Forward Network segment with residual connection
        return x + self.mlp(self.mlp_ln(x))
    
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention framework managing either causal self-attention or cross-attention projections."""

    def __init__(
        self, 
        n_state, 
        n_head
    ):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self, 
        x, 
        xa = None, 
        mask = None, 
        kv_cache = None
    ):
        # Use cross-attention context tensors if provided (xa), otherwise use self sequences (x)
        k, v = (
            self.key(x if xa is None else xa), 
            self.value(x if xa is None else xa)
        ) if kv_cache is None or xa is None or self.key not in kv_cache else (
            kv_cache[self.key], 
            kv_cache[self.value]
        )

        wv, qk = self.qkv_attention(self.query(x), k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q, k, v, mask = None):
        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25 # Scaling factor calculation step 1

        # Reshape projection blocks to separate attention heads
        q, k, v = (
            q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3), 
            k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3), 
            v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        )

        # MatMul score computation with scale factor distribution
        qk = (q * scale) @ (k * scale).transpose(-1, -2)
        if mask is not None: qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        # Compute Softmax weight probabilities and project values back to full sequence shapes
        return (F.softmax(qk, dim=-1).to(q.dtype) @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class SuppressBlank:
    """Logit filter that suppresses structural whitespace/blank space generations right at the very first token."""

    def __init__(self, tokenizer, sample_begin):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits, tokens):
        if tokens.shape[1] == self.sample_begin:
            # Invalidate raw white space token and End of Transcript sequences instantly on start 
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf

class SuppressTokens:
    """Suppresses specific targeted bad or invalid token IDs entirely during text generation."""

    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits, tokens):
        logits[:, self.suppress_tokens] = -np.inf

class PyTorchInference:
    """Standard PyTorch inference runtime implementation preserving and updating historical KV tracking frames."""

    def __init__(self, model, initial_token_length):
        self.model = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

        # Target attention weights across all nested structural layers
        self.kv_modules = [
            block.attn.key 
            for block in self.model.decoder.blocks
        ] + [
            block.attn.value 
            for block in self.model.decoder.blocks
        ]

    def logits(self, tokens, audio_features):
        if not self.kv_cache: self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()
        # Optimize auto-regressive processing loop: Only pass the newest token if historical context is cached
        if tokens.shape[-1] > self.initial_token_length: tokens = tokens[:, -1:]

        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        """Permute or filter the current cached sequences to align correctly with active beam search indices."""

        if source_indices != list(range(len(source_indices))):
            for module in self.kv_modules:
                self.kv_cache[module] = self.kv_cache[module][source_indices].detach()

class MaximumLikelihoodRanker:
    """Ranker selecting the best sequences based on their accumulated log probabilities with length penalization."""

    def __init__(
        self, 
        length_penalty
    ):
        self.length_penalty = length_penalty

    def rank(self, tokens, sum_logprobs):
        def scores(logprobs, lengths):
            result = []

            for logprob, length in zip(logprobs, lengths):
                result.append(
                    # Standard formula applying length constraints penalties safely
                    logprob / (length if self.length_penalty is None else ((5 + length) / 6) ** self.length_penalty)
                )

            return result

        return [
            np.argmax(scores(p, l)) 
            for p, l in zip(sum_logprobs, [[len(t) for t in s] for s in tokens])
        ]

class GreedyDecoder:
    """Simple greedy/categorical sampler generating the single best tokens sequentially."""

    def __init__(
        self, 
        temperature, 
        eot
    ):
        self.temperature = temperature
        self.eot = eot

    def update(self, tokens, logits, sum_logprobs):
        next_tokens = logits.argmax(dim=-1) if self.temperature == 0 else (
            # Sample dynamically using Categorical distributions based on selected temperature configurations
            Categorical(
                logits=(logits / self.temperature).cpu() if config.device.startswith("ocl") else (logits / self.temperature)
            )
        ).sample().to(logits.device)

        # Accumulate log probability values for tracking confidence scores
        logprobs = F.log_softmax(logits.float(), dim=-1)
        sum_logprobs += logprobs[torch.arange(logprobs.shape[0]), next_tokens] * (tokens[:, -1] != self.eot)

        # Lock sequences that have already hit the End-of-Transcript token
        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        return tokens, (tokens[:, -1] == self.eot).all()

    def finalize(self, tokens, sum_logprobs):
        return F.pad(tokens, (0, 1), value=self.eot), sum_logprobs.tolist()

class BeamSearchDecoder:
    """Advanced search tracker preserving up to K alternative high probability token pathways simultaneously."""

    def __init__(
        self, 
        beam_size, 
        eot, 
        inference, 
        patience = None
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates = round(beam_size * self.patience)
        self.finished_sequences = None

        assert (self.max_candidates > 0)

    def reset(self):
        self.finished_sequences = None

    def update(self, tokens, logits, sum_logprobs):
        if tokens.shape[0] % self.beam_size != 0: raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None: self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []

        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # Evaluate options within individual candidate sequence groups
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = (sum_logprobs[idx] + logprob).item()
                    sources[sequence] = idx

            saved = 0
            # Harvest best scoring branches and group them respectively
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot: finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size: break

            finished_sequences.append(finished)

        # Mutate the underlying inference cache positions to reflect survival branches
        self.inference.rearrange_kv_cache(source_indices)
        assert len(self.finished_sequences) == len(finished_sequences)

        # Save finished candidates to list up to max threshold limits
        for previously_finished, newly_finished in zip(self.finished_sequences, finished_sequences):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates: break  
                previously_finished[seq] = newly_finished[seq]

        return (
            torch.tensor(next_tokens, device=tokens.device), 
            all(len(sequences) >= self.max_candidates for sequences in self.finished_sequences)
        )

    def finalize(self, preceding_tokens, sum_logprobs):
        sum_logprobs = sum_logprobs.cpu()

        for i, sequences in enumerate(self.finished_sequences):
            # Fallback handling step: If search completes but lacks finished beams, force-inject fallback records
            if (len(sequences) < self.beam_size):  
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size: break

        return (
            [[torch.tensor(seq) for seq in sequences.keys()] for sequences in self.finished_sequences], 
            [list(sequences.values()) for sequences in self.finished_sequences]
        )

class ApplyTimestampRules:
    """Enforces grammar constraints governing structural timestamp token sequence configurations."""

    def __init__(
        self, 
        tokenizer, 
        sample_begin, 
        max_initial_timestamp_index
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits, tokens):
        if self.tokenizer.no_timestamps is not None: logits[:, self.tokenizer.no_timestamps] = -np.inf

        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = [t for t in sampled_tokens.tolist()]

            last_was_timestamp = (len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin)
            penultimate_was_timestamp = (len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin)

            if last_was_timestamp:
                if penultimate_was_timestamp: logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else: logits[k, : self.tokenizer.eot] = -np.inf

            timestamps = sampled_tokens[sampled_tokens.ge(self.tokenizer.timestamp_begin)]

            if timestamps.numel() > 0: 
                # Ensure timestamp increments monotonically over chronological time steps
                logits[
                    k, 
                    self.tokenizer.timestamp_begin : 
                    timestamps[-1] if last_was_timestamp and not penultimate_was_timestamp else (timestamps[-1] + 1)
                ] = -np.inf

        # Prevent transcription sequences from beginning directly with an empty/unreachable timestamp range
        if tokens.shape[1] == self.sample_begin:
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            if self.max_initial_timestamp_index is not None:
                last_allowed = (self.tokenizer.timestamp_begin + self.max_initial_timestamp_index)
                logits[:, last_allowed + 1 :] = -np.inf

        # Mathematical bias enforcing timestamp generations if their cumulative probability dominates
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            if logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(dim=-1) > logprobs[k, : self.tokenizer.timestamp_begin].max(): 
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf

class DecodingTask:
    """The manager orchestrating the complete cross-attention decoding lifecycle."""

    def __init__(
        self, 
        model, 
        options
    ):
        self.model = model
        language = options.language or "en"

        tokenizer = get_tokenizer(
            model.is_multilingual, 
            num_languages=model.num_languages, 
            language=language, 
            task=options.task
        )

        self.tokenizer = tokenizer
        self.options = self._verify_options(options)
        self.n_group = options.beam_size or options.best_of or 1
        self.n_ctx = model.dims.n_text_ctx
        self.sample_len = options.sample_len or model.dims.n_text_ctx // 2
        self.sot_sequence = tokenizer.sot_sequence

        if self.options.without_timestamps: 
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens = self._get_initial_tokens()
        self.sample_begin = len(self.initial_tokens)
        self.sot_index = self.initial_tokens.index(tokenizer.sot)

        self.inference = PyTorchInference(model, len(self.initial_tokens))
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)
        # Dynamically switch between BeamSearchDecoder and GreedyDecoder based on configuration options
        self.decoder = BeamSearchDecoder(
            options.beam_size, 
            tokenizer.eot, 
            self.inference, 
            options.patience
        ) if options.beam_size is not None else GreedyDecoder(
            options.temperature, 
            tokenizer.eot
        )

        self.logit_filters = []

        if self.options.suppress_blank: 
            self.logit_filters.append(
                SuppressBlank(
                    self.tokenizer, 
                    self.sample_begin
                )
            )

        if self.options.suppress_tokens: 
            self.logit_filters.append(
                SuppressTokens(
                    self._get_suppress_tokens()
                )
            )

        if not options.without_timestamps:
            max_initial_timestamp_index = None

            if options.max_initial_timestamp: 
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / (CHUNK_LENGTH / model.dims.n_audio_ctx)
                )

            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, 
                    self.sample_begin, 
                    max_initial_timestamp_index
                )
            )

    def _verify_options(self, options):
        """Validate argument combinations to avoid conflicting generation requests."""

        if options.beam_size is not None and options.best_of is not None: raise ValueError("Conflict in options: 'beam_size' and 'best_of' cannot be specified simultaneously.")
        if options.temperature == 0 and options.best_of is not None: raise ValueError("Invalid configuration: 'best_of' works with sampling and requires temperature > 0.")
        if options.patience is not None and options.beam_size is None: raise ValueError("Invalid configuration: 'patience' requires 'beam_size' to be set for Beam Search.")
        if options.length_penalty is not None and not (0 <= options.length_penalty <= 1): raise ValueError(f"Invalid range: 'length_penalty' must be a float between 0 and 1, got {options.length_penalty}.")
        return options

    def _get_initial_tokens(self):
        """Assemble prefix prompts and structural start tags into a unified starting sequence context."""

        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (self.tokenizer.encode(" " + prefix.strip()) if isinstance(prefix, str) else prefix)
            if self.sample_len is not None: prefix_tokens = prefix_tokens[-(self.n_ctx // 2 - self.sample_len):]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt: 
            tokens = (
                [self.tokenizer.sot_prev] + 
                (self.tokenizer.encode(" " + prompt.strip()) if isinstance(prompt, str) else prompt)[-(self.n_ctx // 2 - 1) :] + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self):
        """Collect and group explicit vocabulary token IDs designated to be ignored during inference loops."""

        suppress_tokens = self.options.suppress_tokens
        if isinstance(suppress_tokens, str): suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0: suppress_tokens = [] 
        else: assert isinstance(suppress_tokens, list)

        suppress_tokens.extend([
            self.tokenizer.transcribe, 
            self.tokenizer.translate, 
            self.tokenizer.sot, 
            self.tokenizer.sot_prev, 
            self.tokenizer.sot_lm
        ])

        if self.tokenizer.no_speech is not None: suppress_tokens.append(self.tokenizer.no_speech)
        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel):
        """Process and sanitize raw acoustic signals, projecting inputs through the AudioEncoder if necessary."""

        if self.options.fp16: mel = mel.half()

        audio_features = mel if mel.shape[-2:] == (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state) else self.model.encoder(mel)
        if audio_features.dtype != (torch.float16 if self.options.fp16 else torch.float32): return TypeError("Mismatched tensor data types during calculation.")

        return audio_features

    def _detect_language(self, audio_features, tokens):
        """Execute on-the-fly automated identification tracking logic across input components."""

        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(audio_features, self.tokenizer)
            languages = [max(probs, key=probs.get) for probs in lang_probs]

            if self.options.language is None: tokens[:, self.sot_index + 1] = lang_tokens

        return languages, lang_probs

    def _main_loop(self, audio_features, tokens):
        """The core auto-regressive generation execution loop structure."""

        n_batch = tokens.shape[0]
        sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)
                # Capture probability scores designating silence/no-speech boundaries on step zero
                if (i == 0 and self.tokenizer.no_speech is not None):  
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                logits = logits[:, -1] # Isolate trailing predicted token values
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits.cpu() if config.device.startswith("ocl") else logits, tokens)

                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)
                if completed or tokens.shape[-1] > self.n_ctx: break
        finally:
            self.inference.cleanup_caching() # Remove active PyTorch computation hooks safely

        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    def run(self, mel):
        """
        Entry point running the end-to-end decoding orchestrations.

        Args:
            mel (torch.Tensor): Audio Spectrogram inputs.

        Returns:
            List[DecodingResult]: Collected list containing the structured results data outputs.
        """

        self.decoder.reset()
        tokenizer = self.tokenizer
        n_audio = mel.shape[0]

        audio_features = self._get_audio_features(mel)  
        tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id": 
            return [
                DecodingResult(
                    audio_features=features, 
                    language=language, 
                    language_probs=probs
                ) 
                for features, language, probs in zip(
                    audio_features, 
                    languages, 
                    language_probs
                )
            ]

        # Duplicate arrays to align with structural beam grouping counts safely
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # Slice indices down back to standard un-grouped batch layouts
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]

        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        # Strip structural pad tokens and resolve optimal paths
        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens = [[t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens]

        # Use ranker class to fetch the single best survival pathway index
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens = [t[i].tolist() for i, t in zip(selected, tokens)]

        fields = (
            [tokenizer.decode(t).strip() for t in tokens], 
            languages, 
            tokens, 
            audio_features, 
            [lp / (len(t) + 1) for t, lp in zip(tokens, [lp[i] for i, lp in zip(selected, sum_logprobs)])], 
            no_speech_probs
        )
        if len(set(map(len, fields))) != 1: raise RuntimeError(f"Data mapping error: All processed fields fields must share an identical length layout, but got lengths {[len(f) for f in fields]}.")

        # Build final object collection payloads
        return [
            DecodingResult(
                audio_features=features, 
                language=language, 
                tokens=tokens, 
                text=text, 
                avg_logprob=avg_logprob, 
                no_speech_prob=no_speech_prob, 
                temperature=self.options.temperature, 
                compression_ratio=compression_ratio(text)
            ) 
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)
        ]
    
class DecodingResult:
    """The structured data payload wrapping transcription text segments, token tracks, and validation scores."""

    def __init__(
        self, 
        audio_features, 
        language, 
        language_probs = None, 
        tokens = None, 
        text = "", 
        avg_logprob = np.nan, 
        no_speech_prob = np.nan, 
        temperature = np.nan, 
        compression_ratio = np.nan
    ):
        self.audio_features = audio_features
        self.language = language
        self.language_probs = language_probs if language_probs is not None else {}
        self.tokens = tokens if tokens is not None else []
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.temperature = temperature
        self.compression_ratio = compression_ratio

class Tokenizer:
    """
    A comprehensive tokenizer class for text encoding, decoding, and handling 
    special/control tokens for multilingual and multi-task speech models (similar to Whisper).
    """

    def __init__(
        self, 
        encoding_name, 
        num_languages = 2, 
        language = None, 
        task = None, 
        sot_sequence = ()
    ):
        """
        Initializes the Tokenizer with specific encoding settings, language, and task context.

        Args:
            encoding_name (str): The name of the tokenizer encoding pattern to load.
            num_languages (int, optional): Maximum number of languages allowed. Defaults to 2.
            language (str, optional): Target language for text generation. Defaults to None.
            task (str, optional): The task type ('transcribe' or 'translate'). Defaults to None.
            sot_sequence (tuple, optional): Initial start of sequence. Defaults to ().
        """

        # Fetch the core tokenizer based on name and allowed number of languages
        self.encoding = get_encoding(
            name=encoding_name, 
            num_languages=num_languages
        )

        self.num_languages = num_languages
        self.language = language
        self.task = task
        self.sot_sequence = sot_sequence 
        self.special_tokens = {}
        # Cache all special token string-to-ID mappings from the core encoding
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        # Construct the Start Of Transcript (SOT) token sequence dynamically
        sot = self.special_tokens["<|startoftranscript|>"]
        langs = tuple(LANGUAGES.keys())[: self.num_languages]
        sot_sequence = [sot]

        # 1. Append language token if a target language is specified
        if self.language is not None: 
            sot_sequence.append(
                sot + 1 + langs.index(self.language)
            )

        # 2. Append task token based on transcription or translation objective
        if self.task is not None: 
            sot_sequence.append(
                self.special_tokens[
                    "<|transcribe|>"
                ] if self.task == "transcribe" else self.special_tokens[
                    "<|translate|>"
                ]
            )

        # Finalize the SOT sequence as an immutable tuple
        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        """Encodes a string of text into a list of token IDs.

        Args:
            text (str): Input text string to be encoded.
            **kwargs: Extra parameters passed to the underlying encoder.
        
        Returns:
            list: List of integer token IDs.
        """
    
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        """Decodes a list of token IDs back into text, filtering out timestamp tokens.

        Args:
            token_ids (list): List of integer token IDs to decode.
            **kwargs: Extra parameters passed to the underlying decoder.
        
        Returns:
            str: Decoded string excluding timestamps.
        """

        # Exclude any tokens that are equal to or greater than the timestamp boundary
        return self.encoding.decode([
            t 
            for t in token_ids 
            if t < self.timestamp_begin
        ], **kwargs)

    def decode_with_timestamps(self, token_ids, **kwargs):
        """Decodes token IDs into a text string including raw timestamp tags.

        Args:
            token_ids (list): List of integer token IDs.
            **kwargs: Extra parameters passed to the underlying decoder.
        
        Returns:
            str: Full decoded string representation including all special tags.
        """

        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self):
        return self.encoding.eot_token

    @cached_property
    def transcribe(self):
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self):
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self):
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self):
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self):
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self):
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self):
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self):
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self):
        """int: Gets the active language token ID. Raises ValueError if language is unset."""

        if self.language is None: raise ValueError("Language is not specified for this tokenizer instance.")
        return self.to_language_token(self.language)

    def to_language_token(self, language):
        """
        Retrieves the explicit token ID for a given language code string.

        Args:
            language (str): The language identifier (e.g., 'en').

        Returns:
            int: The corresponding language token ID.

        Raises:
            KeyError: If the language code is not found in the special tokens dictionary.
        """

        if token := self.special_tokens.get(f"<|{language}|>", None): return token
        raise KeyError(f"Language token for '<|{language}|>' not found.")

    @cached_property
    def all_language_tokens(self):
        """tuple: A tuple containing all supported language token IDs up to `num_languages`."""

        result = []
        for token, token_id in self.special_tokens.items():
            # Extract language name from bracket format "<|en|>" -> "en"
            if token.strip("<|>") in LANGUAGES: result.append(token_id)

        return tuple(result)[: self.num_languages]

    @cached_property
    def all_language_codes(self):
        """tuple: A tuple containing string language codes derived from all valid language tokens."""

        return tuple(self.decode([_l]).strip("<|>") for _l in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self):
        """tuple: The baseline SOT sequence augmented with a trailing `<|notimestamps|>` token."""

        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @cached_property
    def non_speech_tokens(self):
        """
        tuple: Returns a sorted tuple of token IDs containing punctuation, musical notes, and other non-speech symbols.
        
        Useful for logit filtering / suppressing gibberish or background noises during inference.
        """

        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += ("<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split())

        # Isolated musical notation characters
        miscellaneous = set("♩♪♫♬♭♮♯")
        # Ensure all misc symbols fall inside specific unicode blocks
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # Pre-seed with spaces preceding common hyphen/apostrophe combinations
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        # Accumulate single token variations or versions prefixed with a whitespace character
        for symbol in symbols + list(miscellaneous):
            for tokens in [self.encoding.encode(symbol), self.encoding.encode(" " + symbol)]:
                if len(tokens) == 1 or symbol in miscellaneous: result.add(tokens[0])

        return tuple(sorted(result))

    def split_to_word_tokens(self, tokens):
        """
        Splits a list of token IDs into words based on the tokenizer instance's language.

        Args:
            tokens (list): List of token IDs to split.

        Returns:
            tuple: (list of string words, list of lists of token IDs matching each word)
        """

        # East Asian / Logographic languages do not use spaces as word boundaries
        if self.language in {"zh", "ja", "th", "lo", "my", "yue"}: return self.split_tokens_on_unicode(tokens)
        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(self, tokens):
        """
        Splits tokens incrementally by checking valid UTF-8 boundary reconstructions.

        Prevents broken Unicode multi-byte characters from being isolated during extraction.

        Args:
            tokens (list): List of token IDs.

        Returns:
            tuple: (words, word_tokens)
        """

        replacement_char = "\ufffd" # The standard Unicode Replacement Character used for decoding errors

        words, word_tokens, current_tokens = [], [], []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            # If there's no decoding anomaly (replacement char) OR if the original text truly 
            # had a replacement character at this spot, it's safe to separate into a new unit.
            if (
                replacement_char not in decoded or 
                self.decode_with_timestamps(tokens)[unicode_offset + decoded.index(replacement_char)] == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = [] # Clear buffer for next token sequence
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(self, tokens):
        """
        Group subwords together to construct individual whole words by splitting on space occurrences.

        Args:
            tokens (list): List of token IDs.

        Returns:
            tuple: (words, word_tokens)
        """

        # First isolate character/subword tokens securely via unicode boundary checks
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words, word_tokens = [], []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            # A subword signifies a completely new word structure if:
            # 1. It is a special / control token (>= EOT)
            # 2. It explicitly begins with a whitespace character " "
            # 3. It is a punctuation token
            # 4. Or if it's the very first token in the loop
            if (
                subword_tokens[0] >= self.eot
            ) or (
                subword.startswith(" ")
            ) or (
                subword.strip() in string.punctuation
            ) or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                # Otherwise, append/merge the subword to the previous word index
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens