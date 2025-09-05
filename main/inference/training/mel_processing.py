import torch

import torch.nn.functional as F

from librosa.filters import mel as librosa_mel_fn

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return (torch.clamp(x, min=clip_val) * C).log()

def dynamic_range_decompression_torch(x, C=1):
    return x.exp() / C

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)

mel_basis, hann_window = {}, {}

def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    global hann_window

    wnsize_dtype_device = str(win_size) + "_" + str(y.dtype) + "_" + str(y.device)
    if wnsize_dtype_device not in hann_window: hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    pad = F.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect").squeeze(1)
    if str(y.device).startswith("ocl"): pad = pad.cpu()

    spec = torch.stft(pad, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device].to(pad.device), center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    spec = (spec.real.pow(2) + spec.imag.pow(2) + 1e-6).sqrt()

    return spec.to(y.device)

def spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    global mel_basis

    fmax_dtype_device = str(fmax) + "_" + str(spec.dtype) + "_" + str(spec.device)
    if fmax_dtype_device not in mel_basis: mel_basis[fmax_dtype_device] = torch.from_numpy(librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)).to(dtype=spec.dtype, device=spec.device)
    
    return spectral_normalize_torch(mel_basis[fmax_dtype_device].matmul(spec))

def mel_spectrogram_torch(y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False):
    return spec_to_mel_torch(spectrogram_torch(y, n_fft, hop_size, win_size, center), n_fft, num_mels, sample_rate, fmin, fmax)