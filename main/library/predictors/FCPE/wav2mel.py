import os
import sys
import torch

import numpy as np
import torch.nn.functional as F

from librosa.filters import mel
from torchaudio.transforms import Resample

sys.path.append(os.getcwd())

from main.library.backends import opencl
from main.library.predictors.FCPE.stft import STFT

def spawn_wav2mel(args, device = None):
    _type = args.mel.type
    if (str(_type).lower() == 'none') or (str(_type).lower() == 'default'): _type = 'default'
    elif str(_type).lower() == 'stft': _type = 'stft'
    wav2mel = Wav2MelModule(sr=args.mel.sr, n_mels=args.mel.num_mels, n_fft=args.mel.n_fft, win_size=args.mel.win_size, hop_length=args.mel.hop_size, fmin=args.mel.fmin, fmax=args.mel.fmax, clip_val=1e-05, mel_type=_type)
    
    return wav2mel.to(torch.device(device))

class HannWindow(torch.nn.Module):
    def __init__(self, win_size):
        super().__init__()
        self.register_buffer('window', torch.hann_window(win_size), persistent=False)

    def forward(self):
        return self.window

class MelModule(torch.nn.Module):
    def __init__(self, sr, n_mels, n_fft, win_size, hop_length, fmin = None, fmax = None, clip_val = 1e-5, out_stft = False):
        super().__init__()
        if fmin is None: fmin = 0
        if fmax is None: fmax = sr / 2
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.register_buffer('mel_basis', torch.tensor(mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)).float(), persistent=False)
        self.hann_window = torch.nn.ModuleDict()
        self.out_stft = out_stft

    @torch.no_grad()
    def __call__(self, y, key_shift = 0, speed = 1, center = False, no_cache_window = False):
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        clip_val = self.clip_val
        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))

        y = y.squeeze(-1)
        key_shift_key = str(key_shift)

        if not no_cache_window:
            if key_shift_key in self.hann_window: hann_window = self.hann_window[key_shift_key]
            else:
                hann_window = HannWindow(win_size_new).to(self.mel_basis.device)
                self.hann_window[key_shift_key] = hann_window

            hann_window_tensor = hann_window()
        else: hann_window_tensor = torch.hann_window(win_size_new).to(self.mel_basis.device)

        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.size(-1) - pad_left)

        mode = 'reflect' if pad_right < y.size(-1) else 'constant'
        pad = F.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)

        if str(y.device).startswith("ocl"):
            stft = opencl.STFT(filter_length=n_fft_new, hop_length=hop_length_new, win_length=win_size_new).to(y.device)
            spec = stft.transform(pad, 1e-9)
        else:
            spec = torch.stft(pad, n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=hann_window_tensor, center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
            spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)

        if key_shift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)

            if resize < size: spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * win_size / win_size_new

        spec = spec[:, :512, :] if self.out_stft else torch.matmul(self.mel_basis, spec)
        return torch.log(torch.clamp(spec, min=clip_val) * 1).transpose(-1, -2)

class Wav2MelModule(torch.nn.Module):
    def __init__(self, sr, n_mels, n_fft, win_size, hop_length, fmin = None, fmax = None, clip_val = 1e-5, mel_type="default"):
        super().__init__()
        if fmin is None: fmin = 0
        if fmax is None: fmax = sr / 2
        self.sampling_rate = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.register_buffer('tensor_device_marker', torch.tensor(1.0).float(), persistent=False)
        self.resample_kernel = torch.nn.ModuleDict()
        if mel_type == "default": self.mel_extractor = MelModule(sr, n_mels, n_fft, win_size, hop_length, fmin, fmax, clip_val, out_stft=False)
        elif mel_type == "stft": self.mel_extractor = MelModule(sr, n_mels, n_fft, win_size, hop_length, fmin, fmax, clip_val, out_stft=True)
        self.mel_type = mel_type

    @torch.no_grad()
    def __call__(self, audio, sample_rate, keyshift = 0, no_cache_window = False):
        if sample_rate == self.sampling_rate: audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                if len(self.resample_kernel) > 8: self.resample_kernel.clear()
                self.resample_kernel[key_str] = Resample(sample_rate, self.sampling_rate, lowpass_filter_width=128).to(self.tensor_device_marker.device)

            audio_res = self.resample_kernel[key_str](audio.squeeze(-1)).unsqueeze(-1)

        mel = self.mel_extractor(audio_res, keyshift, no_cache_window=no_cache_window)
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        if n_frames > int(mel.shape[1]): mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]): mel = mel[:, :n_frames, :]

        return mel 

class Wav2Mel:
    def __init__(self, device=None, dtype=torch.float32):
        self.sample_rate = 16000
        self.hop_size = 160
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.stft = STFT(16000, 128, 1024, 1024, 160, 0, 8000)
        self.resample_kernel = {}

    def extract_nvstft(self, audio, keyshift=0, train=False):
        return self.stft.get_mel(audio, keyshift=keyshift, train=train).transpose(1, 2)

    def extract_mel(self, audio, sample_rate, keyshift=0, train=False):
        audio = audio.to(self.dtype).to(self.device)
        if sample_rate == self.sample_rate: audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel: self.resample_kernel[key_str] = Resample(sample_rate, self.sample_rate, lowpass_filter_width=128)
            self.resample_kernel[key_str] = (self.resample_kernel[key_str].to(self.dtype).to(self.device))
            audio_res = self.resample_kernel[key_str](audio)

        mel = self.extract_nvstft(audio_res, keyshift=keyshift, train=train) 
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        mel = (torch.cat((mel, mel[:, -1:, :]), 1) if n_frames > int(mel.shape[1]) else mel)
        return mel[:, :n_frames, :] if n_frames < int(mel.shape[1]) else mel

    def __call__(self, audio, sample_rate, keyshift=0, train=False):
        return self.extract_mel(audio, sample_rate, keyshift=keyshift, train=train)