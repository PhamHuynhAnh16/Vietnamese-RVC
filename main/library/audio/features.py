import torch

import torch.nn.functional as F

def frame(x, frame_length, hop_length, axis = -1):
    if x.shape[axis] < frame_length or hop_length < 1: raise ValueError

    axis = axis % x.ndim
    if axis != x.ndim - 1: x = x.movedim(axis, -1)

    xw = torch.as_strided(x, size=x.shape[:-1] + (1 + (x.shape[-1] - frame_length) // hop_length, frame_length), stride=x.stride()[:-1] + (hop_length * x.stride()[-1], x.stride()[-1]))
    if axis != x.ndim - 1: xw = xw.movedim(-2, axis)

    return xw

def rms(
    y,
    frame_length = 2048,
    hop_length = 512,
    center = True,
    pad_mode = "constant",
    dtype = torch.float32,
    device = "cpu"
):
    y = y.to(device=device, dtype=dtype) if torch.is_tensor(y) else torch.from_numpy(y.copy()).to(device=device, dtype=dtype)
    return frame(torch.nn.functional.pad(y, (frame_length // 2, frame_length // 2), mode=pad_mode) if center else y, frame_length=frame_length, hop_length=hop_length).square().mean(dim=-1, keepdim=True).sqrt().mT

def change_rms(source_audio, source_rate, target_audio, target_rate, rate, device):
    if not torch.is_tensor(target_audio): target_audio = torch.from_numpy(target_audio).to(device=device, dtype=torch.float32)

    rms2 = F.interpolate(
        rms(
            y=target_audio, 
            frame_length=target_rate // 2 * 2, 
            hop_length=target_rate // 2,
            device=device
        ).float().unsqueeze(0), 
        size=target_audio.shape[0], 
        mode="linear"
    ).squeeze()

    return target_audio * (
        F.interpolate(
            rms(
                y=source_audio, 
                frame_length=source_rate // 2 * 2, 
                hop_length=source_rate // 2,
                device=device
            ).float().unsqueeze(0), 
            size=target_audio.shape[0], 
            mode="linear"
        ).squeeze().pow(1 - rate) * rms2.maximum(torch.zeros_like(rms2) + 1e-6).pow(rate - 1)
    )

def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm="slaney", dtype=torch.float32, device=None):
    if fmax is None: fmax = float(sr) / 2
    n_mels = int(n_mels)

    weights = torch.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype, device=device)

    fftfreqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=device).to(dtype)
    mel_f = mel_to_hz(torch.linspace(hz_to_mel(fmin, htk=htk, dtype=dtype, device=device), hz_to_mel(fmax, htk=htk, dtype=dtype, device=device), n_mels + 2, dtype=dtype, device=device), htk=htk, dtype=dtype, device=device)

    fdiff = mel_f.diff()
    ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)

    lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
    weights = lower.minimum(ramps[2:] / fdiff[1:].unsqueeze(1)).clamp(min=0)

    if isinstance(norm, str):
        if norm == "slaney": weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])).unsqueeze(1)
        else: raise ValueError
    else: weights = normalize(weights, norm=norm, axis=-1)

    return weights

def hz_to_mel(frequencies, htk=False, dtype=torch.float32, device=None):
    frequencies = torch.as_tensor(frequencies, dtype=dtype, device=device)
    if htk: return 2595.0 * (1.0 + frequencies / 700.0).log10()

    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp

    logstep = torch.tensor(6.4, dtype=dtype, device=device).log() / 27.0
    return torch.where(frequencies >= min_log_hz, min_log_mel + (frequencies / min_log_hz).log() / logstep, mels)

def mel_to_hz(mels, htk=False, dtype=torch.float32, device=None):
    mels = torch.as_tensor(mels, dtype=dtype, device=device)
    if htk: return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3

    freqs = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = torch.tensor(6.4, dtype=dtype, device=device).log() / 27.0

    return torch.where(mels >= min_log_mel, min_log_hz * (logstep * (mels - min_log_mel)).exp(), freqs)

def normalize(S, norm=torch.inf, axis=0, threshold=None, fill=None):
    S = torch.as_tensor(S)

    if threshold is None:
        x = torch.as_tensor(S)
        threshold = torch.finfo(x.dtype if x.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16) else torch.float32).tiny
    elif threshold <= 0: raise ValueError

    if fill not in [None, False, True] or not torch.all(torch.isfinite(S)): raise ValueError

    mag = S.abs().to(torch.float32)
    fill_norm = 1

    if norm is None: return S
    elif norm == torch.inf: length = mag.amax(dim=axis, keepdim=True)
    elif norm == -torch.inf: length = mag.amin(dim=axis, keepdim=True)
    elif norm == 0:
        if fill is True: raise ValueError
        length = (mag > 0).sum(dim=axis, keepdim=True, dtype=mag.dtype)
    elif isinstance(norm, (int, float)) and norm > 0:
        length = (mag ** norm).sum(dim=axis, keepdim=True) ** (1.0 / norm)
        fill_norm = (mag.numel() ** (-1.0 / norm)) if axis is None else (mag.shape[axis] ** (-1.0 / norm))
    else: raise ValueError

    small_idx = length < threshold

    if fill is None:
        length = length.clone()
        length[small_idx] = 1.0
        Snorm = S / length
    elif fill:
        length = length.clone()
        length[small_idx] = torch.nan
        Snorm = torch.nan_to_num(S / length, nan=fill_norm)
    else:
        length = length.clone()
        length[small_idx] = torch.inf
        Snorm = S / length

    return Snorm