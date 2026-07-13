import torch

import torch.nn.functional as F

def frame(x, frame_length, hop_length, axis = -1):
    """
    Slice a tensor into overlapping frames using stride tricks.

    Args:
        x (torch.Tensor): Input tensor containing the signal data.
        frame_length (int): Number of samples in each frame.
        hop_length (int): Number of samples between consecutive frames.
        axis (int, optional): Axis along which framing is applied. Defaults to the last axis (-1).
    """

    if x.shape[axis] < frame_length or hop_length < 1: raise ValueError("Target axis length must be >= frame_length and hop_length >= 1")
    axis = axis % x.ndim

    # Move target axis to the last dimension for easier processing
    if axis != x.ndim - 1: x = x.movedim(axis, -1)

    # Compute output shape and stride configuration and Create framed tensor without copying memory
    xw = torch.as_strided(x, size=x.shape[:-1] + (1 + (x.shape[-1] - frame_length) // hop_length, frame_length), stride=x.stride()[:-1] + (hop_length * x.stride()[-1], x.stride()[-1]))

    # Restore original axis order if needed
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
    """
    Compute the Root Mean Square (RMS) energy of an audio signal.

    Args:
        y (torch.Tensor or numpy.ndarray): Input audio signal.
        frame_length (int, optional): Length of each analysis frame. Defaults to 2048.
        hop_length (int, optional): Number of samples between frames. Defaults to 512.
        center (bool, optional): If True, pad the signal so frames are centered. Defaults to True.
        pad_mode (str, optional): Padding mode used when centering. Defaults to "constant".
        dtype (torch.dtype, optional): Tensor data type. Defaults to torch.float32.
        device (str, optional): Target device for computation. Defaults to "cpu".
    """
    # Convert input to tensor and move to target device/dtype
    y = y.to(device=device, dtype=dtype) if torch.is_tensor(y) else torch.from_numpy(y.copy()).to(device=device, dtype=dtype)

    return frame(
        torch.nn.functional.pad(y, (frame_length // 2, frame_length // 2), mode=pad_mode) if center else y, # Pad signal so frames are centered
        frame_length=frame_length, 
        hop_length=hop_length
    ).square().mean(dim=-1, keepdim=True).sqrt().mT # Compute mean square energy per frame and convert power to RMS

def change_rms(
    source_audio, 
    source_rate, 
    target_audio, 
    target_rate, 
    rate = 1.0, 
    device = "cpu"
):
    """
    Adjust the RMS level of target_audio to match the RMS of source_audio, with a given blending rate.

    Args:
        source_audio: The source audio signal as a Torch Tensor.
        source_rate: The sampling rate of the source audio.
        target_audio: The target audio signal to adjust.
        target_rate: The sampling rate of the target audio.
        rate: The blending rate between the source and target RMS levels.
        dtype (torch.dtype, optional): Tensor data type. Defaults to torch.float32.
        device (str, optional): Target device for computation. Defaults to "cpu".
    """

    # Convert NumPy input to a float tensor if necessary.
    if not torch.is_tensor(target_audio): target_audio = torch.from_numpy(target_audio).to(device=device, dtype=torch.float32)

    # Compute and upsample the target RMS envelope.
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

    # Scale the target audio using the blended RMS ratio.
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

def mel(
    sr, 
    n_fft, 
    n_mels = 128, 
    fmin = 0.0, 
    fmax = None, 
    htk = False, 
    norm = "slaney", 
    dtype = torch.float32, 
    device = None
):
    """
    This function is used to generate a tensor mel directly, instead of a array mel.
    Construct a Mel filter bank.

    Args:
        sr: Sample rate.
        n_fft: FFT size.
        n_mels: Number of Mel bands.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz). Defaults to Nyquist.
        htk: Use the HTK Mel scale.
        norm: Filter normalization method ("slaney" or numeric norm).
        dtype: Tensor dtype.
        device: Target device.

    Returns:
        torch.Tensor: Mel filter bank with shape (n_mels, n_fft // 2 + 1).
    """

    if fmax is None: fmax = float(sr) / 2
    n_mels = int(n_mels)

    weights = torch.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype, device=device)

    # FFT bin center frequencies.
    fftfreqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=device).to(dtype)

    # Mel band edge frequencies.
    mel_f = mel_to_hz(
        torch.linspace(
            hz_to_mel(fmin, htk=htk, dtype=dtype, device=device), 
            hz_to_mel(fmax, htk=htk, dtype=dtype, device=device), 
            n_mels + 2, 
            dtype=dtype, 
            device=device
        ), 
        htk=htk, 
        dtype=dtype, 
        device=device
    )

    # Compute triangular filter slopes.
    fdiff = mel_f.diff()
    ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)

    lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
    weights = lower.minimum(ramps[2:] / fdiff[1:].unsqueeze(1)).clamp(min=0)

    # Normalize filter areas if requested.
    if isinstance(norm, str):
        if norm == "slaney": weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])).unsqueeze(1)
        else: raise ValueError(f"Unsupported normalization mode: {norm!r}. Expected 'slaney' or a numeric norm.")
    else: weights = normalize(weights, norm=norm, axis=-1)

    return weights

def hz_to_mel(frequencies, htk=False, dtype=torch.float32, device=None):
    """
    Convert frequencies from Hertz to the Mel scale.

    Args:
        frequencies: Frequency values in Hz.
        htk: Use the HTK conversion formula.
        dtype: Tensor dtype.
        device: Target device.

    Returns:
        torch.Tensor: Mel values.
    """

    frequencies = torch.as_tensor(frequencies, dtype=dtype, device=device)
    # HTK Mel conversion.
    if htk: return 2595.0 * (1.0 + frequencies / 700.0).log10()

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0 # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp # same (Mels)

    logstep = torch.tensor(6.4, dtype=dtype, device=device).log() / 27.0 # step size for log region

    return torch.where(frequencies >= min_log_hz, min_log_mel + (frequencies / min_log_hz).log() / logstep, mels)

def mel_to_hz(mels, htk=False, dtype=torch.float32, device=None):
    """
    Convert Mel values back to Hertz.

    Args:
        mels: Mel values.
        htk: Use the HTK conversion formula.
        dtype: Tensor dtype.
        device: Target device.

    Returns:
        torch.Tensor: Frequencies in Hz.
    """

    mels = torch.as_tensor(mels, dtype=dtype, device=device)
    if htk: return 700.0 * (10.0 ** (mels / 2595.0) - 1.0) # HTK inverse conversion.

    # Fill in the linear scale

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale

    min_log_hz = 1000.0 # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp # same (Mels)
    logstep = torch.tensor(6.4, dtype=dtype, device=device).log() / 27.0 # step size for log region

    return torch.where(mels >= min_log_mel, min_log_hz * (logstep * (mels - min_log_mel)).exp(), freqs)

def normalize(S, norm=torch.inf, axis=0, threshold=None, fill=None):
    """
    Normalize a tensor along a specified axis.

    Supports L∞, L-∞, L0, and arbitrary positive Lp norms.

    Args:
        S: Input tensor.
        norm: Normalization type.
        axis: Axis along which to normalize.
        threshold: Minimum norm before considering the vector too small.
        fill: Handling strategy for small norms.
            None  -> leave unchanged.
            False -> output zeros.
            True  -> fill with a uniform normalized vector.

    Returns:
        torch.Tensor: Normalized tensor.
    """

    S = torch.as_tensor(S)

    # Determine numerical threshold.
    if threshold is None:
        x = torch.as_tensor(S)
        threshold = torch.finfo(x.dtype if x.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16) else torch.float32).tiny
    elif threshold <= 0: raise ValueError("threshold must be greater than 0.")

    if fill not in [None, False, True]: raise ValueError("fill must be one of: None, False, or True.")
    if not torch.all(torch.isfinite(S)): raise ValueError("Input tensor contains NaN or Inf values.")

    # All norms only depend on magnitude, let's do that first
    mag = S.abs().to(torch.float32)
    # For max/min norms, filling with 1 works
    fill_norm = 1

    # Compute vector norm.
    if norm is None: return S
    elif norm == torch.inf: length = mag.amax(dim=axis, keepdim=True)
    elif norm == -torch.inf: length = mag.amin(dim=axis, keepdim=True)
    elif norm == 0:
        if fill is True: raise ValueError("fill=True is not supported when norm=0.")
        length = (mag > 0).sum(dim=axis, keepdim=True, dtype=mag.dtype)
    elif isinstance(norm, (int, float)) and norm > 0:
        length = (mag ** norm).sum(dim=axis, keepdim=True) ** (1.0 / norm)
        fill_norm = (mag.numel() ** (-1.0 / norm)) if axis is None else (mag.shape[axis] ** (-1.0 / norm))
    else: raise ValueError(f"Unsupported norm: {norm!r}. Expected None, ±torch.inf, 0, or a positive numeric value.")

    # Detect vectors with extremely small norms.
    small_idx = length < threshold

    if fill is None:
        # Leave small indices un-normalized
        length = length.clone()
        length[small_idx] = 1.0
        Snorm = S / length
    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length = length.clone()
        length[small_idx] = torch.nan
        Snorm = torch.nan_to_num(S / length, nan=fill_norm)
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length = length.clone()
        length[small_idx] = torch.inf
        Snorm = S / length

    return Snorm