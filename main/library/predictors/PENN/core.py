import math
import torch

import torch.nn.functional as F

PITCH_BINS, CENTS_PER_BIN, OCTAVE = 1440, 5, 1200

def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """
    Converts continuous frequencies in Hertz to discrete pitch bin indices.

    Args:
        frequency (torch.Tensor): Tensor containing frequency values in Hz.
        quantize_fn (Callable, optional): Math rounding strategy wrapper. Defaults to torch.floor.

    Returns:
        torch.Tensor: Long tensor containing bounded bin indices.
    """

    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)

def cents_to_bins(cents, quantize_fn=torch.floor):
    """
    Converts logarithmic cent values into discrete index bins bounded within valid ranges.

    Args:
        cents (torch.Tensor): Tensor containing logarithmic values in cents.
        quantize_fn (Callable, optional): Rounding function (e.g., floor or ceil). Defaults to torch.floor.

    Returns:
        torch.Tensor: Clamped tensor array of type torch.long.
    """

    # 1. Map continuous cents space down into discrete bins
    bins = quantize_fn(cents / CENTS_PER_BIN).long()
    # 2. Pin indices strictly inside valid filterbank limits to prevent segmentation faults
    bins[bins < 0] = 0
    bins[bins >= PITCH_BINS] = PITCH_BINS - 1
    return bins

def cents_to_frequency(cents):
    """
    Converts pitch values from logarithmic cents back to continuous Hertz.

    Calculated using the base-frequency equation.
    """

    return 31 * 2 ** (cents / OCTAVE)

def bins_to_cents(bins):
    """Converts discrete bin indices back into continuous logarithmic cents."""

    return CENTS_PER_BIN * bins

def frequency_to_cents(frequency):
    """
    Converts continuous frequency values in Hertz to logarithmic cents.

    Calculated relative to a 31 Hz baseline reference pitch.
    """

    return OCTAVE * (frequency / 31).log2()

def interpolate(pitch, periodicity, value):
    """
    Fills unvoiced gaps by performing linear interpolation over log-frequency fields.

    Forces boundary edges to match the nearest voiced frame to prevent unvoiced leaking.

    Args:
        pitch (torch.Tensor): Extracted continuous pitch tracks in Hertz.
        periodicity (torch.Tensor): Confidence or periodicity values matching the pitch frames.
        value (float): Confidence threshold boundary used to identify voiced segments.

    Returns:
        torch.Tensor: Interpolated continuous pitch curve array in Hertz.
    """

    # 1. Create a boolean mask identifying reliable voiced frames
    voiced = periodicity > value
    if not voiced.any(): return pitch

    # 2. Linear operations require tracking across flat logarithmic spaces
    pitch = pitch.log2()
    # 3. Pin boundary frames to the nearest valid voiced endpoints to ensure stable extrapolation
    pitch[..., 0] = pitch[voiced][..., 0]
    pitch[..., -1] = pitch[voiced][..., -1]
    voiced[..., 0] = True
    voiced[..., -1] = True

    # 4. Interpolate unvoiced indices using coordinates from valid voiced regions
    pitch[~voiced] = _interpolate(
        torch.where(~voiced[0])[0][None], 
        torch.where(voiced[0])[0][None], 
        pitch[voiced][None]
    )

    # 5. Project the values back into standard linear Hertz space
    return 2 ** pitch

def _interpolate(x, xp, fp):
    """Helper function that executes optimized 1D linear matrix interpolation across tensors."""

    if xp.shape[-1] == 0: return x

    if xp.shape[-1] == 1: 
        return torch.full(
            x.shape, 
            fp.squeeze(), 
            device=fp.device, 
            dtype=fp.dtype
        )

    # 1. Calculate the slope (m) and intercept (b) segments
    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])
    b = fp[:, :-1] - (m.mul(xp[:, :-1]))

    # 2. Find the target segment index bounds using comparative operations
    indicies = x[:, :, None].ge(xp[:, None, :]).sum(-1) - 1
    indicies = indicies.clamp(0, m.shape[-1] - 1)

    # 3. Build a sequence line index tracker for safe dynamic assignment
    line_idx = torch.linspace(
        0, 
        indicies.shape[0], 
        1, 
        device=indicies.device
    ).to(torch.long).expand(indicies.shape)

    # 4. Evaluate the linear equations simultaneously
    return m[line_idx, indicies].mul(x) + b[line_idx, indicies]

def entropy(logits):
    """
    Calculates normalized Shannon entropy across pitch logit distributions.
    Can be used as a proxy for pitch tracking uncertainty or unvoiced periodicity.

    Args:
        logits (torch.Tensor): Unnormalized network predictions.

    Returns:
        torch.Tensor: Normalized periodicity values bound between [0.0, 1.0].
    """

    # 1. Map logit values into valid probabilities
    distribution = F.softmax(logits, dim=1)
    # 2. Compute classic normalized negative entropy values
    return (
        1 + 1 / math.log(PITCH_BINS) * (
            distribution * (distribution + 1e-7).log()
        ).sum(dim=1)
    )