import torch

CENTS_PER_BIN, PITCH_BINS = 5, 1440

def bins_to_frequency(bins):
    if str(bins.device).startswith("ocl"): bins = bins.to(torch.float32)

    return cents_to_frequency(bins_to_cents(bins))

def cents_to_frequency(cents):
    return 60 * 2 ** (cents / 1200)

def bins_to_cents(bins):
    return CENTS_PER_BIN * bins

def frequency_to_bins(frequency, quantize_fn=torch.floor):
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)

def cents_to_bins(cents, quantize_fn=torch.floor):
    bins = quantize_fn(cents / CENTS_PER_BIN).long()
    bins[bins < 0] = 0
    bins[bins >= PITCH_BINS] = PITCH_BINS - 1

    return bins

def frequency_to_cents(frequency):
    return 1200 * torch.log2(frequency / 60)

def seconds_to_samples(seconds, sample_rate):
    return seconds * sample_rate