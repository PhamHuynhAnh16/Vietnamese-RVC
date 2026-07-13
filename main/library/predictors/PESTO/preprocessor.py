import torch

import numpy as np

from scipy.signal import get_window

class Preprocessor(torch.nn.Module):
    """
    PESTO feature engineering preprocessor.
    
    Transforms streaming raw audio waveforms into log-magnitude Harmonic Constant-Q 
    Transform (HCQT) representations for pitch decoding.
    """

    def __init__(
        self, 
        hop_size, 
        sampling_rate = None, 
        **hcqt_kwargs
    ):
        """
        Initializes internal processing configurations.

        Args:
            hop_size (float): Frame step increment in milliseconds.
            sampling_rate (int, optional): Audio sampling rate frequency. Defaults to None.
            **hcqt_kwargs: Arbitrary parameters passed down to the HarmonicCQT module.
        """

        super(Preprocessor, self).__init__()
        self.hcqt_sr = None
        self.hcqt_kernels = None
        self.hop_size = hop_size
        self.hcqt_kwargs = hcqt_kwargs
        self.to_log = ToLogMagnitude()
        # Use a non-persistent reference buffer to infer active accelerator devices safely
        self.register_buffer("_device", torch.zeros(()), persistent=False)
        if sampling_rate is not None:
            self.hcqt_sr = sampling_rate
            self._reset_hcqt_kernels()

    def forward(self, x, sr = None):
        """
        Transforms multi-channel temporal audio waves to structured log-magnitude matrices.

        Args:
            x (torch.Tensor): Waveform audio.
            sr (int, optional): Explicit runtime override samplerate. Defaults to None.

        Returns:
            torch.Tensor: Log-magnitude spectral grid maps.
        """

        # Re-arrange sequence indices to line up features
        return self.to_log(self.hcqt(x, sr=sr).permute(0, 3, 1, 2, 4))

    def hcqt(self, audio, sr = None) :
        """Evaluates raw unscaled complex multi-harmonic transform filterbanks."""

        if sr is not None and sr != self.hcqt_sr:
            self.hcqt_sr = sr
            self._reset_hcqt_kernels()

        return self.hcqt_kernels(audio)

    def _reset_hcqt_kernels(self):
        """Re-compiles the internal HarmonicCQT kernel weights when sample rate changes occur."""

        self.hcqt_kernels = HarmonicCQT(
            sr=self.hcqt_sr, 
            hop_length=int(self.hop_size * self.hcqt_sr / 1000 + 0.5), # Calculate discrete hop length size samples from continuous millisecond trackers
            **self.hcqt_kwargs
        ).to(self._device.device)

class ToLogMagnitude(torch.nn.Module):
    """Maps continuous complex fields into safe base-10 decibel scale spaces."""

    def __init__(self):
        super(ToLogMagnitude, self).__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x):
        """Transforms raw energy values into log magnitude footprints."""

        # 1. Resolve complex multi-component dimensions if present
        x = (x[..., 0] ** 2 + x[..., 1] ** 2).sqrt() if x.shape[-1] == 2 else x.abs()
        # 2. Rescale inputs to standard decibels safely: dB = 20 * log10(magnitude)
        x.clamp_(min=self.eps).log10_().mul_(20)

        return x
    
class HarmonicCQT(torch.nn.Module):
    """Orchestrates an array of parallel Constant-Q Transforms tracking multiple harmonics."""

    def __init__(
        self, 
        harmonics, 
        sr = 22050, 
        hop_length = 512, 
        fmin = 32.7, 
        fmax = None, 
        bins_per_semitone = 1, 
        n_bins = 84, 
        center_bins = True, 
        gamma = 0, 
        center = True, 
        streaming = False, 
        mirror = 0, 
        max_batch_size = 1
    ):
        super(HarmonicCQT, self).__init__()
        # Center-align bin configurations by adjusting the base reference frequency if requested
        if center_bins: fmin = fmin / 2 ** ((bins_per_semitone - 1) / (24 * bins_per_semitone))
        # Construct separate parallel scaling filter kernels across every specified integer harmonic fraction
        self.cqt_kernels = torch.nn.ModuleList([
            CQT(
                sr=sr, 
                hop_length=hop_length, 
                fmin=h * fmin, 
                fmax=fmax, 
                n_bins=n_bins, 
                bins_per_octave=12 * bins_per_semitone, 
                gamma=gamma, 
                center=center, 
                streaming=streaming, 
                mirror=mirror, 
                max_batch_size=max_batch_size, 
                output_format="Complex"
            ) 
            for h in harmonics
        ])

    def forward(self, audio_waveforms):
        """Evaluates and stacks parallel multi-harmonic transform outputs."""

        return torch.stack([cqt(audio_waveforms) for cqt in self.cqt_kernels], dim=1)
    
class BaseCQT(torch.nn.Module):
    """Base mathematical foundational layer for execution of Constant-Q Transforms."""

    def __init__(
        self, 
        sr=22050, 
        hop_length=512, 
        fmin=32.70, 
        fmax=None, 
        n_bins=84, 
        bins_per_octave=12, 
        gamma=0, 
        filter_scale=1, 
        norm=1, 
        window="hann", 
        center = True, 
        trainable=False, 
        output_format="Magnitude"
    ):
        super(BaseCQT, self).__init__()
        self.trainable = trainable
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.center = center
        self.output_format = output_format
        # 1. Synthesize static filter coefficients via NumPy execution blocks
        cqt_kernels, self.kernel_width, lengths, freqs = self.create_cqt_kernels(
            # Calculate the bounding quality factor Q parameter constants
            float(filter_scale) / (2 ** (1 / bins_per_octave) - 1), 
            sr, 
            fmin, 
            n_bins, 
            bins_per_octave, 
            norm, window, 
            fmax, 
            gamma=gamma
        )
        # 2. Cache spatial scale factor matrices on the local device graph
        self.sqrt_lengths = lengths.sqrt_().unsqueeze_(-1)
        self.frequencies = freqs
        self.cqt_kernels = torch.from_numpy(cqt_kernels).unsqueeze(1)
    
    def create_cqt_kernels(self, Q, fs, fmin, n_bins=84, bins_per_octave=12, norm=1, window="hann", fmax=None, topbin_check=True, gamma=0):
        """Synthesizes time-domain complex tracking kernel arrays based on spacing frequencies."""

        if (fmax != None) and (n_bins == None):
            n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
            freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
        elif (fmax == None) and (n_bins != None): freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
        else:
            n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
            freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))

        if np.max(freqs) > fs / 2 and topbin_check == True: raise ValueError("Calculated frequencies exceed Nyquist limits.")

        # Compute varying geometric temporal lengths across discrete tracking filters
        lengths = np.ceil(Q * fs / (freqs + gamma / (2.0 ** (1.0 / bins_per_octave) - 1.0)))
        fftLen = int(2 ** (np.ceil(np.log2(int(max(lengths))))))
        tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)

        for k in range(0, int(n_bins)):
            l = lengths[k]
            start = (int(np.ceil(fftLen / 2.0 - l / 2.0)) - 1) if l % 2 == 1 else int(np.ceil(fftLen / 2.0 - l / 2.0))
            N = int(l)

            if isinstance(window, str):
                sig = get_window(window, N, fftbins=True)
            elif isinstance(window, tuple):
                if window[0] == "gaussian":
                    assert window[1] >= 0
                    sig = get_window(("gaussian", np.floor(-N / 2 / np.sqrt(-2 * np.log(10 ** (-window[1] / 20))))), N, fftbins=True)
            else: raise TypeError("Unsupported window data structure type.")

            # Modulate base windows onto complex exponential frequency carriers
            sig = sig * np.exp(np.r_[-l // 2: l // 2] * 1j * 2 * np.pi * freqs[k] / fs) / l

            if norm:
                tempKernel[k, start: start + int(l)] = sig / np.linalg.norm(sig, norm)
            else:
                tempKernel[k, start: start + int(l)] = sig

        return tempKernel, fftLen, torch.tensor(lengths).float(), freqs

    @torch.no_grad()
    def init_weights(self):
        """Converts raw complex arrays to parallel multichannel 1D Convolution parameters."""

        self.conv.weight.copy_(torch.cat((self.cqt_kernels.real, -self.cqt_kernels.imag), dim=0))
        self.conv.weight.requires_grad = self.trainable

    def forward(self, x, output_format=None, normalization_type="librosa"):
        """Processes input audio streams through the mapped multi-channel kernel weight filters."""

        output_format = output_format or self.output_format
        x = self.broadcast_dim(x)
        cqt = self.conv(x).view(x.size(0), 2, self.n_bins, -1)

        if normalization_type == "librosa": cqt *= self.sqrt_lengths.to(cqt.device)
        elif normalization_type == "convolutional": pass
        elif normalization_type == "wrap": cqt *= 2
        else: raise ValueError(f"Unknown scaling normalization type configuration: {normalization_type}")

        if output_format == "Magnitude": return cqt.pow(2).sum(-3).add(1e-8 if self.trainable else 0).sqrt()
        if output_format == "Complex": return cqt.permute(0, 2, 3, 1)

        cqt_real, cqt_imag = cqt.split(self.n_bins, dim=-2)
        if output_format == "Phase": return torch.stack((cqt_imag.atan2(cqt_real).cos(), cqt_imag.atan2(cqt_real).sin()), -1)

        raise ValueError(f"Unsupported target output format schema choice: {output_format}")

    def broadcast_dim(self, x):
        """Normalizes audio dimensions to maintain consistent 3D sequence arrays."""

        if x.dim() == 2: x = x[:, None, :]
        elif x.dim() == 1: x = x[None, None, :]
        elif x.dim() == 3: pass
        else: raise ValueError(f"Invalid structural shape dimension array limits: {x.dim()}")

        return x

class RegularCQT(BaseCQT):
    """Standard execution pass of the Constant-Q Transform tracking full static offline boundaries."""

    def __init__(self, *args, pad_mode="reflect", **kwargs):
        super().__init__(*args, **kwargs)
        padding = self.kernel_width // 2 if self.center else 0
        self.conv = torch.nn.Conv1d(
            1, 
            2 * self.n_bins, 
            kernel_size=self.kernel_width, 
            stride=self.hop_length, 
            padding=padding, 
            padding_mode=pad_mode, 
            bias=False
        )
        self.init_weights()

class StreamingCQT(BaseCQT):
    """Online streaming implementation of the CQT layer tracking state caches over frame steps."""

    def __init__(
        self, 
        *args, 
        mirror = 0, 
        max_batch_size = 1, 
        **kwargs
    ):
        super(StreamingCQT, self).__init__(*args, **kwargs)
        if self.center:
            mirrored_samples = int(mirror * (self.kernel_width - self.hop_length) / 2)
            padding = self.kernel_width - self.hop_length - mirrored_samples
        else:
            mirrored_samples = 0
            padding = 0

        self.conv = CachedConv1d(
            1, 
            2 * self.n_bins, 
            kernel_size=self.kernel_width, 
            stride=self.hop_length, 
            padding=padding, 
            mirror=mirrored_samples, 
            max_batch_size=max_batch_size, 
            bias=False
        )

        self.init_weights()

class CQT:
    """Factory design router class creating standard offline or continuous online streaming variants."""

    regular_only_kwargs = ["pad_mode"]
    streaming_only_kwargs = ["mirror", "max_batch_size"]

    def __new__(cls, *args, **kwargs):
        streaming = kwargs.pop("streaming", False)

        if streaming:
            for kwarg in cls.regular_only_kwargs:
                kwargs.pop(kwarg, None)

            return StreamingCQT(*args, **kwargs)

        for kwarg in cls.streaming_only_kwargs:
            kwargs.pop(kwarg, None)

        return RegularCQT(*args, **kwargs)
    
class CachedConv1d(torch.nn.Conv1d):
    """Convolution layer containing a persistent sliding window ring-buffer cache for real-time tracking."""

    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        kwargs["padding"] = 0
        super(CachedConv1d, self).__init__(*args, **kwargs)
        padding = kwargs.get("padding", 0)
        max_batch_size = kwargs.pop("max_batch_size", 1)
        mirror = kwargs.pop("mirror", 0)
        mirror_fn = kwargs.pop("mirror_fn", "zeros")
        cumulative_delay = kwargs.pop("cumulative_delay", 0)

        if isinstance(padding, int): r_pad = padding
        elif isinstance(padding, list) or isinstance(padding, tuple):
            r_pad = padding[1]
            padding = padding[0] + padding[1]
        else: raise TypeError("Invalid type configuration passed inside spatial padding trackers.")

        s = self.stride[0]
        cd = cumulative_delay

        self.cumulative_delay = (r_pad + ((s - ((r_pad + cd) % s)) % s) + cd) // s
        # Instantiate continuous history tracking padding caches
        self.cache = CachedPadding1d(padding, max_batch_size=max_batch_size)

        if mirror == 0:
            mirroring_fn = torch.nn.Identity
        elif mirror_fn == "reflection":
            mirroring_fn = torch.nn.ReflectionPad1d
        elif mirror_fn == "zeros":
            mirroring_fn = torch.nn.ZeroPad1d
        elif mirror_fn == "refill":
            mirroring_fn = RefillPad1d
        else:
            mirroring_fn = torch.nn.Identity

        self.mirror = mirroring_fn((0, mirror))

    def forward(self, x):
        """Applies spatial transforms across sequential historical feature blocks."""

        return super(CachedConv1d, self).forward(self.mirror(self.cache(x)))
    
class RefillPad1d(torch.nn.Module):
    """Pads tracking segments by duplicating the final sequence sample boundaries on the right."""

    def __init__(
        self, 
        padding
    ):
        super(RefillPad1d, self).__init__()
        self.right_padding = padding[1]

    def forward(self, x):
        """Applies boundary duplication across multi-channel tracking audio tensors."""

        return torch.cat((x, x[..., -self.right_padding:]), dim=-1)
    
class CachedPadding1d(torch.nn.Module):
    """Manages overlapping boundaries across consecutive blocks during real-time streaming."""

    def __init__(
        self, 
        padding, 
        max_batch_size = 1, 
        crop=False
    ):
        super().__init__()
        self.padding = padding
        self.max_batch_size = max_batch_size
        self.crop = crop
        self.init_cache()

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self):
        """Initializes un-tracked baseline zeros mapping arrays inside memory configurations."""

        self.register_buffer("pad", torch.zeros(self.max_batch_size, 1, self.padding), persistent=False)

    def forward(self, x):
        """Appends history cache buffers onto newly received audio packages."""

        bs = x.size(0)
        if self.padding:
            # Prepend historical boundary context and refresh the tracking cache
            x = torch.cat((self.pad[:bs], x), -1)
            self.pad[:bs].copy_(x[..., -self.padding:])

        return x