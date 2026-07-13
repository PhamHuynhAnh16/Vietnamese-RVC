import os
import sys
import torch

from torch.nn.functional import conv1d, conv2d

sys.path.append(os.getcwd())

from main.app.variables import config

@torch.no_grad()
def temperature_sigmoid(x, x0, temp_coeff):
    """
    Computes a temperature-scaled sigmoid function for soft-thresholding.

    Args:
        x (Tensor): Input tensor (activation values).
        x0 (float/Tensor): Threshold center offset (inflection point).
        temp_coeff (float): Scaling factor controlling the slope steepness.

    Returns:
        Tensor: Smooth sigmoidal mask values bounded between 0.0 and 1.0.
    """

    # Scaling by temperature controls mask softness; prevents hard binary clipping artifacts
    return ((x - x0) / temp_coeff).sigmoid()

@torch.no_grad()
def linspace(start, stop, num = 50, endpoint = True, **kwargs):
    """
    Generates a linear sequence of evenly spaced values over a specified interval.

    Args:
        start (float): The starting value of the sequence.
        stop (float): The end value of the sequence.
        num (int): Number of samples to generate. Defaults to 50.
        endpoint (bool): If True, includes the stop value. Defaults to True.
        **kwargs: Additional tensor configuration options (e.g., dtype, device).

    Returns:
        Tensor: 1D linear spacing tensor.
    """

    return (
        torch.linspace(
            start, 
            stop, 
            num, 
            **kwargs
        )
    ) if endpoint else (
        torch.linspace( # If endpoint is excluded, generate num + 1 elements and drop the trailing bound element
            start, 
            stop, 
            num + 1, 
            **kwargs
        )[:-1]
    )

@torch.no_grad()
def amp_to_db(x, eps=torch.finfo(torch.float32).eps, top_db=40):
    """
    Converts a linear amplitude spectrogram to Decibel (dB) scale.

    Args:
        x (Tensor): Magnitude spectrogram tensor.
        eps (float): Small epsilon floor value to prevent log10(0) domain errors.
        top_db (float): Dynamic range threshold in dB relative to the peak magnitude.

    Returns:
        Tensor: Logarithmic scale dB representation capped at the dynamic floor.
    """

    # Compute 20 * log10(Amplitude) with an epsilon floor protection
    x_db = 20 * (x + eps).log10()

    # Apply dynamic range clamping based on the maximum global peak value along the last axis
    return x_db.max(
        (x_db.max(-1).values - top_db).unsqueeze(-1)
    )

class TorchGate(torch.nn.Module):
    """A PyTorch-native Audio Spectral Gate for stationary and non-stationary noise reduction."""

    @torch.no_grad()
    def __init__(
        self, 
        sr, 
        nonstationary = False, 
        n_std_thresh_stationary = 1.5, 
        n_thresh_nonstationary = 1.3, 
        temp_coeff_nonstationary = 0.1, 
        n_movemean_nonstationary = 20, 
        prop_decrease = 1.0, 
        n_fft = 1024, 
        win_length = None, 
        hop_length = None, 
        freq_mask_smooth_hz = 500, 
        time_mask_smooth_ms = 50
    ):
        """
        A PyTorch module that applies a spectral gate to an input signal.

        Args:
            sr {int} -- Sample rate of the input signal.
            nonstationary {bool} -- Whether to use non-stationary or stationary masking (default: {False}).
            n_std_thresh_stationary {float} -- Number of standard deviations above mean to threshold noise for stationary masking (default: {1.5}).
            n_thresh_nonstationary {float} -- Number of multiplies above smoothed magnitude spectrogram. for non-stationary masking (default: {1.3}).
            temp_coeff_nonstationary {float} -- Temperature coefficient for non-stationary masking (default: {0.1}).
            n_movemean_nonstationary {int} -- Number of samples for moving average smoothing in non-stationary masking (default: {20}).
            prop_decrease {float} -- Proportion to decrease signal by where the mask is zero (default: {1.0}).
            n_fft {int} -- Size of FFT for STFT (default: {1024}).
            win_length {[int]} -- Window length for STFT. If None, defaults to `n_fft` (default: {None}).
            hop_length {[int]} -- Hop length for STFT. If None, defaults to `win_length` // 4 (default: {None}).
            freq_mask_smooth_hz {float} -- Frequency smoothing width for mask (in Hz). If None, no smoothing is applied (default: {500}).
            time_mask_smooth_ms {float} -- Time smoothing width for mask (in ms). If None, no smoothing is applied (default: {50}).
        """

        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.nonstationary = nonstationary
        self.prop_decrease = prop_decrease
        self.time_mask_smooth_ms = time_mask_smooth_ms
        self.freq_mask_smooth_hz = freq_mask_smooth_hz
        self.n_thresh_nonstationary = n_thresh_nonstationary
        self.n_std_thresh_stationary = n_std_thresh_stationary
        self.temp_coeff_nonstationary = temp_coeff_nonstationary
        self.n_movemean_nonstationary = n_movemean_nonstationary

        # Enforce range safety invariant for the noise attenuation coefficient
        assert 0.0 <= prop_decrease <= 1.0
        self.stft = None
        self.win_length = self.n_fft if win_length is None else win_length
        self.hop_length = self.win_length // 4 if hop_length is None else hop_length
        self.register_buffer("smoothing_filter", self._generate_mask_smoothing_filter())
        # torch.stft/istft is unsupported on OpenCL and DirectML, use the custom implementation instead.
        self.forward = self._forward_other_backends if config.device.startswith(("ocl", "privateuseone")) else self._forward_torch

    @torch.no_grad()
    def _generate_mask_smoothing_filter(self):
        """
        A PyTorch module that applies a spectral gate to an input signal using the STFT.

        Returns:
            smoothing_filter (torch.Tensor): a 2D tensor representing the smoothing filter,
            with shape (n_grad_freq, n_grad_time), where n_grad_freq is the number of frequency
            bins to smooth and n_grad_time is the number of time frames to smooth.
            If both self.freq_mask_smooth_hz and self.time_mask_smooth_ms are None, returns None.
        """

        if self.freq_mask_smooth_hz is None and self.time_mask_smooth_ms is None: return None

        # Derive the kernel radius length along the frequency axis based on bin resolution width
        n_grad_freq = (1 if self.freq_mask_smooth_hz is None else int(self.freq_mask_smooth_hz / (self.sr / (self.n_fft / 2))))
        if n_grad_freq < 1: raise ValueError(f"freq_mask_smooth_hz needs to be at least {int((self.sr / (self.n_fft / 2)))} Hz")

        # Derive the kernel radius length along the time axis based on the hop length stride window (ms)
        n_grad_time = (1 if self.time_mask_smooth_ms is None else int(self.time_mask_smooth_ms / ((self.hop_length / self.sr) * 1000)))
        if n_grad_time < 1: raise ValueError(f"time_mask_smooth_ms needs to be at least {int((self.hop_length / self.sr) * 1000)} ms")
        if n_grad_time == 1 and n_grad_freq == 1: return None

        # Build a 2D smoothing filter mask by calculating the outer product of two triangular linspace windows
        smoothing_filter = torch.outer(
            torch.cat([
                linspace(0, 1, n_grad_freq + 1, endpoint=False), 
                linspace(1, 0, n_grad_freq + 2)
            ])[1:-1], 
            torch.cat([
                linspace(0, 1, n_grad_time + 1, endpoint=False), 
                linspace(1, 0, n_grad_time + 2)
            ])[1:-1]
        ).unsqueeze(0).unsqueeze(0)

        # Unit-normalize the coefficient matrix sums to guarantee strict energy preservation boundaries
        return smoothing_filter / smoothing_filter.sum()

    @torch.no_grad()
    def _stationary_mask(self, X_db):
        """Calculates a Boolean voice activity gate mask based on stationary noise statistics."""

        # Extract mean profiles and standard deviations across the time frame dimension (final axis)
        std_freq_noise, mean_freq_noise = torch.std_mean(X_db, dim=-1)
        
        # Elements exceeding the statistical floor (mean + n * std) are passed; others are flagged as noise
        return X_db > (mean_freq_noise + std_freq_noise * self.n_std_thresh_stationary).unsqueeze(2)

    @torch.no_grad()
    def _nonstationary_mask(self, X_abs):
        """Calculates a soft gain mask tracking fast-shifting non-stationary noise floors."""

        # Utilize a 1D convolution rolling boxcar filter along the time axis to map local moving averages
        X_smoothed = (
            conv1d(
                X_abs.reshape(-1, 1, X_abs.shape[-1]), 
                torch.ones(
                    self.n_movemean_nonstationary, 
                    dtype=X_abs.dtype, 
                    device=X_abs.device
                ).view(1, 1, -1), 
                padding="same"
            ).view(X_abs.shape) / self.n_movemean_nonstationary
        )

        # Compute signal-to-noise ratio deviations and pass them into the temperature-scaled sigmoid gate
        return temperature_sigmoid(
            ((X_abs - X_smoothed) / X_smoothed), 
            self.n_thresh_nonstationary, 
            self.temp_coeff_nonstationary
        )

    def _forward_torch(self, x):
        """Processes gating functionality using PyTorch native STFT/iSTFT primitives."""

        # Transform 1D time-series signals into a 2D complex time-frequency representation via STFT
        X = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            return_complex=True, 
            pad_mode="constant", 
            center=True, 
            window=torch.hann_window(self.win_length).to(x.device)
        )
            
        # Extract the signal masks based on chosen stationarity tracking configuration modes
        sig_mask = self._nonstationary_mask(X.abs()) if self.nonstationary else self._stationary_mask(amp_to_db(X.abs()))
        # Scale and project the mask tracking space into attenuation limits configured by prop_decrease
        sig_mask = self.prop_decrease * (sig_mask.float() * 1.0 - 1.0) + 1.0

        # Apply 2D spatial convolution filtering to resolve sharp threshold boundaries and block phase variance errors
        if self.smoothing_filter is not None: 
            sig_mask = conv2d(
                sig_mask.unsqueeze(1), 
                self.smoothing_filter.to(sig_mask.dtype), 
                padding="same"
            )

        # Execute point-wise complex spectral multiplication to attenuate noise-dominant frames
        Y = X * sig_mask.squeeze(1)

        # Reconstruct filtered 2D spectral components back into time-domain continuous wave signals
        return (
            torch.istft(
                Y, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                win_length=self.win_length, 
                center=True, 
                window=torch.hann_window(self.win_length).to(Y.device)
            ).to(dtype=x.dtype)
        )

    def _forward_other_backends(self, x):
        """Processes gating functionality using PyTorch native STFT/iSTFT primitives."""

        # Lazy-instantiate the vendor-optimized custom backend STFT handler block if uninitialized
        if self.stft is None: 
            from main.library.backends.utils import STFT

            self.stft = STFT(
                filter_length=self.n_fft, 
                hop_length=self.hop_length, 
                win_length=self.win_length, 
                pad_mode="constant"
            ).to(x.device)

        # Explicitly decouple magnitudes and phase components to handle non-complex hardware registers
        X, phase = self.stft.transform(
            x, 
            eps=1e-9, 
            return_phase=True
        )
            
        # Extract the signal masks based on chosen stationarity tracking configuration modes
        sig_mask = self._nonstationary_mask(X.abs()) if self.nonstationary else self._stationary_mask(amp_to_db(X.abs()))
        # Scale and project the mask tracking space into attenuation limits configured by prop_decrease
        sig_mask = self.prop_decrease * (sig_mask.float() * 1.0 - 1.0) + 1.0

        # Apply 2D spatial convolution filtering to resolve sharp threshold boundaries and block phase variance errors
        if self.smoothing_filter is not None: 
            sig_mask = conv2d(
                sig_mask.unsqueeze(1), 
                self.smoothing_filter.to(sig_mask.dtype), 
                padding="same"
            )

        # Execute point-wise complex spectral multiplication to attenuate noise-dominant frames
        Y = X * sig_mask.squeeze(1)

        # Reconstruct filtered 2D spectral components back into time-domain continuous wave signals
        return (
            self.stft.inverse(
                Y, 
                phase
            )
        )