import torch

import numpy as np
import torch.nn.functional as F

from librosa.util import pad_center
from scipy.signal import get_window

class STFT(torch.nn.Module):
    """
    Short-Time Fourier Transform (STFT) module implemented as a PyTorch Module.
    This class handles both the forward transform (STFT) and inverse transform (iSTFT)
    using 1D convolutions and linear matrix operations.
    """

    def __init__(
        self, 
        filter_length=1024, 
        hop_length=512, 
        win_length=None, 
        window="hann", 
        pad_mode="reflect"
    ):
        """
        Initializes the STFT module and pre-computes the Fourier basis filters.

        Args:
            filter_length (int): Length of the FFT window. Default: 1024.
            hop_length (int): Number of audio samples between adjacent STFT columns. Default: 512.
            win_length (int, optional): Window size. If None, it defaults to filter_length.
            window (str): Type of window function supported by scipy.signal.get_window. Default: "hann".
            pad_mode (str): Padding mode for signal boundaries. Default: "reflect".
        """

        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.pad_amount = int(self.filter_length / 2)
        self.cutoff = int(self.filter_length / 2 + 1) # Nyquist frequency bin index
        self.win_length = win_length
        self.pad_mode = pad_mode
        
        # Generate standard Discrete Fourier Transform (DFT) matrix using NumPy
        fourier_basis = np.fft.fft(
            np.eye(self.filter_length)
        )
        # Split into real and imaginary parts up to the Nyquist frequency (cutoff)
        fourier_basis = np.vstack([
            np.real(fourier_basis[:self.cutoff, :]), 
            np.imag(fourier_basis[:self.cutoff, :])
        ])

        # Convert the basis matrices into PyTorch FloatTensors
        forward_basis = torch.FloatTensor(fourier_basis)
        # Compute the Moore-Penrose pseudo-inverse for the inverse transform
        inverse_basis = torch.FloatTensor(np.linalg.pinv(fourier_basis))

        # Handle fallback for window length
        if win_length is None or not win_length: win_length = filter_length
        assert filter_length >= win_length

        # Generate the window function and pad it to match filter_length
        fft_window = torch.from_numpy(
            pad_center(
                get_window(
                    window, 
                    win_length, 
                    fftbins=True
                ), 
                size=filter_length
            )
        ).float()

        # Apply the windowing function to both forward and inverse bases
        forward_basis *= fft_window
        inverse_basis = (inverse_basis.T * fft_window).T
        # Register bases as buffers so they move automatically to GPU alongside the module
        self.register_buffer("forward_basis", forward_basis.float().unsqueeze(1))
        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("fft_window", fft_window.float())

    def transform(
        self, 
        input_data, 
        eps=1e-9, 
        return_phase=False, 
        center=True
    ):
        """
        Transforms a 1D time-domain signal into its frequency-domain magnitude (and optional phase).

        Args:
            input_data (Tensor): Input signal of shape (Batch, Time).
            eps (float): Small constant to avoid square root of zero or log issues. Default: 1e-9.
            return_phase (bool): If True, returns both magnitude and phase. Default: False.
            center (bool): If True, pads the input signal symmetrically. Default: True.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Magnitude spectrogram, or (magnitude, phase).
        """

        # Symmetrically pad the input signal if center is enabled
        if center: 
            input_data = F.pad(
                input_data, 
                (self.pad_amount, self.pad_amount), 
                mode=self.pad_mode
            )

        # Compute STFT using 1D Convolution (acting as a bank of filters)
        forward_transform = F.conv1d(
            input_data.unsqueeze(1), 
            self.forward_basis, 
            stride=self.hop_length
        )

        # Separate the real and imaginary components from the convolution channels
        real_part = forward_transform[:, :self.cutoff, :]
        imag_part = forward_transform[:, self.cutoff:, :]

        # Calculate the magnitude spectrogram
        magnitude = (real_part**2 + imag_part**2 + eps).sqrt()

        # Compute phase if requested using atan2
        if return_phase:
            phase = imag_part.data.atan2(real_part.data)
            return magnitude, phase

        return magnitude

    def inverse(
        self, 
        magnitude, 
        phase
    ):
        """
        Reconstructs the 1D time-domain signal from its magnitude and phase spectrograms (iSTFT).

        Args:
            magnitude (Tensor): Magnitude spectrogram of shape (Batch, Freq, Time).
            phase (Tensor): Phase spectrogram of shape (Batch, Freq, Time).

        Returns:
            Tensor: Reconstructed 1D time-domain signal.
        """

        # Reconstruct real and imaginary parts, then concatenate along the channel dimension
        cat = torch.cat([
            magnitude * phase.cos(), 
            magnitude * phase.sin()
        ], dim=1)

        # Initialize the Overlap-Add operator (Fold) to reconstruct the original timeline
        fold = torch.nn.Fold(
            output_size=(1, (cat.size(-1) - 1) * self.hop_length + self.filter_length), 
            kernel_size=(1, self.filter_length), 
            stride=(1, self.hop_length)
        )

        # Apply inverse basis matrix and project overlapping frames back to 1D signal space
        inverse_transform = fold(
            self.inverse_basis @ cat
        )[:, 0, 0, self.pad_amount : -self.pad_amount]

        # Calculate the sum of squared windows to compensate for windowing gain artifacts
        window_square_sum = (
            fold(
                self.fft_window.cpu().pow(2).repeat(cat.size(-1), 1).T.unsqueeze(0) # Workaround for OpenCL backend devices if applicable
            )
        ) if str(cat.device).startswith("ocl") else (
            fold(
                self.fft_window.pow(2).repeat(cat.size(-1), 1).T.unsqueeze(0) # Standard routine for DML/Other devices
            )
        )[:, 0, 0, self.pad_amount : -self.pad_amount].to(cat.device)

        # Normalize the inverse signal by dividing by the window square sum
        return inverse_transform / window_square_sum

class GRU(torch.nn.RNNBase):
    """
    A custom implementation of the Gated Recurrent Unit (GRU) extending torch.nn.RNNBase.
    Provides customizable cell-level step looping over time sequences.
    """

    def __init__(
        self, 
        input_size, 
        hidden_size, 
        num_layers=1, 
        bias=True, 
        batch_first=True, 
        dropout=0.0, 
        bidirectional=False, 
        device=None, 
        dtype=None
    ):
        """
        Initializes the custom GRU layer configurations.
        """

        super().__init__(
            "GRU", 
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bias=bias, 
            batch_first=batch_first, 
            dropout=dropout, 
            bidirectional=bidirectional, 
            device=device, 
            dtype=dtype
        )

    def _gru_layer(self, x, hx, weights):
        """
        Processes a single directional GRU layer across all time steps.

        Args:
            x (Tensor): Input tensor of shape (Batch, Time, Input_Dim).
            hx (Tensor): Initial hidden state of shape (Batch, Hidden_Dim).
            weights (tuple): Contains weight matrices and bias tensors (weight_ih, weight_hh, bias_ih, bias_hh).

        Returns:
            Tuple[Tensor, Tensor]: All hidden state outputs across time steps and the final hidden state.
        """

        weight_ih, weight_hh, bias_ih, bias_hh = weights
        B, T, _ = x.shape
        # Precompute the input-to-hidden linear projection for all time steps at once
        gate_x = F.linear(x, weight_ih, bias_ih) 
        outputs = torch.empty(B, T, self.hidden_size, device=x.device, dtype=x.dtype)

        # Segment hidden-to-hidden biases into update, reset, and new gates
        if bias_hh is not None: b_hr, b_hz, b_hn = bias_hh.chunk(3)
        else: b_hr = b_hz = b_hn = None

        # Segment hidden-to-hidden weights into update, reset, and new gates
        w_hr, w_hz, w_hn = weight_hh.chunk(3, dim=0)

        # Explicit loop over each chronological time-step
        for t in range(T):
            # Split the input-to-hidden projections into reset (r), update (z), and new (n) chunks
            x_r, x_z, x_n = gate_x[:, t].chunk(3, dim=1)
            # Compute Reset Gate and Update Gate activations
            r = (x_r + F.linear(hx, w_hr, b_hr)).sigmoid()
            z = (x_z + F.linear(hx, w_hz, b_hz)).sigmoid()
            # Compute Candidate/New Gate state activation
            n = (x_n + r * F.linear(hx, w_hn, b_hn)).tanh()
            # Linear interpolation for the final hidden state output of this step
            hx = n + z * (hx - n)
            outputs[:, t] = hx

        return outputs, hx

    def _gru(self, x, hx):
        """
        Iterates and processes the multi-layer (and optionally bidirectional) GRU architecture.

        Args:
            x (Tensor): Input feature sequences.
            hx (Tensor): Hidden state collections.

        Returns:
            Tuple[Tensor, Tensor]: Output sequences and aggregated stacked final hidden states.
        """

        # Permute if data structure is not configured as batch-first
        if not self.batch_first: x = x.permute(1, 0, 2)
        num_directions = 2 if self.bidirectional else 1

        h_n = []
        output_fwd, output_bwd = x, x

        for layer in range(self.num_layers):
            fwd_idx = layer * num_directions
            bwd_idx = fwd_idx + 1 if self.bidirectional else None
            # Fetch parameters for the forward layer direction
            weights_fwd = self._get_weights(fwd_idx)
            h_fwd = hx[fwd_idx]

            # Execute forward sequence processing
            out_fwd, h_out_fwd = self._gru_layer(
                output_fwd, 
                h_fwd, 
                weights_fwd
            )

            h_n.append(h_out_fwd)

            # Conditional check and execution path for bidirectional layering
            if self.bidirectional:
                weights_bwd = self._get_weights(bwd_idx)
                h_bwd = hx[bwd_idx]

                # Flip time dimension for backward pass execution
                reversed_input = output_bwd.flip(dims=[1])
                out_bwd, h_out_bwd = self._gru_layer(
                    reversed_input, 
                    h_bwd, 
                    weights_bwd
                )

                # Re-reverse output back to original chronological alignment
                out_bwd = out_bwd.flip(dims=[1])
                h_n.append(h_out_bwd)

                # Concatenate forward and backward channel outputs together
                output_fwd = torch.cat([out_fwd, out_bwd], dim=2)
                output_bwd = output_fwd
            else: output_fwd = out_fwd

            # Apply dropout normalization on inner stacked hidden layers
            if layer < self.num_layers - 1 and self.dropout > 0:
                output_fwd = F.dropout(
                    output_fwd, 
                    p=self.dropout, 
                    training=self.training
                )

                if self.bidirectional: output_bwd = output_fwd

        output = output_fwd
        h_n = torch.stack(h_n, dim=0)

        # Restore original non batch-first sequence ordering if applicable
        if not self.batch_first: output = output.permute(1, 0, 2)
        return output, h_n

    def _get_weights(self, layer_idx):
        """
        Helper method to look up weight and bias parameters from internal structures.
        """

        weights = self._all_weights[layer_idx]

        weight_ih = getattr(self, weights[0])
        weight_hh = getattr(self, weights[1])

        bias_ih = getattr(self, weights[2]) if self.bias else None
        bias_hh = getattr(self, weights[3]) if self.bias else None

        return weight_ih, weight_hh, bias_ih, bias_hh

    def forward(self, input, hx=None):
        """
        Defines the computation performed at every call.

        Args:
            input (Tensor): Input tensor.
            hx (Tensor, optional): Initial hidden state.

        Raises:
            ValueError: If input tensor does not have exactly 3 dimensions.
        """

        if input.dim() != 3: raise ValueError(f"Expected 3D input tensor, but got {input.dim()}D tensor instead.")

        batch_size = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1

        # Initialize zero states default tensor if hx isn't explicitly supplied
        if hx is None: 
            hx = torch.zeros(
                self.num_layers * num_directions, 
                batch_size, 
                self.hidden_size, 
                dtype=input.dtype, 
                device=input.device
            )

        # Built-in PyTorch sanity check tool verifying valid parameter alignments
        self.check_forward_args(input, hx, batch_sizes=None)
        return self._gru(input, hx)

class DeviceProperties:
    """
    A simple data structure class representing backend compute hardware specifications.
    """

    def __init__(self, index, name, total_memory):
        """
        Initializes DeviceProperties configuration properties.

        Args:
            index (int): Hardware identifier index assigned to the device.
            name (str): Device model naming moniker.
            total_memory (int or float): Total hardware runtime memory limit availability.
        """

        self.index = index
        self.name = name
        self.total_memory = total_memory