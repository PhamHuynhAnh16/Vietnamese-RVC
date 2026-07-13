import os
import sys
import math
import torch

sys.path.append(os.getcwd())

from main.library.algorithm.modules import WaveNet
from main.library.algorithm.commons import sequence_mask
from main.library.algorithm.normalization import LayerNorm
from main.library.algorithm.attentions import MultiHeadAttention, FFN

class Encoder(torch.nn.Module):
    """
    Transformer-style Encoder backbone block consisting of Multi-Head Attention,
    Feed-Forward Networks (FFN), residual connections, and Layer Normalization.
    Commonly utilized for processing sequence states in sequential text-to-speech models.
    """

    def __init__(
        self, 
        hidden_channels, 
        filter_channels, 
        n_heads, 
        n_layers, 
        kernel_size=1, 
        p_dropout=0.0, 
        window_size=10, 
        onnx=False, 
        **kwargs
    ):
        """
        Initializes the stack of identical Encoder blocks.

        Args:
            hidden_channels (int): Input and output hidden dimensionality of the model.
            filter_channels (int): Intermediate channel depth within the FFN layer block.
            n_heads (int): Count of parallel self-attention execution heads.
            n_layers (int): The overall number of cascading module block repetitions.
            kernel_size (int): Convolution kernel dimension for FFN blocks. Defaults to 1.
            p_dropout (float): Dropout regularizer probability. Defaults to 0.0.
            window_size (int): Context boundary range for localized position embeddings. Defaults to 10.
            onnx (bool): Toggles ONNX-compatible functional paths inside modules. Defaults to False.
        """

        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.drop = torch.nn.Dropout(p_dropout)
        # Multi-Head Attention layers stack
        self.attn_layers = torch.nn.ModuleList([
            MultiHeadAttention(
                hidden_channels, 
                hidden_channels, 
                n_heads, 
                p_dropout=p_dropout, 
                window_size=window_size, 
                onnx=onnx
            )
            for _ in range(n_layers)
        ])

        # Normalization layer stack prior to passing intermediate features forward
        self.norm_layers_1 = torch.nn.ModuleList([
            LayerNorm(
                hidden_channels, 
                onnx=onnx
            )
            for _ in range(n_layers)
        ])

        # Position-wise Feed-Forward Network layer stack
        self.ffn_layers = torch.nn.ModuleList([
            FFN(
                hidden_channels, 
                hidden_channels, 
                filter_channels, 
                kernel_size, 
                p_dropout=p_dropout
            ) 
            for _ in range(n_layers)
        ])

        # Secondary normalization layers following FFN transforms
        self.norm_layers_2 = torch.nn.ModuleList([
            LayerNorm(
                hidden_channels, 
                onnx=onnx
            ) 
            for _ in range(n_layers)
        ])

    def forward(self, x, x_mask):
        """
        Executes sequential transformation computations over hidden sequence maps.

        Args:
            x (torch.Tensor): Hidden states vector sequence tensor.
            x_mask (torch.Tensor): Binary temporal mask padding tensor.

        Returns:
            torch.Tensor: Transformed hidden states sequence matrix.
        """

        # Outer-product step to cast sequence masks into relational 2D matrix planes
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask # Initialize zero-out on padding entries

        # Iterate forward propagation layers along the Transformer hierarchy
        for i in range(self.n_layers):
            # Block 1: Self-Attention layer joined with a standard residual shortcut path
            x = self.norm_layers_1[i](x + self.drop(self.attn_layers[i](x, x, attn_mask)))
            # Block 2: Feature-scaling FFN block paired with a standard residual shortcut path
            x = self.norm_layers_2[i](x + self.drop(self.ffn_layers[i](x, x_mask)))

        return x * x_mask
    
class TextEncoder(torch.nn.Module):
    """
    Phoneme-level textual text encoder that captures textual representations
    and optionally pairs them with explicit fundamental frequency (F0) tracking metrics.
    """

    def __init__(
        self, 
        out_channels, 
        hidden_channels, 
        filter_channels, 
        n_heads, 
        n_layers, 
        kernel_size, 
        p_dropout, 
        embedding_dim, 
        f0=True, 
        onnx=False
    ):
        """
        Initializes the linguistic text representation encoder block mapping.

        Args:
            out_channels (int): Channel capacity for projection parameters (mean and variance).
            hidden_channels (int): Hidden dimension width.
            filter_channels (int): Depth size inside internal FFN channels.
            n_heads (int): Total attention head components.
            n_layers (int): Depth repetitions count for core structural encoder layers.
            kernel_size (int): FFN layer 1D conv width span.
            p_dropout (float): Dropout probability.
            embedding_dim (int): Vector width scale of the text phoneme features.
            f0 (bool): Flag to instantiate structural pitch token embeddings. Defaults to True.
            onnx (bool): ONNX translation compatibility marker. Defaults to False.
        """

        super(TextEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        # Phoneme textual mapping layer
        self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)
        # Quantized F0 pitch state representation layer
        self.emb_pitch = torch.nn.Embedding(256, hidden_channels) if f0 else None
        # Core feature encoder backbone
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), onnx=onnx)
        # Prediction head tracking standard deviation logs and mean parameters
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths):
        """
        Extracts multi-variate statistical distributions representing sequence segments.

        Args:
            phone (torch.Tensor): Continuous phoneme tensor representation of shape (B, T, embedding_dim).
            pitch (torch.Tensor, optional): Discrete fundamental frequency indices of shape (B, T).
            lengths (torch.Tensor): Sequence duration measurements tensor of shape (B,).

        Returns:
            Tuple containing:
                - m (torch.Tensor): Projected distribution mean map tensor of shape (B, out_channels, T).
                - logs (torch.Tensor): Variance standard deviation logarithmic mapping scale of shape (B, out_channels, T).
                - x_mask (torch.Tensor): Output padding tracking logic array of shape (B, 1, T).
        """

        x = self.emb_phone(phone)
        if self.emb_pitch is not None: x += self.emb_pitch(pitch)

        # Scale tensor inputs by hidden width radical prior to shifting axes format
        x = self.lrelu(x * math.sqrt(self.hidden_channels)).transpose(1, -1)
        # Generate masking criteria to discard padding indices during calculations
        x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        # Project contextual embeddings, then split into mean (m) and log variance (logs) matrices
        m, logs = (self.proj(self.encoder(x * x_mask, x_mask)) * x_mask).split(self.out_channels, dim=1)

        return m, logs, x_mask

class TextEncoderSVC(torch.nn.Module):
    """
    Specialized Content-to-Vocal feature variant mapping module tailored for Singing Voice Conversion (SVC).
    Accepts raw pre-extracted audio content feature markers (e.g., ContentVec, Hubert).
    """

    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        onnx=False
    ):
        """Initializes the SVC Text/Content Encoder block."""

        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.f0_emb = torch.nn.Embedding(256, hidden_channels)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)
        # Core encoder configured with a narrower window size for specialized voice audio tracking
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), window_size=4, onnx=onnx)

    def forward(self, x, x_mask, f0=None, noise_scale=1.0):
        """
        Args:
            x (torch.Tensor): Extracted audio content features matrix.
            x_mask (torch.Tensor): Tensor indicating explicit segment sequence length positions.
            f0 (torch.Tensor): Discrete F0 frame metrics arrays mapping pitch profiles.
            noise_scale (float): Scale adjustment factor for Gaussian stochastic reparameterization. Defaults to 1.0.

        Returns:
            Tuple containing:
                - z (torch.Tensor): Latent token sample array generated through reparameterization.
                - m (torch.Tensor): Calculated content probability mean distributions.
                - logs (torch.Tensor): Calculated content probability logarithmic variance tracking scales.
                - x_mask (torch.Tensor): Sequence masking configuration logic matrix.
        """

        # Inject structural musical/vocal pitch embedding markers onto raw acoustic features
        x = x + self.f0_emb(f0).transpose(1, 2)

        # Compute output parameter statistics
        m, logs = (self.proj(self.encoder(x * x_mask, x_mask)) * x_mask).split(self.out_channels, dim=1)
        # Draw random standard Normal distribution variables and apply the Reparameterization Trick
        z = (m + torch.randn_like(m) * logs.exp() * noise_scale) * x_mask

        return z, m, logs, x_mask

class PosteriorEncoder(torch.nn.Module):
    """
    Posterior Encoder module using non-causal WaveNet blocks to map acoustic targets
    (e.g., linear spectrogram magnitudes or Mel features) into prior latent spaces.
    Typically used in VAE-based frameworks like VITS.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        hidden_channels, 
        kernel_size, 
        dilation_rate, 
        n_layers, 
        gin_channels=0
    ):
        """
        Initializes the Posterior WaveNet Spectrogram Feature Encoder.

        Args:
            in_channels (int): Input acoustic map format channels (e.g., linear spec bins).
            out_channels (int): Output dimensionality for latent vectors.
            hidden_channels (int): Hidden channel layer volume within internal layers.
            kernel_size (int): Size of 1D dilated casual convolutional kernels.
            dilation_rate (int): Geometric structural spacing scaling multiplier base.
            n_layers (int): Comprehensive layer length inside the WaveNet core block.
            gin_channels (int): Global conditioning channel length (e.g., speaker embeddings). Defaults to 0.
        """

        super(PosteriorEncoder, self).__init__()
        self.out_channels = out_channels
        # Initial dimension adjustment module
        self.pre = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        # Dilated residual convolution block backbone
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        # Output mean and variance parameter estimation projection layer
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g = None):
        """
        Transforms acoustic spectrum values into sound distribution variables.

        Args:
            x (torch.Tensor): Linear audio spectrogram values sequence matrix.
            x_lengths (torch.Tensor): Real unpadded step sequence time vectors.
            g (torch.Tensor, optional): Speaker identity condition vectors matrix. Defaults to None.

        Returns:
            Tuple containing:
                - z (torch.Tensor): Sampled random output latent target vectors.
                - m (torch.Tensor): Target feature standard mean values projection.
                - logs (torch.Tensor): Projected logarithmic standard deviation scale variance.
                - x_mask (torch.Tensor): Tracking array indicating temporal sequence lengths.
        """

        # Form sequence masking filters
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        # Run multi-layer forward execution pipelines
        m, logs = (
            self.proj(
                self.enc(
                    self.pre(x) * x_mask, 
                    x_mask, 
                    g=g
                )
            ) * x_mask
        ).split(self.out_channels, dim=1)

        # Perform random latent generation using standard VAE reparameterization
        return (m + torch.randn_like(m) * logs.exp()) * x_mask, m, logs, x_mask

    def remove_weight_norm(self):
        """
        Unbinds and removes weight normalization tracking constants from internal layers
        to optimize execution performance during inference.
        """

        self.enc.remove_weight_norm()