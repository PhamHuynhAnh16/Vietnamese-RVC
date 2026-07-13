import os
import sys

import torch.nn as nn

sys.path.append(os.getcwd())

from main.library.predictors.DJCM.utils import init_bn
from main.library.predictors.DJCM.decoder import PE_Decoder, SVS_Decoder
from main.library.predictors.DJCM.encoder import ResEncoderBlock, Encoder

class LatentBlocks(nn.Module):
    """
    Latent space processing blocks at the UNet bottleneck.

    Applies a series of residual blocks without any spatial downsampling/pooling
    to deep feature maps to further extract complex abstract representations.
    """

    def __init__(
        self, 
        n_blocks, 
        latent_layers
    ):
        """
        Initializes the LatentBlocks layer.

        Args:
            n_blocks (int): Number of internal residual blocks per layer.
            latent_layers (int): The total number of consecutive sequence layers.
        """

        super(LatentBlocks, self).__init__()
        # Chain multiple ResEncoderBlocks together with pooling explicitly deactivated (None)
        self.latent_blocks = nn.ModuleList([
            ResEncoderBlock(
                384, 
                384, 
                n_blocks, 
                None
            ) 
            for _ in range(latent_layers)
        ])

    def forward(self, x):
        """
        Executes the forward pass of the bottleneck latent representation layer.

        Args:
            x (torch.Tensor): Bottleneck feature tensor.

        Returns:
            torch.Tensor: Enhanced deep latent space feature map tensor.
        """

        # Sequentially evaluate each unpooled residual block layer
        for layer in self.latent_blocks:
            x = layer(x)

        return x

class DJCMM(nn.Module):
    """DJCMM Main Module: A Deep Joint Cascade Model for Singing Voice Separation (SVS) and Vocal Pitch Estimation (PE).

    This model can work in two modes:
    1. Pitch Estimation Only (svs=False): Directly predicts pitch tracks from input spectrograms.
    2. Joint Cascade Mode (svs=True): First separates the singing voice from background audio/music, then routes the cleaned vocal spectrogram into the pitch tracker for high-precision F0 extraction.
    """

    def __init__(
        self, 
        in_channels, 
        n_blocks, 
        latent_layers, 
        svs=False, 
        window_length=1024, 
        n_class=360
    ):
        """
        Initializes the integrated DJCMM model.

        Args:
            in_channels (int): Input feature map channel dimensions.
            n_blocks (int): Number of residual blocks per encoder/decoder stage.
            latent_layers (int): Number of unpooled blocks situated at the bottlenecks.
            svs (bool): Enables the integrated Singing Voice Separation cascade. Default is False.
            window_length (int): FFT window size constraint parameter. Default is 1024.
            n_class (int): Number of final target pitch bin classification classes. Default is 360.
        """

        super(DJCMM, self).__init__()
        # Batch Normalization layer dedicated to input frequency configurations
        self.bn = nn.BatchNorm2d(
            window_length // 2 + 1, 
            momentum=0.01
        )

        # Core Vocal Pitch Estimation (PE) network components initialization
        self.pe_encoder = Encoder(
            in_channels, 
            n_blocks
        )
        self.pe_latent = LatentBlocks(
            n_blocks, 
            latent_layers
        )
        self.pe_decoder = PE_Decoder(
            n_blocks, 
            window_length=window_length, 
            n_class=n_class
        )

        self.svs = svs
        # Conditionally initialize Singing Voice Separation (SVS) components if cascading is active
        if svs:
            self.svs_encoder = Encoder(
                in_channels, 
                n_blocks
            )
            self.svs_latent = LatentBlocks(
                n_blocks, 
                latent_layers
            )
            self.svs_decoder = SVS_Decoder(
                in_channels, 
                n_blocks
            )

        init_bn(self.bn)

    def spec(self, x, spec_m):
        """
        Applies decoded SVS separation masks to filter and isolate the raw spectrogram.

        Args:
            x (torch.Tensor): Unprocessed multitrack representation from the SVS decoder.
            spec_m (torch.Tensor): Original reference mixed audio input spectrogram.

        Returns:
            torch.Tensor: Refined and masked isolated vocal spectrogram.
        """

        bs, c, time_steps, freqs_steps = x.shape
        # Split channels into structured segments representing masking values and linear additives
        x = x.reshape(bs, c // 4, 4, time_steps, freqs_steps)

        mask_spec = x[:, :, 0, :, :].sigmoid() # Generate soft masking bounds between [0, 1]
        linear_spec = x[:, :, 3, :, :] # Extract linear residue offsets

        # Apply mask directly onto target mixture and inject linear residue additions
        out_spec = (
            spec_m.detach() * mask_spec + linear_spec
        ).relu()

        return out_spec

    def forward(self, spec):
        """
        Executes the complete cascade forward pass.

        Args:
            spec (torch.Tensor): Input audio mix spectrogram tensor.

        Returns:
            torch.Tensor: Pitch bin activation probabilities profile.
        """

        # Normalize input frequency bands across axis boundaries, then clip trailing pad tokens
        x = self.bn(
            spec.transpose(1, 3)
        ).transpose(1, 3)[..., :-1]

        # Stage 1 Cascade Branch: Singing Voice Separation (SVS) processing pipeline
        if self.svs:
            # Upsample and extract isolated targets from input representation mappings
            x, concat_tensors = self.svs_encoder(x)

            x = self.svs_decoder(
                self.svs_latent(x), 
                concat_tensors
            )

            # Re-pad and reconstruct mask outputs to extract purified vocal components
            x = self.spec(
                nn.functional.pad(x, pad=(0, 1)), 
                spec
            )[..., :-1]

        # Stage 2 Cascade Branch: Core Vocal Pitch Estimation (PE) processing pipeline
        # processes either the raw mix spectrogram (svs=False) or the isolated vocal spectrogram (svs=True)
        x, concat_tensors = self.pe_encoder(x)

        pe_out = self.pe_decoder(
            self.pe_latent(x), 
            concat_tensors
        )

        return pe_out