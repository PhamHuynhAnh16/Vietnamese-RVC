import os
import sys

import torch.nn as nn

sys.path.append(os.getcwd())

from main.library.predictors.DJCM.utils import init_bn
from main.library.predictors.DJCM.decoder import PE_Decoder, SVS_Decoder
from main.library.predictors.DJCM.encoder import ResEncoderBlock, Encoder

class LatentBlocks(nn.Module):
    def __init__(
        self, 
        n_blocks, 
        latent_layers
    ):
        super(LatentBlocks, self).__init__()
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
        for layer in self.latent_blocks:
            x = layer(x)

        return x

class DJCMM(nn.Module):
    def __init__(
        self, 
        in_channels, 
        n_blocks, 
        latent_layers, 
        svs=False, 
        window_length=1024, 
        n_class=360
    ):
        super(DJCMM, self).__init__()
        self.bn = nn.BatchNorm2d(
            window_length // 2 + 1, 
            momentum=0.01
        )
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
        bs, c, time_steps, freqs_steps = x.shape
        x = x.reshape(bs, c // 4, 4, time_steps, freqs_steps)

        mask_spec = x[:, :, 0, :, :].sigmoid()
        linear_spec = x[:, :, 3, :, :]

        out_spec = (
            spec_m.detach() * mask_spec + linear_spec
        ).relu()

        return out_spec

    def forward(self, spec):
        x = self.bn(
            spec.transpose(1, 3)
        ).transpose(1, 3)[..., :-1]

        if self.svs:
            x, concat_tensors = self.svs_encoder(x)

            x = self.svs_decoder(
                self.svs_latent(x), 
                concat_tensors
            )

            x = self.spec(
                nn.functional.pad(x, pad=(0, 1)), 
                spec
            )[..., :-1]

        x, concat_tensors = self.pe_encoder(x)

        pe_out = self.pe_decoder(
            self.pe_latent(x), 
            concat_tensors
        )

        return pe_out