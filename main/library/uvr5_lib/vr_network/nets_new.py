import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.uvr5_lib.vr_network import layers_new as layers

class BaseNet(nn.Module):
    """
    Standard U-Net backbone integrated with Atrous Spatial Pyramid Pooling (ASPP) 
    and a Bi-directional LSTM layer for progressive spatial-temporal audio features.
    """

    def __init__(
        self, 
        nin, 
        nout, 
        nin_lstm, 
        nout_lstm, 
        dilations=((4, 2), (8, 4), (12, 6))
    ):
        """
        Args:
            nin (int): Number of input channels.
            nout (int): Baseline output channel scale for the first encoder.
            nin_lstm (int): Frequency bin feature sequence length/size for the LSTM.
            nout_lstm (int): Total hidden size of the bidirectional LSTM layer.
            dilations (Tuple, optional): Dilation configurations for the ASPP module branches.
        """

        super(BaseNet, self).__init__()
        # Encoder Layers
        self.enc1 = layers.Conv2DBNActiv(
            nin, 
            nout, 
            3, 
            1, 
            1
        )
        self.enc2 = layers.Encoder(
            nout, 
            nout * 2, 
            3, 
            2, 
            1
        )
        self.enc3 = layers.Encoder(
            nout * 2, 
            nout * 4, 
            3, 
            2, 
            1
        )
        self.enc4 = layers.Encoder(
            nout * 4, 
            nout * 6, 
            3, 
            2, 
            1
        )
        self.enc5 = layers.Encoder(
            nout * 6, 
            nout * 8, 
            3, 
            2, 
            1
        )
        # Bottleneck Multi-Scale Context Aggregator
        self.aspp = layers.ASPPModule(
            nout * 8, 
            nout * 8, 
            dilations, 
            dropout=True
        )
        # Decoder Layers
        # nin format: current upsampled channels + skip connection channel width
        self.dec4 = layers.Decoder(
            nout * (6 + 8), 
            nout * 6, 
            3, 
            1, 
            1
        )
        self.dec3 = layers.Decoder(
            nout * (4 + 6), 
            nout * 4, 
            3, 
            1, 
            1
        )
        self.dec2 = layers.Decoder(
            nout * (2 + 4), 
            nout * 2, 
            3, 
            1, 
            1
        )
        # Recurrent Temporal Processing Refinement
        self.lstm_dec2 = layers.LSTMModule(
            nout * 2, 
            nin_lstm, 
            nout_lstm
        )
        # Final decoder joins standard features, LSTM temporal features, and skip connections (+1 channel)
        self.dec1 = layers.Decoder(
            nout * (1 + 2) + 1, 
            nout * 1, 
            3, 
            1, 
            1
        )

    def __call__(self, input_tensor):
        """
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Input audio spectrogram maps.

        Returns:
            torch.Tensor: Refined and decoded audio representations.
        """

        # Execute hierarchical contracting path (Encoder)
        encoded1 = self.enc1(input_tensor)
        encoded2 = self.enc2(encoded1)
        encoded3 = self.enc3(encoded2)
        encoded4 = self.enc4(encoded3)

        # Bottom layer extraction followed by progressive upsampling and skip-connections matching
        bottleneck = self.dec2(
            self.dec3(
                self.dec4(
                    self.aspp(self.enc5(encoded4)), 
                    encoded4
                ), 
                encoded3
            ), 
            encoded2
        )
        # Concatenate spatial layout maps with 1D sequential temporal features from the LSTM
        bottleneck = self.dec1(
            torch.cat([
                bottleneck, 
                self.lstm_dec2(bottleneck)
            ], dim=1), 
            encoded1
        )

        return bottleneck

class CascadedNet(nn.Module):
    """
    Cascaded Multi-Band Architecture designed for full audio separation tasks.
    Splits the spectrogram into low/high sub-bands, processes them across stages, 
    and predicts an ideal separation filter/mask.
    """

    def __init__(
        self, 
        n_fft, 
        nn_arch_size=51000, 
        nout=32, 
        nout_lstm=128
    ):
        """
        Args:
            n_fft (int): Fast Fourier Transform (FFT) frequency window footprint.
            nn_arch_size (int, optional): Preset capacity key dictating filter depth limits.
            nout (int, optional): Core feature channels size. Defaults to 32.
            nout_lstm (int, optional): Inner hidden sequence limit sizes for recurrent blocks.
        """

        super(CascadedNet, self).__init__()
        # Determine internal frequency dimensions ignoring the complex symmetric conjugate block
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64 # Overlap padding boundary crop parameter
        # Dynamic structural sizing check based on architecture footprint definitions
        nout = 64 if nn_arch_size == 218409 else nout

        # STAGE 1: Independent Sub-Band Nets
        self.stg1_low_band_net = nn.Sequential(
            BaseNet(
                2, 
                nout // 2, 
                self.nin_lstm // 2, 
                nout_lstm
            ), 
            layers.Conv2DBNActiv(
                nout // 2, 
                nout // 4, 
                1, 
                1, 
                0
            )
        )

        self.stg1_high_band_net = BaseNet(
            2, 
            nout // 4, 
            self.nin_lstm // 2, 
            nout_lstm // 2
        )

        # STAGE 2: Dense-Linked Sub-Band Nets
        self.stg2_low_band_net = nn.Sequential(
            BaseNet(
                nout // 4 + 2, 
                nout, 
                self.nin_lstm // 2, nout_lstm
            ), 
            layers.Conv2DBNActiv(
                nout, 
                nout // 2, 
                1, 
                1, 
                0
            )
        )

        self.stg2_high_band_net = BaseNet(
            nout // 4 + 2, 
            nout // 2, 
            self.nin_lstm // 2, 
            nout_lstm // 2
        )

        # STAGE 3: Full-Band Cross-Attention / Aggregation Net
        self.stg3_full_band_net = BaseNet(
            3 * nout // 4 + 2, 
            nout, 
            self.nin_lstm, 
            nout_lstm
        )

        # Linear Projection Output Probes
        self.out = nn.Conv2d(
            nout, 
            2, 
            1, 
            bias=False
        )
        self.aux_out = nn.Conv2d(
            3 * nout // 4, 
            2, 
            1, 
            bias=False
        )

    def forward(self, input_tensor):
        """
        Forward pass. Splits, predicts masks hierarchically, and maps results to original sizes.

        Args:
            input_tensor (torch.Tensor): Audio spectrogram batch of size.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                Primary mask, plus auxiliary mask if module is set to training mode.
        """

        # Crop to standard max frequency limits
        input_tensor = input_tensor[:, :, : self.max_bin]
        bandw = input_tensor.size()[2] // 2

        # Squeeze split into distinct low-band and high-band modules
        l1_in = input_tensor[:, :, :bandw]
        h1_in = input_tensor[:, :, bandw:]

        # Execute Stage 1 processing
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        # Cascade features down to Stage 2 inputs using dense connections
        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)

        # Execute Stage 2 processing
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)

        aux2 = torch.cat([l2, h2], dim=2)
        # Fuse Stage 1, Stage 2, and initial inputs together for Full-Band global resolution
        f3_in = torch.cat([input_tensor, aux1, aux2], dim=1)
        f3 = self.stg3_full_band_net(f3_in)

        # Compute target isolation filter mask [bounded 0.0 to 1.0]
        mask = self.out(f3).sigmoid()
        # Pad back to match original input frequency output bin lengths using edge replication
        mask = F.pad(input=mask, pad=(0, 0, 0, self.output_bin - mask.size()[2]), mode="replicate")

        if self.training:
            # Generate intermediate auxiliary mask profiles for training optimization constraints
            aux = self.aux_out(torch.cat([aux1, aux2], dim=1)).sigmoid()
            aux = F.pad(input=aux, pad=(0, 0, 0, self.output_bin - aux.size()[2]), mode="replicate")

            return mask, aux
        else:
            return mask

    def predict_mask(self, input_tensor):
        """
        Generates and trims the separation mask by slicing window padding offsets.

        Args:
            input_tensor (torch.Tensor): Audio spectrogram tensor.

        Returns:
            torch.Tensor: Trimmed audio mask.
        """

        mask = self.forward(input_tensor)
        # Slice off overlapping contextual boundaries along the time dimension if offset is active
        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]
            assert mask.size()[3] > 0

        return mask

    def predict(self, input_tensor):
        """
        Extracts targeted source signal magnitude values from the audio mix.

        Args:
            input_tensor (torch.Tensor): Audio input mix spectrogram.

        Returns:
            torch.Tensor: Isolated target source magnitude spectrogram.
        """

        mask = self.forward(input_tensor)
        # Apply element-wise multiplication to mask out unwanted frequency components
        pred_mag = input_tensor * mask

        # Remove temporary window overlap boundaries along the time dimension
        if self.offset > 0:
            pred_mag = pred_mag[:, :, :, self.offset : -self.offset]
            assert pred_mag.size()[3] > 0

        return pred_mag