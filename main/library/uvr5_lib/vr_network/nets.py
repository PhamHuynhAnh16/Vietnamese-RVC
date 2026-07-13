import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.uvr5_lib.vr_network import layers

class BaseASPPNet(nn.Module):
    """
    U-Net style sub-network backbone integrated with an Atrous Spatial Pyramid Pooling 
    (ASPP) module. Dynamically allocates an extra depth layer for deeper model configurations.
    """

    def __init__(
        self, 
        nn_architecture, 
        nin, 
        ch, 
        dilations=(4, 8, 16)
    ):
        """
        Args:
            nn_architecture (int): Configuration ID determining the architectural path.
            nin (int): Number of input channels.
            ch (int): Base channel depth multiplier.
            dilations (Tuple[int, int, int], optional): ASPP module dilation rates. Defaults to (4, 8, 16).
        """

        super(BaseASPPNet, self).__init__()
        self.nn_architecture = nn_architecture

        # Baseline Encoder contracting path (Downsampling)
        self.enc1 = layers.Encoder(
            nin, 
            ch, 
            3, 
            2, 
            1
        )
        self.enc2 = layers.Encoder(
            ch, 
            ch * 2, 
            3, 
            2, 
            1
        )
        self.enc3 = layers.Encoder(
            ch * 2, 
            ch * 4, 
            3, 
            2, 
            1
        )
        self.enc4 = layers.Encoder(
            ch * 4, 
            ch * 8, 
            3, 
            2, 
            1
        )

        # Dynamic expansion branch based on architecture key
        if self.nn_architecture == 129605:
            self.enc5 = layers.Encoder(
                ch * 8, 
                ch * 16, 
                3, 
                2, 
                1
            )
            self.aspp = layers.ASPPModule(
                nn_architecture, 
                ch * 16, 
                ch * 32, 
                dilations
            )
            self.dec5 = layers.Decoder(
                ch * (16 + 32), 
                ch * 16, 
                3, 
                1, 
                1
            )
        else:
            self.aspp = layers.ASPPModule(
                nn_architecture, 
                ch * 8, 
                ch * 16, 
                dilations
            )

        # Baseline Decoder expanding path (Upsampling)
        self.dec4 = layers.Decoder(
            ch * (8 + 16), 
            ch * 8, 
            3, 
            1, 
            1
        )
        self.dec3 = layers.Decoder(
            ch * (4 + 8), 
            ch * 4, 
            3, 
            1, 
            1
        )
        self.dec2 = layers.Decoder(
            ch * (2 + 4), 
            ch * 2, 
            3, 
            1, 
            1
        )
        self.dec1 = layers.Decoder(
            ch * (1 + 2), 
            ch, 
            3, 
            1, 
            1
        )

    def __call__(self, input_tensor):
        """
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Input feature maps.

        Returns:
            torch.Tensor: Reconstructed full-resolution feature maps.
        """

        # Execute downsampling encoder sequence and store intermediate states for skip connections
        hidden_state, encoder_output1 = self.enc1(input_tensor)
        hidden_state, encoder_output2 = self.enc2(hidden_state)
        hidden_state, encoder_output3 = self.enc3(hidden_state)
        hidden_state, encoder_output4 = self.enc4(hidden_state)

        # Apply multi-scale context bottleneck according to preset configuration depth
        if self.nn_architecture == 129605:
            hidden_state, encoder_output5 = self.enc5(hidden_state)
            hidden_state = self.dec5(self.aspp(hidden_state), encoder_output5)
        else:
            hidden_state = self.aspp(hidden_state)

        # Execute upsampling decoder sequence utilizing skip connections
        hidden_state = self.dec1(
            self.dec2(
                self.dec3(self.dec4(hidden_state, encoder_output4), encoder_output3), 
                encoder_output2
            ), 
            encoder_output1
        )
        return hidden_state

def determine_model_capacity(n_fft_bins, nn_architecture):
    """
    Factory function that retrieves hyperparameters based on model capacity profiles,
    instantiates, and returns a CascadedASPPNet instance.

    Args:
        n_fft_bins (int): The number of raw frequency bins from the FFT process.
        nn_architecture (int): Configuration ID flag identifying capacity constraints.

    Raises:
        ValueError: If an unexpected or unknown architecture ID is provided.

    Returns:
        nn.Module: Configured CascadedASPPNet model instance ready for weight loading or training.
    """

    # Architecture capacity categories
    sp_model_arch = [31191, 33966, 129605]
    hp_model_arch = [123821, 123812]
    hp2_model_arch = [537238, 537227]

    if nn_architecture in sp_model_arch:
        # Standard Profile structural tuples
        model_capacity_data = [
            (2, 16), 
            (2, 16), 
            (18, 8, 1, 1, 0), 
            (8, 16), 
            (34, 16, 1, 1, 0), 
            (16, 32), 
            (32, 2, 1), 
            (16, 2, 1), 
            (16, 2, 1)
        ]

    if nn_architecture in hp_model_arch:
        # High Profile structural tuples
        model_capacity_data = [
            (2, 32), 
            (2, 32), 
            (34, 16, 1, 1, 0), 
            (16, 32), 
            (66, 32, 1, 1, 0), 
            (32, 64), 
            (64, 2, 1), 
            (32, 2, 1), 
            (32, 2, 1)
        ]

    if nn_architecture in hp2_model_arch:
        # High Profile 2 structural tuples
        model_capacity_data = [
            (2, 64), 
            (2, 64), 
            (66, 32, 1, 1, 0), 
            (32, 64), 
            (130, 64, 1, 1, 0), 
            (64, 128), 
            (128, 2, 1), 
            (64, 2, 1), 
            (64, 2, 1)
        ]

    # Initialize the target network wrapper mapping selected structural data configurations
    model = CascadedASPPNet(
        n_fft_bins, 
        model_capacity_data, 
        nn_architecture
    )

    return model

class CascadedASPPNet(nn.Module):
    """
    Multi-stage cascaded separation architecture using parallel ASPP modules.
    Splits audio into low/high sub-bands before fusing them progressively 
    into full-band resolution estimators.
    """

    def __init__(
        self, 
        n_fft, 
        model_capacity_data, 
        nn_architecture
    ):
        """
        Args:
            n_fft (int): Total FFT size determining spectrogram bin width.
            model_capacity_data (List[Tuple]): Hyperparameters unpacking structure for layer dims.
            nn_architecture (int): Configuration token for BaseASPPNet variants.
        """

        super(CascadedASPPNet, self).__init__()
        # STAGE 1: Independent Sub-Band Processing Nets
        self.stg1_low_band_net = BaseASPPNet(
            nn_architecture, 
            *model_capacity_data[0]
        )
        self.stg1_high_band_net = BaseASPPNet(
            nn_architecture, 
            *model_capacity_data[1]
        )
        # STAGE 2: Channel Dimensionality Bridge and Full-Band Fusion Net
        self.stg2_bridge = layers.Conv2DBNActiv(*model_capacity_data[2])
        self.stg2_full_band_net = BaseASPPNet(
            nn_architecture, 
            *model_capacity_data[3]
        )
        # STAGE 3: Final Dense Aggregation Bridge and Full-Band Net
        self.stg3_bridge = layers.Conv2DBNActiv(*model_capacity_data[4])
        self.stg3_full_band_net = BaseASPPNet(
            nn_architecture, 
            *model_capacity_data[5]
        )
        # Projection Heads
        self.out = nn.Conv2d(
            *model_capacity_data[6], 
            bias=False
        )
        self.aux1_out = nn.Conv2d(
            *model_capacity_data[7], 
            bias=False
        )
        self.aux2_out = nn.Conv2d(
            *model_capacity_data[8], 
            bias=False
        )
        # Boundary calculations and padding configuration constants
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.offset = 128

    def forward(self, input_tensor):
        """
        Forward pass. Computes cascading target sub-band components and recovers shapes via padding.

        Args:
            input_tensor (torch.Tensor): Audio mixture spectrogram tensor of shape (N, 2, Bins, Frames).

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                In evaluation mode: Primary isolated signal tensor.
                In training mode: Tuple containing masked target signal and two multi-stage auxiliary predictions.
        """

        # Maintain an isolated detached pointer of the reference mix spectrogram
        mix = input_tensor.detach()
        # Slice down to target structural maximum frequency limits
        input_tensor = input_tensor.clone()
        input_tensor = input_tensor[:, :, : self.max_bin]
        # Calculate midpoint split threshold across frequency axis
        bandwidth = input_tensor.size()[2] // 2
        # Stage 1: Process independent sub-bands, then concat along frequency axis (dim=2)
        aux1 = torch.cat([
            self.stg1_low_band_net(input_tensor[:, :, :bandwidth]), 
            self.stg1_high_band_net(input_tensor[:, :, bandwidth:])
        ], dim=2)
        # Stage 2: Concatenate initial mix with stage 1 features along channels (dim=1)
        hidden_state = torch.cat([input_tensor, aux1], dim=1)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(hidden_state))
        # Stage 3: Merge cumulative multi-stage features for comprehensive full-band inference
        hidden_state = torch.cat([input_tensor, aux1, aux2], dim=1)
        hidden_state = self.stg3_full_band_net(self.stg3_bridge(hidden_state))
        # Project output estimation and scale back to original STFT footprint via edge replication
        mask = self.out(hidden_state).sigmoid()
        mask = F.pad(
            input=mask, 
            pad=(0, 0, 0, self.output_bin - mask.size()[2]), 
            mode="replicate"
        )

        if self.training:
            # Format and pad Stage 1 auxiliary output profiles for optimization tracking
            aux1 = self.aux1_out(aux1).sigmoid()
            aux1 = F.pad(
                input=aux1, 
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]), 
                mode="replicate"
            )
            # Format and pad Stage 2 auxiliary output profiles for optimization tracking
            aux2 = self.aux2_out(aux2).sigmoid()
            aux2 = F.pad(
                input=aux2, 
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]), 
                mode="replicate"
            )

            # Return element-wise masked component segments multiplied directly by the original mix
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            return mask

    def predict_mask(self, input_tensor):
        """
        Helper method that invokes the forward pass and clips frame overlap boundary margins.

        Args:
            input_tensor (torch.Tensor): Audio mix input spectrogram.

        Returns:
            torch.Tensor: Cropped evaluation isolation filter mask.
        """

        mask = self.forward(input_tensor)
        # Remove sliding window frame overlap margins along the time dimension if configured
        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]

        return mask