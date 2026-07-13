import os
import sys
import torch

import torch.nn as nn

sys.path.append(os.getcwd())

from main.library.predictors.RMVPE.deepunet import DeepUnet, HPADeepUnet

N_MELS, N_CLASS = 128, 360

class BiGRU(nn.Module):
    """
    Bidirectional Gated Recurrent Unit (BiGRU) wrapper network block.

    This module encloses a standard PyTorch GRU configured for bidirectional execution.
    It includes an embedded error handling fallback mechanism to temporarily disable 
    cuDNN backend execution if it encounters memory layout or driver compatibility errors.
    """

    def __init__(
        self, 
        input_features, 
        hidden_features, 
        num_layers
    ):
        """
        Initializes the BiGRU block.

        Args:
            input_features (int): Dimensionality size of the incoming feature sequence vectors.
            hidden_features (int): Size of internal hidden state memory dimensions per direction.
            num_layers (int): Total number of stacked recurrent network layers.
        """

        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features, 
            hidden_features, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )

    def forward(self, x):
        """
        Executes sequential data forward pass processing with a cuDNN safety fallback shield.

        Args:
            x (torch.Tensor): Sequenced input tensor.

        Returns:
            torch.Tensor: Combined hidden sequence representations from both directions.
        """

        try:
            return self.gru(x)[0]
        except:
            # Fallback routine: temporarily disable cuDNN to calculate on vanilla backend if an exception occurs
            # Exceptions typically occur when the input is too large.
            torch.backends.cudnn.enabled = False
            return self.gru(x)[0]
        
class E2E(nn.Module):
    """
    End-to-End (E2E) Neural Network Architecture for the RMVPE Pitch Tracker.

    This architecture processes input Mel-spectrogram vectors using a choice of standard 
    DeepUnet or a High-Performance Hypergraph-enhanced (HPA) DeepUnet encoder backbone. 
    The intermediate outputs are refined using 2D CNN mapping and decoded via temporal 
    recurrent blocks (BiGRU) or feedforward linear blocks into multi-class pitch salience vectors.
    """

    def __init__(
        self, 
        n_blocks, 
        n_gru, 
        kernel_size, 
        en_de_layers=5, 
        inter_layers=4, 
        in_channels=1, 
        en_out_channels=16, 
        hpa=False
    ):
        """Initializes the End-to-End (E2E) feature processing architecture.

        Args:
            n_blocks (int): Number of internal layer blocks inside standard DeepUnet modules.
            n_gru (int): Total recurrent layer stack count inside the BiGRU network. If 0, replaces GRUs with Feedforward layers.
            kernel_size (int or tuple): Transposed convolution kernel window specifications.
            en_de_layers (int, default=5): Total layer depths of encoder-decoder chains.
            inter_layers (int, default=4): Total layer counts inside bottleneck layers.
            in_channels (int, default=1): Number of structural input channels for 2D convolutions.
            en_out_channels (int, default=16): Number of target feature channels leaving the network encoder block.
            hpa (bool, default=False): Flag to trigger the YOLOv13-based Hypergraph and FullPAD backbone variant.
        """

        super(E2E, self).__init__()
        # Branch dynamic instantiation of target network structural backbone
        self.unet = (
            HPADeepUnet(
                in_channels=in_channels, 
                en_out_channels=en_out_channels, 
                base_channels=64, 
                hyperace_k=2, 
                hyperace_l=1, 
                num_hyperedges=16, 
                num_heads=4
            ) 
        ) if hpa else (
            DeepUnet(
                kernel_size, 
                n_blocks, 
                en_de_layers, 
                inter_layers, 
                in_channels, 
                en_out_channels
            )
        )
        # 2D CNN projection to collapse spatial representations into standard mapping boundaries
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        # Configure the final classification head based on the requested GRU recurrent setup
        self.fc = (
            nn.Sequential(
                BiGRU(3 * 128, 256, n_gru), # Input features mapped across 3 channels * 128 Mel frequencies
                nn.Linear(512, N_CLASS),  # Map combined bidirectional outputs (256 * 2 = 512) to N_CLASS (360)
                nn.Dropout(0.25), 
                nn.Sigmoid() # Output normalized independent frame-level probabilities
            )
        ) if n_gru else (
            nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS), 
                nn.Dropout(0.25), 
                nn.Sigmoid()
            )
        )

    def forward(self, mel):
        """Processes input Mel spectrogram features to predict F0 pitch salience profiles.

        Args:
            mel (torch.Tensor): Input Mel-spectrogram tensor maps with shape (batch, channels, time_frames).

        Returns:
            torch.Tensor: Multi-class fundamental frequency classification matrix of shape (batch, time_frames, N_CLASS).
        """

        # Step 3: Classify sequence arrays across target pitch classes using the selected FC header network block
        return self.fc(
            # Step 2: Reduce channel spaces down using local convolutional wrappers
            self.cnn(
                # Step 1: Feed into structural Unet backbone networks
                self.unet(
                    mel.transpose(-1, -2).unsqueeze(1)
                )
            ).transpose(1, 2).flatten(-2)
        )