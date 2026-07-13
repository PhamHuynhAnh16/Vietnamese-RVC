import torch

class FCN(torch.nn.Sequential):
    """
    Fully Convolutional Network (FCN) model tailored for PENN pitch estimation.

    Processes overlapping contextual audio frames through a series of specialized 
    1D convolution layers to output discrete pitch bin probability logits.
    """

    def __init__(
        self, 
        channels = 256, 
        pitch_bins = 1440, 
        pooling = (2, 2)
    ):
        """
        Builds and orchestrates the deep sequential network layers.

        Args:
            channels (int): Base configuration multiplier for channel width tracking. Defaults to 256.
            pitch_bins (int): Final output logit classes (total discrete pitch bins). Defaults to 1440.
            pooling (Tuple[int, int]): Pooling factor configuration passed to feature blocks.
        """

        super().__init__(*(
            # Stage 1: Dense multi-channel projection with early structural downsampling
            Block(
                1, 
                channels, 
                481, 
                pooling
            ), 
            Block(
                channels, 
                channels // 8, 
                225, 
                pooling
            ), 
            Block(
                channels // 8, 
                channels // 8, 
                97, 
                pooling
            ), 
            # Stage 2: Feature expansion without spatial downsampling
            Block(
                channels // 8, 
                channels // 2, 
                66
            ), 
            Block(
                channels // 2, 
                channels, 
                35
            ), 
            Block(
                channels, 
                channels * 2, 
                4
            ), 
            # Stage 3: Final linear mapping down to target pitch bin structures
            torch.nn.Conv1d(
                channels * 2, 
                pitch_bins, 
                4
            )
        ))

    def forward(self, frames):
        """Executes the forward network tracking logic.

        Args:
            frames (torch.Tensor): Audio frames tensor.

        Returns:
            torch.Tensor: Unnormalized output logits.
        """

        # Crop explicit boundary margins to clip off unaligned context padding elements
        # Formulates structural inputs mapping exactly from 1024 down to 993 samples
        return super().forward(
            frames[:, :, 16:-15]
        )
    
class Block(torch.nn.Sequential):
    """
    A core building block for the PENN FCN pitch estimation architecture.

    Consists of a 1D Convolution, a ReLU activation, an optional MaxPool1d layer,
    and a trailing LayerNorm layer tracking precise sequence dimensions.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        length=1, 
        pooling=None, 
        kernel_size=32
    ):
        """
        Initializes structural block layers.

        Args:
            in_channels (int): Number of input feature maps or audio channels.
            out_channels (int): Number of output filters or hidden channels.
            length (int): Expected temporal length of the feature map for LayerNorm tracking.
            pooling (Tuple[int, int], optional): MaxPool1d configuration (kernel_size, stride).
            kernel_size (int): Size of the convolving 1D filter. Defaults to 32.
        """

        # 1. Instantiate basic convolution projection and its non-linear activation
        layers = (
            torch.nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size
            ), 
            torch.nn.ReLU()
        )

        # 2. Append downsampling pool layers if specified
        if pooling is not None: 
            layers += (
                torch.nn.MaxPool1d(*pooling),
            )

        # 3. Add spatial LayerNorm tracking exactly across (Channels, Length) dimensions
        layers += (
            torch.nn.LayerNorm((out_channels, length)),
        )

        super().__init__(*layers)