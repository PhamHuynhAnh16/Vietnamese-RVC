import torch

PITCH_BINS = 360

class CREPEE(torch.nn.Module):
    """
    CREPEE model architecture for pitch estimation using deep convolutional networks.

    This class implements the core neural network of CREPE, supporting five model sizes:
    'full', 'large', 'medium', 'small', and 'tiny'. It processes 1D raw audio signals mapped
    into 2D space, applying convolutional blocks, max-pooling, and a linear classifier.
    """

    def __init__(self, model='full'):
        """
        Initializes the CREPEE model architecture according to the specified size.

        Args:
            model (str): The configuration capacity size. Options are 'full', 'large', 'medium', 'small', or 'tiny'. Default is 'full'.
        """
    
        super().__init__()
        # Define the layer configuration dictionaries mapped to model capacity
        in_channels = {"full": [1, 1024, 128, 128, 128, 256], "large": [1, 768, 96, 96, 96, 192], "medium": [1, 512, 64, 64, 64, 128], "small": [1, 256, 32, 32, 32, 64], "tiny": [1, 128, 16, 16, 16, 32]}[model]
        out_channels = {"full": [1024, 128, 128, 128, 256, 512], "large": [768, 96, 96, 96, 192, 384], "medium": [512, 64, 64, 64, 128, 256], "small": [256, 32, 32, 32, 64, 128], "tiny": [128, 16, 16, 16, 32, 64]}[model]
        self.in_features = {"full": 2048, "large": 1536, "medium": 1024, "small": 512, "tiny": 256}[model]

        # Convolutional parameters setup: first layer has a large receptive field (512, 1)
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        # Layer 1 Block (Processes raw audio waveforms down-sampled temporally by stride 4)
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_sizes[0], stride=strides[0])
        self.conv1_BN = torch.nn.BatchNorm2d(num_features=out_channels[0], eps=0.0010000000474974513, momentum=0.0)
        # Layer 2 Block
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_sizes[1], stride=strides[1])
        self.conv2_BN = torch.nn.BatchNorm2d(num_features=out_channels[1], eps=0.0010000000474974513, momentum=0.0)
        # Layer 3 Block
        self.conv3 = torch.nn.Conv2d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_sizes[2], stride=strides[2])
        self.conv3_BN = torch.nn.BatchNorm2d(num_features=out_channels[2], eps=0.0010000000474974513, momentum=0.0)
        # Layer 4 Block
        self.conv4 = torch.nn.Conv2d(in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=kernel_sizes[3], stride=strides[3])
        self.conv4_BN = torch.nn.BatchNorm2d(num_features=out_channels[3], eps=0.0010000000474974513, momentum=0.0)
        # Layer 5 Block
        self.conv5 = torch.nn.Conv2d(in_channels=in_channels[4], out_channels=out_channels[4], kernel_size=kernel_sizes[4], stride=strides[4])
        self.conv5_BN = torch.nn.BatchNorm2d(num_features=out_channels[4], eps=0.0010000000474974513, momentum=0.0)
        # Layer 6 Block
        self.conv6 = torch.nn.Conv2d(in_channels=in_channels[5], out_channels=out_channels[5], kernel_size=kernel_sizes[5], stride=strides[5])
        self.conv6_BN = torch.nn.BatchNorm2d(num_features=out_channels[5], eps=0.0010000000474974513, momentum=0.0)
        
        # Classification linear head maps latent features to pitch bin activations
        self.classifier = torch.nn.Linear(in_features=self.in_features, out_features=PITCH_BINS)

    def forward(self, x, embed=False):
        """Executes the forward pass of the network.

        Args:
            x (torch.Tensor): Input audio frames tensor.
            embed (bool): If True, returns the latent embeddings from layer 5 instead of pitch bins.

        Returns:
            torch.Tensor: Normalized class probabilities (via Sigmoid) or latent representations.
        """

        # Extract features up to layer 5
        x = self.embed(x)
        if embed: return x

        # Execute final conv layer, flatten spatial dimensions, and map to activation values
        return self.classifier(self.layer(x, self.conv6, self.conv6_BN).permute(0, 2, 1, 3).reshape(-1, self.in_features)).sigmoid()

    def embed(self, x):
        """Processes the tensor through the first 5 convolutional layer blocks.

        Args:
            x (torch.Tensor): Raw 1D input tensor of shape (batch, width).

        Returns:
            torch.Tensor: Latent feature maps block tensor.
        """

        # Reshape 1D audio sequence into a 4D tensor
        x = x[:, None, :, None]
        # Chain consecutive block processing pipelines sequentially
        return self.layer(self.layer(self.layer(self.layer(self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254)), self.conv2, self.conv2_BN), self.conv3, self.conv3_BN), self.conv4, self.conv4_BN), self.conv5, self.conv5_BN)

    def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
        """A reusable helper function representing a single CREPE network layer block.

        Applies asymmetrical manual padding, 2D convolution, ReLU activation, 
        Batch Normalization, and spatial (2, 1) down-sampling Max Pooling.

        Args:
            x (torch.Tensor): Input activation map tensor.
            conv (torch.nn.Conv2d): Convolutional layer module to evaluate.
            batch_norm (torch.nn.BatchNorm2d): Associated batch normalization module.
            padding (tuple): Left, Right, Top, Bottom manual zero padding configuration.

        Returns:
            torch.Tensor: Block downsampled representation matrix.
        """

        # Pad -> Conv -> ReLU -> BatchNorm -> MaxPool along the temporal dimension
        return torch.nn.functional.max_pool2d(batch_norm(torch.nn.functional.relu(conv(torch.nn.functional.pad(x, padding)))), (2, 1), (2, 1))