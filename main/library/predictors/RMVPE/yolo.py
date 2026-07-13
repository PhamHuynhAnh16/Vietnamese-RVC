import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None):
    """
    Computes the padding value dynamically to retain identical spatial dimensions (same padding).

    Args:
        k (int or list): Kernel size used in the convolution layer.
        p (int or list, optional): Explicit padding value. Defaults to None.

    Returns:
        int or list: Dynamically calculated padding value.
    """

    if p is None: p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """
    Standard Convolutional Block utilizing the SiLU activation function.

    Consists of a structured sequence: 2D Convolution -> 2D Batch Normalization -> SiLU/Identity.
    """

    def __init__(
        self, 
        c1, 
        c2, 
        k=1, 
        s=1, 
        p=None, 
        g=1, 
        act=True
    ):
        """
        Initializes the standard Conv block.

        Args:
            c1 (int): Input channel count.
            c2 (int): Output channel count.
            k (int): Convolution kernel dimension. Default is 1.
            s (int): Convolution stride dimension. Default is 1.
            p (int, optional): Spatial padding. Evaluated automatically if None.
            g (int): Grouped convolution configurations. Default is 1.
            act (bool): Flag toggling SiLU activation vs Identity. Default is True.
        """

        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2) 
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        """Executes sequential Conv2d -> BatchNorm -> Activation operations."""

        return self.act(
            self.bn(
                self.conv(x)
            )
        )

class DSConv(nn.Module):
    """
    Depthwise Separable Convolution (DSConv) Block.

    Reduces computational complexity by splitting filtering and combination steps into a Depthwise Conv (spatial) 
    followed by a Pointwise Conv (channel cross-mapping).
    """

    def __init__(
        self, 
        c1, 
        c2, 
        k=3, 
        s=1, 
        p=None, 
        act=True
    ):
        """
        Initializes the Depthwise Separable Convolution block.

        Args:
            c1 (int): Input channel dimension.
            c2 (int): Output channel dimension.
            k (int): Depthwise kernel dimension size. Default is 3.
            s (int): Depthwise stride tracking. Default is 1.
            p (int, optional): Spatial padding parameters.
            act (bool): Toggles SiLU activation constraints. Default is True.
        """

        super().__init__()
        # Channel dimension-wise isolated spatial filters (groups=c1)
        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        # Pointwise 1x1 cross-channel combination layer
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        """Applies depthwise processing followed by pointwise cross-channel expansion."""

        return self.act(
            self.bn(
                self.pwconv(
                    self.dwconv(x)
                )
            )
        )

class DS_Bottleneck(nn.Module):
    """
    Standard Bottleneck Block built using Depthwise Separable Convolutions.

    Utilizes residual skip-connections if input and output dimensions match identically.
    """

    def __init__(
        self, 
        c1, 
        c2, 
        k=3, 
        shortcut=True
    ):
        """
        Initializes the DS_Bottleneck block.

        Args:
            c1 (int): Incoming spatial channel dimensions.
            c2 (int): Outgoing spatial channel dimensions.
            k (int): Target kernel window constraint for the second DSConv layer. Default is 3.
            shortcut (bool): Triggers residual identity pathing logic. Default is True.
        """

        super().__init__()
        self.dsconv1 = DSConv(c1, c1, k=3, s=1)
        self.dsconv2 = DSConv(c1, c2, k=k, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        """Passes inputs through dual DSConvs with an optional residual identity shortcut path."""

        return x + self.dsconv2(self.dsconv1(x)) if self.shortcut else self.dsconv2(self.dsconv1(x))

class DS_C3k(nn.Module):
    """YOLOv13 Cross-Stage Partial (CSP) block built via Depthwise Separable Bottlenecks."""

    def __init__(
        self, 
        c1, 
        c2, 
        n=1, 
        k=3, 
        e=0.5
    ):
        """
        Initializes the DS_C3k module.

        Args:
            c1 (int): Input channel dimension.
            c2 (int): Output channel dimension.
            n (int): Multi-stack count for internal DS_Bottleneck layers. Default is 1.
            k (int): Kernel settings inside bottleneck blocks. Default is 3.
            e (float): Scaling factor adjusting bottleneck channel width. Default is 0.5.
        """

        super().__init__()
        self.cv1 = Conv(c1, int(c2 * e), 1, 1)
        self.cv2 = Conv(c1, int(c2 * e), 1, 1)
        self.cv3 = Conv(2 * int(c2 * e), c2, 1, 1)
        self.m = nn.Sequential(
            *[
                DS_Bottleneck(
                    int(c2 * e), 
                    int(c2 * e), 
                    k=k, 
                    shortcut=True
                ) 
                for _ in range(n)
            ]
        )

    def forward(self, x):
        """Fuses deep bottleneck features and shallow skip features before spatial projection."""

        return self.cv3(
            torch.cat(
                (self.m(self.cv1(x)), self.cv2(x)), 
                dim=1
            )
        )

class DS_C3k2(nn.Module):
    """Advanced YOLOv13 CSP Variant wrapping a nested DS_C3k module block."""

    def __init__(
        self, 
        c1, 
        c2, 
        n=1, 
        k=3, 
        e=0.5
    ):
        """
        Initializes the DS_C3k2 module wrapper.

        Args:
            c1 (int): Input channel count.
            c2 (int): Output channel count.
            n (int): Depth loop multiplier count passed to the inner DS_C3k block. Default is 1.
            k (int): Kernel size for bottlenecks. Default is 3.
            e (float): Ratio constraint adjusting the hidden channel depth profiles. Default is 0.5.
        """

        super().__init__()
        self.cv1 = Conv(c1, int(c2 * e), 1, 1)
        self.m = DS_C3k(int(c2 * e), int(c2 * e), n=n, k=k, e=1.0)
        self.cv2 = Conv(int(c2 * e), c2, 1, 1)

    def forward(self, x):
        """Routes execution through cascading Conv -> DS_C3k -> Conv pipelines."""

        return self.cv2(
            self.m(
                self.cv1(x)
            )
        )

class AdaptiveHyperedgeGeneration(nn.Module):
    """
    Generates an adaptive incidence matrix to model relationships across spatial points.

    Combines dynamic context maps via global average/max pooling alongside global hyperedge 
    prototypes, mapping high-order correlations through a multi-head projection space.
    """

    def __init__(
        self, 
        in_channels, 
        num_hyperedges, 
        num_heads
    ):
        """
        Initializes the Hyperedge Generator.

        Args:
            in_channels (int): Input feature depth dimensions.
            num_hyperedges (int): Total number of hyperedges (M) to generate dynamically.
            num_heads (int): Head splits for the multi-head correlation space.
        """

        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = max(1, in_channels // num_heads)
        # Learnable global dictionary tracking reference structural shapes
        self.global_proto = nn.Parameter(torch.randn(num_hyperedges, in_channels))
        # Maps pooled statistical descriptors back into localized scaling weights
        self.context_mapper = nn.Linear(2 * in_channels, num_hyperedges * in_channels, bias=False)
        self.query_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """Generates the soft hypergraph incidence matrix"""
        B, N, C = x.shape
        P = (
            # Step 2: Combine local context projections with the global learnable prototypes
            self.global_proto.unsqueeze(0) + 
            self.context_mapper(
                # Step 1: Accumulate spatial context statistics using parallel Pooling operations
                torch.cat(
                    (
                        F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1).squeeze(-1), 
                        F.adaptive_max_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
                    ), 
                    dim=1
                )
            ).view(B, self.num_hyperedges, C))

        # Step 3: Scale affinities, average across multiple heads, and apply Softmax normalization
        return F.softmax((
            (
                # Step 3: Project inputs into query states and calculate dot-product affinity maps across heads
                self.query_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) @ P.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
            ) * self.scale
        ).mean(dim=1).permute(0, 2, 1), dim=-1)

class HypergraphConvolution(nn.Module):
    """
    Executes high-order spatial message-passing updates over a soft hypergraph network.

    Applies the mathematical projection sequence: Vertex -> Hyperedge -> Vertex.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels
    ):
        """
        Initializes the Hypergraph Convolution block.

        Args:
            in_channels (int): Incoming token feature dimensions.
            out_channels (int): Outgoing token feature dimensions.
        """

        super().__init__()
        self.W_e = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.SiLU()

    def forward(self, x, A):
        """Performs graph-based vertex feature updates using the incidence matrix."""

        # Vertex-to-Hyperedge aggregation followed by Hyperedge-to-Vertex distribution
        return x + self.act(self.W_v(A.transpose(1, 2).bmm(self.act(self.W_e(A.bmm(x))))))

class AdaptiveHypergraphComputation(nn.Module):
    """Orchestrates hypergraph generation and structural convolution over 2D feature maps."""

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        num_hyperedges, 
        num_heads
    ):
        super().__init__()
        self.adaptive_hyperedge_gen = AdaptiveHyperedgeGeneration(in_channels, num_hyperedges, num_heads)
        self.hypergraph_conv = HypergraphConvolution(in_channels, out_channels)

    def forward(self, x):
        """Flattens 2D feature grids into sequence tokens for hypergraph operations."""

        B, _, H, W = x.shape
        # Flatten spatial dimensions
        x_flat = x.flatten(2).permute(0, 2, 1)

        # Generate hypergraph structures, run graph convolutions, and reconstruct the 2D grid
        return self.hypergraph_conv(x_flat, self.adaptive_hyperedge_gen(x_flat)).permute(0, 2, 1).view(B, -1, H, W)

class C3AH(nn.Module):
    """Cross-Stage Partial (CSP) module integrated with Adaptive Hypergraph Computation (AHC)."""

    def __init__(
        self, 
        c1, 
        c2, 
        num_hyperedges, 
        num_heads, 
        e=0.5
    ):
        super().__init__()
        self.cv1 = Conv(c1, int(c1 * e), 1, 1)
        self.cv2 = Conv(c1, int(c1 * e), 1, 1)
        self.ahc = AdaptiveHypergraphComputation(int(c1 * e), int(c1 * e), num_hyperedges, num_heads)
        self.cv3 = Conv(2 * int(c1 * e), c2, 1, 1)

    def forward(self, x):
        """Splits the feature routing path into an active AHC stream and a standard skip stream."""

        return self.cv3(
            torch.cat(
                (self.ahc(self.cv2(x)), self.cv1(x)), 
                dim=1
            )
        )

class HyperACE(nn.Module):
    """
    Hypergraph-based Adaptive Correlation Enhancement (HyperACE) block from YOLOv13.

    Aggregates multi-scale visual features from all four encoder stages onto a single reference grid. 
    It then splits processing into a high-order hypergraph relationship branch and a low-order 
    local spatial convolution branch to capture comprehensive cross-frequency associations.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        num_hyperedges=16, 
        num_heads=8, 
        k=2, 
        l=1, 
        c_h=0.5, 
        c_l=0.25
    ):
        """
        Initializes the multi-branch HyperACE module.

        Args:
            in_channels (list): A list containing channel counts across the four encoder stages [P2, P3, P4, P5].
            out_channels (int): Output channel configuration leaving the fusion layer.
            num_hyperedges (int): Hyperedge count allocations. Default is 16.
            num_heads (int): Head splits inside AHC submodules. Default is 8.
            k (int): Number of stacked high-order hypergraph processing layers. Default is 2.
            l (int): Number of stacked low-order local DS_C3k convolution layers. Default is 1.
            c_h (float): Channel width ratio assigned to the high-order branch. Default is 0.5.
            c_l (float): Channel width ratio assigned to the low-order branch. Default is 0.25.
        """

        super().__init__()
        c2, c3, c4, c5 = in_channels 
        c_mid = c4 # Target intermediate destination reference channel footprint width
        self.fuse_conv = Conv(c2 + c3 + c4 + c5, c_mid, 1, 1) 
        # Calculate split slice channel allocations
        self.c_h = int(c_mid * c_h)
        self.c_l = int(c_mid * c_l)
        self.c_s = c_mid - self.c_h - self.c_l # Residual skip channel remainder size
        # Branch A: High-Order Structural Relationships (Cascaded C3AH Hypergraph layers)
        self.high_order_branch = nn.ModuleList([
            C3AH(
                self.c_h, 
                self.c_h, 
                num_hyperedges=num_hyperedges, 
                num_heads=num_heads, e=1.0
            ) 
            for _ in range(k)
        ])
        self.high_order_fuse = Conv(self.c_h * k, self.c_h, 1, 1)
        # Branch B: Low-Order Structural Relationships (Stacked Local DS Convolutions)
        self.low_order_branch = nn.Sequential(
            *[
                DS_C3k(
                    self.c_l, 
                    self.c_l, 
                    n=1, 
                    k=3, 
                    e=1.0
                ) 
                for _ in range(l)
            ]
        )
        self.final_fuse = Conv(self.c_h + self.c_l + self.c_s, out_channels, 1, 1)

    def forward(self, x):
        """Rescales and groups multi-scale features to compute adaptive high/low-order correlations."""

        B2, B3, B4, B5 = x 
        _, _, H4, W4 = B4.shape

        # Interpolate and concatenate all feature maps onto the reference grid resolution of stage B4
        x_h, x_l, x_s = self.fuse_conv(
            torch.cat(
                (
                    F.interpolate(
                        B2, 
                        size=(H4, W4), 
                        mode='bilinear', 
                        align_corners=False
                    ), 
                    F.interpolate(
                        B3, 
                        size=(H4, W4), 
                        mode='bilinear', 
                        align_corners=False
                    ), 
                    B4, 
                    F.interpolate(
                        B5, 
                        size=(H4, W4), 
                        mode='bilinear', 
                        align_corners=False
                    )
                ), 
                dim=1
            )
        ).split([self.c_h, self.c_l, self.c_s], dim=1) # Split into separate processing streams: high-order, low-order, and direct identity skip

        # Execute concurrent branch operations and combine the outputs through the final fusion layer
        return self.final_fuse(
            torch.cat(
                (
                    self.high_order_fuse(torch.cat([m(x_h) for m in self.high_order_branch], dim=1)), 
                    self.low_order_branch(x_l), 
                    x_s
                ), 
                dim=1
            )
        )

class GatedFusion(nn.Module):
    """Applies a learnable residual gate parameter ($\gamma$) to dynamically fuse two feature fields."""

    def __init__(
        self, 
        in_channels
    ):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, f_in, h):
        # Initialized to zero so that the network initially relies on the base feature stream
        return f_in + self.gamma * h

class YOLO13Encoder(nn.Module):
    """
    Multi-stage Feature Downsampling Encoder Backbone styled after YOLOv13.

    Sequentially processes raw spectrogram maps down across 4 spatial dimensions [P2, P3, P4, P5], 
    utilizing Depthwise Separable Convolutions to optimize efficiency.
    """

    def __init__(
        self, 
        in_channels, 
        base_channels=32
    ):
        super().__init__()
        # Initial downsampling block
        self.stem = DSConv(
            in_channels, 
            base_channels, 
            k=3, 
            s=1
        )
        # Stage P2: Downsample and process features
        self.p2 = nn.Sequential(
            DSConv(
                base_channels, 
                base_channels*2, k=3, s=(2, 2)), 
            DS_C3k2(
                base_channels*2, 
                base_channels*2, 
                n=1
            )
        )
        # Stage P3
        self.p3 = nn.Sequential(
            DSConv(
                base_channels*2, 
                base_channels*4, 
                k=3, 
                s=(2, 2)
            ), 
            DS_C3k2(
                base_channels*4, 
                base_channels*4, 
                n=2
            )
        )
        # Stage P4
        self.p4 = nn.Sequential(
            DSConv(
                base_channels*4, 
                base_channels*8, 
                k=3, 
                s=(2, 2)
            ), 
            DS_C3k2(
                base_channels*8, 
                base_channels*8, 
                n=2
            )
        )
        # Stage P5
        self.p5 = nn.Sequential(
            DSConv(
                base_channels*8, 
                base_channels*16, 
                k=3, 
                s=(2, 2)
            ), 
            DS_C3k2(
                base_channels*16, 
                base_channels*16, 
                n=1
            )
        )
        
        self.out_channels = [base_channels*2, base_channels*4, base_channels*8, base_channels*16]

    def forward(self, x):
        """Passes inputs forward through the cascading encoder stages, returning all intermediate multi-scale feature maps."""
    
        p2 = self.p2(self.stem(x))
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        return [p2, p3, p4, p5]

class YOLO13FullPADDecoder(nn.Module):
    """
    Full-Pipeline Aggregation-and-Distribution (FullPAD) Decoder module.

    Implements top-down feature distribution pathways by dynamically incorporating 
    global relational cues from the HyperACE module into each decoder stage 
    using learnable GatedFusion gates.
    """

    def __init__(self, encoder_channels, hyperace_out_c, out_channels_final):
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels
        c_d5, c_d4, c_d3, c_d2 = c_p5, c_p4, c_p3, c_p2
        
        # Projection convolutions mapping HyperACE features to respective decoder stage depths
        self.h_to_d5 = Conv(
            hyperace_out_c, 
            c_d5, 
            1, 
            1
        )
        self.h_to_d4 = Conv(
            hyperace_out_c, 
            c_d4, 
            1, 
            1
        )
        self.h_to_d3 = Conv(
            hyperace_out_c, 
            c_d3, 
            1, 
            1
        )
        self.h_to_d2 = Conv(
            hyperace_out_c, 
            c_d2, 
            1, 
            1
        )

        # Dynamic gated fusion blocks
        self.fusion_d5 = GatedFusion(c_d5)
        self.fusion_d4 = GatedFusion(c_d4)
        self.fusion_d3 = GatedFusion(c_d3)
        self.fusion_d2 = GatedFusion(c_d2)

        # Lateral skip-connection layers mapping encoder features to decoder spaces
        self.skip_p5 = Conv(
            c_p5, 
            c_d5, 
            1, 
            1
        )
        self.skip_p4 = Conv(
            c_p4, 
            c_d4, 
            1, 
            1
        )
        self.skip_p3 = Conv(
            c_p3, 
            c_d3, 
            1, 
            1
        )
        self.skip_p2 = Conv(
            c_p2, 
            c_d2, 
            1, 
            1
        )

        # Top-down upsampling CSP convolvers
        self.up_d5 = DS_C3k2(
            c_d5, 
            c_d4, 
            n=1
        )
        self.up_d4 = DS_C3k2(
            c_d4, 
            c_d3, 
            n=1
        )
        self.up_d3 = DS_C3k2(
            c_d3, 
            c_d2, 
            n=1
        )
        
        self.final_d2 = DS_C3k2(
            c_d2, 
            c_d2, 
            n=1
        )
        self.final_conv = Conv(
            c_d2,
            out_channels_final, 
            1, 
            1
        )

    def forward(self, enc_feats, h_ace):
        """Executes FullPAD multi-level top-down aggregation loops with hypergraph feature injection."""

        p2, p3, p4, p5 = enc_feats
        d5 = self.skip_p5(p5)

        d4 = self.up_d5(
            F.interpolate(
                self.fusion_d5(d5, self.h_to_d5(F.interpolate(h_ace, size=d5.shape[2:], mode='bilinear', align_corners=False))), 
                size=p4.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        ) + self.skip_p4(p4)

        d3 = self.up_d4(
            F.interpolate(
                self.fusion_d4(d4, self.h_to_d4(F.interpolate(h_ace, size=d4.shape[2:], mode='bilinear', align_corners=False))), 
                size=p3.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        ) + self.skip_p3(p3)

        d2 = self.up_d3(
            F.interpolate(
                self.fusion_d3(d3, self.h_to_d3(F.interpolate(h_ace, size=d3.shape[2:], mode='bilinear', align_corners=False))), 
                size=p2.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        ) + self.skip_p2(p2)

        return self.final_conv(
            self.final_d2(
                self.fusion_d2(
                    d2, 
                    self.h_to_d2(
                        F.interpolate(h_ace, size=d2.shape[2:], mode='bilinear', align_corners=False)
                    )
                )
            )
        )