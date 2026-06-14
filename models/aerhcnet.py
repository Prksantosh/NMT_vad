import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


###############################################
# Channel Shuffle
###############################################
class ChannelShuffle(nn.Module):
    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups

        if c % g != 0:
            raise ValueError(f"Number of channels ({c}) must be divisible by groups ({g})")

        x = x.view(b, g, c // g, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, c, h, w)
        return x


###############################################
# Deformable Convolution Block
###############################################
class DeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()

        padding = kernel // 2
        offset_channels = 2 * kernel * kernel

        self.offset = nn.Conv2d(
            in_channels,
            offset_channels,
            kernel_size=kernel,
            padding=padding
        )

        self.deform = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            padding=padding
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        offset = self.offset(x)
        x = self.deform(x, offset)
        x = self.bn(x)
        return x


###############################################
# Updated RHC Block
###############################################
class RHCBlock(nn.Module):
    """
    Updated design:
      Input
      -> Regular Conv (3 -> 16)
      -> Split into 8 + 8
      -> Branch 1: Conv(8->16) + BN + Conv(16->32) + BN + ReLU
      -> Branch 2: DefConv(8->16) + DefConv(16->32) + ReLU
      -> Concat => 64 channels
      -> Channel Shuffle
      -> 1x1 fusion Conv (64 -> 32)
      -> Skip fusion from regular conv output (16 -> 32)
      -> BN + ReLU
      -> Output 32 channels
    """
    def __init__(self, in_channels=3, mid_channels=16, out_channels=32):
        super().__init__()

        if mid_channels % 2 != 0:
            raise ValueError("mid_channels must be even for channel split.")

        half = mid_channels // 2

        # Initial regular convolution: 3 -> 16
        self.initial = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Branch 1: regular convolution path
        self.branch1 = nn.Sequential(
            nn.Conv2d(half, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True)
        )

        # Branch 2: deformable convolution path
        self.branch2_def1 = DeformConvBlock(half, 16)
        self.branch2_def2 = DeformConvBlock(16, 32)
        self.branch2_relu = nn.ReLU(inplace=True)

        # Concatenated: 32 + 32 = 64
        self.shuffle = ChannelShuffle(groups=2)

        # Reduce concatenated output 64 -> 32
        self.fuse = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

        # Skip projection from regular conv output: 16 -> 32
        self.skip_proj = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Initial regular conv
        reg_out = self.initial(x)                 # (B,16,H,W)

        # Split into two 8-channel tensors
        x1, x2 = torch.chunk(reg_out, 2, dim=1)  # (B,8,H,W), (B,8,H,W)

        # Branch 1: regular conv branch
        b1 = self.branch1(x1)                    # (B,32,H,W)

        # Branch 2: deformable conv branch
        b2 = self.branch2_def1(x2)               # (B,16,H,W)
        b2 = self.branch2_def2(b2)               # (B,32,H,W)
        b2 = self.branch2_relu(b2)

        # Concatenate dual branches
        out = torch.cat([b1, b2], dim=1)         # (B,64,H,W)

        # Shuffle channels
        out = self.shuffle(out)                  # (B,64,H,W)

        # Fuse to 32 channels
        out = self.fuse(out)                     # (B,32,H,W)

        # Skip fusion from regular conv output
        skip = self.skip_proj(reg_out)           # (B,32,H,W)

        out = out + skip
        out = self.norm(out)
        out = self.relu(out)

        return out


###############################################
# SE Attention Block
###############################################
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        reduced = max(in_channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, reduced, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape

        squeeze = self.pool(x).view(b, c)
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(b, c, 1, 1)

        return x * excitation


###############################################
# AE-RHCNet Block (UPDATED)
###############################################
class AE_RHCNet(nn.Module):

    def __init__(self, in_channels, mid_channels=16, out_channels=32):
        super().__init__()

        # Updated RHC Block
        self.rhc = RHCBlock(
            in_channels=in_channels,
            mid_channels=mid_channels,
            out_channels=out_channels
        )

        self.se1 = SEBlock(out_channels)
        self.se2 = SEBlock(out_channels)

        # Residual projection (important!)
        self.identity_proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):

        identity = self.identity_proj(x)

        out = self.rhc(x)

        out = self.se1(out)
        out = self.se2(out)

        out = out + identity

        return out


###############################################
# Quick test
###############################################
if __name__ == "__main__":
    model = AE_RHCNet(in_channels=64, out_channels=128)
    x = torch.randn(3, 64, 58, 58)
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)