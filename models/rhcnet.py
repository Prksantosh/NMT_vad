import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


# =========================================================
# Channel Shuffle
# =========================================================
class ChannelShuffle(nn.Module):
    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups

        assert c % g == 0, "Channels must be divisible by groups"

        x = x.view(b, g, c // g, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, c, h, w)

        return x


# =========================================================
# Deformable Conv Block
# =========================================================
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


# =========================================================
# RHC Block (Updated)
# =========================================================
class RHCBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        # USE out_channels dynamically
        self.initial = nn.Conv2d(in_channels, mid_channels, 3, padding=1)

        half = mid_channels // 2

        self.branch1 = nn.Sequential(
            nn.Conv2d(half, out_channels//2, 3, padding=1),
            nn.BatchNorm2d(out_channels//2),

            nn.Conv2d(out_channels//2, out_channels//2, 3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )

        self.def1 = DeformConvBlock(half, out_channels//2)
        self.def2 = DeformConvBlock(out_channels//2, out_channels//2)

        self.shuffle = ChannelShuffle(2)

        self.fuse = nn.Conv2d(out_channels, out_channels, 1)

        self.skip_proj = nn.Conv2d(mid_channels, out_channels, 1)

        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        reg = self.initial(x)              # (B,16,H,W)

        x1, x2 = torch.chunk(reg, 2, dim=1)  # (B,8,H,W)

        # Branch 1
        b1 = self.branch1(x1)            # (B,32,H,W)

        # Branch 2
        b2 = self.def1(x2)
        b2 = self.def2(b2)
        b2 = self.relu(b2)              # (B,32,H,W)

        # Concat
        out = torch.cat([b1, b2], dim=1)   # (B,64,H,W)

        # Shuffle
        out = self.shuffle(out)

        # Reduce channels
        out = self.fuse(out)              # (B,32,H,W)

        # Skip connection
        skip = self.skip_proj(reg)

        out = out + skip
        out = self.norm(out)
        out = self.relu(out)

        return out


