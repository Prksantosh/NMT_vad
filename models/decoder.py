
import torch.nn as nn
from models.rhcnet import RHCBlock


class DecoderStage(nn.Module):

    def __init__(self, in_ch, out_ch, mid_ch=16):

        super().__init__()

     
        self.rhc = RHCBlock(
            in_channels=in_ch,
            mid_channels=mid_ch,
            out_channels=in_ch   
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.rhc(x)   # (B, in_ch, H, W)

        x = self.up(x)    # (B, out_ch, 2H, 2W)

        return x
