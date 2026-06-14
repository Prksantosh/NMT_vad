import torch.nn as nn
#from models.rhcnet import RHCBlock
from models.aerhcnet import AE_RHCNet

###############################################
# Encoder Stage (Fixed)
###############################################
class EncoderStage(nn.Module):

    def __init__(self, in_ch, out_ch, mid_ch=16):

        super().__init__()

        # AE-RHCNet now outputs out_ch channels
        self.aerhc = AE_RHCNet(
            in_channels=in_ch,
            mid_channels=mid_ch,
            out_channels=out_ch
        )

        # Downsample AFTER channel expansion
        self.down = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.aerhc(x)   # (B, out_ch, H, W)
        x = self.down(x)    # (B, out_ch, H/2, W/2)

        return x
