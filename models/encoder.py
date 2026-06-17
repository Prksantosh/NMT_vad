import torch.nn as nn
from models.aerhcnet import AE_RHCNet


class EncoderStage(nn.Module):

    def __init__(self, in_ch, out_ch, mid_ch=16):

        super().__init__()

     
        self.aerhc = AE_RHCNet(
            in_channels=in_ch,
            mid_channels=mid_ch,
            out_channels=out_ch
        )


        self.down = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.aerhc(x)  
        x = self.down(x)    

        return x
