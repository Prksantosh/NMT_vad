import torch.nn as nn
from models.encoder import EncoderStage
from models.decoder import DecoderStage
from models.timestamp import TimestampTransform
from models.convlstm import ConvLSTM


###############################################
# Full Autoencoder Model
###############################################
class RHCNetAutoencoder(nn.Module):

    def __init__(self, seq_len=4):

        super().__init__()

        self.seq_len = seq_len

        self.initial = nn.Conv2d(3, 64, 3, padding=1)

        self.enc1 = EncoderStage(64, 64)
        self.enc2 = EncoderStage(64, 128)
        self.enc3 = EncoderStage(128, 256)
        self.enc4 = EncoderStage(256, 512)

        self.timestamp = TimestampTransform()

        self.lstm1 = ConvLSTM(512, 512)
        self.lstm2 = ConvLSTM(512, 512)      

        self.dec1 = DecoderStage(512, 256)
        self.dec2 = DecoderStage(256, 128)
        self.dec3 = DecoderStage(128, 64)
        self.dec4 = DecoderStage(64, 32)

        self.final = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):

        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        x = self.initial(x)

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.timestamp(x, B, T)

        x, _ = self.lstm1(x)
        x, h = self.lstm2(x)

        x = h

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        x = self.final(x)

        return x
