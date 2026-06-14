# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:00:37 2026

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import EncoderStage
from models.decoder import DecoderStage
from models.emu_timestamp import TimestampTransform
from models.temp_trans import MemoryGuidedTemporalTransformer


class RHCNetAutoencoder(nn.Module):

    def __init__(self, seq_len=3):
        super().__init__()

        self.seq_len = seq_len

        self.initial = nn.Conv2d(3, 24, 3, padding=1)

        # Encoder
        self.enc1 = EncoderStage(24, 48)
        self.enc2 = EncoderStage(48, 96)
        self.enc3 = EncoderStage(96, 192)
        self.enc4 = EncoderStage(192, 256)

        self.timestamp = TimestampTransform()

        self.temporal = MemoryGuidedTemporalTransformer(
            channels=256,
            num_heads=8,
            memory_slots=150,
            num_layers=2
        )

        # Decoder
        self.dec1 = DecoderStage(256, 192)
        self.dec2 = DecoderStage(192, 96)
        self.dec3 = DecoderStage(96, 48)
        self.dec4 = DecoderStage(48, 24)

        # Skip fusion layers
        self.skip1 = nn.Conv2d(192 + 256, 192, kernel_size=1)
        self.skip2 = nn.Conv2d(96 + 192, 96, kernel_size=1)
        self.skip3 = nn.Conv2d(48 + 96, 48, kernel_size=1)
        self.skip4 = nn.Conv2d(24 + 48, 24, kernel_size=1)

        self.final = nn.Conv2d(24, 3, 3, padding=1)

    def get_last_timestep(self, feat, B, T):
        """
        Convert encoder feature from (B*T, C, H, W)
        to last time-step feature (B, C, H, W)
        """
        _, C, H, W = feat.shape
        feat = feat.view(B, T, C, H, W)
        return feat[:, -1]

    def align_and_concat(self, x, skip):
        """
        Align spatial size of skip with decoder feature map.
        """
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(
                skip,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        x = self.initial(x)

        # Encoder with direct skip storage
        e1 = self.enc1(x)      # 64 channels
        e2 = self.enc2(e1)     # 128 channels
        e3 = self.enc3(e2)     # 256 channels
        e4 = self.enc4(e3)     # 256 channels

        # Convert encoder skips to last time-step only
        s1 = self.get_last_timestep(e1, B, T)
        s2 = self.get_last_timestep(e2, B, T)
        s3 = self.get_last_timestep(e3, B, T)
        s4 = self.get_last_timestep(e4, B, T)

        # Temporal modeling
        x = self.timestamp(e4, B, T)     # (B, C, T, H, W)
        x = self.temporal(x)
        x = x[:, :, -1]                 # (B, C, H, W)

        # Decoder with aligned direct skip connections
        x = self.dec1(x)
        x = self.align_and_concat(x, s4)
        x = self.skip1(x)

        x = self.dec2(x)
        x = self.align_and_concat(x, s3)
        x = self.skip2(x)

        x = self.dec3(x)
        x = self.align_and_concat(x, s2)
        x = self.skip3(x)

        x = self.dec4(x)
        x = self.align_and_concat(x, s1)
        x = self.skip4(x)

        x = self.final(x)

        return x