# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:11:34 2026

@author: USER
"""

import torch
import torch.nn as nn



###############################################
# Utility: LayerNorm for last dimension tensors
###############################################
class LayerNormLastDim(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)


###############################################
# Local Temporal Enhancement Branch
# Captures short-range temporal continuity
###############################################
class LocalTemporalConv(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.dw_conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False
        )

        self.pw_conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm3d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C, T, H, W)
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        out = self.bn(out)
        out = self.act(out)
        return out


###############################################
# Memory-Guided Temporal Attention
# Attention across time + learnable memory bank
###############################################
class MemoryGuidedTemporalAttention(nn.Module):
    def __init__(self, channels, num_heads=4, memory_slots=16, dropout=0.1):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.memory_slots = memory_slots

        self.norm = LayerNormLastDim(channels)

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)

        # Learnable normal-pattern memory bank
        self.memory = nn.Parameter(torch.randn(memory_slots, channels))

        self.mem_k_proj = nn.Linear(channels, channels, bias=False)
        self.mem_v_proj = nn.Linear(channels, channels, bias=False)

        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Memory fusion gate
        self.gate_proj = nn.Linear(channels * 2, channels)

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        return: (B, C, T, H, W)
        """

        B, C, T, H, W = x.shape
        N = H * W

        # Rearrange to per-spatial-location temporal tokens
        # (B, C, T, H, W) -> (B, H, W, T, C) -> (B*N, T, C)
        x_tokens = x.permute(0, 3, 4, 2, 1).contiguous().view(B * N, T, C)

        x_norm = self.norm(x_tokens)

        q = self.q_proj(x_norm)  # (B*N, T, C)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Prepare memory keys/values
        mem = self.memory.unsqueeze(0).expand(B * N, -1, -1)  # (B*N, M, C)
        mem_k = self.mem_k_proj(mem)
        mem_v = self.mem_v_proj(mem)

        # Concatenate temporal tokens with memory tokens
        k_all = torch.cat([k, mem_k], dim=1)  # (B*N, T+M, C)
        v_all = torch.cat([v, mem_v], dim=1)

        # Multi-head reshape
        q = q.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_all = k_all.view(B * N, T + self.memory_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v_all = v_all.view(B * N, T + self.memory_slots, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k_all.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_all)  # (B*N, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B * N, T, C)

        # Explicit memory read for anomaly-aware fusion
        mem_attn = torch.matmul(
            self.q_proj(x_norm),
            self.memory.t()
        ) * (C ** -0.5)
        mem_attn = torch.softmax(mem_attn, dim=-1)  # (B*N, T, M)
        mem_read = torch.matmul(mem_attn, self.memory)  # (B*N, T, C)

        fused = torch.cat([out, mem_read], dim=-1)
        gate = torch.sigmoid(self.gate_proj(fused))
        out = gate * out + (1.0 - gate) * mem_read

        out = self.out_proj(out)

        # Back to (B, C, T, H, W)
        out = out.view(B, H, W, T, C).permute(0, 4, 3, 1, 2).contiguous()

        return out


###############################################
# Feed Forward Network for 3D features
###############################################
class TemporalFeedForward(nn.Module):
    def __init__(self, channels, expansion=4, dropout=0.1):
        super().__init__()

        hidden = channels * expansion

        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


###############################################
# One Transformer Block
###############################################
class MemoryGuidedTemporalBlock(nn.Module):
    def __init__(self, channels, num_heads=4, memory_slots=16, dropout=0.1):
        super().__init__()

        self.local_branch = LocalTemporalConv(channels)
        self.attn_branch = MemoryGuidedTemporalAttention(
            channels=channels,
            num_heads=num_heads,
            memory_slots=memory_slots,
            dropout=dropout
        )

        self.fusion = nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)

        self.ffn = TemporalFeedForward(channels, expansion=4, dropout=dropout)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        # x: (B, C, T, H, W)

        local_feat = self.local_branch(x)
        attn_feat = self.attn_branch(x)

        fused = torch.cat([local_feat, attn_feat], dim=1)
        fused = self.fusion(fused)

        x = self.bn1(x + fused)
        x = self.bn2(x + self.ffn(x))

        return x


###############################################
# Final Novel Temporal Module
# Drop-in replacement for E3D-LSTM
###############################################
class MemoryGuidedTemporalTransformer(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=4,
        memory_slots=16,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            MemoryGuidedTemporalBlock(
                channels=channels,
                num_heads=num_heads,
                memory_slots=memory_slots,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        return: (B, C, T, H, W)
        """
        for layer in self.layers:
            x = layer(x)
        return x