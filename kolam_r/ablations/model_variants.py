"""Ablation model variants for KOLAM-R.

Includes:
1. VisionToGrammarSequenceOnly: No auxiliary heads (trained with pure sequence CE).
2. VisionToGrammarResNet: ResNet-18 visual encoder backbone + Transformer decoder.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

from kolam_r.grammar.vocabulary import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    VOCAB_TOKENS,
)

VOCAB_SIZE = len(VOCAB_TOKENS)
SOS_ID = BOS_IDX
EOS_ID = EOS_IDX
PAD_ID = PAD_IDX


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, max_len: int = 128) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class VisionToGrammarSequenceOnly(nn.Module):
    """Ablation 1: Sequence-Only Model with NO auxiliary parameter heads."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Visual Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # Transformer Decoder
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.encoder_conv(images)  # (B, d_model, 4, 4)
        b, c, h, w = feat.shape
        return feat.view(b, c, h * w).permute(0, 2, 1)  # (B, 16, d_model)

    def forward(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        memory = self.encode_visual(images)
        tgt_emb = self.token_embed(tgt_tokens) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.lm_head(out)
        return logits, {}

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_length: int = 64) -> tuple[torch.Tensor, dict]:
        self.eval()
        b = images.size(0)
        device = images.device
        memory = self.encode_visual(images)
        curr = torch.full((b, 1), SOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_length):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(curr.size(1), device=device)
            tgt_emb = self.token_embed(curr) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            out = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            next_logits = self.lm_head(out[:, -1, :])
            next_tokens = next_logits.argmax(dim=-1)
            curr = torch.cat([curr, next_tokens.unsqueeze(1)], dim=1)
            finished |= (next_tokens == EOS_ID)
            if finished.all():
                break

        return curr, {}


class ResNetBlock(nn.Module):
    """Classic 2-conv residual block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class VisionToGrammarResNet(nn.Module):
    """Ablation 2: ResNet Visual Backbone + Transformer Decoder + Aux Heads."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # ResNet Backbone (4 residual stages: 64x64 -> 4x4)
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResNetBlock(32, 64, stride=2)    # -> 32x32
        self.layer2 = ResNetBlock(64, 128, stride=2)   # -> 16x16
        self.layer3 = ResNetBlock(128, 256, stride=2)  # -> 8x8
        self.layer4 = ResNetBlock(256, d_model, stride=2)  # -> 4x4

        # Auxiliary Heads
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_symmetry = nn.Linear(d_model, 6)
        self.head_grid_size = nn.Linear(d_model, 4)
        self.head_depth = nn.Linear(d_model, 4)
        self.head_angle = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

        # Transformer Decoder
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode_visual(self, images: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x = self.in_conv(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)  # (B, d_model, 4, 4)

        # Global aux predictions
        pooled = self.global_pool(feat_map).view(images.size(0), -1)
        aux_preds = {
            "pred_symmetry": self.head_symmetry(pooled).argmax(dim=-1),
            "pred_grid_size": self.head_grid_size(pooled).argmax(dim=-1),
            "pred_depth_direct": self.head_depth(pooled).argmax(dim=-1) + 1,
            "pred_angle": self.head_angle(pooled).squeeze(-1) * 180.0,
        }

        b, c, h, w = feat_map.shape
        memory = feat_map.view(b, c, h * w).permute(0, 2, 1)
        return memory, aux_preds

    def forward(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory, _ = self.encode_visual(images)
        pooled = self.global_pool(self.layer4(self.layer3(self.layer2(self.layer1(self.in_conv(images)))))).view(images.size(0), -1)
        
        tgt_emb = self.token_embed(tgt_tokens) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        token_logits = self.lm_head(out)
        
        return {
            "token_logits": token_logits,
            "logits_symmetry": self.head_symmetry(pooled),
            "logits_grid_size": self.head_grid_size(pooled),
            "logits_depth_direct": self.head_depth(pooled),
            "pred_angle_norm": self.head_angle(pooled).squeeze(-1),
        }

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_length: int = 64) -> tuple[torch.Tensor, dict]:
        self.eval()
        b = images.size(0)
        device = images.device
        memory, aux_preds = self.encode_visual(images)
        curr = torch.full((b, 1), SOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_length):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(curr.size(1), device=device)
            tgt_emb = self.token_embed(curr) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            out = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            next_logits = self.lm_head(out[:, -1, :])
            next_tokens = next_logits.argmax(dim=-1)
            curr = torch.cat([curr, next_tokens.unsqueeze(1)], dim=1)
            finished |= (next_tokens == EOS_ID)
            if finished.all():
                break

        return curr, aux_preds
