"""Vision-to-Grammar Encoder-Decoder Architecture for KOLAM-R.

Visual Encoder: 4-block ConvNet feature extractor mapping (B, 1, 64, 64) -> (B, 16, 256) visual tokens.
Grammar Decoder: Autoregressive Cross-Attention Transformer Decoder over constrained 24-token vocabulary.
Auxiliary Heads: Geometric attributes (symmetry, angle, grid size, direct depth baseline).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kolam_r.grammar.vocabulary import BOS_IDX, EOS_IDX, PAD_IDX, VOCAB_TOKENS
from kolam_r.models.cnn_baseline import ConvBlock
from kolam_r.schema import VALID_SYMMETRIES


class VisionToGrammarModel(nn.Module):
    """End-to-end Neural Vision-to-Grammar L-system program synthesizer."""

    def __init__(
        self,
        vocab_size: int = len(VOCAB_TOKENS),  # 24
        d_model: int = 256,
        nhead: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        max_seq_len: int = 110,
        dropout: float = 0.1,
        num_symmetries: int = len(VALID_SYMMETRIES),  # 6
        num_grid_sizes: int = 4,                      # 4 (3, 5, 7, 9)
        num_depths: int = 4,                          # 4 (d=1, 2, 3, 4)
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx = PAD_IDX
        self.bos_idx = BOS_IDX
        self.eos_idx = EOS_IDX

        # 1. Visual Feature Extractor (64x64 -> 4x4 spatial grid with 256 channels)
        self.conv_layer1 = ConvBlock(1, 32)
        self.conv_layer2 = ConvBlock(32, 64)
        self.conv_layer3 = ConvBlock(64, 128)
        self.conv_layer4 = ConvBlock(128, d_model)

        # 16 visual spatial tokens + learnable 2D positional embeddings
        self.visual_pos_embed = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)

        # 2. Grammar Token & Positional Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.seq_pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # 4. Token Output Projection
        self.lm_head = nn.Linear(d_model, vocab_size)

        # 5. Auxiliary Geometric Parameter Heads (pooled from visual features)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_aux = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.head_symmetry = nn.Linear(d_model, num_symmetries)
        self.head_grid_size = nn.Linear(d_model, num_grid_sizes)
        self.head_depth_direct = nn.Linear(d_model, num_depths)  # Protocol C baseline control
        self.head_angle = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Normalized [0, 1] -> 0° to 180°
        )

    def encode_image(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract spatial visual tokens and pooled latent vector.

        Args:
            x: Input image tensor (B, 1, 64, 64).

        Returns:
            memory_tokens: (B, 16, 256) spatial visual memory tokens.
            pooled_feats: (B, 256) global visual representation.
        """
        feats = self.conv_layer1(x)
        feats = self.conv_layer2(feats)
        feats = self.conv_layer3(feats)
        feats = self.conv_layer4(feats)  # (B, 256, 4, 4)

        B, C, H, W = feats.shape
        # Flatten spatial dimensions to 16 tokens: (B, 16, 256)
        spatial_tokens = feats.flatten(2).transpose(1, 2)
        memory_tokens = spatial_tokens + self.visual_pos_embed

        pooled_feats = self.fc_aux(self.global_pool(feats))
        return memory_tokens, pooled_feats

    def forward(
        self,
        images: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forcing forward pass for training.

        Args:
            images: (B, 1, 64, 64)
            target_tokens: (B, T) token IDs including <BOS> and <EOS>.

        Returns:
            Dictionary containing token logits and auxiliary predictions.
        """
        B, T = target_tokens.shape
        memory_tokens, pooled_feats = self.encode_image(images)

        # Token embeddings + positional embeddings
        tok_emb = self.token_embed(target_tokens)
        pos_emb = self.seq_pos_embed[:, :T, :]
        tgt_embed = self.embed_dropout(tok_emb + pos_emb)

        # Causal mask for autoregressive self-attention
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=images.device)
        tgt_key_padding_mask = (target_tokens == self.pad_idx)

        # Transformer decoding with visual cross-attention
        dec_out = self.transformer_decoder(
            tgt=tgt_embed,
            memory=memory_tokens,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        token_logits = self.lm_head(dec_out)

        # Auxiliary head predictions
        logits_sym = self.head_symmetry(pooled_feats)
        logits_grid = self.head_grid_size(pooled_feats)
        logits_depth_direct = self.head_depth_direct(pooled_feats)
        pred_angle_norm = self.head_angle(pooled_feats).squeeze(-1)

        return {
            "token_logits": token_logits,
            "logits_symmetry": logits_sym,
            "logits_grid_size": logits_grid,
            "logits_depth_direct": logits_depth_direct,
            "pred_angle_norm": pred_angle_norm,
        }

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 64,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Autoregressive greedy sequence generation at inference time.

        Args:
            images: Input images (B, 1, 64, 64).
            max_length: Maximum decoding length.

        Returns:
            generated_token_ids: (B, T) tensor of decoded token IDs.
            aux_predictions: Dictionary of auxiliary attribute predictions.
        """
        B = images.size(0)
        device = images.device
        memory_tokens, pooled_feats = self.encode_image(images)

        # Auxiliary predictions
        aux_predictions = {
            "pred_symmetry": self.head_symmetry(pooled_feats).argmax(dim=-1),
            "pred_grid_size": self.head_grid_size(pooled_feats).argmax(dim=-1),
            "pred_depth_direct": self.head_depth_direct(pooled_feats).argmax(dim=-1) + 1,
            "pred_angle": self.head_angle(pooled_feats).squeeze(-1) * 180.0,
        }

        # Initialize sequence with <BOS>
        current_tokens = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(1, max_length):
            T = current_tokens.size(1)
            tok_emb = self.token_embed(current_tokens)
            pos_emb = self.seq_pos_embed[:, :T, :]
            tgt_embed = tok_emb + pos_emb

            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

            dec_out = self.transformer_decoder(
                tgt=tgt_embed,
                memory=memory_tokens,
                tgt_mask=causal_mask,
            )

            next_token_logits = self.lm_head(dec_out[:, -1, :])  # (B, vocab_size)
            # Mask out <PAD> and <BOS> from generation
            next_token_logits[:, self.pad_idx] = -float("inf")
            next_token_logits[:, self.bos_idx] = -float("inf")

            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # If finished, output <PAD>
            next_token = torch.where(finished.unsqueeze(1), self.pad_idx, next_token)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == self.eos_idx)
            if finished.all():
                break

        return current_tokens, aux_predictions
