"""Multi-task loss formulation for Vision-to-Grammar model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kolam_r.grammar.vocabulary import PAD_IDX
from kolam_r.training.losses import GRID_SIZE_TO_IDX


class GrammarRecoveryLoss(nn.Module):
    """Calibrated sequence + auxiliary geometric loss for Grammar Recovery."""

    def __init__(
        self,
        weight_seq: float = 1.0,
        weight_sym: float = 0.5,
        weight_grid: float = 0.5,
        weight_angle: float = 1.5,
        weight_depth_direct: float = 0.5,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.w_seq = weight_seq
        self.w_sym = weight_sym
        self.w_grid = weight_grid
        self.w_angle = weight_angle
        self.w_depth_direct = weight_depth_direct

        self.seq_loss_fn = nn.CrossEntropyLoss(
            ignore_index=PAD_IDX, label_smoothing=label_smoothing
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_tokens: torch.Tensor,
        target_meta: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute sequence and auxiliary losses.

        Args:
            outputs: Model outputs containing 'token_logits' (B, T, vocab_size) and auxiliary heads.
            target_tokens: Ground truth token IDs (B, T). Target for token_logits[:, :-1] is target_tokens[:, 1:].
            target_meta: Dictionary of ground truth parameters.
        """
        token_logits = outputs["token_logits"]  # (B, T, V)

        # Shift sequence for standard autoregressive language modeling
        # Input tokens: y_0 ... y_{T-2}
        # Target tokens: y_1 ... y_{T-1}
        shift_logits = token_logits[:, :-1, :].contiguous()
        shift_targets = target_tokens[:, 1:].contiguous()

        loss_seq = self.seq_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1)
        )

        # Auxiliary heads
        loss_sym = self.ce_loss(outputs["logits_symmetry"], target_meta["symmetry"])

        grid_target = target_meta["grid_size"]
        if grid_target.max() > 3:
            grid_idx = torch.zeros_like(grid_target)
            for val, idx in GRID_SIZE_TO_IDX.items():
                grid_idx[grid_target == val] = idx
            grid_target = grid_idx
        loss_grid = self.ce_loss(outputs["logits_grid_size"], grid_target)

        target_angle_norm = target_meta["angle"].float() / 180.0
        loss_angle = self.smooth_l1(outputs["pred_angle_norm"], target_angle_norm)

        depth_target = target_meta["recursion_depth"] - 1
        loss_depth_direct = self.ce_loss(outputs["logits_depth_direct"], depth_target)

        total_loss = (
            self.w_seq * loss_seq
            + self.w_sym * loss_sym
            + self.w_grid * loss_grid
            + self.w_angle * loss_angle
            + self.w_depth_direct * loss_depth_direct
        )

        return {
            "total_loss": total_loss,
            "loss_seq": loss_seq,
            "loss_sym": loss_sym,
            "loss_grid": loss_grid,
            "loss_angle": loss_angle,
            "loss_depth_direct": loss_depth_direct,
        }
