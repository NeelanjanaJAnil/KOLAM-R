"""Multi-task loss formulation for KOLAM-R.

Combines categorical cross-entropy losses for discrete parameter heads
with normalized continuous regression loss for turning angle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Map grid size integer (3, 5, 7, 9) to class index (0, 1, 2, 3)
GRID_SIZE_TO_IDX = {3: 0, 5: 1, 7: 2, 9: 3}
IDX_TO_GRID_SIZE = {0: 3, 1: 5, 2: 7, 3: 9}


class MultiTaskKolamLoss(nn.Module):
    """Calibrated multi-task loss for Kolam parameter prediction."""

    def __init__(
        self,
        weight_rule: float = 1.0,
        weight_sym: float = 1.0,
        weight_motif: float = 0.5,
        weight_depth: float = 1.0,
        weight_grid: float = 1.0,
        weight_angle: float = 2.0,
        angle_max_deg: float = 180.0,
    ) -> None:
        super().__init__()
        self.w_rule = weight_rule
        self.w_sym = weight_sym
        self.w_motif = weight_motif
        self.w_depth = weight_depth
        self.w_grid = weight_grid
        self.w_angle = weight_angle
        self.angle_max_deg = angle_max_deg

        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute individual and weighted total loss.

        Args:
            outputs: Dictionary of model predictions.
            targets: Dictionary of target tensors.

        Returns:
            Dictionary containing 'total_loss' and all component loss scalars.
        """
        # 1. Production Rule loss (Categorical: 6 classes)
        loss_rule = self.ce_loss(outputs["logits_rule"], targets["rule_id"])

        # 2. Symmetry loss (Categorical: 6 classes)
        loss_sym = self.ce_loss(outputs["logits_symmetry"], targets["symmetry"])

        # 3. Motif loss (Categorical: 4 classes)
        loss_motif = self.ce_loss(outputs["logits_motif"], targets["motif"])

        # 4. Recursion Depth loss (Categorical: depths 1..4 -> indices 0..3)
        depth_target = targets["recursion_depth"] - 1
        loss_depth = self.ce_loss(outputs["logits_depth"], depth_target)

        # 5. Grid Size loss (Categorical: 3, 5, 7, 9 -> indices 0..3)
        grid_target = targets["grid_size"]
        # Map values if tensor contains raw sizes
        if grid_target.max() > 3:
            grid_idx = torch.zeros_like(grid_target)
            for val, idx in GRID_SIZE_TO_IDX.items():
                grid_idx[grid_target == val] = idx
            grid_target = grid_idx
        loss_grid = self.ce_loss(outputs["logits_grid_size"], grid_target)

        # 6. Angle regression loss (Continuous: normalized to [0, 1])
        target_angle_norm = targets["angle"].float() / self.angle_max_deg
        loss_angle = self.smooth_l1(outputs["pred_angle_norm"], target_angle_norm)

        # Total combined loss
        total_loss = (
            self.w_rule * loss_rule
            + self.w_sym * loss_sym
            + self.w_motif * loss_motif
            + self.w_depth * loss_depth
            + self.w_grid * loss_grid
            + self.w_angle * loss_angle
        )

        return {
            "total_loss": total_loss,
            "loss_rule": loss_rule,
            "loss_symmetry": loss_sym,
            "loss_motif": loss_motif,
            "loss_depth": loss_depth,
            "loss_grid_size": loss_grid,
            "loss_angle": loss_angle,
        }
