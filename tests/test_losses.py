"""Tests for multi-task loss computation."""

import torch
import pytest

from kolam_r.training.losses import MultiTaskKolamLoss


class TestMultiTaskLoss:
    """Tests for MultiTaskKolamLoss."""

    def test_loss_computation_and_keys(self):
        """Loss must return total_loss and individual components as valid scalars."""
        criterion = MultiTaskKolamLoss()
        batch_size = 4

        outputs = {
            "logits_rule": torch.randn(batch_size, 6),
            "logits_symmetry": torch.randn(batch_size, 6),
            "logits_motif": torch.randn(batch_size, 4),
            "logits_depth": torch.randn(batch_size, 4),
            "logits_grid_size": torch.randn(batch_size, 4),
            "pred_angle_norm": torch.sigmoid(torch.randn(batch_size)),
        }

        targets = {
            "rule_id": torch.tensor([0, 1, 2, 3]),
            "symmetry": torch.tensor([0, 1, 2, 3]),
            "motif": torch.tensor([0, 1, 2, 3]),
            "recursion_depth": torch.tensor([1, 2, 3, 4]),
            "grid_size": torch.tensor([3, 5, 7, 9]),
            "angle": torch.tensor([45.0, 90.0, 45.0, 25.0]),
        }

        loss_dict = criterion(outputs, targets)

        assert "total_loss" in loss_dict
        assert "loss_rule" in loss_dict
        assert "loss_symmetry" in loss_dict
        assert "loss_motif" in loss_dict
        assert "loss_depth" in loss_dict
        assert "loss_grid_size" in loss_dict
        assert "loss_angle" in loss_dict

        total = loss_dict["total_loss"]
        assert total.item() > 0
        assert not torch.isnan(total)
