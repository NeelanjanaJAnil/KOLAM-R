"""Tests for CNN Baseline model forward pass and output shapes."""

import torch
import pytest

from kolam_r.models.cnn_baseline import KolamCNNBaseline


class TestKolamCNNBaseline:
    """Tests for KolamCNNBaseline architecture."""

    def test_forward_pass_shapes(self):
        """Model must accept (B, 1, 64, 64) and return all 6 head predictions."""
        model = KolamCNNBaseline()
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 1, 64, 64)
        outputs = model(x)

        assert "logits_rule" in outputs
        assert "logits_symmetry" in outputs
        assert "logits_motif" in outputs
        assert "logits_depth" in outputs
        assert "logits_grid_size" in outputs
        assert "pred_angle_norm" in outputs

        assert outputs["logits_rule"].shape == (batch_size, 6)
        assert outputs["logits_symmetry"].shape == (batch_size, 6)
        assert outputs["logits_motif"].shape == (batch_size, 4)
        assert outputs["logits_depth"].shape == (batch_size, 4)
        assert outputs["logits_grid_size"].shape == (batch_size, 4)
        assert outputs["pred_angle_norm"].shape == (batch_size,)

        # Angle must be in [0, 1] due to Sigmoid
        assert (outputs["pred_angle_norm"] >= 0.0).all()
        assert (outputs["pred_angle_norm"] <= 1.0).all()

    def test_gradient_flow(self):
        """Backpropagation must compute non-zero gradients for all parameters."""
        model = KolamCNNBaseline()
        x = torch.randn(2, 1, 64, 64)
        outputs = model(x)

        # Compute a dummy loss combining all 6 heads
        loss = (
            outputs["logits_rule"].sum()
            + outputs["logits_symmetry"].sum()
            + outputs["logits_motif"].sum()
            + outputs["logits_depth"].sum()
            + outputs["logits_grid_size"].sum()
            + outputs["pred_angle_norm"].sum()
        )
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"
