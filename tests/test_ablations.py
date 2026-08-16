"""Unit tests for Stage 8 Ablation Models and Search Metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kolam_r.ablations.model_variants import (
    VisionToGrammarResNet,
    VisionToGrammarSequenceOnly,
)
from kolam_r.ablations.search_metrics import search_depth_multi_metric
from kolam_r.grammar.executor import ParsedGrammar


class TestAblationModels:
    """Test shapes, forward passes, and generation for ablation models."""

    def test_sequence_only_forward_pass(self):
        """SequenceOnly model outputs (B, T, vocab_size) and empty aux dict."""
        model = VisionToGrammarSequenceOnly(d_model=64, nhead=2, num_decoder_layers=2)
        imgs = torch.randn(2, 1, 64, 64)
        tokens = torch.randint(0, 10, (2, 8))
        logits, aux = model(imgs, tokens)

        assert logits.shape == (2, 8, 24)
        assert aux == {}

    def test_resnet_forward_pass(self):
        """ResNet model outputs logits and populated aux dictionary."""
        model = VisionToGrammarResNet(d_model=64, nhead=2, num_decoder_layers=2)
        imgs = torch.randn(2, 1, 64, 64)
        tokens = torch.randint(0, 10, (2, 8))
        logits, aux = model(imgs, tokens)

        assert logits.shape == (2, 8, 24)
        assert "symmetry" in aux
        assert "motif" in aux
        assert "depth" in aux
        assert "grid_size" in aux
        assert "pred_angle" in aux

    def test_search_depth_multi_metric(self):
        """Test search_depth_multi_metric across all 4 metrics (ncc, ssim, iou, mse)."""
        parsed = ParsedGrammar(
            axiom="F",
            productions={"F": "F+F-F"},
            is_valid=True,
        )
        dummy_img = np.zeros((64, 64), dtype=np.uint8)
        dummy_img[20:44, 20:44] = 255

        for m in ("ncc", "ssim", "iou", "mse"):
            d, score = search_depth_multi_metric(parsed, dummy_img, angle=90.0, search_metric=m)
            assert 1 <= d <= 4
            assert isinstance(score, float)
