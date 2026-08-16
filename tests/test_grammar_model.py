"""Tests for VisionToGrammarModel architecture and generation."""

import torch
import pytest

from kolam_r.models.seq2seq_grammar import VisionToGrammarModel


class TestVisionToGrammarModel:
    """Tests for Vision-to-Grammar model."""

    def test_forward_pass_shapes(self):
        model = VisionToGrammarModel()
        model.eval()

        batch_size = 2
        seq_len = 16
        imgs = torch.randn(batch_size, 1, 64, 64)
        targets = torch.randint(0, 24, (batch_size, seq_len))

        outputs = model(imgs, targets)
        assert "token_logits" in outputs
        assert outputs["token_logits"].shape == (batch_size, seq_len, 24)
        assert outputs["logits_symmetry"].shape == (batch_size, 6)
        assert outputs["pred_angle_norm"].shape == (batch_size,)

    def test_autoregressive_generate(self):
        model = VisionToGrammarModel()
        model.eval()

        imgs = torch.randn(2, 1, 64, 64)
        gen_tokens, aux = model.generate(imgs, max_length=20)

        assert gen_tokens.shape[0] == 2
        assert gen_tokens.shape[1] <= 20
        assert "pred_symmetry" in aux
        assert "pred_angle" in aux
