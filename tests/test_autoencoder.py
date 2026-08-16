"""Unit tests for Image-to-Image Autoencoder Baseline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kolam_r.baselines.autoencoder import KolamAutoencoder


class TestKolamAutoencoder:
    """Test autoencoder architecture, shapes, and inference."""

    def test_forward_pass_shapes(self):
        """Input (B, 1, 64, 64) -> Output (B, 1, 64, 64)."""
        model = KolamAutoencoder(latent_dim=64)
        x = torch.randn(4, 1, 64, 64)
        out = model(x)
        assert out.shape == (4, 1, 64, 64)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_predict_and_reconstruct(self):
        """Test end-to-end numpy inference."""
        model = KolamAutoencoder(latent_dim=64)
        target = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

        res = model.predict_and_reconstruct(target, device="cpu")
        assert res.image.shape == (64, 64)
        assert res.mask.shape == (64, 64)
        assert res.metrics.psnr >= 0.0
        assert res.parameters["grammar"] == "N/A"
        assert res.parameters["depth"] == "N/A"
