"""End-to-End Image-to-Image Neural Autoencoder Baseline for KOLAM-R.

A purely neural raster autoencoder predicting direct output pixels
without any symbolic L-system, discrete grammar, or parameter intermediate.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from kolam_r.reconstruction.metrics import (
    ReconstructionMetrics,
    compute_reconstruction_metrics,
)
from kolam_r.reconstruction.pipeline import ReconstructionResult
from kolam_r.schema import BoundingBox


class KolamAutoencoder(nn.Module):
    """Convolutional Image-to-Image Autoencoder."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 64x64 -> 4x4x256 -> latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fc_enc = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)

        # Decoder: latent_dim -> 4x4x256 -> 64x64
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder_conv(x)
        feat_flat = feat.view(feat.size(0), -1)
        return self.fc_enc(feat_flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        feat_flat = self.fc_dec(z)
        feat = feat_flat.view(feat_flat.size(0), 256, 4, 4)
        return self.decoder_conv(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def predict_and_reconstruct(
        self,
        target_image: np.ndarray,
        device: str = "cpu",
    ) -> ReconstructionResult:
        """Run image through autoencoder to produce direct pixel reconstruction."""
        self.eval()
        img_norm = (target_image.astype(np.float32) / 255.0)
        t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out_t = self(t)

        out_img = (out_t[0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        recon_mask = (out_img > 30).astype(np.uint8) * 255

        metrics = compute_reconstruction_metrics(
            target_img=target_image,
            recon_img=out_img,
            target_segments=[],
            recon_segments=[],
        )

        bbox = BoundingBox(
            min_x=0.0, min_y=0.0, max_x=64.0, max_y=64.0
        )

        return ReconstructionResult(
            method="image_to_image_autoencoder",
            image=out_img,
            mask=recon_mask,
            segments=[],
            bbox=bbox,
            metrics=metrics,
            grammar_string="N/A",
            parameters={
                "model_type": "ImageToImageAutoencoder",
                "grammar": "N/A",
                "depth": "N/A",
                "symmetry": "N/A",
            },
        )
