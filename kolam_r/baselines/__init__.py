"""KOLAM-R Baseline Models Package.

Contains:
- classical_cv: Classical computer vision symmetry detection and template matching
- autoencoder: End-to-end convolutional image-to-image autoencoder
- cnn_baseline: Multi-task direct parameter regression baseline (from Stage 3)
"""

from kolam_r.baselines.autoencoder import KolamAutoencoder
from kolam_r.baselines.classical_cv import (
    ClassicalCVBaseline,
    detect_symmetry_autocorrelation,
)

__all__ = [
    "ClassicalCVBaseline",
    "detect_symmetry_autocorrelation",
    "KolamAutoencoder",
]
