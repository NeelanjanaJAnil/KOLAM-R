"""Tests for evaluation metrics and confusion matrix generation."""

import numpy as np
import pytest

from kolam_r.evaluation.metrics import compute_benchmark_metrics


class TestMetrics:
    """Tests for benchmark metrics computation."""

    def test_perfect_predictions(self):
        """Perfect predictions must produce 100% accuracy and 0.0 angle MAE."""
        n = 10
        predictions = {
            "rule_id": np.zeros(n, dtype=int),
            "symmetry": np.zeros(n, dtype=int),
            "motif": np.zeros(n, dtype=int),
            "recursion_depth": np.ones(n, dtype=int),
            "grid_size": np.full(n, 5, dtype=int),
            "angle": np.full(n, 45.0, dtype=float),
        }
        targets = {
            "rule_id": np.zeros(n, dtype=int),
            "symmetry": np.zeros(n, dtype=int),
            "motif": np.zeros(n, dtype=int),
            "recursion_depth": np.ones(n, dtype=int),
            "grid_size": np.full(n, 5, dtype=int),
            "angle": np.full(n, 45.0, dtype=float),
        }

        metrics = compute_benchmark_metrics(predictions, targets)
        assert metrics["structural_exact_match_accuracy"] == 1.0
        assert metrics["full_exact_match_accuracy"] == 1.0
        assert metrics["structural_parameters"]["rule_id_accuracy"] == 1.0
        assert metrics["structural_parameters"]["angle_mae_degrees"] == 0.0

    def test_imperfect_structural_match(self):
        """If one structural head fails, structural exact match must drop."""
        n = 10
        predictions = {
            "rule_id": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  # 1 mistake
            "symmetry": np.zeros(n, dtype=int),
            "motif": np.zeros(n, dtype=int),
            "recursion_depth": np.ones(n, dtype=int),
            "grid_size": np.full(n, 5, dtype=int),
            "angle": np.full(n, 45.0, dtype=float),
        }
        targets = {
            "rule_id": np.zeros(n, dtype=int),
            "symmetry": np.zeros(n, dtype=int),
            "motif": np.zeros(n, dtype=int),
            "recursion_depth": np.ones(n, dtype=int),
            "grid_size": np.full(n, 5, dtype=int),
            "angle": np.full(n, 45.0, dtype=float),
        }

        metrics = compute_benchmark_metrics(predictions, targets)
        assert metrics["structural_exact_match_accuracy"] == 0.9
        assert metrics["structural_parameters"]["rule_id_accuracy"] == 0.9
