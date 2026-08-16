"""KOLAM-R benchmark evaluation and diagnostic metrics."""

from kolam_r.evaluation.evaluator import BaselineEvaluator
from kolam_r.evaluation.metrics import (
    compute_benchmark_metrics,
    plot_confusion_matrices,
)

__all__ = [
    "BaselineEvaluator",
    "compute_benchmark_metrics",
    "plot_confusion_matrices",
]
