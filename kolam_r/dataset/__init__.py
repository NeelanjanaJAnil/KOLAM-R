"""Dataset generation, splitting, corruptions, loading, and statistical analysis."""

from kolam_r.dataset.corruption import CorruptionType, apply_corruption
from kolam_r.dataset.generator import DatasetGenerator, generate_full_dataset
from kolam_r.dataset.loader import KolamDataset
from kolam_r.dataset.splits import DatasetSplitter, SplitManifest
from kolam_r.dataset.statistics import DatasetStatisticsEngine

__all__ = [
    "DatasetGenerator",
    "generate_full_dataset",
    "DatasetSplitter",
    "SplitManifest",
    "CorruptionType",
    "apply_corruption",
    "KolamDataset",
    "DatasetStatisticsEngine",
]
