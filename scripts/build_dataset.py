"""End-to-end dataset build script for KOLAM-R Stage 2.

Generates the complete 1,920-sample synthetic dataset, partitions it into
reproducible train/val/test splits, computes dataset statistics, and exports
distribution charts.

Usage:
    python scripts/build_dataset.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kolam_r.dataset.generator import DatasetGenerator
from kolam_r.dataset.splits import DatasetSplitter
from kolam_r.dataset.statistics import DatasetStatisticsEngine


def main() -> None:
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    splits_dir = data_dir / "splits"
    stats_dir = data_dir / "stats"

    for d in [raw_dir, splits_dir, stats_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KOLAM-R Stage 2 — Full Dataset Builder")
    print("=" * 70)

    # 1. Generate 1,920 samples
    start_time = time.time()
    print(f"\n[1/3] Generating 1,920 canonical synthetic samples...")
    print(f"      Output directory: {raw_dir}")

    def progress(current: int, total: int) -> None:
        print(f"      Progress: {current}/{total} samples ({(current/total)*100:.1f}%)")

    generator = DatasetGenerator(raw_dir)
    results = generator.generate_all(progress_callback=progress)

    elapsed = time.time() - start_time
    print(f"      Generated {len(results)} images & metadata in {elapsed:.2f}s.")

    # 2. Partition into reproducible splits
    print(f"\n[2/3] Partitioning dataset into reproducible splits...")
    splitter = DatasetSplitter(seed=42, train_ratio=0.70, val_ratio=0.15, test_iid_ratio=0.15)
    split_paths = splitter.save_splits(raw_dir, splits_dir)

    for split_name, path in split_paths.items():
        print(f"      Split [{split_name:18s}]: {path}")

    # 3. Compute statistics and export plots
    print(f"\n[3/3] Computing dataset statistics & R02 vs R05 overlap metrics...")
    stats_engine = DatasetStatisticsEngine(data_dir / "raw")
    # Point splits_dir to data/splits
    stats_engine.splits_dir = splits_dir

    stats_json_path = stats_engine.export_statistics_json(stats_dir / "dataset_statistics.json")
    plot_path = stats_engine.plot_distributions(stats_dir / "distributions.png")

    print(f"      Statistics JSON: {stats_json_path}")
    print(f"      Distribution Plots: {plot_path}")

    stats = stats_engine.compute_all_statistics()
    r02_r05 = stats.get("r02_vs_r05_overlap_metrics", {})

    print("\n" + "-" * 70)
    print("DATASET SUMMARY")
    print("-" * 70)
    print(f"Total Images: {stats['total_images']}")
    for split_name, info in stats["splits"].items():
        print(f"  - {split_name:18s}: {info['count']:4d} samples ({info['percentage']:5.1f}%) | {info['num_unique_structures']:3d} unique structures")

    print("\nR02 vs R05 AMBIGUITY METRICS:")
    print(f"  - Overall Mean SSIM: {r02_r05.get('overall_mean_ssim', 0.0):.4f}")
    print(f"  - Overall Mean MSE:  {r02_r05.get('overall_mean_mse', 0.0):.4f}")
    print(f"  - Overall Mean NCC:  {r02_r05.get('overall_mean_ncc', 0.0):.4f}")
    for depth_key, metrics in r02_r05.get("by_depth", {}).items():
        print(f"    * {depth_key}: SSIM={metrics['mean_ssim']:.4f}, MSE={metrics['mean_mse']:.4f}, NCC={metrics['mean_ncc']:.4f}")

    print("=" * 70)
    print("Stage 2 Dataset Build Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
