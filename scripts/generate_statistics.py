"""Generate comprehensive dataset statistics and distribution charts.

Usage:
    python scripts/generate_statistics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kolam_r.dataset.statistics import DatasetStatisticsEngine


def main() -> None:
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    splits_dir = data_dir / "splits"
    stats_dir = data_dir / "stats"

    if not splits_dir.exists():
        print("Error: Splits directory not found. Please run scripts/build_dataset.py first.")
        sys.exit(1)

    engine = DatasetStatisticsEngine(raw_dir)
    engine.splits_dir = splits_dir

    json_path = engine.export_statistics_json(stats_dir / "dataset_statistics.json")
    plot_path = engine.plot_distributions(stats_dir / "distributions.png")

    print(f"Exported statistics JSON to: {json_path}")
    print(f"Exported distribution plots to: {plot_path}")


if __name__ == "__main__":
    main()
