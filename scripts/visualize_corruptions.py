"""Visualize clean test samples against the 5 evaluation-only corruptions.

Generates a multi-panel visual artifact comparing:
    Clean vs. Gaussian Noise, Rotation, Affine, Blur, and Stroke Dropout.

Usage:
    python scripts/visualize_corruptions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kolam_r.dataset.corruption import CorruptionType, apply_corruption


def main() -> None:
    data_dir = project_root / "data"
    splits_dir = data_dir / "splits"
    raw_images_dir = data_dir / "raw" / "images"
    output_path = data_dir / "stats" / "corrupted_samples_grid.png"

    test_file = splits_dir / "test_iid.json"
    if not test_file.exists():
        print("Error: test_iid.json not found. Run scripts/build_dataset.py first.")
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    # Pick 5 distinct samples from test_iid
    selected = records[:6]

    corruptions = [
        ("Clean", None),
        ("Gaussian Noise", CorruptionType.GAUSSIAN_NOISE),
        ("Rotation (±12°)", CorruptionType.ROTATION),
        ("Affine Shear", CorruptionType.AFFINE),
        ("Gaussian Blur", CorruptionType.BLUR),
        ("Stroke Dropout", CorruptionType.STROKE_DROPOUT),
    ]

    nrows = len(selected)
    ncols = len(corruptions)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.8))
    fig.suptitle(
        "KOLAM-R — Evaluation-Only Corruption Benchmark Suite\n(Strictly for Testing Robustness; Never Used in Training)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for r_idx, rec in enumerate(selected):
        img_path = Path(rec["image_path"])
        if not img_path.exists():
            img_path = raw_images_dir / f"{rec['image_id']}.png"

        clean_img = np.array(Image.open(img_path).convert("L"))

        for c_idx, (c_name, c_type) in enumerate(corruptions):
            ax = axes[r_idx, c_idx]
            if c_type is None:
                display_img = clean_img
            else:
                display_img = apply_corruption(clean_img, c_type, severity=1.0, seed=42 + r_idx)

            ax.imshow(display_img, cmap="gray", vmin=0, vmax=255)
            if r_idx == 0:
                ax.set_title(c_name, fontsize=10, fontweight="bold", pad=6)

            if c_idx == 0:
                ax.set_ylabel(
                    f"{rec['image_id']}\n{rec['production_rule_id']} d={rec['recursion_depth']}\n{rec['symmetry']}",
                    fontsize=8,
                    fontweight="bold",
                )

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Corruptions comparison grid saved to: {output_path}")


if __name__ == "__main__":
    main()
