"""Create visualization grids of generated Kolam examples.

Reads generated images and metadata from the output directory
and creates annotated grid plots for visual inspection.

Usage:
    cd kolam_r
    python scripts/visualize_grid.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_examples(output_dir: Path):
    """Load all generated examples with their metadata."""
    meta_dir = output_dir / "metadata"
    hires_dir = output_dir / "images_hires"
    lo_dir = output_dir / "images"

    examples = []
    for meta_file in sorted(meta_dir.glob("*.json")):
        with open(meta_file, "r") as f:
            meta = json.load(f)

        image_id = meta["image_id"]

        # Prefer hi-res image for visualization
        hires_path = hires_dir / f"{image_id}.png"
        lo_path = lo_dir / f"{image_id}.png"

        if hires_path.exists():
            img = np.array(Image.open(hires_path).convert("L"))
        elif lo_path.exists():
            img = np.array(Image.open(lo_path).convert("L"))
        else:
            continue

        examples.append({"image": img, "metadata": meta})

    return examples


def create_grid(examples: list, output_path: Path, use_hires: bool = True):
    """Create an annotated grid visualization."""
    n = len(examples)
    if n == 0:
        print("No examples found.")
        return

    # Calculate grid dimensions
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 3.8),
        squeeze=False,
    )
    fig.suptitle(
        "KOLAM-R Stage 1 — Generated Examples",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for idx, ax_row in enumerate(axes.flat):
        if idx < n:
            ex = examples[idx]
            meta = ex["metadata"]
            params = meta.get("params", meta)  # Handle nested or flat format

            ax_row.imshow(ex["image"], cmap="gray", vmin=0, vmax=255)

            # Build label
            rule_id = params.get("production_rule_id", "?")
            depth = params.get("recursion_depth", "?")
            sym = params.get("symmetry", "?")
            motif = params.get("motif", "?")
            grid = params.get("grid_size", "?")
            angle = params.get("angle", "?")
            img_id = meta.get("image_id", "?")

            label = (
                f"{img_id}\n"
                f"{rule_id} d={depth} {sym}\n"
                f"{motif} g={grid} a={angle}°"
            )
            ax_row.set_title(label, fontsize=7, pad=4, fontfamily="monospace")
        else:
            ax_row.set_visible(False)

        ax_row.set_xticks([])
        ax_row.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Grid saved to: {output_path}")


def create_rule_comparison_grid(examples: list, output_path: Path):
    """Create a grid organized by production rule for comparison."""
    # Group by rule
    by_rule = {}
    for ex in examples:
        params = ex["metadata"].get("params", ex["metadata"])
        rule_id = params.get("production_rule_id", "?")
        by_rule.setdefault(rule_id, []).append(ex)

    rules = sorted(by_rule.keys())
    max_per_rule = max(len(v) for v in by_rule.values())

    fig, axes = plt.subplots(
        len(rules), max_per_rule,
        figsize=(max_per_rule * 2.8, len(rules) * 3.2),
        squeeze=False,
    )
    fig.suptitle(
        "KOLAM-R — Production Rules Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    for row_idx, rule_id in enumerate(rules):
        rule_examples = by_rule[rule_id]
        for col_idx in range(max_per_rule):
            ax = axes[row_idx, col_idx]
            if col_idx < len(rule_examples):
                ex = rule_examples[col_idx]
                meta = ex["metadata"]
                params = meta.get("params", meta)

                ax.imshow(ex["image"], cmap="gray", vmin=0, vmax=255)

                depth = params.get("recursion_depth", "?")
                sym = params.get("symmetry", "?")
                motif = params.get("motif", "?")
                ax.set_title(
                    f"d={depth} {sym} {motif}",
                    fontsize=7,
                    pad=2,
                    fontfamily="monospace",
                )
            else:
                ax.set_visible(False)

            ax.set_xticks([])
            ax.set_yticks([])

        # Row label
        axes[row_idx, 0].set_ylabel(
            rule_id, fontsize=10, fontweight="bold", rotation=0, labelpad=30
        )

    plt.tight_layout(rect=[0.05, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Rule comparison grid saved to: {output_path}")


def main():
    output_dir = project_root / "output"

    if not (output_dir / "metadata").exists():
        print("No metadata found. Run generate_examples.py first.")
        sys.exit(1)

    examples = load_examples(output_dir)
    print(f"Loaded {len(examples)} examples.")

    # Main grid
    create_grid(examples, output_dir / "grid.png")

    # Hi-res grid
    create_grid(examples, output_dir / "grid_hires.png", use_hires=True)

    # Rule comparison grid
    create_rule_comparison_grid(examples, output_dir / "rule_comparison.png")

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
