"""Generate 30 example Kolam patterns spanning the parameter space.

This script generates diverse examples using all 6 production rules
with varying recursion depths, symmetries, and motifs.

Usage:
    cd kolam_r
    python scripts/generate_examples.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kolam_r.generator import KolamGenerator
from kolam_r.schema import KolamParams, RULE_DEFAULT_ANGLES
from kolam_r.metadata import reset_counter


def generate_examples():
    """Generate 30 diverse Kolam examples."""
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = KolamGenerator()
    reset_counter(0)

    # Define 30 configurations spanning the parameter space
    configs = [
        # --- R01: Krishna Anklets (45 deg) ---
        # Vary recursion depth
        {"production_rule_id": "R01", "angle": 45.0, "recursion_depth": 1, "symmetry": "C1", "motif": "M1", "grid_size": 5},
        {"production_rule_id": "R01", "angle": 45.0, "recursion_depth": 2, "symmetry": "C1", "motif": "M1", "grid_size": 5},
        {"production_rule_id": "R01", "angle": 45.0, "recursion_depth": 3, "symmetry": "C1", "motif": "M1", "grid_size": 7},
        # Vary symmetry
        {"production_rule_id": "R01", "angle": 45.0, "recursion_depth": 2, "symmetry": "C4", "motif": "M1", "grid_size": 7},
        {"production_rule_id": "R01", "angle": 45.0, "recursion_depth": 2, "symmetry": "D4", "motif": "M2", "grid_size": 7},

        # --- R02: Snake Kolam (90 deg) ---
        {"production_rule_id": "R02", "angle": 90.0, "recursion_depth": 1, "symmetry": "C1", "motif": "M1", "grid_size": 5},
        {"production_rule_id": "R02", "angle": 90.0, "recursion_depth": 2, "symmetry": "C1", "motif": "M1", "grid_size": 5},
        {"production_rule_id": "R02", "angle": 90.0, "recursion_depth": 3, "symmetry": "C1", "motif": "M1", "grid_size": 7},
        {"production_rule_id": "R02", "angle": 90.0, "recursion_depth": 2, "symmetry": "C4", "motif": "M2", "grid_size": 7},
        {"production_rule_id": "R02", "angle": 90.0, "recursion_depth": 2, "symmetry": "D2", "motif": "M3", "grid_size": 5},

        # --- R03: Kolam Tile (45 deg) ---
        {"production_rule_id": "R03", "angle": 45.0, "recursion_depth": 1, "symmetry": "C1", "motif": "M1", "grid_size": 5},
        {"production_rule_id": "R03", "angle": 45.0, "recursion_depth": 2, "symmetry": "C1", "motif": "M1", "grid_size": 7},
        {"production_rule_id": "R03", "angle": 45.0, "recursion_depth": 2, "symmetry": "C4", "motif": "M1", "grid_size": 9},
        {"production_rule_id": "R03", "angle": 45.0, "recursion_depth": 2, "symmetry": "D4", "motif": "M2", "grid_size": 7},
        {"production_rule_id": "R03", "angle": 45.0, "recursion_depth": 1, "symmetry": "D1", "motif": "M4", "grid_size": 5},

        # --- R04: Mango Leaf (45 deg) ---
        {"production_rule_id": "R04", "angle": 45.0, "recursion_depth": 1, "symmetry": "C1", "motif": "M1", "grid_size": 3},
        {"production_rule_id": "R04", "angle": 45.0, "recursion_depth": 2, "symmetry": "C1", "motif": "M1", "grid_size": 5},
        {"production_rule_id": "R04", "angle": 45.0, "recursion_depth": 3, "symmetry": "C1", "motif": "M1", "grid_size": 7},
        {"production_rule_id": "R04", "angle": 45.0, "recursion_depth": 4, "symmetry": "C1", "motif": "M1", "grid_size": 9},
        {"production_rule_id": "R04", "angle": 45.0, "recursion_depth": 2, "symmetry": "C4", "motif": "M3", "grid_size": 7},

        # --- R05: Hilbert Meander (90 deg) ---
        {"production_rule_id": "R05", "angle": 90.0, "recursion_depth": 1, "symmetry": "C1", "motif": "M1", "grid_size": 3},
        {"production_rule_id": "R05", "angle": 90.0, "recursion_depth": 2, "symmetry": "C1", "motif": "M1", "grid_size": 5},
        {"production_rule_id": "R05", "angle": 90.0, "recursion_depth": 3, "symmetry": "C1", "motif": "M1", "grid_size": 7},
        {"production_rule_id": "R05", "angle": 90.0, "recursion_depth": 4, "symmetry": "C1", "motif": "M1", "grid_size": 9},
        {"production_rule_id": "R05", "angle": 90.0, "recursion_depth": 2, "symmetry": "C2", "motif": "M2", "grid_size": 5},

        # --- R06: Branching Floral (25 deg) ---
        {"production_rule_id": "R06", "angle": 25.0, "recursion_depth": 1, "symmetry": "C1", "motif": "M1", "grid_size": 3},
        {"production_rule_id": "R06", "angle": 25.0, "recursion_depth": 2, "symmetry": "C1", "motif": "M1", "grid_size": 5},
        {"production_rule_id": "R06", "angle": 25.0, "recursion_depth": 3, "symmetry": "C1", "motif": "M1", "grid_size": 5},
        {"production_rule_id": "R06", "angle": 25.0, "recursion_depth": 4, "symmetry": "C1", "motif": "M1", "grid_size": 7},
        {"production_rule_id": "R06", "angle": 25.0, "recursion_depth": 3, "symmetry": "D4", "motif": "M4", "grid_size": 7},
    ]

    print(f"Generating {len(configs)} Kolam examples...")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    results = []
    for i, cfg in enumerate(configs):
        params = KolamParams(**cfg)
        result = generator.generate_and_save(params, output_dir)
        results.append(result)

        print(
            f"  [{i+1:2d}/{len(configs)}] {result.metadata.image_id} | "
            f"{cfg['production_rule_id']} depth={cfg['recursion_depth']} "
            f"sym={cfg['symmetry']} motif={cfg['motif']} "
            f"| {len(result.segments_raw)} raw -> {len(result.segments_final)} final segs "
            f"| str_len={len(result.expanded_string)}"
        )

    print("-" * 60)
    print(f"Done. Generated {len(results)} images.")
    print(f"  Images (64x64):  {output_dir / 'images'}")
    print(f"  Images (256x256): {output_dir / 'images_hires'}")
    print(f"  Metadata:        {output_dir / 'metadata'}")

    return results


if __name__ == "__main__":
    generate_examples()
