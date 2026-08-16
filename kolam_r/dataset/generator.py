"""Full-factorial parameterized dataset generator for KOLAM-R.

Generates the complete 1,920-sample canonical dataset (480 mathematical
structures x 4 motif rendering styles) with complete ground truth metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from PIL import Image
import numpy as np

from kolam_r.generator import KolamGenerator, KolamResult
from kolam_r.metadata import next_image_id, reset_counter, save_metadata
from kolam_r.schema import (
    RULE_DEFAULT_ANGLES,
    VALID_MOTIFS,
    VALID_RULES,
    VALID_SYMMETRIES,
    KolamParams,
)

# Canonical parameter space
VALID_GRID_SIZES = (3, 5, 7, 9)

# Valid recursion depths per rule based on max_safe_depth and aesthetic validity
RULE_VALID_DEPTHS: dict[str, tuple[int, ...]] = {
    "R01": (1, 2, 3),        # Krishna Anklets (max_safe_depth=4)
    "R02": (1, 2, 3),        # Snake Kolam (max_safe_depth=3)
    "R03": (1, 2),           # Kolam Tile (max_safe_depth=3, d=3 string is 12k+ chars)
    "R04": (1, 2, 3, 4),     # Mango Leaf (max_safe_depth=5)
    "R05": (1, 2, 3, 4),     # Hilbert Meander (max_safe_depth=5)
    "R06": (1, 2, 3, 4),     # Branching Floral (max_safe_depth=5)
}


@dataclass(frozen=True)
class StructuralConfig:
    """A unique mathematical generative structure (independent of motif)."""

    production_rule_id: str
    recursion_depth: int
    symmetry: str
    grid_size: int
    angle: float

    @property
    def structure_id(self) -> str:
        """Deterministic identifier for the geometric structure."""
        return f"{self.production_rule_id}_d{self.recursion_depth}_{self.symmetry}_g{self.grid_size}"


def enumerate_structural_configs() -> list[StructuralConfig]:
    """Enumerate all 480 unique mathematical structures in the canonical space."""
    configs: list[StructuralConfig] = []
    for rule_id in VALID_RULES:
        angle = RULE_DEFAULT_ANGLES[rule_id]
        depths = RULE_VALID_DEPTHS[rule_id]
        for depth in depths:
            for sym in VALID_SYMMETRIES:
                for grid in VALID_GRID_SIZES:
                    configs.append(
                        StructuralConfig(
                            production_rule_id=rule_id,
                            recursion_depth=depth,
                            symmetry=sym,
                            grid_size=grid,
                            angle=angle,
                        )
                    )
    return configs


def enumerate_all_params() -> list[KolamParams]:
    """Enumerate all 1,920 full parameter configurations (480 structures x 4 motifs)."""
    structures = enumerate_structural_configs()
    all_params: list[KolamParams] = []
    for struct in structures:
        for motif in VALID_MOTIFS:
            all_params.append(
                KolamParams(
                    grid_size=struct.grid_size,
                    symmetry=struct.symmetry,
                    angle=struct.angle,
                    motif=motif,
                    production_rule_id=struct.production_rule_id,
                    recursion_depth=struct.recursion_depth,
                    dot_spacing=1.0,
                    step_length=1.0,
                    random_seed=42,
                )
            )
    return all_params


class DatasetGenerator:
    """Generates and saves the full synthetic Kolam dataset to disk."""

    def __init__(
        self,
        output_dir: Path | str,
        generator: KolamGenerator | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.generator = generator or KolamGenerator()

    def generate_all(
        self,
        progress_callback: callable | None = None,
    ) -> list[KolamResult]:
        """Generate all 1,920 images and write them with metadata to output_dir."""
        img_dir = self.output_dir / "images"
        hires_dir = self.output_dir / "images_hires"
        meta_dir = self.output_dir / "metadata"

        for d in [img_dir, hires_dir, meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        params_list = enumerate_all_params()
        total = len(params_list)
        reset_counter(0)

        results: list[KolamResult] = []
        for idx, params in enumerate(params_list):
            image_id = f"K{idx + 1:06d}"
            result = self.generator.generate(params, image_id=image_id)
            results.append(result)

            # Save 64x64 image
            Image.fromarray(result.image_64, mode="L").save(
                img_dir / f"{image_id}.png"
            )

            # Save 256x256 reference image
            Image.fromarray(result.image_256, mode="L").save(
                hires_dir / f"{image_id}.png"
            )

            # Save JSON metadata
            save_metadata(result.metadata, meta_dir / f"{image_id}.json")

            if progress_callback is not None and (idx + 1) % 100 == 0:
                progress_callback(idx + 1, total)

        return results


def generate_full_dataset(
    output_dir: Path | str,
    progress_callback: callable | None = None,
) -> list[KolamResult]:
    """Convenience function to generate the complete dataset."""
    gen = DatasetGenerator(output_dir)
    return gen.generate_all(progress_callback=progress_callback)
