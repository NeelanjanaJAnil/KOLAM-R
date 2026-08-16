"""Reproducible dataset partitioning into Train, Val, Test_IID, and Test_Heldout_Depth.

Ensures zero parameter leakage and strict isolation of held-out recursion depths.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from kolam_r.dataset.generator import (
    RULE_VALID_DEPTHS,
    StructuralConfig,
    enumerate_all_params,
    enumerate_structural_configs,
)
from kolam_r.schema import KolamParams

# Explicit specification of held-out depths for recursion extrapolation tests
HELDOUT_DEPTH_RULES: dict[str, int] = {
    "R01": 3,  # Krishna Anklets max depth
    "R02": 3,  # Snake Kolam max depth
    "R04": 4,  # Mango Leaf max depth
    "R05": 4,  # Hilbert Meander max depth
    "R06": 4,  # Branching Floral max depth
}


@dataclass
class DatasetSampleRecord:
    """Index entry for a single dataset sample."""

    image_id: str
    image_path: str
    metadata_path: str
    structure_id: str
    production_rule_id: str
    recursion_depth: int
    symmetry: str
    grid_size: int
    motif: str
    angle: float
    split: str


@dataclass
class SplitManifest:
    """Summary and item list for a dataset partition."""

    split_name: str
    total_samples: int
    num_unique_structures: int
    sample_ids: list[str]
    records: list[dict]


class DatasetSplitter:
    """Partitions the synthetic Kolam dataset into reproducible splits."""

    def __init__(
        self,
        seed: int = 42,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_iid_ratio: float = 0.15,
    ) -> None:
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_iid_ratio = test_iid_ratio

    def is_heldout_depth(self, rule_id: str, depth: int) -> bool:
        """Check if a (rule_id, depth) combination belongs to held-out depth test set."""
        return HELDOUT_DEPTH_RULES.get(rule_id) == depth

    def partition_structures(
        self,
    ) -> tuple[list[StructuralConfig], list[StructuralConfig], list[StructuralConfig], list[StructuralConfig]]:
        """Partition the 480 structures into train, val, test_iid, and test_heldout_depth."""
        all_structures = enumerate_structural_configs()

        heldout_structures: list[StructuralConfig] = []
        standard_structures: list[StructuralConfig] = []

        for struct in all_structures:
            if self.is_heldout_depth(struct.production_rule_id, struct.recursion_depth):
                heldout_structures.append(struct)
            else:
                standard_structures.append(struct)

        # Shuffle standard structures deterministically
        rng = random.Random(self.seed)
        shuffled = list(standard_structures)
        rng.shuffle(shuffled)

        n_std = len(shuffled)
        n_train = int(round(n_std * self.train_ratio))
        n_val = int(round(n_std * self.val_ratio))

        train_structs = shuffled[:n_train]
        val_structs = shuffled[n_train : n_train + n_val]
        test_iid_structs = shuffled[n_train + n_val :]

        return train_structs, val_structs, test_iid_structs, heldout_structures

    def build_split_records(
        self,
        dataset_dir: Path | str,
    ) -> dict[str, SplitManifest]:
        """Build all split manifests and individual sample records."""
        dataset_dir = Path(dataset_dir)
        (
            train_structs,
            val_structs,
            test_iid_structs,
            heldout_structs,
        ) = self.partition_structures()

        struct_to_split: dict[str, str] = {}
        for s in train_structs:
            struct_to_split[s.structure_id] = "train"
        for s in val_structs:
            struct_to_split[s.structure_id] = "val"
        for s in test_iid_structs:
            struct_to_split[s.structure_id] = "test_iid"
        for s in heldout_structs:
            struct_to_split[s.structure_id] = "test_heldout_depth"

        all_params = enumerate_all_params()
        split_records: dict[str, list[DatasetSampleRecord]] = {
            "train": [],
            "val": [],
            "test_iid": [],
            "test_heldout_depth": [],
        }

        for idx, params in enumerate(all_params):
            image_id = f"K{idx + 1:06d}"
            struct_id = (
                f"{params.production_rule_id}_d{params.recursion_depth}_"
                f"{params.symmetry}_g{params.grid_size}"
            )
            split_name = struct_to_split[struct_id]

            rec = DatasetSampleRecord(
                image_id=image_id,
                image_path=str(dataset_dir / "images" / f"{image_id}.png"),
                metadata_path=str(dataset_dir / "metadata" / f"{image_id}.json"),
                structure_id=struct_id,
                production_rule_id=params.production_rule_id,
                recursion_depth=params.recursion_depth,
                symmetry=params.symmetry,
                grid_size=params.grid_size,
                motif=params.motif,
                angle=params.angle,
                split=split_name,
            )
            split_records[split_name].append(rec)

        manifests: dict[str, SplitManifest] = {}
        struct_lists = {
            "train": train_structs,
            "val": val_structs,
            "test_iid": test_iid_structs,
            "test_heldout_depth": heldout_structs,
        }

        for split_name, recs in split_records.items():
            manifests[split_name] = SplitManifest(
                split_name=split_name,
                total_samples=len(recs),
                num_unique_structures=len(struct_lists[split_name]),
                sample_ids=[r.image_id for r in recs],
                records=[asdict(r) for r in recs],
            )

        return manifests

    def save_splits(
        self,
        dataset_dir: Path | str,
        output_dir: Path | str | None = None,
    ) -> dict[str, Path]:
        """Save split manifests as JSON files."""
        dataset_dir = Path(dataset_dir)
        splits_dir = Path(output_dir) if output_dir else dataset_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        manifests = self.build_split_records(dataset_dir)
        saved_paths: dict[str, Path] = {}

        for split_name, manifest in manifests.items():
            path = splits_dir / f"{split_name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "split_name": manifest.split_name,
                        "total_samples": manifest.total_samples,
                        "num_unique_structures": manifest.num_unique_structures,
                        "sample_ids": manifest.sample_ids,
                        "records": manifest.records,
                    },
                    f,
                    indent=2,
                )
            saved_paths[split_name] = path

        return saved_paths
