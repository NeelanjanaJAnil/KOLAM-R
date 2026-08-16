"""Dataset loader for KOLAM-R supporting PyTorch and pure NumPy.

Provides typed dataset loading, parameter encoding/decoding, and
optional on-the-fly evaluation corruptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from kolam_r.dataset.corruption import CorruptionType, apply_corruption
from kolam_r.schema import VALID_MOTIFS, VALID_RULES, VALID_SYMMETRIES

# Parameter label encoding mappings
RULE_TO_IDX: dict[str, int] = {r: i for i, r in enumerate(VALID_RULES)}
IDX_TO_RULE: dict[int, str] = {i: r for i, r in enumerate(VALID_RULES)}

SYM_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(VALID_SYMMETRIES)}
IDX_TO_SYM: dict[int, str] = {i: s for i, s in enumerate(VALID_SYMMETRIES)}

MOTIF_TO_IDX: dict[str, int] = {m: i for i, m in enumerate(VALID_MOTIFS)}
IDX_TO_MOTIF: dict[int, str] = {i: m for i, m in enumerate(VALID_MOTIFS)}


class KolamDataset:
    """Dataset class for loading Kolam images and parameter ground truth.

    Compatible with standard Python iteration and PyTorch DataLoader.
    """

    def __init__(
        self,
        split_path: Path | str,
        transform: Callable[[np.ndarray], Any] | None = None,
        corruption: CorruptionType | str | None = None,
        corruption_severity: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.split_path = Path(split_path)
        self.transform = transform
        self.corruption = corruption
        self.corruption_severity = corruption_severity
        self.seed = seed

        with open(self.split_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        self.split_name: str = manifest_data.get("split_name", "unknown")
        self.records: list[dict] = manifest_data["records"]

    def __len__(self) -> int:
        return len(self.records)

    def get_sample_record(self, idx: int) -> dict:
        return self.records[idx]

    def load_image(self, idx: int) -> np.ndarray:
        """Load and return the 64x64 grayscale image as a uint8 NumPy array."""
        rec = self.records[idx]
        img_path = Path(rec["image_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)

        # Apply evaluation corruption if specified
        if self.corruption is not None:
            # Deterministic per-sample seed for reproducibility
            sample_seed = self.seed + idx
            img = apply_corruption(
                img,
                self.corruption,
                severity=self.corruption_severity,
                seed=sample_seed,
            )

        return img

    def encode_targets(self, rec: dict) -> dict[str, Any]:
        """Encode metadata parameters into ML-ready tensor targets."""
        rule_idx = RULE_TO_IDX[rec["production_rule_id"]]
        sym_idx = SYM_TO_IDX[rec["symmetry"]]
        motif_idx = MOTIF_TO_IDX[rec["motif"]]
        depth = int(rec["recursion_depth"])
        angle = float(rec["angle"])
        grid_size = int(rec["grid_size"])

        return {
            "rule_id": rule_idx,
            "symmetry": sym_idx,
            "motif": motif_idx,
            "recursion_depth": depth,
            "angle": angle,
            "grid_size": grid_size,
        }

    def __getitem__(self, idx: int) -> tuple[Any, dict[str, Any]]:
        """Return (image, targets_dict) for sample idx."""
        img = self.load_image(idx)
        rec = self.records[idx]
        targets = self.encode_targets(rec)

        if self.transform is not None:
            img = self.transform(img)
        else:
            # Default: float32 in [0, 1] with channel dim (1, 64, 64)
            img = (img.astype(np.float32) / 255.0)[np.newaxis, ...]

        return img, targets
