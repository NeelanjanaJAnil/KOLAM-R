"""Metadata creation and serialization for generated Kolam images.

Handles the creation, serialization, and loading of KolamMetadata
records that accompany every generated image.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from kolam_r.schema import BoundingBox, KolamMetadata, KolamParams


# Global image counter for sequential ID generation
_image_counter: int = 0


def reset_counter(start: int = 0) -> None:
    """Reset the global image counter."""
    global _image_counter
    _image_counter = start


def next_image_id() -> str:
    """Generate the next sequential image ID (e.g. K000001)."""
    global _image_counter
    _image_counter += 1
    return f"K{_image_counter:06d}"


def create_metadata(
    params: KolamParams,
    expanded_string_length: int,
    num_segments: int,
    num_segments_after_symmetry: int,
    connectivity: str,
    bounding_box: BoundingBox | None = None,
    image_id: str | None = None,
) -> KolamMetadata:
    """Create a complete metadata record for a generated Kolam image.

    Args:
        params: The generator parameters used.
        expanded_string_length: Length of the expanded L-system string.
        num_segments: Number of line segments from turtle interpretation.
        num_segments_after_symmetry: Number of segments after symmetry transform.
        connectivity: Connectivity type from the production rule.
        bounding_box: Bounding box of the generated geometry.
        image_id: Optional explicit image ID. If None, auto-generated.

    Returns:
        Complete KolamMetadata record.
    """
    if image_id is None:
        image_id = next_image_id()

    return KolamMetadata(
        image_id=image_id,
        params=params,
        generator_version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        expanded_string_length=expanded_string_length,
        num_segments=num_segments,
        num_segments_after_symmetry=num_segments_after_symmetry,
        connectivity=connectivity,
        bounding_box=bounding_box,
    )


def save_metadata(metadata: KolamMetadata, path: Path) -> None:
    """Save metadata as a JSON file.

    Args:
        metadata: The metadata record to save.
        path: Output file path (should end in .json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2, default=str)


def load_metadata(path: Path) -> KolamMetadata:
    """Load metadata from a JSON file.

    Args:
        path: Path to the JSON metadata file.

    Returns:
        Deserialized KolamMetadata record.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return KolamMetadata(**data)
