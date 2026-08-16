"""Top-level Kolam generator orchestrator.

Coordinates the full generation pipeline:
    1. Look up production rule
    2. Expand L-system string
    3. Interpret turtle geometry
    4. Apply symmetry transform
    5. Render to image
    6. Build metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from kolam_r.geometry.fitter import fit_segments_piecewise_parametric
from kolam_r.lsystem.engine import LSystemEngine
from kolam_r.lsystem.rules import get_rule
from kolam_r.metadata import create_metadata, save_metadata
from kolam_r.renderer.image_renderer import render_kolam
from kolam_r.schema import BoundingBox, GeometricRepresentation, KolamMetadata, KolamParams
from kolam_r.symmetry.transforms import apply_symmetry
from kolam_r.turtle.interpreter import LineSegment, TurtleInterpreter, TurtleResult


@dataclass
class KolamResult:
    """Result of Kolam generation.

    Contains the generated image(s), metadata, and intermediate
    representations for inspection and validation.
    """

    image_64: np.ndarray           # 64x64 grayscale image
    image_256: np.ndarray          # 256x256 grayscale image (for visual inspection)
    metadata: KolamMetadata        # Complete mathematical ground truth
    segments_raw: list[LineSegment]  # Segments before symmetry transform
    segments_final: list[LineSegment]  # Segments after symmetry transform
    expanded_string: str           # The fully expanded L-system string
    turtle_result: TurtleResult    # Full turtle interpretation result


class KolamGenerator:
    """Orchestrates the Kolam generation pipeline.

    Usage:
        generator = KolamGenerator()
        params = KolamParams(production_rule_id="R01", recursion_depth=2, ...)
        result = generator.generate(params)
        result.image_64  # 64x64 numpy array
    """

    def __init__(self, max_string_length: int = 100_000) -> None:
        self._engine = LSystemEngine(max_string_length=max_string_length)
        self._turtle = TurtleInterpreter()

    def generate(
        self,
        params: KolamParams,
        image_id: str | None = None,
    ) -> KolamResult:
        """Generate a Kolam pattern from the given parameters.

        Args:
            params: Complete parameter specification.
            image_id: Optional explicit image ID for metadata.

        Returns:
            KolamResult with images, metadata, and intermediate data.
        """
        # Step 1: Look up production rule
        rule = get_rule(params.production_rule_id)

        # Step 2: Expand L-system string
        expanded = self._engine.expand(
            axiom=rule.axiom,
            productions=rule.productions,
            depth=params.recursion_depth,
        )

        # Step 3: Interpret turtle geometry
        turtle_result = self._turtle.interpret(
            instructions=expanded,
            angle=params.angle,
            step_length=params.step_length,
        )
        segments_raw = list(turtle_result.segments)

        # Step 4: Apply symmetry transform
        segments_final = apply_symmetry(
            segments=segments_raw,
            symmetry=params.symmetry,
        )

        # Step 5: Render to images
        image_64 = render_kolam(
            segments=segments_final,
            image_size=64,
            padding=4,
            motif=params.motif,
            grid_size=params.grid_size,
            dot_spacing=params.dot_spacing,
        )
        image_256 = render_kolam(
            segments=segments_final,
            image_size=256,
            padding=16,
            motif=params.motif,
            grid_size=params.grid_size,
            dot_spacing=params.dot_spacing,
        )

        # Step 6: Extract continuous geometry & recover mathematical equations
        geom_rep = fit_segments_piecewise_parametric(segments_final)

        # Step 7: Build metadata
        bbox = None
        if turtle_result.min_x != float("inf"):
            bbox = BoundingBox(
                min_x=turtle_result.min_x,
                min_y=turtle_result.min_y,
                max_x=turtle_result.max_x,
                max_y=turtle_result.max_y,
            )

        metadata = create_metadata(
            params=params,
            expanded_string_length=len(expanded),
            num_segments=len(segments_raw),
            num_segments_after_symmetry=len(segments_final),
            connectivity=rule.connectivity,
            bounding_box=bbox,
            image_id=image_id,
            geometric_representation=geom_rep,
        )

        return KolamResult(
            image_64=image_64,
            image_256=image_256,
            metadata=metadata,
            segments_raw=segments_raw,
            segments_final=segments_final,
            expanded_string=expanded,
            turtle_result=turtle_result,
        )

    def generate_and_save(
        self,
        params: KolamParams,
        output_dir: Path,
        image_id: str | None = None,
    ) -> KolamResult:
        """Generate a Kolam and save images + metadata to disk.

        Directory structure:
            output_dir/
                images/K000001.png        (64x64)
                images_hires/K000001.png   (256x256)
                metadata/K000001.json

        Args:
            params: Complete parameter specification.
            output_dir: Root output directory.
            image_id: Optional explicit image ID.

        Returns:
            KolamResult with images, metadata, and intermediate data.
        """
        result = self.generate(params, image_id=image_id)
        mid = result.metadata.image_id

        # Create directories
        img_dir = output_dir / "images"
        hires_dir = output_dir / "images_hires"
        meta_dir = output_dir / "metadata"
        for d in [img_dir, hires_dir, meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Save 64x64 image
        Image.fromarray(result.image_64, mode="L").save(img_dir / f"{mid}.png")

        # Save 256x256 image
        Image.fromarray(result.image_256, mode="L").save(hires_dir / f"{mid}.png")

        # Save metadata
        save_metadata(result.metadata, meta_dir / f"{mid}.json")

        return result
