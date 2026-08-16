"""Parameter schema and validation for KOLAM-R generator.

Defines the parameter space for the constrained Kolam L-system generator.
All parameters are validated using Pydantic v2 models.
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# Valid parameter values
VALID_SYMMETRIES = ("C1", "C2", "C4", "D1", "D2", "D4")
VALID_MOTIFS = ("M1", "M2", "M3", "M4")
VALID_RULES = ("R01", "R02", "R03", "R04", "R05", "R06")

# Default angles per rule (from published sources)
RULE_DEFAULT_ANGLES: dict[str, float] = {
    "R01": 45.0,   # Krishna Anklets (Prusinkiewicz & Hanan 1989)
    "R02": 90.0,   # Snake Kolam
    "R03": 45.0,   # Kolam Tile (Prusinkiewicz & Hanan)
    "R04": 45.0,   # Mango Leaf Kolam
    "R05": 90.0,   # Hilbert Meander
    "R06": 25.0,   # Branching Floral
}


class KolamParams(BaseModel):
    """Parameters for generating a single Kolam pattern.

    Defines the complete mathematical specification needed to
    deterministically generate a Kolam image via L-system expansion,
    turtle geometry interpretation, symmetry transform, and rendering.
    """

    grid_size: int = Field(
        default=5,
        ge=3,
        le=11,
        description="Lattice grid dimension (NxN dots on square orthogonal lattice)",
    )
    dot_spacing: float = Field(
        default=1.0,
        gt=0,
        description="Distance between adjacent dots in abstract units",
    )
    symmetry: Literal["C1", "C2", "C4", "D1", "D2", "D4"] = Field(
        default="C1",
        description="Symmetry group applied to the base pattern",
    )
    angle: float = Field(
        default=90.0,
        gt=0,
        lt=360,
        description="Turtle turning angle in degrees",
    )
    motif: Literal["M1", "M2", "M3", "M4"] = Field(
        default="M1",
        description="Rendering-level motif primitive (stroke style, not geometric structure)",
    )
    production_rule_id: Literal["R01", "R02", "R03", "R04", "R05", "R06"] = Field(
        default="R01",
        description="Production rule identifier from the curated registry",
    )
    recursion_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of L-system expansion iterations",
    )
    step_length: float = Field(
        default=1.0,
        gt=0,
        description="Turtle forward step length in abstract units",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility (reserved for future stochastic extensions)",
    )

    @model_validator(mode="after")
    def warn_angle_mismatch(self) -> "KolamParams":
        """Warn if the specified angle doesn't match the rule's published default."""
        expected = RULE_DEFAULT_ANGLES.get(self.production_rule_id)
        if expected is not None and abs(self.angle - expected) > 1e-6:
            warnings.warn(
                f"Angle {self.angle}° does not match the published default "
                f"({expected}°) for rule {self.production_rule_id}. "
                f"The rule may not produce its characteristic pattern.",
                UserWarning,
                stacklevel=2,
            )
        return self

    model_config = {"frozen": True}


class BoundingBox(BaseModel):
    """Axis-aligned bounding box for generated geometry."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


class EquationTerm(BaseModel):
    """Mathematical equation definition for a single continuous parametric stroke or curve."""

    subpath_index: int = Field(description="0-indexed identifier for the continuous stroke path")
    expression_x: str = Field(description="Analytical formula for x(t), e.g., 'x(t) = a0 + a1*t + a2*t^2 + a3*t^3'")
    expression_y: str = Field(description="Analytical formula for y(t), e.g., 'y(t) = b0 + b1*t + b2*t^2 + b3*t^3'")
    t_min: float = Field(default=0.0, description="Start of parameter interval")
    t_max: float = Field(default=1.0, description="End of parameter interval")
    degree: int = Field(default=1, description="Polynomial degree (1=linear, 2=quadratic, 3=cubic)")
    coefficients_x: list[float] = Field(description="Coefficients [c0, c1, ...] for x(t)")
    coefficients_y: list[float] = Field(description="Coefficients [c0, c1, ...] for y(t)")
    r_squared_x: float = Field(default=1.0, description="Coefficient of determination for x(t)")
    r_squared_y: float = Field(default=1.0, description="Coefficient of determination for y(t)")
    max_error: float = Field(default=0.0, description="Maximum Euclidean coordinate fitting error")


class GeometricRepresentation(BaseModel):
    """Complete mathematical geometric representation of a Kolam pattern."""

    representation_type: Literal[
        "piecewise_parametric_linear",
        "piecewise_parametric_cubic",
        "parametric_fourier_harmonic",
        "recursive_grammar",
    ] = Field(description="Mathematical representation class")
    num_subpaths: int = Field(description="Total number of continuous mathematical subpaths")
    equations: list[EquationTerm] = Field(default_factory=list, description="Explicit equation definitions for each subpath")
    mean_fitting_error: float = Field(default=0.0, description="Mean Euclidean error across all curve samples")
    max_fitting_error: float = Field(default=0.0, description="Peak Euclidean error across all curve samples")
    is_closed_loop: bool = Field(default=False, description="Whether the curve forms a closed topological loop")


class KolamMetadata(BaseModel):
    """Complete metadata for a generated Kolam image.

    Every generated image has an associated metadata record that
    provides the full mathematical ground truth.
    """

    image_id: str = Field(description="Unique identifier, e.g. K000001")
    params: KolamParams
    generator_version: str = Field(default="0.1.0")
    timestamp: str = Field(description="ISO 8601 generation timestamp")
    expanded_string_length: int = Field(
        ge=0, description="Length of the fully expanded L-system string"
    )
    num_segments: int = Field(
        ge=0, description="Number of line segments after turtle interpretation"
    )
    num_segments_after_symmetry: int = Field(
        ge=0, description="Number of line segments after symmetry transform"
    )
    connectivity: str = Field(
        description="Connectivity type from the production rule: single_stroke | multi_stroke | branching"
    )
    bounding_box: Optional[BoundingBox] = None
    geometric_representation: Optional[GeometricRepresentation] = None
