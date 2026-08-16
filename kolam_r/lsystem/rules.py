"""Production rule registry for KOLAM-R.

Defines 6 curated L-system production rules (R01–R06) sourced from
published Kolam L-system research. Each rule produces a distinct
family of Kolam-like patterns when interpreted through turtle geometry.

SCIENTIFIC NOTE: These rules are mathematical generative formalisms
chosen for controlled synthetic data generation. They do NOT claim to
represent the historical construction process used by Kolam artists.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProductionRule:
    """A single L-system production rule specification.

    Attributes:
        rule_id: Unique identifier (R01–R06).
        name: Human-readable name.
        axiom: Starting string for L-system expansion.
        productions: Mapping of symbol -> replacement string.
        default_angle: Default turtle turning angle in degrees.
        description: Description of the pattern and its visual characteristics.
        max_safe_depth: Maximum recursion depth before string length explosion.
            Beyond this depth, the expanded string may exceed memory limits
            or produce unresolvable detail at 64×64 resolution.
        connectivity: Expected connectivity type of the generated pattern.
        source: Academic citation or source for the rule.
    """

    rule_id: str
    name: str
    axiom: str
    productions: dict[str, str]
    default_angle: float
    description: str
    max_safe_depth: int
    connectivity: str  # "single_stroke" | "multi_stroke" | "branching"
    source: str


def _build_registry() -> dict[str, ProductionRule]:
    """Build the curated production rule registry."""
    rules = [
        ProductionRule(
            rule_id="R01",
            name="Krishna Anklets",
            axiom="F--XF--F--XF",
            productions={"X": "XF+F+XF--F--XF+F+X"},
            default_angle=45.0,
            description=(
                "Anklets of Krishna (Brahma Mudi variant). A recursive continuous "
                "looping pattern around expanding diamond dot grids. At higher depths, "
                "produces woven 8-lobed lotus loop structures."
            ),
            max_safe_depth=4,
            connectivity="single_stroke",
            source="Prusinkiewicz & Hanan 1989; Paul Bourke fractal archive",
        ),
        ProductionRule(
            rule_id="R02",
            name="Snake Kolam",
            axiom="F+XF+F+XF",
            productions={"X": "XF-F-F+XF+F+XF-F-F+X"},
            default_angle=90.0,
            description=(
                "Snake Kolam (Naga Kolam). Orthogonal grid-filling loops that "
                "weave between rows of dots, forming interlaced ribbon patterns. "
                "The 90-degree angle produces strictly orthogonal meanders."
            ),
            max_safe_depth=3,
            connectivity="single_stroke",
            source="Paul Bourke fractal archive; traditional Naga Kolam family",
        ),
        ProductionRule(
            rule_id="R03",
            name="Kolam Tile",
            axiom="(-D--D)",
            productions={
                "A": "F++FFFF--F--FFFF++F++FFFF--F",
                "B": "F--FFFF++F++FFFF--F--FFFF++F",
                "C": "BFA--BFA",
                "D": "CFC--CFC",
            },
            default_angle=45.0,
            description=(
                "Kolam Tile (Sikku Kolam mat). Intricate braided pattern from "
                "Prusinkiewicz & Hanan's hierarchical tile system. Uses 4 production "
                "rules (A, B, C, D) to build multi-level structure. The axiom starts "
                "with parenthesized expression for initial orientation."
            ),
            max_safe_depth=3,
            connectivity="single_stroke",
            source="Prusinkiewicz & Hanan 1989, Lindenmayer Systems, Fractals, and Plants",
        ),
        ProductionRule(
            rule_id="R04",
            name="Mango Leaf",
            axiom="-X--X",
            productions={"X": "XFX--XFX"},
            default_angle=45.0,
            description=(
                "Mango Leaf Kolam (Mavilai Kolam). Recursive expansion producing "
                "diamond-shaped mango-leaf motifs with increasing detail at each "
                "depth. Simple rule with elegant self-similar structure."
            ),
            max_safe_depth=5,
            connectivity="single_stroke",
            source="Traditional Mavilai Kolam; Paul Bourke archive",
        ),
        ProductionRule(
            rule_id="R05",
            name="Hilbert Meander",
            axiom="L",
            productions={
                "L": "+RF-LFL-FR+",
                "R": "-LF+RFR+FL-",
            },
            default_angle=90.0,
            description=(
                "Hilbert space-filling curve. A continuous non-intersecting path "
                "that meanders through the dot grid, visiting every cell. Included "
                "as a well-studied mathematical reference curve. Not a traditional "
                "Kolam pattern, but shares the single-stroke continuous path property."
            ),
            max_safe_depth=5,
            connectivity="single_stroke",
            source="Hilbert 1891; Prusinkiewicz & Lindenmayer, ABOP 1990",
        ),
        ProductionRule(
            rule_id="R06",
            name="Branching Floral",
            axiom="A",
            productions={"A": "F+[[A]-A]-F[-FA]+A"},
            default_angle=25.0,
            description=(
                "Branching Kolam with floral/tree-like motifs. Produces fractal "
                "branching structures using push/pop stack operations. Uses a "
                "non-standard 25-degree angle. Represents Kolam patterns with "
                "floral emanations rather than continuous loop structure."
            ),
            max_safe_depth=5,
            connectivity="branching",
            source="Prusinkiewicz & Lindenmayer, ABOP 1990 (plant branching model adapted)",
        ),
    ]
    return {r.rule_id: r for r in rules}


# Module-level registry (immutable after construction)
RULE_REGISTRY: dict[str, ProductionRule] = _build_registry()
RULES_BY_ID: dict[str, ProductionRule] = RULE_REGISTRY
ALL_RULES: list[ProductionRule] = list(RULE_REGISTRY.values())


def get_rule(rule_id: str) -> ProductionRule:
    """Retrieve a production rule by its ID.

    Args:
        rule_id: Rule identifier (e.g. "R01").

    Returns:
        The corresponding ProductionRule.

    Raises:
        KeyError: If the rule_id is not in the registry.
    """
    if rule_id not in RULE_REGISTRY:
        valid = ", ".join(sorted(RULE_REGISTRY.keys()))
        raise KeyError(f"Unknown rule_id '{rule_id}'. Valid rules: {valid}")
    return RULE_REGISTRY[rule_id]


def list_rules() -> list[str]:
    """Return a sorted list of all registered rule IDs."""
    return sorted(RULE_REGISTRY.keys())
