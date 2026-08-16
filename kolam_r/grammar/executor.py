"""Executable grammar parser, symbolic validator, and Analysis-by-Synthesis depth search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from kolam_r.lsystem.engine import LSystemEngine
from kolam_r.lsystem.rules import ProductionRule
from kolam_r.renderer.image_renderer import render_kolam
from kolam_r.schema import BoundingBox, KolamParams
from kolam_r.symmetry.transforms import apply_symmetry
from kolam_r.turtle.interpreter import LineSegment, TurtleInterpreter


@dataclass(frozen=True)
class ParsedGrammar:
    """Represents a syntactically parsed L-system grammar."""

    is_valid: bool
    axiom: str = ""
    productions: dict[str, str] = None
    error_message: str | None = None

    def to_lsystem_rule(self, name: str = "recovered", default_angle: float = 90.0) -> ProductionRule | None:
        """Convert to executable ProductionRule if valid."""
        if not self.is_valid or not self.axiom or not self.productions:
            return None
        return ProductionRule(
            rule_id="REC",
            name=name,
            axiom=self.axiom,
            productions=self.productions,
            default_angle=default_angle,
            description="Recovered grammar rule",
            max_safe_depth=4,
            connectivity="closed_loop",
            source="recovered",
        )


class GrammarExecutor:
    """Parses grammar strings, validates syntax, and executes geometric synthesis."""

    def __init__(self) -> None:
        self.engine = LSystemEngine()
        self.interpreter = TurtleInterpreter()

    @staticmethod
    def parse_grammar_string(grammar_str: str) -> ParsedGrammar:
        """Parse a canonical grammar string into an axiom and production dictionary.

        Expected format:
            AXIOM: <axiom> ; RULES: <v1> -> <succ1> ; <v2> -> <succ2>
        """
        clean_str = grammar_str.strip()
        if not clean_str.startswith("AXIOM:"):
            return ParsedGrammar(
                is_valid=False,
                error_message="String must begin with 'AXIOM:' header.",
            )

        if "RULES:" not in clean_str:
            return ParsedGrammar(
                is_valid=False,
                error_message="String missing 'RULES:' section header.",
            )

        try:
            # Split into Axiom section and Rules section
            parts = clean_str.split("RULES:")
            axiom_section = parts[0].replace("AXIOM:", "").strip()
            rules_section = parts[1].strip()

            # Clean axiom
            if axiom_section.endswith(";"):
                axiom_section = axiom_section[:-1].strip()
            axiom = axiom_section.replace(" ", "")

            if not axiom:
                return ParsedGrammar(is_valid=False, error_message="Empty axiom string.")

            # Parse production rules separated by ';'
            productions: dict[str, str] = {}
            rule_clauses = [c.strip() for c in rules_section.split(";") if c.strip()]

            for clause in rule_clauses:
                if "->" not in clause:
                    continue
                lhs, rhs = clause.split("->", 1)
                var = lhs.strip().replace(" ", "")
                succ = rhs.strip().replace(" ", "")
                if var and succ:
                    productions[var] = succ

            if not productions:
                return ParsedGrammar(
                    is_valid=False,
                    error_message="No valid production rules (var -> succ) parsed.",
                )

            return ParsedGrammar(
                is_valid=True,
                axiom=axiom,
                productions=productions,
                error_message=None,
            )

        except Exception as e:
            return ParsedGrammar(is_valid=False, error_message=f"Parsing error: {str(e)}")

    def execute_grammar_to_segments(
        self,
        parsed: ParsedGrammar,
        depth: int,
        angle: float,
        symmetry: str = "C1",
    ) -> tuple[list[LineSegment], BoundingBox]:
        """Expand grammar and interpret turtle commands into symmetrical line segments."""
        if not parsed.is_valid or not parsed.axiom or not parsed.productions:
            return [], BoundingBox(0.0, 0.0, 0.0, 0.0)

        rule = parsed.to_lsystem_rule(default_angle=angle)
        expanded_str = self.engine.expand_rule(rule, depth=depth)
        turtle_result = self.interpreter.interpret(expanded_str, angle=angle)
        sym_segments = apply_symmetry(turtle_result.segments, symmetry=symmetry)
        bbox = BoundingBox(
            min_x=turtle_result.min_x,
            min_y=turtle_result.min_y,
            max_x=turtle_result.max_x,
            max_y=turtle_result.max_y,
        )
        return sym_segments, bbox

    def render_grammar(
        self,
        parsed: ParsedGrammar,
        depth: int,
        angle: float,
        symmetry: str = "C1",
        grid_size: int = 5,
        motif: str = "M1",
        canvas_size: int = 64,
    ) -> np.ndarray:
        """Render parsed grammar to grayscale canvas."""
        sym_segments, bbox = self.execute_grammar_to_segments(
            parsed, depth=depth, angle=angle, symmetry=symmetry
        )
        return render_kolam(
            segments=sym_segments,
            image_size=canvas_size,
            motif=motif,
            grid_size=grid_size,
        )

    def search_depth_ncc(
        self,
        parsed: ParsedGrammar,
        target_image: np.ndarray,
        angle: float,
        symmetry: str = "C1",
        grid_size: int = 5,
        motif: str = "M1",
        max_safe_depth: int = 4,
    ) -> tuple[int, float]:
        """Protocol B: Bounded Analysis-by-Synthesis Depth Search via Normalized Cross-Correlation.

        Sweeps candidate depths d in [1, max_safe_depth], renders the candidate pattern,
        and selects d_hat = argmax_d NCC(Render(G, d), target_image).
        """
        if not parsed.is_valid:
            return 1, 0.0

        best_depth = 1
        best_ncc = -1.0

        t_img = target_image.astype(np.float32)
        t_norm = (t_img - np.mean(t_img)) / (np.std(t_img) + 1e-8)

        for d in range(1, max_safe_depth + 1):
            try:
                cand_img = self.render_grammar(
                    parsed=parsed,
                    depth=d,
                    angle=angle,
                    symmetry=symmetry,
                    grid_size=grid_size,
                    motif=motif,
                    canvas_size=target_image.shape[0],
                )
                c_img = cand_img.astype(np.float32)
                c_norm = (c_img - np.mean(c_img)) / (np.std(c_img) + 1e-8)
                ncc = float(np.mean(t_norm * c_norm))

                if ncc > best_ncc:
                    best_ncc = ncc
                    best_depth = d
            except Exception:
                continue

        return best_depth, max(best_ncc, 0.0)
