"""Tests for the L-system expansion engine."""

import pytest

from kolam_r.lsystem.engine import LSystemEngine, LSystemExpansionError
from kolam_r.lsystem.rules import get_rule, list_rules, RULE_REGISTRY


class TestLSystemEngine:
    """Tests for L-system string expansion."""

    def setup_method(self):
        self.engine = LSystemEngine()

    def test_depth_zero_returns_axiom(self):
        """Depth 0 should return the axiom unchanged."""
        result = self.engine.expand("F+F", {"F": "FF"}, depth=0)
        assert result == "F+F"

    def test_simple_expansion_depth_1(self):
        """Simple single-rule expansion at depth 1."""
        result = self.engine.expand("F", {"F": "F+F"}, depth=1)
        assert result == "F+F"

    def test_simple_expansion_depth_2(self):
        """Expansion at depth 2 applies the rule twice."""
        result = self.engine.expand("F", {"F": "F+F"}, depth=2)
        # F -> F+F -> F+F+F+F
        assert result == "F+F+F+F"

    def test_multiple_productions(self):
        """Multiple production rules should apply in parallel."""
        result = self.engine.expand("AB", {"A": "AB", "B": "A"}, depth=1)
        # A->AB, B->A => ABA
        assert result == "ABA"

    def test_non_terminal_passthrough(self):
        """Symbols without production rules are kept as-is."""
        result = self.engine.expand("F+F", {"F": "FF"}, depth=1)
        assert result == "FF+FF"

    def test_negative_depth_raises(self):
        """Negative depth should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            self.engine.expand("F", {}, depth=-1)

    def test_safety_guard_triggers(self):
        """String explosion should trigger the safety guard."""
        engine = LSystemEngine(max_string_length=50)
        with pytest.raises(LSystemExpansionError, match="exceeded"):
            # This rule doubles the string each iteration
            engine.expand("F", {"F": "FF+FF"}, depth=10)

    def test_all_rules_expand_depth_1(self):
        """All registered rules should expand without error at depth 1."""
        for rule_id in list_rules():
            rule = get_rule(rule_id)
            result = self.engine.expand(rule.axiom, rule.productions, depth=1)
            assert len(result) > 0, f"Rule {rule_id} produced empty string"

    def test_all_rules_expand_depth_2(self):
        """All registered rules should expand without error at depth 2."""
        for rule_id in list_rules():
            rule = get_rule(rule_id)
            result = self.engine.expand(rule.axiom, rule.productions, depth=2)
            assert len(result) > len(rule.axiom), f"Rule {rule_id} did not grow"

    def test_deterministic(self):
        """Same inputs should always produce the same output."""
        result1 = self.engine.expand("F", {"F": "F+F-F"}, depth=3)
        result2 = self.engine.expand("F", {"F": "F+F-F"}, depth=3)
        assert result1 == result2

    def test_expand_rule_by_id(self):
        """expand_rule should work with a valid rule ID."""
        result = self.engine.expand_rule("R04", depth=1)
        # R04 axiom is "-X--X", production X -> XFX--XFX
        assert "F" in result
        assert len(result) > len(get_rule("R04").axiom)

    def test_expand_rule_invalid_id(self):
        """expand_rule with invalid ID should raise KeyError."""
        with pytest.raises(KeyError):
            self.engine.expand_rule("R99")

    def test_r01_krishna_anklets_depth1(self):
        """R01 Krishna Anklets: verify expansion at depth 1 manually."""
        rule = get_rule("R01")
        result = self.engine.expand(rule.axiom, rule.productions, depth=1)
        # Axiom: F--XF--F--XF
        # X -> XF+F+XF--F--XF+F+X
        # So F--XF+F+XF--F--XF+F+XF--F--XF+F+XF--F--XF+F+XF
        assert "XF+F+XF--F--XF+F+X" in result

    def test_r03_kolam_tile_has_all_productions(self):
        """R03 should have productions for A, B, C, and D."""
        rule = get_rule("R03")
        assert set(rule.productions.keys()) == {"A", "B", "C", "D"}

    def test_estimate_length(self):
        """Length estimation should produce a positive integer."""
        rule = get_rule("R01")
        est = self.engine.estimate_length(rule, depth=2)
        assert est > 0
