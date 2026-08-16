"""Tests for GrammarExecutor and Analysis-by-Synthesis depth search."""

import numpy as np
import pytest

from kolam_r.grammar.executor import GrammarExecutor, ParsedGrammar
from kolam_r.grammar.tokenizer import GrammarTokenizer
from kolam_r.lsystem.rules import ALL_RULES, RULES_BY_ID


class TestGrammarExecutor:
    """Tests for parsing, executing, and depth searching grammars."""

    def test_parse_valid_rules(self):
        executor = GrammarExecutor()
        tokenizer = GrammarTokenizer()

        for rule in ALL_RULES:
            g_str = tokenizer.rule_to_grammar_string(rule)
            parsed = executor.parse_grammar_string(g_str)
            assert parsed.is_valid, f"Failed to parse rule {rule.rule_id}: {parsed.error_message}"
            assert parsed.axiom == rule.axiom
            assert parsed.productions == rule.productions

    def test_parse_invalid_string(self):
        executor = GrammarExecutor()
        parsed = executor.parse_grammar_string("NOT A GRAMMAR")
        assert not parsed.is_valid
        assert parsed.error_message is not None

    def test_execute_grammar_to_segments(self):
        executor = GrammarExecutor()
        tokenizer = GrammarTokenizer()
        rule = RULES_BY_ID["R01"]
        g_str = tokenizer.rule_to_grammar_string(rule)
        parsed = executor.parse_grammar_string(g_str)

        segments, bbox = executor.execute_grammar_to_segments(parsed, depth=1, angle=45.0, symmetry="C1")
        assert len(segments) > 0
        assert bbox.width > 0

    def test_search_depth_ncc(self):
        executor = GrammarExecutor()
        tokenizer = GrammarTokenizer()
        rule = RULES_BY_ID["R01"]
        g_str = tokenizer.rule_to_grammar_string(rule)
        parsed = executor.parse_grammar_string(g_str)

        # Render true target at depth 2
        target_img = executor.render_grammar(
            parsed=parsed, depth=2, angle=45.0, symmetry="C1", grid_size=5
        )

        # Depth search should find depth 2 with high NCC (~1.0)
        found_depth, ncc = executor.search_depth_ncc(
            parsed=parsed,
            target_image=target_img,
            angle=45.0,
            symmetry="C1",
            grid_size=5,
            max_safe_depth=3,
        )
        assert found_depth == 2
        assert ncc > 0.95
