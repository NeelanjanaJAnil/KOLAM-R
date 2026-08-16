"""Canonical tokenizer for KOLAM-R L-system grammar strings."""

from __future__ import annotations

import re
from kolam_r.grammar.vocabulary import BOS_IDX, EOS_IDX, PAD_IDX, GrammarVocabulary
from kolam_r.lsystem.rules import ProductionRule, RULES_BY_ID


class GrammarTokenizer:
    """Encodes and decodes between ProductionRule objects, canonical strings, and token ID sequences."""

    def __init__(self, vocab: GrammarVocabulary | None = None) -> None:
        self.vocab = vocab or GrammarVocabulary()

    @staticmethod
    def rule_to_grammar_string(rule: ProductionRule) -> str:
        """Serialize a ProductionRule into a canonical whitespace-separated grammar string.

        Format:
            AXIOM: <axiom> ; RULES: <v1> -> <succ1> ; <v2> -> <succ2>
        """
        axiom_str = rule.axiom.strip()
        parts = ["AXIOM:", axiom_str, ";", "RULES:"]

        # Sort variable keys alphabetically for deterministic canonical string representation
        for var in sorted(rule.productions.keys()):
            succ = rule.productions[var].strip()
            parts.extend([var, "->", succ, ";"])

        # Remove trailing semicolon if present
        if parts[-1] == ";":
            parts.pop()

        return " ".join(parts)

    def tokenize(self, grammar_str: str) -> list[str]:
        """Tokenize a canonical grammar string into a list of individual symbolic tokens."""
        raw_tokens = grammar_str.strip().split()
        final_tokens: list[str] = []

        keywords = {"AXIOM:", "RULES:", "->", ";"}

        for token in raw_tokens:
            if token in keywords:
                final_tokens.append(token)
            else:
                # Split character sequence into individual atomic symbols (terminals / variables)
                for char in token:
                    final_tokens.append(char)

        return final_tokens

    def encode(
        self,
        grammar_str: str,
        max_length: int = 110,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Convert a canonical grammar string into a fixed-length list of integer token IDs."""
        tokens = self.tokenize(grammar_str)
        token_ids: list[int] = []

        if add_special_tokens:
            token_ids.append(self.vocab.bos_idx)

        for t in tokens:
            token_ids.append(self.vocab.encode_token(t))

        if add_special_tokens:
            token_ids.append(self.vocab.eos_idx)

        # Truncate or Pad
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            if add_special_tokens:
                token_ids[-1] = self.vocab.eos_idx
        else:
            token_ids.extend([self.vocab.pad_idx] * (max_length - len(token_ids)))

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        remove_special_tokens: bool = True,
    ) -> str:
        """Convert a list of integer token IDs back into a reconstructed grammar string."""
        tokens: list[str] = []
        for tid in token_ids:
            if tid == self.vocab.pad_idx and remove_special_tokens:
                continue
            if tid == self.vocab.bos_idx and remove_special_tokens:
                continue
            if tid == self.vocab.eos_idx and remove_special_tokens:
                break
            tokens.append(self.vocab.decode_id(tid))

        # Reconstruct canonical spacing
        reconstructed: list[str] = []
        for t in tokens:
            if t in {"AXIOM:", "RULES:", "->", ";"}:
                reconstructed.append(t)
            else:
                # If preceding element is a structural keyword, start new block; else append char
                if not reconstructed or reconstructed[-1] in {"AXIOM:", "RULES:", "->", ";"}:
                    reconstructed.append(t)
                else:
                    reconstructed[-1] += t

        return " ".join(reconstructed)
