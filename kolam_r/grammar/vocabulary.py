"""Constrained grammar vocabulary for KOLAM-R.

A closed, minimal symbolic vocabulary of 24 tokens covering all authentic
Kolam L-system axioms and production rewrite rules.
"""

from __future__ import annotations

# Constrained Vocabulary Tokens
VOCAB_TOKENS: list[str] = [
    "<PAD>",     # 0: Padding token
    "<BOS>",     # 1: Beginning of sequence
    "<EOS>",     # 2: End of sequence
    "F",         # 3: Forward draw
    "f",         # 4: Forward move without draw
    "+",         # 5: Turn right
    "-",         # 6: Turn left
    "[",         # 7: Push state
    "]",         # 8: Pop state
    "|",         # 9: Reverse direction (180°)
    "(",         # 10: Special loop open
    ")",         # 11: Special loop close
    "L",         # 12: Loop terminal marker
    "A",         # 13: Non-terminal A
    "B",         # 14: Non-terminal B
    "C",         # 15: Non-terminal C
    "D",         # 16: Non-terminal D
    "X",         # 17: Non-terminal X
    "Y",         # 18: Non-terminal Y
    "R",         # 19: Non-terminal R
    "AXIOM:",    # 20: Axiom section header
    "RULES:",    # 21: Rules section header
    "->",        # 22: Production rewrite arrow
    ";",         # 23: Section / rule delimiter
]

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


class GrammarVocabulary:
    """Vocabulary mapping and token-id conversion engine."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens = tokens or list(VOCAB_TOKENS)
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for i, t in enumerate(self.tokens)}
        self.pad_idx = PAD_IDX
        self.bos_idx = BOS_IDX
        self.eos_idx = EOS_IDX

    def __len__(self) -> int:
        return len(self.tokens)

    def encode_token(self, token: str) -> int:
        if token not in self.token_to_id:
            raise KeyError(f"Unknown grammar token: '{token}'. Must be in vocabulary.")
        return self.token_to_id[token]

    def decode_id(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, "<UNK>")
