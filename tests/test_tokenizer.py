"""Tests for GrammarTokenizer and Vocabulary."""

import pytest

from kolam_r.grammar.tokenizer import GrammarTokenizer
from kolam_r.grammar.vocabulary import GrammarVocabulary
from kolam_r.lsystem.rules import ALL_RULES, RULES_BY_ID


class TestGrammarTokenizer:
    """Tests for GrammarTokenizer serialization, encoding, and roundtrips."""

    def test_vocabulary_tokens(self):
        vocab = GrammarVocabulary()
        assert len(vocab) == 24
        assert vocab.encode_token("<PAD>") == 0
        assert vocab.encode_token("<BOS>") == 1
        assert vocab.encode_token("<EOS>") == 2
        assert vocab.encode_token("AXIOM:") == 20
        assert vocab.encode_token("RULES:") == 21

    def test_all_rules_serialize_and_tokenize(self):
        tokenizer = GrammarTokenizer()
        for rule in ALL_RULES:
            g_str = tokenizer.rule_to_grammar_string(rule)
            assert g_str.startswith("AXIOM:")
            assert "RULES:" in g_str

            tokens = tokenizer.tokenize(g_str)
            assert len(tokens) > 0
            # All tokens must be in vocabulary
            for t in tokens:
                assert t in tokenizer.vocab.token_to_id, f"Token {t} not in vocab"

    def test_encode_decode_roundtrip(self):
        tokenizer = GrammarTokenizer()
        for rule in ALL_RULES:
            g_str = tokenizer.rule_to_grammar_string(rule)
            token_ids = tokenizer.encode(g_str, max_length=110, add_special_tokens=True)
            assert len(token_ids) == 110
            assert token_ids[0] == tokenizer.vocab.bos_idx

            decoded = tokenizer.decode(token_ids, remove_special_tokens=True)
            # Re-tokenization must match exactly
            assert tokenizer.tokenize(decoded) == tokenizer.tokenize(g_str)
