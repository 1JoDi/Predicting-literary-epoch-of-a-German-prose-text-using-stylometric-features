"""Tests for the four stylometric features on hand-made spaCy Docs.

The Docs are built directly from words, POS tags, sentence starts and
entity tags, so the expected values can be counted by hand and the tests
do not depend on a statistical model.
"""

from collections import Counter

import pytest
import spacy
from spacy.tokens import Doc

from src.features import (character_density, count_characters,
                          extract_features, function_word_ratio,
                          median_sentence_length, normalise_person_name,
                          standardised_ttr)

VOCAB = spacy.blank("de").vocab
TITLES = ["herr", "frau", "von"]


def make_doc():
    """Four sentences with 2, 5, 1 and 2 words (plus punctuation)."""
    words = ["Effi", "lachte", ".",
             "Herr", "Innstetten", "kam", "in", "das", ".",
             "Haus", ".",
             "Effi", "schlief", "."]
    pos = ["PROPN", "VERB", "PUNCT",
           "NOUN", "PROPN", "VERB", "ADP", "DET", "PUNCT",
           "NOUN", "PUNCT",
           "PROPN", "VERB", "PUNCT"]
    sent_starts = [True, False, False,
                   True, False, False, False, False, False,
                   True, False,
                   True, False, False]
    ents = ["B-PER", "O", "O",
            "B-PER", "I-PER", "O", "O", "O", "O",
            "O", "O",
            "B-PER", "O", "O"]
    return Doc(VOCAB, words=words, pos=pos, sent_starts=sent_starts,
               ents=ents)


def test_median_sentence_length_ignores_punctuation():
    # Sentence lengths in words: 2, 5, 1, 2 -> median 2.0
    assert median_sentence_length(make_doc()) == 2.0


def test_function_word_ratio():
    # 10 words, of which "in" (ADP) and "das" (DET) are function words.
    assert function_word_ratio(make_doc(), ["DET", "ADP"]) == 0.2


def test_standardised_ttr_averages_windows():
    words = ["a", "a", "b", "c", "d", "d", "d", "d", "x"]
    doc = Doc(VOCAB, words=words)
    # Windows of 4: [a a b c] -> 3/4, [d d d d] -> 1/4; "x" is ignored.
    assert standardised_ttr(doc, 4) == pytest.approx(0.5)


def test_standardised_ttr_is_case_insensitive():
    doc = Doc(VOCAB, words=["Der", "der", "Mann", "mann"])
    assert standardised_ttr(doc, 4) == pytest.approx(0.5)


def test_standardised_ttr_raises_for_short_text():
    with pytest.raises(ValueError):
        standardised_ttr(Doc(VOCAB, words=["kurz"]), 10)


def test_normalise_person_name_removes_titles_and_punctuation():
    assert normalise_person_name("Herr von Innstetten,", TITLES) \
        == "innstetten"


def test_count_characters_merges_variants():
    names = Counter({"effi briest": 3, "effi": 5, "effis": 1,
                     "innstetten": 4, "roswitha": 1})
    # effi/effis/effi briest -> one character; roswitha has 1 mention.
    assert count_characters(names, min_mentions=2) == 2
    assert count_characters(names, min_mentions=1) == 3


def test_character_density():
    # PER: "Effi" x2, "Herr Innstetten" x1 -> with min_mentions=1 two
    # characters in 10 words = 200 per 1000 words.
    assert character_density(make_doc(), TITLES, 1) == pytest.approx(200)
    assert character_density(make_doc(), TITLES, 2) == pytest.approx(100)


def test_extract_features_returns_all_four():
    config = {"sttr_window": 5, "function_word_pos": ["DET", "ADP"],
              "person_titles": TITLES, "min_person_mentions": 1}
    features = extract_features(make_doc(), config)
    assert list(features) == ["median_sentence_length", "sttr",
                              "function_word_ratio", "character_density"]
