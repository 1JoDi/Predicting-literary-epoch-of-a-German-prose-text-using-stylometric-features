"""Tests for boilerplate removal, cleaning and sampling."""

from src.cleaning import (clean_text, count_words, extract_sample,
                          strip_gutenberg_boilerplate)

RAW = """The Project Gutenberg eBook of Test
License text that must disappear.
*** START OF THE PROJECT GUTENBERG EBOOK TEST ***
Erstes Kapitel.

Es war einmal ein
_kleiner_ Mann.[Illustration: Ein Mann]

* * * * *

Er ging nach Hause.
*** END OF THE PROJECT GUTENBERG EBOOK TEST ***
More license text.
"""


def test_strip_boilerplate():
    body = strip_gutenberg_boilerplate(RAW)
    assert "License" not in body
    assert "More license" not in body
    assert "Es war einmal" in body


def test_strip_boilerplate_without_markers_keeps_text():
    assert strip_gutenberg_boilerplate("Nur Text.") == "Nur Text."


def test_clean_text_joins_lines_and_removes_markup():
    cleaned = clean_text(strip_gutenberg_boilerplate(RAW))
    assert cleaned == ("Erstes Kapitel.\n\n"
                       "Es war einmal ein kleiner Mann.\n\n"
                       "Er ging nach Hause.")


def test_count_words():
    assert count_words("ein  zwei\ndrei") == 3


def test_extract_sample_has_exact_length_and_skips_start():
    paragraphs = [f"p{index} " + "wort " * 9 for index in range(10)]
    text = "\n\n".join(paragraph.strip() for paragraph in paragraphs)
    sample = extract_sample(text, skip_fraction=0.2, sample_words=25)
    assert count_words(sample) == 25
    # 20 % of 100 words = 20 words = the first two paragraphs.
    assert sample.startswith("p2 ")


def test_extract_sample_returns_none_for_short_texts():
    assert extract_sample("zu kurz", 0.0, 100) is None
