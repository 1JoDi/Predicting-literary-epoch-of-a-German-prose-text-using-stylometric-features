"""Tests for parsing and filtering the Gutenberg catalog."""

import pandas as pd

from src.catalog import (contains_keyword, filter_catalog, normalise_title,
                         parse_author, split_authors)
from tests.test_epochs import EPOCHS

CORPUS = {
    "language": "de",
    "locc_prefix": "PT",
    "exclude_keywords": ["drama", "gedicht", "briefe"],
    "seed": 1,
}


def test_parse_author_with_life_dates():
    author = parse_author("Goethe, Johann Wolfgang von, 1749-1832")
    assert author["name"] == "Goethe, Johann Wolfgang von"
    assert author["birth_year"] == 1749
    assert author["death_year"] == 1832
    assert author["role"] is None


def test_parse_author_with_uncertain_and_missing_years():
    assert parse_author("Heine, Heinrich, 1797?-1856")["death_year"] == 1856
    assert parse_author("Wolff, Julius, 1834-")["death_year"] is None
    assert parse_author("Anonymous")["death_year"] is None


def test_parse_author_with_role():
    author = parse_author("Doré, Gustave, 1832-1883 [Illustrator]")
    assert author["role"] == "Illustrator"
    assert author["death_year"] == 1883


def test_split_authors():
    authors = split_authors("Storm, Theodor, 1817-1888; "
                            "Meyer, Hans, 1850-1920 [Editor]")
    assert [author["name"] for author in authors] == ["Storm, Theodor",
                                                      "Meyer, Hans"]


def test_contains_keyword_matches_word_starts_only():
    assert contains_keyword("Ausgewählte Gedichte", ["gedicht"])
    assert not contains_keyword("Die Preise", ["reise"])


def test_normalise_title():
    assert normalise_title("Effi  Briest: Roman") == "effi briest roman"


def test_filter_catalog():
    catalog = pd.DataFrame({
        "Text#": ["1", "2", "3", "4", "5", "6", "7"],
        "Type": ["Text"] * 7,
        "Language": ["de", "de", "en", "de", "de", "de", "de"],
        "Title": ["Effi Briest", "Gedichte", "Effi Briest", "Faust: Drama",
                  "Der Prozess", "Effi Briest", "Unbekannt"],
        "Authors": ["Fontane, Theodor, 1819-1898",
                    "Heine, Heinrich, 1797-1856",
                    "Fontane, Theodor, 1819-1898",
                    "Goethe, Johann Wolfgang von, 1749-1832",
                    "Kafka, Franz, 1883-1924",
                    "Fontane, Theodor, 1819-1898",
                    "Anonymous"],
        "Subjects": [""] * 7,
        "Bookshelves": [""] * 7,
        "LoCC": ["PT"] * 7,
    })
    candidates = filter_catalog(catalog, CORPUS, EPOCHS)
    # Kept: 1 (Effi Briest) and 5 (Kafka).  Dropped: poetry (2), English
    # (3), drama (4), duplicate edition (6), unknown author (7).
    assert sorted(candidates["gutenberg_id"]) == [1, 5]
    epochs = dict(zip(candidates["gutenberg_id"], candidates["epoch"]))
    assert epochs == {1: "Realismus", 5: "Moderne"}


def test_filter_catalog_drops_translations():
    catalog = pd.DataFrame({
        "Text#": ["1"], "Type": ["Text"], "Language": ["de"],
        "Title": ["Oliver Twist"],
        "Authors": ["Dickens, Charles, 1812-1870; "
                    "Scheibe, Anon, 1800-1870 [Translator]"],
        "Subjects": [""], "Bookshelves": [""], "LoCC": ["PT"],
    })
    assert filter_catalog(catalog, CORPUS, EPOCHS).empty
