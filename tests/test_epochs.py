"""Tests for the death year -> epoch mapping."""

import pandas as pd

from src.epochs import death_year_to_epoch, distribution_table, epoch_names

EPOCHS = {
    "min_death_year": 1720,
    "bins": [
        {"name": "Goethezeit", "max_death_year": 1860},
        {"name": "Realismus", "max_death_year": 1910},
        {"name": "Moderne", "max_death_year": 1960},
    ],
}


def test_epoch_names_are_chronological():
    assert epoch_names(EPOCHS) == ["Goethezeit", "Realismus", "Moderne"]


def test_death_year_to_epoch_inside_bins():
    assert death_year_to_epoch(1832, EPOCHS) == "Goethezeit"   # Goethe
    assert death_year_to_epoch(1898, EPOCHS) == "Realismus"    # Fontane
    assert death_year_to_epoch(1924, EPOCHS) == "Moderne"      # Kafka


def test_death_year_to_epoch_boundaries_are_inclusive():
    assert death_year_to_epoch(1860, EPOCHS) == "Goethezeit"
    assert death_year_to_epoch(1861, EPOCHS) == "Realismus"


def test_death_year_outside_or_missing():
    assert death_year_to_epoch(None, EPOCHS) is None
    assert death_year_to_epoch(1600, EPOCHS) is None
    assert death_year_to_epoch(1990, EPOCHS) is None


def test_distribution_table_counts_texts_and_authors():
    corpus = pd.DataFrame({
        "author": ["A", "A", "B", "C"],
        "epoch": ["Moderne", "Moderne", "Moderne", "Goethezeit"],
    })
    table = distribution_table(corpus, epoch_names(EPOCHS))
    assert table.loc["Moderne", "texts"] == 3
    assert table.loc["Moderne", "authors"] == 2
    assert table.loc["Realismus", "texts"] == 0
