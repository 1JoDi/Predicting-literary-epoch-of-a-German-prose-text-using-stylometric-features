"""Construction of the feature table (one row per text).

For every text in the corpus, the raw file is cleaned, a fixed-size sample
is extracted and processed with spaCy, and the four features are computed.
The results are appended to ``data/processed/features.csv`` line by line,
so an interrupted run can be resumed: texts that are already in the table
are skipped.  Only the numbers are stored, never the spaCy ``Doc``.
"""

import csv

import pandas as pd
import spacy

from src.cleaning import (clean_text, extract_sample,
                          strip_gutenberg_boilerplate)
from src.download import raw_text_path
from src.features import FEATURE_NAMES, extract_features

METADATA_COLUMNS = ["gutenberg_id", "author", "title", "birth_year",
                    "death_year", "epoch"]
# The lemmatizer is not needed for any feature; disabling it saves time.
UNUSED_PIPES = ["lemmatizer"]


def iter_samples(corpus, config, done_ids):
    """Yields the text sample of every corpus text that is not done yet.

    This is a generator: only one text is held in memory at a time, which
    matters because complete novels are large.

    Args:
      corpus (pandas.DataFrame): The corpus metadata.
      config (dict): The full configuration.
      done_ids (set of int): Gutenberg ids that are already processed.

    Yields:
      tuple: ``(sample_text, metadata)`` where ``metadata`` is a dict with
        the columns in ``METADATA_COLUMNS``.
    """
    sampling = config["sampling"]
    for _, row in corpus.iterrows():
        if row["gutenberg_id"] in done_ids:
            continue
        path = raw_text_path(config["paths"]["texts_dir"], row["gutenberg_id"])
        text = clean_text(strip_gutenberg_boilerplate(
            path.read_text(encoding="utf-8")))
        sample = extract_sample(text, sampling["skip_fraction"],
                                sampling["sample_words"])
        if sample is None:
            print(f"  ! {row['gutenberg_id']} is too short, skipped")
            continue
        yield sample, {column: row[column] for column in METADATA_COLUMNS}


def load_done_ids(features_file):
    """Reads the ids of texts that already have features.

    Args:
      features_file (pathlib.Path): The feature table.

    Returns:
      set of int: The Gutenberg ids in the table (empty if no table).
    """
    if not features_file.exists():
        return set()
    return set(pd.read_csv(features_file)["gutenberg_id"])


def build_feature_table(config):
    """Runs spaCy over all corpus texts and writes ``features.csv``.

    Args:
      config (dict): The full configuration.

    Returns:
      pandas.DataFrame: The complete feature table.
    """
    paths = config["paths"]
    feature_config = config["features"]
    corpus = pd.read_csv(paths["corpus_file"])
    done_ids = load_done_ids(paths["features_file"])
    n_todo = len(set(corpus["gutenberg_id"]) - done_ids)
    print(f"{len(done_ids)} texts already processed, {n_todo} to go.")

    nlp = spacy.load(feature_config["spacy_model"], disable=UNUSED_PIPES)

    write_header = not paths["features_file"].exists()
    with open(paths["features_file"], "a", encoding="utf-8",
              newline="") as out_file:
        writer = csv.DictWriter(out_file,
                                fieldnames=METADATA_COLUMNS + FEATURE_NAMES)
        if write_header:
            writer.writeheader()

        docs = nlp.pipe(iter_samples(corpus, config, done_ids),
                        as_tuples=True,
                        batch_size=feature_config["batch_size"],
                        n_process=feature_config["n_process"])
        for index, (doc, metadata) in enumerate(docs, start=1):
            row = {**metadata, **extract_features(doc, feature_config)}
            writer.writerow(row)
            out_file.flush()
            print(f"  [{index:>4}/{n_todo}] {metadata['author'][:30]:<30} "
                  f"{metadata['title'][:40]}")

    features = pd.read_csv(paths["features_file"])
    print(f"Feature table has {len(features)} rows: "
          f"{paths['features_file']}")
    return features
