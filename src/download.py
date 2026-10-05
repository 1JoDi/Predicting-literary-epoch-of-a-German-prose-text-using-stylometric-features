"""Download of the selected Gutenberg texts and assembly of the corpus.

The download step walks through the (shuffled) candidate list and keeps a
text if

* its author does not yet have ``max_texts_per_author`` texts,
* its epoch does not yet have ``max_texts_per_epoch`` texts, and
* the cleaned text is long enough (``min_words``).

Raw files are cached in ``data/raw/texts/<id>.txt``; a second run does not
download anything that is already on disk.  The final corpus is written
to ``data/processed/corpus_metadata.csv``.
"""

import time
from collections import Counter

import pandas as pd
import requests

from src.cleaning import clean_text, count_words, strip_gutenberg_boilerplate


def raw_text_path(texts_dir, gutenberg_id):
    """Returns the path of the cached raw file for one ebook.

    Args:
      texts_dir (pathlib.Path): Directory with the raw texts.
      gutenberg_id (int): The Gutenberg ebook number.

    Returns:
      pathlib.Path: ``<texts_dir>/<gutenberg_id>.txt``.
    """
    return texts_dir / f"{gutenberg_id}.txt"


def fetch_text(gutenberg_id, gutenberg_config, texts_dir):
    """Returns the raw text of an ebook, downloading it if necessary.

    Args:
      gutenberg_id (int): The Gutenberg ebook number.
      gutenberg_config (dict): The ``gutenberg`` section of the config.
      texts_dir (pathlib.Path): Directory with the cached raw texts.

    Returns:
      str or None: The raw text, or None if the download failed.
    """
    path = raw_text_path(texts_dir, gutenberg_id)
    if path.exists():
        return path.read_text(encoding="utf-8")

    url = gutenberg_config["text_url"].format(id=gutenberg_id)
    try:
        response = requests.get(
            url,
            headers={"User-Agent": gutenberg_config["user_agent"]},
            timeout=gutenberg_config["timeout"],
        )
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"  ! could not download {gutenberg_id}: {error}")
        return None
    finally:
        # Wait after every request, also after failed ones.
        time.sleep(gutenberg_config["request_delay"])

    response.encoding = "utf-8"
    path.write_text(response.text, encoding="utf-8")
    return response.text


def build_corpus(config):
    """Downloads texts until every epoch is full and writes the corpus.

    Args:
      config (dict): The full configuration.

    Returns:
      pandas.DataFrame: The corpus metadata (one row per kept text, with
        the additional column ``n_words``).
    """
    paths = config["paths"]
    corpus_config = config["corpus"]
    paths["texts_dir"].mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(paths["candidates_file"])
    texts_per_author = Counter()
    texts_per_epoch = Counter()
    kept_rows = []

    for _, candidate in candidates.iterrows():
        if texts_per_author[candidate["author"]] \
                >= corpus_config["max_texts_per_author"]:
            continue
        if texts_per_epoch[candidate["epoch"]] \
                >= corpus_config["max_texts_per_epoch"]:
            continue

        raw_text = fetch_text(candidate["gutenberg_id"], config["gutenberg"],
                              paths["texts_dir"])
        if raw_text is None:
            continue
        n_words = count_words(clean_text(
            strip_gutenberg_boilerplate(raw_text)))
        if n_words < corpus_config["min_words"]:
            continue

        texts_per_author[candidate["author"]] += 1
        texts_per_epoch[candidate["epoch"]] += 1
        kept_rows.append({**candidate.to_dict(), "n_words": n_words})
        print(f"  [{len(kept_rows):>4}] {candidate['epoch']:<11} "
              f"{candidate['author'][:30]:<30} {candidate['title'][:40]}")

    corpus = pd.DataFrame(kept_rows)
    corpus.to_csv(paths["corpus_file"], index=False)
    print(f"Wrote {len(corpus)} texts to {paths['corpus_file']}")
    return corpus
