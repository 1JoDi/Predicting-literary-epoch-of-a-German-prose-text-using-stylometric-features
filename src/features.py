"""The four stylometric features.

All features are computed from a spaCy ``Doc`` of one text sample:

1. ``median_sentence_length`` -- median number of words per sentence.
2. ``sttr`` -- standardised type-token ratio: the mean TTR of consecutive
   windows of equal size.  The raw TTR falls with text length and would
   not be comparable between texts (Covington & McFall 2010).
3. ``function_word_ratio`` -- share of determiners, adpositions and
   conjunctions among all words.
4. ``character_density`` -- number of distinct characters (PER entities,
   after merging name variants) per 1000 words.

"Words" are all tokens that are neither punctuation nor whitespace.
Every feature function only depends on token attributes, so it can be
unit-tested with hand-made ``Doc`` objects (see ``tests/test_features.py``).
"""

import statistics
from collections import Counter

FEATURE_NAMES = [
    "median_sentence_length",
    "sttr",
    "function_word_ratio",
    "character_density",
]


def word_tokens(doc_or_span):
    """Returns all tokens that are neither punctuation nor whitespace.

    Args:
      doc_or_span (spacy.tokens.Doc or spacy.tokens.Span): The text.

    Returns:
      list of spacy.tokens.Token: The word tokens.
    """
    return [token for token in doc_or_span
            if not token.is_punct and not token.is_space]


def median_sentence_length(doc):
    """Computes the median sentence length in words.

    The median is used instead of the mean because a few extremely long
    sentences (or segmentation errors) would distort the mean.

    Args:
      doc (spacy.tokens.Doc): A parsed text.

    Returns:
      float: The median number of words per sentence.
    """
    lengths = [len(word_tokens(sentence)) for sentence in doc.sents]
    lengths = [length for length in lengths if length > 0]
    return float(statistics.median(lengths))


def standardised_ttr(doc, window_size):
    """Computes the standardised type-token ratio (STTR).

    The text is cut into consecutive windows of ``window_size`` words; the
    TTR (distinct lower-cased words / words) is computed for every
    complete window and averaged.  An incomplete last window is ignored.

    Args:
      doc (spacy.tokens.Doc): A parsed text.
      window_size (int): Number of words per window.

    Returns:
      float: The mean TTR over all windows.

    Raises:
      ValueError: If the text is shorter than one window.
    """
    words = [token.lower_ for token in word_tokens(doc)]
    n_windows = len(words) // window_size
    if n_windows == 0:
        raise ValueError(
            f"Text has {len(words)} words, fewer than one window "
            f"({window_size})."
        )

    ratios = []
    for window_index in range(n_windows):
        window = words[window_index * window_size:
                       (window_index + 1) * window_size]
        ratios.append(len(set(window)) / window_size)
    return sum(ratios) / n_windows


def function_word_ratio(doc, function_pos):
    """Computes the share of function words among all words.

    Function words are identified by their universal POS tag (by default
    DET, ADP, CCONJ and SCONJ), not by a word list, so that spelling
    variants of older texts are covered as well.

    Args:
      doc (spacy.tokens.Doc): A parsed text.
      function_pos (list of str): The POS tags that count as function
        words.

    Returns:
      float: Number of function words divided by number of words.
    """
    words = word_tokens(doc)
    n_function_words = sum(1 for token in words
                           if token.pos_ in function_pos)
    return n_function_words / len(words)


def normalise_person_name(name, titles):
    """Normalises a person name for counting distinct characters.

    Lower-cases the name, removes punctuation and drops titles such as
    "Herr" or "Gräfin".

    Args:
      name (str): The text of a PER entity, e.g. "Herr von Instetten".
      titles (list of str): Lower-case words to remove.

    Returns:
      str: The normalised name (possibly empty), e.g. "instetten".
    """
    parts = []
    for part in name.lower().split():
        part = part.strip(".,;:!?\"'()-»«›‹„“”‚‘’")
        if part and part not in titles:
            parts.append(part)
    return " ".join(parts)


def count_characters(name_counts, min_mentions):
    """Counts distinct characters after merging name variants.

    Merging rules (deliberately simple and transparent):

    1. A genitive form is merged with its base form if the base form
       also occurs ("fabians" -> "fabian").
    2. Multi-word names are grouped by their last word (the surname):
       "effi briest" and "frau briest" -> "briest".
    3. A one-word name is added to the group of the first multi-word name
       that contains it ("effi" -> group "briest").
    4. Groups with fewer than ``min_mentions`` mentions are discarded.

    Args:
      name_counts (collections.Counter): Normalised name -> mentions.
      min_mentions (int): Minimum mentions for a character to count.

    Returns:
      int: The number of distinct characters.

    Example:
      >>> count_characters(Counter({"effi briest": 3, "effi": 5,
      ...                           "innstetten": 4, "roswitha": 1}), 2)
      2
    """
    names = Counter()
    for name, count in name_counts.items():
        if name.endswith("s") and name[:-1] in name_counts:
            names[name[:-1]] += count
        elif name:
            names[name] += count

    multi_word_names = sorted(name for name in names if " " in name)
    groups = Counter()
    for name, count in names.items():
        if " " in name:
            groups[name.split()[-1]] += count
            continue
        containing = [multi for multi in multi_word_names
                      if name in multi.split()]
        key = containing[0].split()[-1] if containing else name
        groups[key] += count

    return sum(1 for count in groups.values() if count >= min_mentions)


def character_density(doc, titles, min_mentions):
    """Computes the number of distinct characters per 1000 words.

    Args:
      doc (spacy.tokens.Doc): A parsed text with named entities.
      titles (list of str): Titles to remove from names.
      min_mentions (int): Minimum mentions for a character to count.

    Returns:
      float: Distinct characters per 1000 words.
    """
    name_counts = Counter(
        normalise_person_name(entity.text, titles)
        for entity in doc.ents if entity.label_ == "PER"
    )
    n_characters = count_characters(name_counts, min_mentions)
    return n_characters / len(word_tokens(doc)) * 1000


def extract_features(doc, feature_config):
    """Computes all four features for one text.

    Args:
      doc (spacy.tokens.Doc): A parsed text sample.
      feature_config (dict): The ``features`` section of the config.

    Returns:
      dict: Feature name -> value, in the order of ``FEATURE_NAMES``.
    """
    return {
        "median_sentence_length": median_sentence_length(doc),
        "sttr": standardised_ttr(doc, feature_config["sttr_window"]),
        "function_word_ratio": function_word_ratio(
            doc, feature_config["function_word_pos"]),
        "character_density": character_density(
            doc, feature_config["person_titles"],
            feature_config["min_person_mentions"]),
    }
