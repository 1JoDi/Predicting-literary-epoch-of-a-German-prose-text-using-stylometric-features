"""Cleaning of raw Gutenberg texts and extraction of fixed-size samples.

Every Gutenberg file starts with a license header and ends with a license
footer, which must be removed before any analysis.  Inside the text there
are transcription artefacts (``[Illustration]``, footnote markers,
``_emphasis_``) and hard line breaks every ~70 characters.

To make the texts comparable, only a contiguous passage of a fixed number
of words is analysed later on.  This (a) removes text length as a
confounding variable, (b) keeps the spaCy processing time manageable and
(c) avoids the front matter by skipping the beginning of the book.
"""

import re

START_PATTERN = re.compile(
    r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK[^\n]*",
    re.IGNORECASE,
)
END_PATTERN = re.compile(
    r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK",
    re.IGNORECASE,
)
# Editorial notes in square brackets, e.g. "[Illustration: ...]" or
# "[Fußnote 3: ...]".  Limited length so that a stray "[" cannot delete
# half of the book.
BRACKET_NOTE_PATTERN = re.compile(r"\[[^\[\]]{0,1000}\]")
# Emphasis markup used by Gutenberg transcribers: _italic_, =spaced=,
# and runs of separator characters such as "* * *" or "-----".
MARKUP_PATTERN = re.compile(r"[_=]")
SEPARATOR_LINE_PATTERN = re.compile(r"^[\s*\-.=_~]+$", re.MULTILINE)


def strip_gutenberg_boilerplate(raw_text):
    """Removes the Project Gutenberg license header and footer.

    Args:
      raw_text (str): The complete content of a Gutenberg ``.txt`` file.

    Returns:
      str: The text between the START and END markers.  If a marker is
        missing, the text is kept from the beginning or to the end.
    """
    start_match = START_PATTERN.search(raw_text)
    end_match = END_PATTERN.search(raw_text)
    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(raw_text)
    return raw_text[start:end]


def clean_text(text):
    """Removes transcription artefacts and joins hard-wrapped lines.

    Paragraphs (blocks separated by an empty line) are kept and separated
    by exactly one empty line; line breaks inside a paragraph become
    spaces.

    Args:
      text (str): The text without the Gutenberg boilerplate.

    Returns:
      str: The cleaned text.

    Example:
      >>> clean_text("Es war\\neinmal _ein_ Mann.\\n\\n[Illustration]")
      'Es war einmal ein Mann.'
    """
    text = text.replace("\r\n", "\n")
    text = BRACKET_NOTE_PATTERN.sub(" ", text)
    text = SEPARATOR_LINE_PATTERN.sub("", text)
    text = MARKUP_PATTERN.sub("", text)

    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [" ".join(paragraph.split()) for paragraph in paragraphs]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def count_words(text):
    """Counts whitespace-separated words.

    Args:
      text (str): Any text.

    Returns:
      int: The number of words.
    """
    return len(text.split())


def extract_sample(text, skip_fraction, sample_words):
    """Extracts a contiguous passage with exactly ``sample_words`` words.

    The passage starts at the first paragraph after ``skip_fraction`` of
    all words, so that title pages, prefaces and tables of contents are
    skipped.  Paragraph boundaries are kept because they help the sentence
    segmenter.

    Args:
      text (str): A cleaned text (paragraphs separated by empty lines).
      skip_fraction (float): Fraction of words to skip at the beginning.
      sample_words (int): Number of words in the sample.

    Returns:
      str or None: The sample, or None if the text is too short.
    """
    paragraphs = text.split("\n\n")
    total_words = count_words(text)
    words_to_skip = int(total_words * skip_fraction)

    skipped = 0
    sample = []
    sample_size = 0
    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        if skipped < words_to_skip:
            skipped += len(paragraph_words)
            continue
        missing = sample_words - sample_size
        sample.append(" ".join(paragraph_words[:missing]))
        sample_size += min(len(paragraph_words), missing)
        if sample_size >= sample_words:
            return "\n\n".join(sample)
    return None
