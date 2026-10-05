"""Download and filtering of the Project Gutenberg catalog.

This is the data access script of the project.  It downloads the official
catalog file (``pg_catalog.csv``, one row per ebook), keeps German
narrative prose by authors with a known death year and writes the
resulting list of candidate texts to ``data/processed/candidates.csv``.

Filtering steps (each one is logged with the number of remaining books):

1. ``Type`` is "Text" and ``Language`` is exactly "de".
2. The Library of Congress class (``LoCC``) starts with "PT" (German
   literature).  This removes translations and non-literary books.
3. No book with a translator (translations are not the author's style).
4. None of the exclusion keywords (drama, poetry, letters, ...) occurs in
   the title, subjects or bookshelves.
5. The first author has a known death year that falls into one of the
   predetermined epochs.
6. Duplicate editions (same author and title) are removed.
"""

import re

import pandas as pd
import requests

from src.epochs import death_year_to_epoch

# "Goethe, Johann Wolfgang von, 1749-1832" --> birth 1749, death 1832.
# Birth or death year may be missing ("-1832", "1749-") or uncertain
# ("1797?-1856"). Everything before the last comma is considered part
# of the name.
LIFESPAN_PATTERN = re.compile(
    r",\s*(?P<birth>\d{1,4})?\??\s*-\s*(?P<death>\d{1,4})?\??\s*$"
)
# Role annotations such as "[Translator]" or "[Illustrator]".
ROLE_PATTERN = re.compile(r"\[(?P<role>[^\]]+)\]")
# Columns of pg_catalog.csv that the filter relies on.
REQUIRED_COLUMNS = ["Text#", "Type", "Title", "Language", "Authors",
                    "Subjects", "LoCC", "Bookshelves"]
CANDIDATE_COLUMNS = ["gutenberg_id", "title", "author", "birth_year",
                     "death_year", "epoch"]


def download_catalog(url, target_path, user_agent, timeout):
    """Downloads the Gutenberg catalog unless it is already on disk.

    Args:
      url (str): URL of ``pg_catalog.csv``.
      target_path (pathlib.Path): Where the file is stored.
      user_agent (str): User-Agent header sent with the request.
      timeout (float): Request timeout in seconds.

    Returns:
      pathlib.Path: The path of the catalog file.

    Raises:
      requests.HTTPError: If the server answers with an error status.
    """
    if target_path.exists():
        print(f"Catalog already downloaded: {target_path}")
        return target_path

    print(f"Downloading catalog from {url} ...")
    response = requests.get(
        url, headers={"User-Agent": user_agent}, timeout=timeout
    )
    response.raise_for_status()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(response.content)
    return target_path


def parse_author(author_entry):
    """Splits one author entry of the catalog into name and life dates.

    Args:
      author_entry (str): One author as written in the ``Authors``
        column, e.g. ``"Fontane, Theodor, 1819-1898"``.

    Returns:
      dict: With the keys ``name`` (str), ``role`` (str or None),
        ``birth_year`` and ``death_year`` (int or None).

    Example:
      >>> parse_author("Fontane, Theodor, 1819-1898")["death_year"]
      1898
    """
    role_match = ROLE_PATTERN.search(author_entry)
    role = role_match.group("role") if role_match else None
    entry = ROLE_PATTERN.sub("", author_entry).strip()

    lifespan_match = LIFESPAN_PATTERN.search(entry)
    if lifespan_match is None:
        return {"name": entry, "role": role,
                "birth_year": None, "death_year": None}

    name = entry[:lifespan_match.start()].strip()
    birth = lifespan_match.group("birth")
    death = lifespan_match.group("death")
    return {
        "name": name,
        "role": role,
        "birth_year": int(birth) if birth else None,
        "death_year": int(death) if death else None,
    }


def split_authors(authors_field):
    """Parses the complete ``Authors`` field of a catalog row.

    Args:
      authors_field (str): Semicolon-separated author entries.

    Returns:
      list of dict: One parsed entry (see ``parse_author``) per person.
    """
    return [
        parse_author(entry)
        for entry in authors_field.split(";")
        if entry.strip()
    ]


def contains_keyword(text, keywords):
    """Checks whether a word in the text starts with one of the keywords.

    Matching is case-insensitive and anchored at a word start, so
    "gedicht" matches "Gedichte" but "reise" does not match "Preise".

    Args:
      text (str): The text to search in.
      keywords (list of str): Lower-case keywords.

    Returns:
      bool: True if at least one keyword occurs.
    """
    text = text.lower()
    return any(re.search(r"\b" + re.escape(keyword), text)
               for keyword in keywords)


def normalise_title(title):
    """Returns a simplified title used to detect duplicate editions.

    Args:
      title (str): The title as given in the catalog.

    Returns:
      str: Lower-case title without punctuation and extra whitespace.
    """
    title = re.sub(r"[^\w\s]", " ", title.lower())
    return " ".join(title.split())


def _log_step(description, table, log):
    """Prints and records how many books are left after a filtering step."""
    print(f"  {description:<45} {len(table):>6} books")
    log.append({"step": description, "books": len(table)})


def filter_catalog(catalog, corpus_config, epoch_config, log=None):
    """Selects German prose texts with a known author death year.

    Args:
      catalog (pandas.DataFrame): The raw Gutenberg catalog.
      corpus_config (dict): The ``corpus`` section of the configuration.
      epoch_config (dict): The ``epochs`` section of the configuration.
      log (list, optional): If given, one dict per filtering step with
        the number of remaining books is appended to it.

    Returns:
      pandas.DataFrame: One row per candidate text with the columns
        ``gutenberg_id``, ``title``, ``author``, ``birth_year``,
        ``death_year`` and ``epoch``, in random (but reproducible) order.
    """
    if log is None:
        log = []
    catalog = catalog.fillna("")
    _log_step("full catalog", catalog, log)

    books = catalog[(catalog["Type"] == "Text")
                    & (catalog["Language"] == corpus_config["language"])]
    _log_step("German texts", books, log)

    books = books[books["LoCC"].str.contains(
        corpus_config["locc_prefix"], regex=False)]
    _log_step("German literature (LoCC PT)", books, log)

    books = books[~books["Authors"].str.contains("Translator", regex=False)]
    _log_step("without translator", books, log)

    searchable = books["Title"] + " " + books["Subjects"] + " " \
        + books["Bookshelves"]
    is_excluded = searchable.apply(
        contains_keyword, keywords=corpus_config["exclude_keywords"])
    books = books[~is_excluded]
    _log_step("without drama/poetry/non-fiction keywords", books, log)

    rows = []
    for _, book in books.iterrows():
        authors = split_authors(book["Authors"])
        if not authors:
            continue
        # The first listed person is the author of the text.
        author = authors[0]
        epoch = death_year_to_epoch(author["death_year"], epoch_config)
        if epoch is None:
            continue
        rows.append({
            "gutenberg_id": int(book["Text#"]),
            "title": " ".join(book["Title"].split()),
            "author": author["name"],
            "birth_year": author["birth_year"],
            "death_year": author["death_year"],
            "epoch": epoch,
        })
    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    _log_step("author death year inside an epoch", candidates, log)

    candidates["title_key"] = candidates["title"].apply(normalise_title)
    candidates = candidates.drop_duplicates(subset=["author", "title_key"])
    candidates = candidates.drop(columns="title_key")
    _log_step("without duplicate editions", candidates, log)

    # Shuffle once with a fixed seed: the download step walks through
    # this list in order, so the selection is random but reproducible.
    return candidates.sample(frac=1, random_state=corpus_config["seed"])


def build_candidate_list(config):
    """Runs the complete catalog step and writes ``candidates.csv``.

    Args:
      config (dict): The full configuration.

    Returns:
      pandas.DataFrame: The candidate texts.
    """
    paths = config["paths"]
    gutenberg = config["gutenberg"]
    download_catalog(gutenberg["catalog_url"], paths["catalog_file"],
                     gutenberg["user_agent"], gutenberg["timeout"])

    catalog = pd.read_csv(paths["catalog_file"], dtype=str,
                          keep_default_na=False)
    missing = [name for name in REQUIRED_COLUMNS
               if name not in catalog.columns]
    if missing:
        raise KeyError(f"The catalog file lacks the columns {missing}; "
                       f"delete {paths['catalog_file']} and download it "
                       f"again.")
    print("Filtering the catalog:")
    filter_log = []
    candidates = filter_catalog(catalog, config["corpus"], config["epochs"],
                                filter_log)

    paths["candidates_file"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filter_log).to_csv(
        paths["candidates_file"].parent / "filter_counts.csv", index=False)
    candidates.to_csv(paths["candidates_file"], index=False)
    print(f"Wrote {len(candidates)} candidates to "
          f"{paths['candidates_file']}")
    return candidates
