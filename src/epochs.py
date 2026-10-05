"""Mapping from author death years to literary epochs.

Project Gutenberg does not record when a text was first published (the
catalog field ``Issued`` is the date of the digitisation).  The author's
death year is therefore used as a proxy: a text is assigned to the first
epoch whose upper boundary is not earlier than the death year.  The
boundaries are defined in ``config.yaml``.
"""


def epoch_names(epoch_config):
    """Returns the epoch names in chronological order.

    Args:
      epoch_config (dict): The ``epochs`` section of the configuration.

    Returns:
      list of str: The epoch names, oldest first.
    """
    return [epoch_bin["name"] for epoch_bin in epoch_config["bins"]]


def death_year_to_epoch(death_year, epoch_config):
    """Assigns an epoch label to an author death year.

    Args:
      death_year (int or None): The death year of the author.
      epoch_config (dict): The ``epochs`` section of the configuration,
        with the keys ``min_death_year`` and ``bins`` (a list of dicts
        with ``name`` and ``max_death_year``, sorted chronologically).

    Returns:
      str or None: The epoch name, or None if the year is missing or lies
        outside all epochs.

    Example:
      >>> death_year_to_epoch(1832, {"min_death_year": 1720, "bins": [
      ...     {"name": "Goethezeit", "max_death_year": 1860}]})
      'Goethezeit'
    """
    if death_year is None or death_year < epoch_config["min_death_year"]:
        return None
    for epoch_bin in epoch_config["bins"]:
        if death_year <= epoch_bin["max_death_year"]:
            return epoch_bin["name"]
    return None


def distribution_table(corpus, epoch_order):
    """Counts texts and distinct authors per epoch.

    Args:
      corpus (pandas.DataFrame): Table with the columns ``epoch`` and
        ``author``.
      epoch_order (list of str): The epochs in chronological order.

    Returns:
      pandas.DataFrame: One row per epoch with the columns ``texts`` and
        ``authors``.
    """
    table = corpus.groupby("epoch").agg(
        texts=("author", "size"),
        authors=("author", "nunique"),
    )
    return table.reindex(epoch_order, fill_value=0)
