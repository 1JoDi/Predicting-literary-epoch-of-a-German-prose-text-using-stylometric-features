"""Descriptive statistics of the features per epoch.

Before any classifier is trained, this step answers the simpler question
whether the feature distributions differ between the epochs at all.  For
every feature it reports mean, standard deviation and median per epoch
and a Kruskal-Wallis H-test (a rank-based test that does not assume
normally distributed values).
"""

from scipy.stats import kruskal

from src.features import FEATURE_NAMES


def summary_by_epoch(features, epoch_order):
    """Computes mean, standard deviation and median per epoch.

    Args:
      features (pandas.DataFrame): The feature table.
      epoch_order (list of str): The epochs in chronological order.

    Returns:
      pandas.DataFrame: Rows = epochs, columns = (feature, statistic).
    """
    summary = features.groupby("epoch")[FEATURE_NAMES].agg(
        ["mean", "std", "median"])
    return summary.reindex(epoch_order).round(4)


def kruskal_tests(features, epoch_order):
    """Tests for every feature whether its distribution differs by epoch.

    Args:
      features (pandas.DataFrame): The feature table.
      epoch_order (list of str): The epochs in chronological order.

    Returns:
      list of dict: One entry per feature with ``feature``, ``H`` and
        ``p_value``.
    """
    results = []
    for feature in FEATURE_NAMES:
        groups = [features.loc[features["epoch"] == epoch, feature]
                  for epoch in epoch_order]
        statistic, p_value = kruskal(*groups)
        results.append({"feature": feature, "H": round(statistic, 3),
                        "p_value": p_value})
    return results
