"""All figures used in the report.

Colours follow one rule: each epoch always has the same colour in every
figure (blue, orange, green in chronological order), so the reader does
not have to re-learn the legend.  Magnitudes (confusion matrices) use a
single-hue scale from light to dark.
"""

import matplotlib

matplotlib.use("Agg")  # Write files only, no window (works on servers).
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.features import FEATURE_NAMES  # noqa: E402

EPOCH_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SECONDARY_COLOUR = "#52514e"
FEATURE_LABELS = {
    "median_sentence_length": "Median sentence length (words)",
    "sttr": "Standardised type-token ratio",
    "function_word_ratio": "Function word ratio",
    "character_density": "Characters per 1000 words",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 9,
})


def _save(figure, path):
    """Saves a figure tightly cropped and closes it."""
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_epoch_distribution(table, path):
    """Bar chart of texts and distinct authors per epoch.

    Args:
      table (pandas.DataFrame): Output of ``epochs.distribution_table``.
      path (pathlib.Path): Output file.
    """
    figure, axis = plt.subplots(figsize=(5, 3))
    positions = np.arange(len(table))
    width = 0.38
    for offset, column, colour in [(-width / 2, "texts", EPOCH_COLOURS[0]),
                                   (width / 2, "authors", SECONDARY_COLOUR)]:
        bars = axis.bar(positions + offset, table[column], width * 0.94,
                        color=colour, label=column)
        axis.bar_label(bars, padding=2, fontsize=8)
    axis.set_xticks(positions, table.index)
    axis.set_ylabel("count")
    axis.grid(axis="x", visible=False)
    axis.legend(frameon=False)
    axis.set_title("Corpus: texts and authors per epoch", loc="left")
    _save(figure, path)


def plot_feature_boxplots(features, epoch_order, path):
    """Box plots of every feature, one box per epoch.

    Args:
      features (pandas.DataFrame): The feature table.
      epoch_order (list of str): The epochs in chronological order.
      path (pathlib.Path): Output file.
    """
    figure, axes = plt.subplots(2, 2, figsize=(8, 6))
    rng = np.random.default_rng(0)
    for axis, feature in zip(axes.flat, FEATURE_NAMES):
        values = [features.loc[features["epoch"] == epoch, feature]
                  for epoch in epoch_order]
        boxes = axis.boxplot(values, patch_artist=True, widths=0.55,
                             showfliers=False,
                             medianprops={"color": "#0b0b0b"})
        for box, colour in zip(boxes["boxes"], EPOCH_COLOURS):
            box.set_facecolor(colour)
            box.set_alpha(0.35)
            box.set_edgecolor(colour)
        # Individual texts as jittered dots on top of the boxes.
        for position, (epoch_values, colour) in enumerate(
                zip(values, EPOCH_COLOURS), start=1):
            jitter = rng.uniform(-0.15, 0.15, len(epoch_values))
            axis.scatter(position + jitter, epoch_values, s=8,
                         color=colour, alpha=0.7, linewidths=0)
        axis.set_xticks(range(1, len(epoch_order) + 1), epoch_order)
        axis.set_title(FEATURE_LABELS[feature], loc="left")
        axis.grid(axis="x", visible=False)
    figure.tight_layout()
    _save(figure, path)


def plot_confusion_matrix(matrix, epoch_order, title, path):
    """Heat map of a confusion matrix with counts and row percentages.

    Args:
      matrix (numpy.ndarray): Rows = true epoch, columns = predicted.
      epoch_order (list of str): The epochs in chronological order.
      title (str): Figure title.
      path (pathlib.Path): Output file.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, row_sums, out=np.zeros(matrix.shape),
                       where=row_sums > 0)

    figure, axis = plt.subplots(figsize=(4.2, 3.6))
    axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    axis.grid(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ink = "white" if shares[row, column] > 0.55 else "#0b0b0b"
            axis.text(column, row,
                      f"{matrix[row, column]}\n"
                      f"({shares[row, column]:.0%})",
                      ha="center", va="center", color=ink, fontsize=8)
    axis.set_xticks(range(len(epoch_order)), epoch_order)
    axis.set_yticks(range(len(epoch_order)), epoch_order)
    axis.set_xlabel("predicted epoch")
    axis.set_ylabel("true epoch")
    axis.set_title(title, loc="left")
    _save(figure, path)


def plot_cv_comparison(cv_results, path):
    """Compares cross-validated macro-F1 of author-disjoint vs. random splits.

    Args:
      cv_results (pandas.DataFrame): Output of
        ``classify.cross_validate_models`` for both split types.
      path (pathlib.Path): Output file.
    """
    models = list(dict.fromkeys(cv_results["model"]))
    positions = np.arange(len(models))
    width = 0.38
    figure, axis = plt.subplots(figsize=(6, 3.2))
    for offset, split, colour in [
            (-width / 2, "author-disjoint", EPOCH_COLOURS[0]),
            (width / 2, "random (leaky)", SECONDARY_COLOUR)]:
        rows = cv_results[cv_results["split"] == split].set_index("model")
        rows = rows.loc[models]
        bars = axis.bar(positions + offset, rows["macro_f1_mean"],
                        width * 0.94, yerr=rows["macro_f1_sd"], capsize=2,
                        color=colour, label=split,
                        error_kw={"ecolor": "#8a8984", "lw": 0.8})
        axis.bar_label(bars, fmt="%.2f", label_type="center",
                       color="white", fontsize=7)
    axis.set_xticks(positions, models)
    axis.set_ylabel("macro-F1 (5-fold CV)")
    axis.set_ylim(0, 1)
    axis.grid(axis="x", visible=False)
    axis.legend(frameon=False, loc="upper left")
    axis.set_title("Cross-validation: author-disjoint vs. random folds",
                   loc="left")
    _save(figure, path)


def plot_feature_importance(coefficients, importance, epoch_order, path):
    """Logistic regression coefficients and permutation importance.

    Args:
      coefficients (pandas.DataFrame): Rows = epochs, columns = features.
      importance (pandas.DataFrame): Output of the permutation importance
        with the columns ``feature``, ``importance_mean``, ``importance_sd``.
      epoch_order (list of str): The epochs in chronological order.
      path (pathlib.Path): Output file.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
    labels = [FEATURE_LABELS[name] for name in FEATURE_NAMES]
    positions = np.arange(len(FEATURE_NAMES))

    height = 0.8 / len(epoch_order)
    for index, (epoch, colour) in enumerate(zip(epoch_order,
                                                EPOCH_COLOURS)):
        offset = (index - (len(epoch_order) - 1) / 2) * height
        left.barh(positions + offset, coefficients.loc[epoch],
                  height * 0.92, color=colour, label=epoch)
    left.axvline(0, color="#8a8984", linewidth=0.8)
    left.set_yticks(positions, labels)
    left.invert_yaxis()
    left.set_xlabel("standardised coefficient")
    left.grid(axis="y", visible=False)
    left.legend(frameon=False, fontsize=8)
    left.set_title("Logistic regression coefficients", loc="left")

    right.barh(positions, importance["importance_mean"], 0.55,
               xerr=importance["importance_sd"], color=SECONDARY_COLOUR,
               error_kw={"ecolor": "#8a8984", "lw": 0.8})
    right.axvline(0, color="#8a8984", linewidth=0.8)
    right.set_yticks(positions, [""] * len(labels))
    right.invert_yaxis()
    right.set_xlabel("drop in macro-F1 when shuffled")
    right.grid(axis="y", visible=False)
    right.set_title("Permutation importance (test set)", loc="left")
    figure.tight_layout()
    _save(figure, path)
