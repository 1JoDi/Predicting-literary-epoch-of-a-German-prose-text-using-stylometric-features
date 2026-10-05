"""Entry point of the pipeline: stylometric epoch classification.

Usage (from the project root):

    python main.py                   # run all steps
    python main.py --step catalog    # only one step
    python main.py --config my.yaml  # use another configuration file

Steps (each step reads the output files of the previous one, so they can
be run separately):

    catalog   download + filter the Gutenberg catalog -> candidates.csv
    download  download texts, build the corpus -> corpus_metadata.csv
    features  spaCy + the four features -> features.csv
    analyse   descriptive statistics, classification, figures -> results/
"""

import argparse
import json

import pandas as pd

from src import catalog, classify, dataset, describe, download, plots
from src.config import load_config
from src.epochs import distribution_table, epoch_names

STEPS = ["catalog", "download", "features", "analyse"]


def parse_arguments():
    """Reads the command-line arguments.

    Returns:
      argparse.Namespace: With the attributes ``config`` and ``step``.
    """
    parser = argparse.ArgumentParser(
        description="Classify German prose texts by literary epoch "
                    "from four stylometric features.")
    parser.add_argument("--config", default="config.yaml",
                        help="path to the YAML configuration file "
                             "(default: config.yaml)")
    parser.add_argument("--step", choices=STEPS + ["all"], default="all",
                        help="pipeline step to run (default: all)")
    return parser.parse_args()


def run_analysis(config):
    """Runs descriptive statistics, classification and all figures.

    Args:
      config (dict): The full configuration.

    Returns:
      None
    """
    results_dir = config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    epoch_order = epoch_names(config["epochs"])
    features = pd.read_csv(config["paths"]["features_file"])

    print("Corpus distribution:")
    distribution = distribution_table(features, epoch_order)
    print(distribution.to_string())
    distribution.to_csv(results_dir / "epoch_distribution.csv")
    plots.plot_epoch_distribution(distribution,
                                  results_dir / "epoch_distribution.png")

    print("Descriptive statistics:")
    summary = describe.summary_by_epoch(features, epoch_order)
    summary.to_csv(results_dir / "feature_summary.csv")
    tests = describe.kruskal_tests(features, epoch_order)
    pd.DataFrame(tests).to_csv(results_dir / "kruskal_tests.csv",
                               index=False)
    for test in tests:
        print(f"  {test['feature']:<24} H = {test['H']:>8}  "
              f"p = {test['p_value']:.2e}")
    plots.plot_feature_boxplots(features, epoch_order,
                                results_dir / "feature_boxplots.png")

    results = classify.run_classification(features, config, epoch_order)
    plots.plot_cv_comparison(results["cv_results"],
                             results_dir / "cv_comparison.png")
    best = results["best_model"]
    plots.plot_confusion_matrix(
        results["matrices"][best], epoch_order,
        f"Test set: {best}", results_dir / "confusion_matrix.png")
    plots.plot_feature_importance(
        results["coefficients"], results["importance"], epoch_order,
        results_dir / "feature_importance.png")

    print("Test set results:")
    print(results["scores"].round(3).to_string(index=False))
    print(json.dumps(results["summary"], indent=2))
    print(f"All results were written to {results_dir}")
    return None


def main():
    """Runs the requested pipeline step(s)."""
    arguments = parse_arguments()
    config = load_config(arguments.config)
    steps = STEPS if arguments.step == "all" else [arguments.step]

    for step in steps:
        print(f"\n===== step: {step} =====")
        if step == "catalog":
            catalog.build_candidate_list(config)
        elif step == "download":
            download.build_corpus(config)
        elif step == "features":
            dataset.build_feature_table(config)
        elif step == "analyse":
            run_analysis(config)


# The guard is required: spaCy's multiprocessing (n_process > 1) starts
# new Python processes that import this file, especially on Windows.
if __name__ == "__main__":
    main()
