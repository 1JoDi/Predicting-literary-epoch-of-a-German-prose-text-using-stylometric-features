"""Training and evaluation of the epoch classifiers.

The texts are split into a training set and a test set so that no
author appears in both. Four models are compared on the training set
with cross-validation: a majority baseline, the nearest-centroid
classifier, logistic regression and a random forest. The cross-
validation is run twice, once with author-separated folds and once
with ordinary random folds.

Each model is then trained on the full training set and evaluated
once on the test set. The model with the best cross-validated macro-F1
counts as the best model. The file also measures how much each feature
contributes and saves all scores, predictions, misclassified texts and
the best model to the results folder.
"""


import json
import pickle

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_validate)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.centroid import NearestCentroidClassifier
from src.features import FEATURE_NAMES

BASELINE = "majority baseline"


def make_models(seed):
    """Creates the (unfitted) models that are compared.

    Args:
      seed (int): Random seed for reproducible results.

    Returns:
      dict: Model name -> scikit-learn compatible estimator.
    """
    return {
        BASELINE: DummyClassifier(strategy="most_frequent"),
        "nearest centroid": NearestCentroidClassifier(),
        "logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=seed),
    }


def author_disjoint_split(features, n_folds, seed):
    """Splits the texts into training and test set, grouped by author.

    ``StratifiedGroupKFold`` builds ``n_folds`` folds in which no author
    occurs in two folds and the epoch proportions are similar; the first
    fold is used as test set.

    Args:
      features (pandas.DataFrame): The feature table.
      n_folds (int): 1 / n_folds of the data become the test set.
      seed (int): Random seed.

    Returns:
      tuple: ``(train, test)`` as two DataFrames.
    """
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                    random_state=seed)
    train_index, test_index = next(splitter.split(
        features, features["epoch"], groups=features["author"]))
    return features.iloc[train_index], features.iloc[test_index]


def cross_validate_models(models, train, cv_folds, seed, grouped=True):
    """Cross-validates every model on the training set.

    Args:
      models (dict): Model name -> estimator.
      train (pandas.DataFrame): The training texts.
      cv_folds (int): Number of folds.
      seed (int): Random seed.
      grouped (bool): If True, folds are author-disjoint.  If False, an
        ordinary stratified split is used (to measure author leakage).

    Returns:
      pandas.DataFrame: Mean and standard deviation of accuracy and
        macro-F1 per model.
    """
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True,
                                        random_state=seed)
    else:
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                   random_state=seed)

    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, train[FEATURE_NAMES], train["epoch"],
            groups=train["author"] if grouped else None,
            cv=splitter, scoring=["accuracy", "f1_macro"])
        rows.append({
            "model": name,
            "split": "author-disjoint" if grouped else "random (leaky)",
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_sd": scores["test_accuracy"].std(),
            "macro_f1_mean": scores["test_f1_macro"].mean(),
            "macro_f1_sd": scores["test_f1_macro"].std(),
        })
    return pd.DataFrame(rows)


def feature_ablation(train, cv_folds, seed):
    """Measures how much each feature contributes (logistic regression).

    For every feature, the cross-validated macro-F1 is computed (a) with
    this feature alone and (b) with all features except this one.

    Args:
      train (pandas.DataFrame): The training texts.
      cv_folds (int): Number of folds.
      seed (int): Random seed.

    Returns:
      pandas.DataFrame: One row per feature.
    """
    splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True,
                                    random_state=seed)

    def cv_macro_f1(columns):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000))
        scores = cross_validate(model, train[columns], train["epoch"],
                                groups=train["author"], cv=splitter,
                                scoring="f1_macro")
        return scores["test_score"].mean()

    all_features_f1 = cv_macro_f1(FEATURE_NAMES)
    rows = []
    for feature in FEATURE_NAMES:
        others = [name for name in FEATURE_NAMES if name != feature]
        without_f1 = cv_macro_f1(others)
        rows.append({
            "feature": feature,
            "macro_f1_only_this": cv_macro_f1([feature]),
            "macro_f1_without_this": without_f1,
            "drop_when_removed": all_features_f1 - without_f1,
        })
    return pd.DataFrame(rows)


def logistic_coefficients(model, epoch_order):
    """Extracts the standardised coefficients of the logistic regression.

    Because the features are standardised, the coefficients are
    comparable: a positive value means that a higher feature value makes
    the epoch more likely.

    Args:
      model (sklearn.pipeline.Pipeline): The fitted scaler + regression.
      epoch_order (list of str): The epochs in chronological order.

    Returns:
      pandas.DataFrame: Rows = epochs, columns = features.
    """
    regression = model[-1]
    coefficients = pd.DataFrame(regression.coef_, index=regression.classes_,
                                columns=FEATURE_NAMES)
    return coefficients.reindex(epoch_order)


def evaluate_on_test(models, train, test, epoch_order):
    """Fits every model on the training set and evaluates it on the test set.

    Args:
      models (dict): Model name -> estimator (fitted in place).
      train (pandas.DataFrame): The training texts.
      test (pandas.DataFrame): The held-out texts.
      epoch_order (list of str): The epochs in chronological order.

    Returns:
      tuple: ``(scores, predictions, reports, matrices)`` -- a DataFrame
        with accuracy and macro-F1 per model, the test set with one
        prediction column per model, the classification reports (dict of
        str) and the confusion matrices (dict of numpy arrays).
    """
    predictions = test.copy()
    scores, reports, matrices = [], {}, {}
    for name, model in models.items():
        model.fit(train[FEATURE_NAMES], train["epoch"])
        predicted = model.predict(test[FEATURE_NAMES])
        predictions[f"pred_{name}"] = predicted
        scores.append({
            "model": name,
            "accuracy": accuracy_score(test["epoch"], predicted),
            "macro_f1": f1_score(test["epoch"], predicted, average="macro"),
        })
        reports[name] = classification_report(
            test["epoch"], predicted, labels=epoch_order, zero_division=0)
        matrices[name] = confusion_matrix(test["epoch"], predicted,
                                          labels=epoch_order)
    return pd.DataFrame(scores), predictions, reports, matrices


def run_classification(features, config, epoch_order):
    """Runs the complete classification experiment and saves all results.

    Args:
      features (pandas.DataFrame): The feature table.
      config (dict): The full configuration.
      epoch_order (list of str): The epochs in chronological order.

    Returns:
      dict: Summary of the most important numbers (also written to
        ``results/metrics.json``).
    """
    settings = config["classification"]
    seed = settings["seed"]
    results_dir = config["paths"]["results_dir"]

    train, test = author_disjoint_split(features, settings["n_test_folds"],
                                        seed)
    print(f"Training set: {len(train)} texts, {train['author'].nunique()} "
          f"authors; test set: {len(test)} texts, "
          f"{test['author'].nunique()} authors.")
    shared_authors = set(train["author"]) & set(test["author"])
    assert not shared_authors, "author leakage between train and test"

    models = make_models(seed)
    print("Cross-validating on the training set ...")
    cv_results = pd.concat([
        cross_validate_models(models, train, settings["cv_folds"], seed),
        cross_validate_models(models, train, settings["cv_folds"], seed,
                              grouped=False),
    ])
    cv_results.round(4).to_csv(results_dir / "cv_results.csv", index=False)

    grouped_cv = cv_results[(cv_results["split"] == "author-disjoint")
                            & (cv_results["model"] != BASELINE)]
    best_name = grouped_cv.sort_values("macro_f1_mean").iloc[-1]["model"]
    print(f"Best model by cross-validated macro-F1: {best_name}")

    print("Evaluating on the held-out test set ...")
    scores, predictions, reports, matrices = evaluate_on_test(
        models, train, test, epoch_order)
    scores.round(4).to_csv(results_dir / "test_results.csv", index=False)
    predictions.to_csv(results_dir / "test_predictions.csv", index=False)
    with open(results_dir / "classification_reports.txt", "w",
              encoding="utf-8") as report_file:
        for name, report in reports.items():
            report_file.write(f"=== {name} ===\n{report}\n")

    misclassified = predictions[
        predictions["epoch"] != predictions[f"pred_{best_name}"]]
    misclassified.to_csv(results_dir / "misclassified.csv", index=False)

    print("Analysing feature importance ...")
    coefficients = logistic_coefficients(models["logistic regression"],
                                         epoch_order)
    coefficients.round(4).to_csv(results_dir / "logreg_coefficients.csv")
    importance = permutation_importance(
        models[best_name], test[FEATURE_NAMES], test["epoch"],
        scoring="f1_macro", n_repeats=settings["permutation_repeats"],
        random_state=seed)
    importance_table = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance_mean": importance.importances_mean,
        "importance_sd": importance.importances_std,
    })
    importance_table.round(4).to_csv(
        results_dir / "permutation_importance.csv", index=False)
    ablation = feature_ablation(train, settings["cv_folds"], seed)
    ablation.round(4).to_csv(results_dir / "feature_ablation.csv",
                             index=False)

    with open(results_dir / "best_model.pkl", "wb") as model_file:
        pickle.dump(models[best_name], model_file)

    best_scores = scores.set_index("model").loc[best_name]
    summary = {
        "n_texts": len(features),
        "n_authors": int(features["author"].nunique()),
        "n_train": len(train),
        "n_test": len(test),
        "best_model": best_name,
        "test_accuracy": round(float(best_scores["accuracy"]), 4),
        "test_macro_f1": round(float(best_scores["macro_f1"]), 4),
        "baseline_test_accuracy": round(float(
            scores.set_index("model").loc[BASELINE, "accuracy"]), 4),
    }
    with open(results_dir / "metrics.json", "w",
              encoding="utf-8") as metrics_file:
        json.dump(summary, metrics_file, indent=2)

    return {"summary": summary, "cv_results": cv_results, "scores": scores,
            "matrices": matrices, "coefficients": coefficients,
            "importance": importance_table, "ablation": ablation,
            "best_model": best_name}
