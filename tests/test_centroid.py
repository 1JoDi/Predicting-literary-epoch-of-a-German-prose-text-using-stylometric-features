"""Tests for the NumPy nearest-centroid classifier."""

import numpy as np

from src.centroid import NearestCentroidClassifier


def test_predicts_nearest_centroid():
    features = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0],
                         [10.0, 11.0]])
    labels = np.array(["early", "early", "late", "late"])
    model = NearestCentroidClassifier().fit(features, labels)
    predicted = model.predict(np.array([[1.0, 1.0], [9.0, 9.0]]))
    assert list(predicted) == ["early", "late"]


def test_standardisation_prevents_scale_dominance():
    # Feature 1 separates the classes, feature 2 is large-scale noise.
    features = np.array([[0.0, 1000.0], [0.0, 0.0],
                         [1.0, 1000.0], [1.0, 0.0]])
    labels = np.array(["a", "a", "b", "b"])
    model = NearestCentroidClassifier().fit(features, labels)
    assert list(model.predict(np.array([[0.9, 1000.0]]))) == ["b"]


def test_constant_feature_does_not_divide_by_zero():
    features = np.array([[1.0, 5.0], [2.0, 5.0]])
    model = NearestCentroidClassifier().fit(features, ["x", "y"])
    assert np.isfinite(model.distances(features)).all()
