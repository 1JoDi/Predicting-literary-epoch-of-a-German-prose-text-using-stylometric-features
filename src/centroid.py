"""A nearest-centroid ("stylistic fingerprint") classifier in NumPy.

This implements the idea of the course's stylometry project: every epoch
gets a fingerprint -- the mean feature vector of its training texts -- and
a new text is assigned to the epoch with the closest fingerprint.  The
features have very different scales (a sentence length of ~15 vs. a ratio
of ~0.3), so they are z-standardised with the training mean and standard
deviation before distances are computed.

The class inherits from scikit-learn's ``BaseEstimator`` and
``ClassifierMixin`` only for the common interface (``fit``/``predict``/
``score``); this lets it be cross-validated exactly like the library
models.  The actual computation uses nothing but NumPy.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class NearestCentroidClassifier(ClassifierMixin, BaseEstimator):
    """Assigns each sample to the class with the nearest mean vector.

    Attributes (after ``fit``):
      classes_ (numpy.ndarray): The class labels, sorted.
      centroids_ (numpy.ndarray): One standardised mean vector per class,
        shape (n_classes, n_features).
      mean_ (numpy.ndarray): Training mean of every feature.
      scale_ (numpy.ndarray): Training standard deviation of every feature.
    """

    def fit(self, features, labels):
        """Computes the standardisation and one centroid per class.

        Args:
          features (array-like): Shape (n_samples, n_features).
          labels (array-like): Shape (n_samples,).

        Returns:
          NearestCentroidClassifier: The fitted classifier (self).
        """
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels)

        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        # A constant feature would lead to a division by zero.
        self.scale_[self.scale_ == 0] = 1.0

        standardised = (features - self.mean_) / self.scale_
        self.classes_ = np.unique(labels)
        self.centroids_ = np.array([
            standardised[labels == label].mean(axis=0)
            for label in self.classes_
        ])
        return self

    def distances(self, features):
        """Computes the Euclidean distance of every sample to every centroid.

        Args:
          features (array-like): Shape (n_samples, n_features).

        Returns:
          numpy.ndarray: Shape (n_samples, n_classes).
        """
        standardised = (np.asarray(features, dtype=float) - self.mean_) \
            / self.scale_
        # Broadcasting: (n_samples, 1, n_features) - (n_classes, n_features)
        differences = standardised[:, np.newaxis, :] - self.centroids_
        return np.sqrt((differences ** 2).sum(axis=2))

    def predict(self, features):
        """Predicts the class with the nearest centroid for every sample.

        Args:
          features (array-like): Shape (n_samples, n_features).

        Returns:
          numpy.ndarray: The predicted labels, shape (n_samples,).
        """
        nearest = self.distances(features).argmin(axis=1)
        return self.classes_[nearest]
