"""Small NumPy k-means implementation for the video's sampled HSV pixels."""

from __future__ import annotations

import numpy as np


class KMeans:
    def __init__(self, n_clusters: int, random_state: int = 7, n_init: int = 10):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.cluster_centers_: np.ndarray | None = None

    def fit_predict(self, pixels: np.ndarray) -> np.ndarray:
        data = np.asarray(pixels, dtype=np.float64)
        if data.ndim != 2 or len(data) < self.n_clusters:
            raise ValueError("Need at least one pixel per cluster")
        rng = np.random.default_rng(self.random_state)
        best_labels = None
        best_centers = None
        best_error = float("inf")
        for _ in range(max(1, self.n_init)):
            centers = data[rng.choice(len(data), self.n_clusters, replace=False)].copy()
            for _ in range(50):
                distance = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                labels = np.argmin(distance, axis=1)
                updated = centers.copy()
                for index in range(self.n_clusters):
                    members = data[labels == index]
                    updated[index] = members.mean(axis=0) if len(members) else data[rng.integers(len(data))]
                if np.allclose(updated, centers, atol=1e-5):
                    centers = updated
                    break
                centers = updated
            distance = np.sum((data - centers[labels]) ** 2, axis=1)
            error = float(np.sum(distance))
            if error < best_error:
                best_error, best_labels, best_centers = error, labels.copy(), centers.copy()
        self.cluster_centers_ = best_centers
        assert best_labels is not None
        return best_labels
