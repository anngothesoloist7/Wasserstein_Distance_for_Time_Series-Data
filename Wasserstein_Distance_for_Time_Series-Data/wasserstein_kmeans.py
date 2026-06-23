"""
Wasserstein k-means algorithm for market regime clustering.

Based on: "Clustering Market Regimes Using the Wasserstein Distance"
by B. Horvath, Z. Issa, and A. Muguruza (2021)

This module implements:
1. Wasserstein distance computation for empirical measures
2. Wasserstein barycenter computation
3. WK-means clustering algorithm
4. MK-means (moment-based) benchmark algorithm
5. MMD validation metrics
"""

import os
import sys

import numpy as np
from scipy import stats
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import warnings

# Make the repository-root shared package importable regardless of the working
# directory the scripts are launched from (e.g. ``python main_synthetic.py`` run
# inside this project folder).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Shared, reusable primitives (identical implementations, factored out so both
# this project and the TiOT project share one source of truth).
from ot_common import (
    wasserstein_distance_1d,
    wasserstein_barycenter_1d,
    gaussian_kernel,
    compute_mmd_biased,
    compute_mmd_fast,
)


@dataclass
class ClusteringResult:
    """Container for clustering results."""
    labels: np.ndarray  # Cluster assignments for each distribution
    centroids: List[np.ndarray]  # Centroid distributions (sorted atoms)
    n_iterations: int
    loss_history: List[float]


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute log returns from price series.

    r_i = log(S_{i+1}) - log(S_i)

    Args:
        prices: Array of prices

    Returns:
        Array of log returns (length = len(prices) - 1)
    """
    return np.diff(np.log(prices))


def create_sliding_windows(returns: np.ndarray, h1: int, h2: int) -> List[np.ndarray]:
    """
    Apply stream lift to create overlapping windows of returns.

    Definition 1.2 from paper:
    ℓ^i(x) = (x_{1+h2(i-1)}, ..., x_{1+h1+h2(i-1)}) for i = 1, ..., M

    Args:
        returns: Array of log returns
        h1: Window length (number of returns per segment)
        h2: Sliding window offset (overlap = h1 - h2)

    Returns:
        List of return windows (each is an empirical distribution)
    """
    if h2 >= h1:
        raise ValueError("h2 must be less than h1")

    windows = []
    i = 0
    while i + h1 <= len(returns):
        window = returns[i:i + h1]
        windows.append(window)
        i += h2 if h2 > 0 else h1

    return windows


class WassersteinKMeans:
    """
    Wasserstein k-means (WK-means) clustering algorithm.

    Clusters empirical distributions using the p-Wasserstein distance
    and Wasserstein barycenter for centroid updates.

    Reference: Definition 2.7 and Algorithm 1 in paper.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        p: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_init: int = 10,
        random_state: Optional[int] = None
    ):
        """
        Initialize WK-means.

        Args:
            n_clusters: Number of clusters (k)
            p: Order of Wasserstein distance
            max_iter: Maximum number of iterations
            tol: Convergence tolerance for loss function
            n_init: Number of initializations to try
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.centroids_: Optional[List[np.ndarray]] = None
        self.labels_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.inertia_: float = np.inf

    def _init_centroids(
        self,
        distributions: List[np.ndarray],
        rng: np.random.Generator
    ) -> List[np.ndarray]:
        """Initialize centroids by random sampling from distributions."""
        indices = rng.choice(len(distributions), size=self.n_clusters, replace=False)
        return [np.sort(distributions[i].copy()) for i in indices]

    def _assign_clusters(
        self,
        distributions: List[np.ndarray],
        centroids: List[np.ndarray]
    ) -> np.ndarray:
        """
        Assign each distribution to nearest centroid.

        C_l^n := {μ_i ∈ K : argmin_{j=1,...,k} W_p(μ_i, μ̄_j^{n-1}) = l}
        """
        labels = np.zeros(len(distributions), dtype=int)

        for i, dist in enumerate(distributions):
            distances = [
                wasserstein_distance_1d(dist, centroid, self.p)
                for centroid in centroids
            ]
            labels[i] = np.argmin(distances)

        return labels

    def _update_centroids(
        self,
        distributions: List[np.ndarray],
        labels: np.ndarray
    ) -> List[np.ndarray]:
        """Update centroids as Wasserstein barycenters of assigned distributions."""
        new_centroids = []

        for k in range(self.n_clusters):
            cluster_dists = [distributions[i] for i in range(len(distributions)) if labels[i] == k]

            if len(cluster_dists) == 0:
                # Keep old centroid if cluster is empty
                new_centroids.append(self.centroids_[k] if self.centroids_ else np.zeros(1))
            else:
                barycenter = wasserstein_barycenter_1d(cluster_dists, self.p)
                new_centroids.append(barycenter)

        return new_centroids

    def _compute_loss(
        self,
        old_centroids: List[np.ndarray],
        new_centroids: List[np.ndarray]
    ) -> float:
        """
        Compute loss function.

        l(μ̄^{n-1}, μ̄^n) = Σ_{i=1}^{k} W_p(μ̄_i^{n-1}, μ̄_i^n)
        """
        return sum(
            wasserstein_distance_1d(old, new, self.p)
            for old, new in zip(old_centroids, new_centroids)
        )

    def _compute_inertia(
        self,
        distributions: List[np.ndarray],
        labels: np.ndarray,
        centroids: List[np.ndarray]
    ) -> float:
        """Compute total within-cluster Wasserstein distance."""
        inertia = 0
        for i, dist in enumerate(distributions):
            inertia += wasserstein_distance_1d(dist, centroids[labels[i]], self.p) ** self.p
        return inertia

    def fit(self, distributions: List[np.ndarray]) -> 'WassersteinKMeans':
        """
        Fit WK-means to a set of empirical distributions.

        Args:
            distributions: List of empirical distributions (each is array of atoms)

        Returns:
            self
        """
        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        for init in range(self.n_init):
            # Initialize centroids
            centroids = self._init_centroids(distributions, rng)

            loss_history = []

            for iteration in range(self.max_iter):
                # Assign clusters
                labels = self._assign_clusters(distributions, centroids)

                # Update centroids
                new_centroids = self._update_centroids(distributions, labels)

                # Compute loss
                loss = self._compute_loss(centroids, new_centroids)
                loss_history.append(loss)

                # Check convergence
                if loss < self.tol:
                    centroids = new_centroids
                    break

                centroids = new_centroids

            # Compute inertia for this run
            inertia = self._compute_inertia(distributions, labels, centroids)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = iteration + 1

        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.n_iter_ = best_n_iter
        self.inertia_ = best_inertia

        return self

    def predict(self, distributions: List[np.ndarray]) -> np.ndarray:
        """Predict cluster labels for new distributions."""
        if self.centroids_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._assign_clusters(distributions, self.centroids_)

    def fit_predict(self, distributions: List[np.ndarray]) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(distributions)
        return self.labels_


class MomentKMeans:
    """
    Moment k-means (MK-means) clustering algorithm.

    Clusters distributions based on their first p statistical moments
    using standard Euclidean k-means.

    Reference: Definition 3.1 in paper.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        n_moments: int = 4,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_init: int = 10,
        random_state: Optional[int] = None
    ):
        """
        Initialize MK-means.

        Args:
            n_clusters: Number of clusters
            n_moments: Number of moments to use (p in paper)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            n_init: Number of initializations
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.n_moments = n_moments
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.scaler_mean_: Optional[np.ndarray] = None
        self.scaler_std_: Optional[np.ndarray] = None

    def _compute_moments(self, distribution: np.ndarray) -> np.ndarray:
        """
        Compute first p raw moments of a distribution.

        φ^p(μ) = (1/n! ∫ x^n μ(dx))_{1≤n≤p}
        """
        moments = np.zeros(self.n_moments)
        for n in range(1, self.n_moments + 1):
            # Raw moment: E[X^n]
            moments[n-1] = np.mean(distribution ** n)
        return moments

    def _distributions_to_moments(self, distributions: List[np.ndarray]) -> np.ndarray:
        """Convert list of distributions to moment matrix."""
        return np.array([self._compute_moments(d) for d in distributions])

    def _standardize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Standardize moment features component-wise.

        As per Remark 3.2: E[{x_j^i}] = 0 and Var({x_j^i}) = 1
        """
        if fit:
            self.scaler_mean_ = np.mean(X, axis=0)
            self.scaler_std_ = np.std(X, axis=0)
            self.scaler_std_[self.scaler_std_ == 0] = 1  # Avoid division by zero

        return (X - self.scaler_mean_) / self.scaler_std_

    def fit(self, distributions: List[np.ndarray]) -> 'MomentKMeans':
        """Fit MK-means to distributions."""
        from sklearn.cluster import KMeans

        # Convert to moment representation
        X = self._distributions_to_moments(distributions)

        # Standardize
        X_scaled = self._standardize(X, fit=True)

        # Apply standard k-means
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            random_state=self.random_state
        )

        self.labels_ = kmeans.fit_predict(X_scaled)
        self.centroids_ = kmeans.cluster_centers_

        return self

    def predict(self, distributions: List[np.ndarray]) -> np.ndarray:
        """Predict cluster labels for new distributions."""
        if self.centroids_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._distributions_to_moments(distributions)
        X_scaled = self._standardize(X, fit=False)

        # Assign to nearest centroid
        distances = np.linalg.norm(X_scaled[:, np.newaxis] - self.centroids_, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, distributions: List[np.ndarray]) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(distributions)
        return self.labels_


def compute_self_similarity(
    cluster_distributions: List[np.ndarray],
    n_samples: int = 1000,
    sigma: float = 0.1,
    random_state: Optional[int] = None
) -> float:
    """
    Compute within-cluster self-similarity score using MMD.

    Definition 1.9:
    Sim(X) = Median((MMD²_b[F, x_i, y_i])_{1≤i≤n})

    Args:
        cluster_distributions: List of distributions in the cluster
        n_samples: Number of pairwise samples to compute
        sigma: Gaussian kernel bandwidth
        random_state: Random seed

    Returns:
        Self-similarity score (lower is better - more similar)
    """
    if len(cluster_distributions) < 2:
        return 0.0

    rng = np.random.default_rng(random_state)
    n_dists = len(cluster_distributions)

    mmd_scores = []

    for _ in range(min(n_samples, n_dists * (n_dists - 1) // 2)):
        # Sample two different distributions
        idx = rng.choice(n_dists, size=2, replace=False)
        mmd = compute_mmd_fast(
            cluster_distributions[idx[0]],
            cluster_distributions[idx[1]],
            sigma
        )
        mmd_scores.append(mmd ** 2)

    return np.median(mmd_scores)


def compute_between_cluster_mmd(
    cluster1: List[np.ndarray],
    cluster2: List[np.ndarray],
    n_samples: int = 1000,
    sigma: float = 0.1,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Compute distribution of MMD scores between two clusters.

    Args:
        cluster1, cluster2: Lists of distributions in each cluster
        n_samples: Number of pairwise samples
        sigma: Gaussian kernel bandwidth
        random_state: Random seed

    Returns:
        Array of MMD scores
    """
    rng = np.random.default_rng(random_state)

    mmd_scores = []

    for _ in range(n_samples):
        idx1 = rng.integers(len(cluster1))
        idx2 = rng.integers(len(cluster2))

        mmd = compute_mmd_fast(cluster1[idx1], cluster2[idx2], sigma)
        mmd_scores.append(mmd ** 2)

    return np.array(mmd_scores)


def order_clusters_by_variance(
    distributions: List[np.ndarray],
    labels: np.ndarray,
    centroids: List[np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Reorder clusters so that cluster 0 has lower variance (bull/normal regime).

    This ensures consistent interpretation across runs.

    Returns:
        Reordered labels and centroids
    """
    cluster_variances = []

    for k in range(len(centroids)):
        cluster_dists = [distributions[i] for i in range(len(distributions)) if labels[i] == k]
        if cluster_dists:
            avg_var = np.mean([np.var(d) for d in cluster_dists])
        else:
            avg_var = np.var(centroids[k])
        cluster_variances.append(avg_var)

    # Sort by variance
    order = np.argsort(cluster_variances)

    # Create mapping
    mapping = {old: new for new, old in enumerate(order)}

    new_labels = np.array([mapping[l] for l in labels])
    new_centroids = [centroids[order[i]] for i in range(len(centroids))]

    return new_labels, new_centroids
