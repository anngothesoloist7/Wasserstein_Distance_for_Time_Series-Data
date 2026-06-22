"""
General-purpose 1D optimal-transport / Wasserstein primitives.

These functions are pure (NumPy/SciPy only) and shared across projects.
Behaviour is byte-for-byte identical to the original implementations that
previously lived in:
  - Wasserstein_Distance_for_Time_Series-Data/wasserstein_kmeans.py
  - TiOT/TiOT_lib.py  (``normalization``)
"""

from typing import List

import numpy as np


def normalization(x, y):
    """
    Normalize input time series by formular
    x = (x - mean(x)) / std(x)

    Parameters
        x (ndarray): 1D array of shape (n,)
        y (ndarray): 1D array of shape (m,)

    Returns:
        (tuple): normalized x, normalized y
    """

    return (x - np.mean(x)) / np.std(x), (y - np.mean(y)) / np.std(y)


def wasserstein_distance_1d(mu: np.ndarray, nu: np.ndarray, p: int = 1) -> float:
    """
    Compute p-Wasserstein distance between two 1D empirical distributions.

    From Equation (21) in paper:
    W_p(μ, ν)^p = (1/N) * Σ_{i=1}^{N} |α_i - β_i|^p

    where (α_i) and (β_i) are sorted atoms of μ and ν respectively.

    Args:
        mu: Atoms of first distribution (will be sorted)
        nu: Atoms of second distribution (will be sorted)
        p: Order of Wasserstein distance (default 1)

    Returns:
        p-Wasserstein distance
    """
    # Sort atoms (order statistics)
    alpha = np.sort(mu)
    beta = np.sort(nu)

    # Handle different sizes by interpolating quantiles
    if len(alpha) != len(beta):
        n_points = max(len(alpha), len(beta))
        quantiles = np.linspace(0, 1, n_points + 1)[1:]  # Exclude 0

        # Compute quantile functions
        alpha_quantiles = np.quantile(mu, quantiles)
        beta_quantiles = np.quantile(nu, quantiles)
    else:
        alpha_quantiles = alpha
        beta_quantiles = beta

    # Compute Wasserstein distance
    dist = np.mean(np.abs(alpha_quantiles - beta_quantiles) ** p) ** (1/p)

    return dist


def wasserstein_barycenter_1d(distributions: List[np.ndarray], p: int = 1) -> np.ndarray:
    """
    Compute Wasserstein barycenter for a set of 1D empirical distributions.

    From Proposition 2.6:
    - For p=1: a_j = Median(α_j^1, ..., α_j^M) for j = 1, ..., N
    - For p>1: a_j = Mean(α_j^1, ..., α_j^M)

    Args:
        distributions: List of empirical distributions (arrays of atoms)
        p: Order of Wasserstein distance (1 for median, >1 for mean)

    Returns:
        Barycenter distribution (sorted atoms)
    """
    if len(distributions) == 0:
        raise ValueError("Cannot compute barycenter of empty set")

    # Sort all distributions
    sorted_dists = [np.sort(d) for d in distributions]

    # Resample to common size if needed
    n_atoms = max(len(d) for d in sorted_dists)

    # Compute quantile functions at common points
    quantiles = np.linspace(0, 1, n_atoms + 1)[1:]

    quantile_values = np.zeros((len(distributions), n_atoms))
    for i, d in enumerate(distributions):
        quantile_values[i] = np.quantile(d, quantiles)

    # Compute barycenter atoms
    if p == 1:
        barycenter = np.median(quantile_values, axis=0)
    else:
        barycenter = np.mean(quantile_values, axis=0)

    return barycenter
