"""
Maximum Mean Discrepancy (MMD) helpers and Gaussian kernel.

Pure NumPy. Behaviour is byte-for-byte identical to the original
implementations in Wasserstein_Distance_for_Time_Series-Data/wasserstein_kmeans.py.
"""

import numpy as np


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 0.1) -> float:
    """
    Compute Gaussian kernel.

    κ_G(x, y) = exp(-||x - y||^2 / (2σ^2))

    Args:
        x, y: Input vectors
        sigma: Kernel bandwidth

    Returns:
        Kernel value
    """
    return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))


def compute_mmd_biased(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 0.1
) -> float:
    """
    Compute biased empirical MMD estimate.

    From Equation (53):
    MMD_b[F, x, y] = [1/n² Σ κ(x_i, x_j) - 2/(mn) Σ κ(x_i, y_j) + 1/m² Σ κ(y_i, y_j)]^{1/2}

    Args:
        x: First sample (n x d array or 1d array)
        y: Second sample (m x d array or 1d array)
        sigma: Gaussian kernel bandwidth

    Returns:
        Biased MMD estimate
    """
    x = np.atleast_2d(x).T if x.ndim == 1 else x
    y = np.atleast_2d(y).T if y.ndim == 1 else y

    n, m = len(x), len(y)

    # Compute kernel matrices
    xx = np.sum([
        [gaussian_kernel(x[i], x[j], sigma) for j in range(n)]
        for i in range(n)
    ])

    yy = np.sum([
        [gaussian_kernel(y[i], y[j], sigma) for j in range(m)]
        for i in range(m)
    ])

    xy = np.sum([
        [gaussian_kernel(x[i], y[j], sigma) for j in range(m)]
        for i in range(n)
    ])

    mmd_squared = xx / (n * n) - 2 * xy / (n * m) + yy / (m * m)

    return np.sqrt(max(mmd_squared, 0))


def compute_mmd_fast(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 0.1
) -> float:
    """
    Fast vectorized computation of biased MMD.

    Args:
        x: First sample (1d array - atoms of empirical distribution)
        y: Second sample (1d array - atoms of empirical distribution)
        sigma: Gaussian kernel bandwidth

    Returns:
        Biased MMD estimate
    """
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Compute pairwise squared distances
    xx_dist = np.sum((x[:, np.newaxis] - x) ** 2, axis=2)
    yy_dist = np.sum((y[:, np.newaxis] - y) ** 2, axis=2)
    xy_dist = np.sum((x[:, np.newaxis] - y) ** 2, axis=2)

    # Apply Gaussian kernel
    gamma = 1 / (2 * sigma ** 2)
    K_xx = np.exp(-gamma * xx_dist)
    K_yy = np.exp(-gamma * yy_dist)
    K_xy = np.exp(-gamma * xy_dist)

    n, m = len(x), len(y)
    mmd_squared = np.sum(K_xx) / (n * n) - 2 * np.sum(K_xy) / (n * m) + np.sum(K_yy) / (m * m)

    return np.sqrt(max(mmd_squared, 0))
