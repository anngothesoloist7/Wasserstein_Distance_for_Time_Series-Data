"""
Cost-matrix builders shared by optimal-transport solvers.

Pure NumPy. Behaviour is byte-for-byte identical to the original
implementations in TiOT/TiOT_lib.py.
"""

import numpy as np

from ._wasserstein1d import normalization


def spat_temp_cost(x, y, eps=1):
    """Spatial (squared-Euclidean) and temporal cost matrices for x, y."""
    n, m = len(x), len(y)
    t, s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    if x.ndim == 1 and y.ndim == 1:
        spatial_cost = (x[:, None] - y[None, :])**2 / eps
    else:
        x_norm2 = np.sum(x**2, axis=1)[:, None]
        y_norm2 = np.sum(y**2, axis=1)[None, :]
        spatial_cost = (x_norm2 + y_norm2 - 2 * x @ y.T) / eps
    temporal_cost = (t[:, None] - s[None, :])**2 / eps
    return spatial_cost, temporal_cost


def costmatrix0(x, y, w) -> np.ndarray:
    """
    Cost matrix define in the TAOT's way: C.

    Parameters:
        x (ndarray): 1D array of shape (n,)
        y (ndarray): 1D array of shape (m,)
        w (float)  : parameter to calculate C_ij

    Returns:
        ndarray: 2D array of shape (n,m)
    """

    spatial_cost, temporal_cost = spat_temp_cost(x, y)
    C = spatial_cost + w * temporal_cost
    C = C / np.median(C)
    return C


def costmatrix1(x, y, w) -> np.ndarray:
    """
    Cost matrix define in the TiOT's way: C.

    Parameters:
        x (ndarray): 1D array of shape (n,)
        y (ndarray): 1D array of shape (m,)
        w (float)  : parameter to calculate C_ij

    Returns:
        ndarray: 2D array of shape (n,m)
    """
    x, y = normalization(x, y)
    spatial_cost, temporal_cost = spat_temp_cost(x, y)
    return w * spatial_cost + (1-w)*temporal_cost
