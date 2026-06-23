"""
ot_common — shared optimal-transport / Wasserstein primitives.

A small, dependency-light library of reusable functions used across the
projects in this repository (TiOT and the Wasserstein K-Means market-regime
clustering). Extracting these here removes duplication and lets each project
import the same, tested implementation:

    from ot_common import wasserstein_distance_1d, normalization, emd2_distance

Layers (by dependency):
  - _wasserstein1d : closed-form 1D Wasserstein + barycenter, z-score
                     normalization              (NumPy only)
  - _cost          : spatial/temporal cost matrices (TiOT)   (NumPy only)
  - _mmd           : Gaussian kernel + MMD estimators         (NumPy only)
  - _solvers       : exact OT via POT (``ot``)                (POT, lazy import)

Note: ``emd2_distance`` imports POT (``ot``) lazily, so importing this package
does not require POT for projects that only use the pure NumPy/SciPy helpers.
"""

from ._wasserstein1d import (
    normalization,
    wasserstein_distance_1d,
    wasserstein_barycenter_1d,
)
from ._cost import (
    spat_temp_cost,
    costmatrix0,
    costmatrix1,
)
from ._mmd import (
    gaussian_kernel,
    compute_mmd_biased,
    compute_mmd_fast,
)
from ._solvers import emd2_distance

__all__ = [
    # 1D Wasserstein primitives
    "normalization",
    "wasserstein_distance_1d",
    "wasserstein_barycenter_1d",
    # cost matrices
    "spat_temp_cost",
    "costmatrix0",
    "costmatrix1",
    # MMD
    "gaussian_kernel",
    "compute_mmd_biased",
    "compute_mmd_fast",
    # exact OT solver (POT)
    "emd2_distance",
]
