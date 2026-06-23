"""
General optimal-transport solver wrappers built on POT (``ot``).

``ot`` is imported lazily inside the function so that importing this package
does not require POT to be installed for projects that only use the pure
NumPy/SciPy primitives (e.g. the 1D Wasserstein / MMD helpers).
"""


def emd2_distance(a, b, M, num_iter_max=10**8):
    """
    Exact OT (earth mover's) distance and transport plan via POT's ``emd2``.

    This is the core computation used by TiOT's ``TAOT`` solver, factored out
    for reuse. Returns the same ``(distance, transport_plan)`` pair as before:

        result = ot.lp.emd2(a, b, M, numItermax=num_iter_max, return_matrix=True)
        distance       = result[0]
        transport_plan = result[1]['G']

    Args:
        a (ndarray): source weights of shape (n,)
        b (ndarray): target weights of shape (m,)
        M (ndarray): cost matrix of shape (n, m)
        num_iter_max (int): maximum number of iterations for the LP solver

    Returns:
        tuple: (distance, transport_plan)
    """
    import ot  # lazy import: POT only needed when this solver is actually called

    result = ot.lp.emd2(a, b, M, numItermax=num_iter_max, return_matrix=True)
    distance = result[0]
    transport_plan = result[1]['G']
    return distance, transport_plan
