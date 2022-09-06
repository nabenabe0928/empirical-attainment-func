from typing import List, Optional

from fast_pareto import is_pareto_front
from fast_pareto.pareto import _change_directions

import numpy as np


def _get_pf_set_list(costs: np.ndarray) -> List[np.ndarray]:
    """
    Get the list of Pareto front sets.

    Args:
        costs (np.ndarray):
            The costs obtained in the observations.
            The shape must be (n_independent_runs, n_samples, n_obj).
            For now, we only support n_obj == 2.

    Returns:
        pf_set_list (List[np.ndarray]):
            The list of the Pareto front sets.
            The shape is (trial number, Pareto solution index, objective index).
            Note that each pareto front set is sorted based on the ascending order of
            the first objective.
    """
    pf_set_list: List[np.ndarray] = []
    for _costs in costs:
        order = np.argsort(_costs[:, 0])
        _costs = _costs[order]
        pf_set_list.append(_costs[is_pareto_front(_costs, filter_duplication=True)])
    return pf_set_list


def _compute_emp_att_surf(X: np.ndarray, pf_set_list: List[np.ndarray], levels: np.ndarray) -> np.ndarray:
    """
    Compute the empirical attainment surface of the given Pareto front sets.

    Args:
        x (np.ndarray):
            The first objective values appeared in pf_set_list.
            This array is sorted in the ascending order.
            The shape is (number of possible values, ).
        levels (np.ndarray):
            A list of `level` described below:
                Control the k in the k-% attainment surface.
                    k = level / n_independent_runs
                must hold.
                level must be in [1, n_independent_runs].
                level=1 leads to the best attainment surface,
                level=n_independent_runs leads to the worst attainment surface,
                level=n_independent_runs//2 leads to the median attainment surface.
        pf_set_list (List[np.ndarray]):
            The list of the Pareto front sets.
            The shape is (trial number, Pareto solution index, objective index).
            Note that each pareto front set is sorted based on the ascending order of
            the first objective.

    Returns:
        emp_att_surfs (np.ndarray):
            The vertices of the empirical attainment surfaces for each level.
            If emp_att_surf[i, j, 1] takes np.inf, this is not actually on the surface.
            The shape is (levels.size, X.size, 2).

    Reference:
        Title: On the Computation of the Empirical Attainment Function
        Authors: Carlos M. Fonseca et al.
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.705.1929&rep=rep1&type=pdf

    NOTE:
        Algorithm is different, but the result will be same.
        When we define N = n_independent_runs, K = X.size, and S = n_samples,
        the original algorithm requires O(K log K + NK)
        and this algorithm requires O(NK log (NK)).
        Although our algorithm is a bit worse than the original algorithm,
        since the enumerating Pareto solutions requires O(NS log S),
        which might be smaller complexity but will take more time in Python,
        the time complexity will not dominate the whole process.
    """
    n_levels = len(levels)
    emp_att_surfs = np.zeros((n_levels, X.size, 2))
    emp_att_surfs[..., 0] = X
    n_independent_runs = len(pf_set_list)
    y_candidates = np.zeros((X.size, n_independent_runs))
    for i, pf_set in enumerate(pf_set_list):
        ub = np.searchsorted(pf_set[:, 0], X, side="right")
        y_min = np.minimum.accumulate(np.hstack([np.inf, pf_set[:, 1]]))
        y_candidates[:, i] = y_min[ub]
    else:
        y_candidates = np.sort(y_candidates, axis=-1)

    y_sol = y_candidates[:, levels - 1].T
    emp_att_surfs[..., 1] = y_sol

    for emp_att_surf in emp_att_surfs:
        idx = np.sum(emp_att_surf[:, 1] == np.inf)
        emp_att_surf[:idx, 0] = emp_att_surf[idx, 0]

    return emp_att_surfs


def get_empirical_attainment_surface(
    costs: np.ndarray,
    levels: List[int],
    larger_is_better_objectives: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Get the empirical attainment surface given the costs observations.

    Args:
        costs (np.ndarray):
            The costs obtained in the observations.
            The shape must be (n_independent_runs, n_samples, n_obj).
            For now, we only support n_obj == 2.
        levels (List[int]):
            A list of `level` described below:
                Control the k in the k-% attainment surface.
                    k = level / n_independent_runs
                must hold.
                level must be in [1, n_independent_runs].
                level=1 leads to the best attainment surface,
                level=n_independent_runs leads to the worst attainment surface,
                level=n_independent_runs//2 leads to the median attainment surface.
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.

    Returns:
        emp_att_surfs (np.ndarray):
            The costs attained by (level / n_independent_runs) * 100% of the trials.
            In other words, (level / n_independent_runs) * 100% of runs dominate
            or at least include those solutions in their Pareto front.
            Note that we only return the Pareto front of attained solutions.
    """

    if len(costs.shape) != 3:
        # costs.shape = (n_independent_runs, n_samples, n_obj)
        raise ValueError(f"costs must have the shape of (n_independent_runs, n_samples, n_obj), but got {costs.shape}")

    (n_independent_runs, _, n_obj) = costs.shape
    if n_obj != 2:
        raise NotImplementedError("Three or more objectives are not supported.")
    if not all(1 <= level <= n_independent_runs for level in levels):
        raise ValueError(f"All elements in levels must be in [1, n_independent_runs], but got {levels}")
    if not np.all(np.maximum.accumulate(levels) == levels):
        raise ValueError(f"levels must be an increasing sequence, but got {levels}")
    if larger_is_better_objectives is not None:
        costs = _change_directions(costs, larger_is_better_objectives=larger_is_better_objectives)

    pf_set_list = _get_pf_set_list(costs)
    pf_sols = np.vstack(pf_set_list)
    X = np.unique(np.hstack([-np.inf, pf_sols[:, 0], np.inf]))

    emp_att_surfs = _compute_emp_att_surf(X=X, pf_set_list=pf_set_list, levels=np.asarray(levels))
    if larger_is_better_objectives is not None:
        emp_att_surfs = _change_directions(emp_att_surfs, larger_is_better_objectives=larger_is_better_objectives)

    return emp_att_surfs
