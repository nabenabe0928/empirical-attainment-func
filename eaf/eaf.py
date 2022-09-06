from typing import List, Optional

from fast_pareto import is_pareto_front

import numpy as np


def _get_pf_set_list(
    costs: np.ndarray,
    larger_is_better_objectives: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """
    Get the list of Pareto front sets.

    Args:
        costs (np.ndarray):
            The costs obtained in the observations.
            The shape must be (n_trials, n_samples, n_obj).
            For now, we only support n_obj == 2.
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.

    Returns:
        pf_set_list (List[np.ndarray]):
            The list of the Pareto front sets.
            The shape is (trial number, Pareto solution index, objective index).
            Note that each pareto front set is sorted based on the ascending order of
            the first objective.
    """
    pf_set_list: List[np.ndarray] = []
    kwargs = dict(larger_is_better_objectives=larger_is_better_objectives, filter_duplication=True)
    for _costs in costs:
        order = np.argsort(_costs[:, 0])
        _costs = _costs[order]
        pf_set_list.append(_costs[is_pareto_front(_costs, **kwargs)])
    return pf_set_list


def _compute_emp_att_surf(X: np.ndarray, pf_set_list: List[np.ndarray], level: int) -> np.ndarray:
    """
    Compute the empirical attainment surface of the given Pareto front sets.

    Args:
        x (np.ndarray):
            The first objective values appeared in pf_set_list.
            This array is sorted in the ascending order.
            The shape is (number of possible values, ).
        level (int):
            Control the k in the k-% attainment surface.
                k = level / n_trials
            must hold.
            level must be in [1, n_trials].
            level=1 leads to the best attainment surface,
            level=n_trials leads to the worst attainment surface,
            level=n_trials//2 leads to the median attainment surface.
        pf_set_list (List[np.ndarray]):
            The list of the Pareto front sets.
            The shape is (trial number, Pareto solution index, objective index).
            Note that each pareto front set is sorted based on the ascending order of
            the first objective.

    Reference:
        Title: On the Computation of the Empirical Attainment Function
        Authors: Carlos M. Fonseca et al.
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.705.1929&rep=rep1&type=pdf

    NOTE:
        Algorithm is different, but the result will be same.
    """
    emp_att_surf = np.zeros((X.size, 2))
    emp_att_surf[:, 0] = X
    n_trials = len(pf_set_list)
    y_candidates = np.zeros((X.size, n_trials))
    inf_array = np.asarray([np.inf])
    for i, pf_set in enumerate(pf_set_list):
        ub = np.searchsorted(pf_set[:, 0], X, side="right")
        y_min = np.minimum.accumulate(np.hstack([inf_array, pf_set[:, 1]]))
        y_candidates[:, i] = y_min[ub]
    else:
        y_candidates = np.sort(y_candidates, axis=-1)

    emp_att_surf[:, 1] = y_candidates[:, level - 1]
    return emp_att_surf


def get_empirical_attainment_surface(
    costs: np.ndarray,
    level: int,
    larger_is_better_objectives: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Get the empirical attainment surface given the costs observations.

    Args:
        costs (np.ndarray):
            The costs obtained in the observations.
            The shape must be (n_trials, n_samples, n_obj).
            For now, we only support n_obj == 2.
        level (int):
            Control the k in the k-% attainment surface.
                k = level / n_trials
            must hold.
            level must be in [1, n_trials].
            level=1 leads to the best attainment surface,
            level=n_trials leads to the worst attainment surface,
            level=n_trials//2 leads to the median attainment surface.
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.

    Returns:
        emp_att_surf (np.ndarray):
            The costs attained by (100 * quantile)% of the trials.
            In other words, (100 * quantile)% of trials dominate
            or at least include those solutions in their Pareto front.
            Note that we only return the Pareto front of attained solutions.

    TODO:
        * Activate `larger_is_better_objectives` (it is not valid now)
    """

    if len(costs.shape) != 3:
        # costs.shape = (n_trials, n_samples, n_obj)
        raise ValueError(f"costs must have the shape of (n_trials, n_samples, n_obj), but got {costs.shape}")

    (n_trials, _, n_obj) = costs.shape
    if n_obj != 2:
        raise NotImplementedError("Three or more objectives are not supported.")
    if not (1 <= level <= n_trials):
        raise ValueError(f"level must be in [1, n_trials], but got {level}")
    if larger_is_better_objectives is not None:
        raise NotImplementedError("Not available yet")

    pf_set_list = _get_pf_set_list(costs, larger_is_better_objectives=larger_is_better_objectives)
    pf_sols = np.vstack(pf_set_list)
    X = np.unique(pf_sols[:, 0])

    return _compute_emp_att_surf(X=X, pf_set_list=pf_set_list, level=level)
