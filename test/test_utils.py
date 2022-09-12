import unittest

from fast_pareto import is_pareto_front

import matplotlib.pyplot as plt

import pytest

import numpy as np

from eaf import pareto_front_to_surface, EmpiricalAttainmentFuncPlot
from eaf.utils import LOGEPS, _check_surface, _compute_hypervolume2d, _step_direction


def func(X: np.ndarray) -> np.ndarray:
    f1 = np.sum(X**2, axis=-1)
    f2 = np.sum((X - 2) ** 2, axis=-1)
    return np.stack([f1, f2], axis=-1)


def get_dummy_dataset() -> np.ndarray:
    dim, n_samples = 2, 300
    X = np.random.random((n_samples, dim)) * 10 - 5
    costs = func(X)
    return costs


def test_compute_hypervolume2d() -> None:
    for _ in range(20):
        costs_array = np.random.random((20, 10, 2))
        ref_point = np.ones(2)
        sol = _compute_hypervolume2d(costs_array=costs_array, ref_point=ref_point)

        for i, costs in enumerate(costs_array):
            costs = costs[is_pareto_front(costs)]
            order = np.lexsort((-costs[:, 1], costs[:, 0]))
            sorted_costs = costs[order]
            w = np.hstack([sorted_costs[1:, 0], ref_point[0]]) - sorted_costs[:, 0]
            h = ref_point[1] - np.minimum.accumulate(sorted_costs[:, 1])
            assert np.allclose(sol[i], w @ h)

    with pytest.raises(ValueError):
        _compute_hypervolume2d(costs_array, ref_point=np.array([0.5, 0.5]))


def test_raise_errors_in_pareto_front_to_surface() -> None:
    with pytest.raises(ValueError):
        costs = get_dummy_dataset()
        pareto_front = costs[is_pareto_front(costs)]
        # Shape error
        pareto_front_to_surface(pareto_front[None])

    with pytest.raises(ValueError):
        costs = get_dummy_dataset()
        pareto_front = costs[is_pareto_front(costs)]
        pareto_front[0, 0] = 1000
        # Out of bound error
        pareto_front_to_surface(pareto_front, x_max=100)


def test_values_in_pareto_front_to_surface() -> None:
    costs = get_dummy_dataset()
    pareto_front = costs[is_pareto_front(costs)]
    modified_pf = pareto_front_to_surface(pareto_front, x_min=0, x_max=100, y_min=0, y_max=100)
    assert modified_pf[0, 0] == pareto_front[:, 0].min()
    assert modified_pf[0, 1] == 100
    assert modified_pf[-1, 0] == 100
    assert modified_pf[-1, 1] == pareto_front[:, 1].min()

    modified_pf = pareto_front_to_surface(pareto_front, x_max=100, y_max=100, log_scale=[0])
    assert modified_pf[0, 0] == pareto_front[:, 0].min()
    assert modified_pf[0, 1] == 100
    assert modified_pf[-1, 0] == 100
    assert modified_pf[-1, 1] == pareto_front[:, 1].min()


def test_pareto_front_to_surface() -> None:
    costs = get_dummy_dataset()
    pareto_front = costs[is_pareto_front(costs)]
    modified_pf = pareto_front_to_surface(pareto_front)
    assert np.allclose(np.maximum.accumulate(modified_pf[:, 0]), modified_pf[:, 0])
    assert np.allclose(np.minimum.accumulate(modified_pf[:, 1]), modified_pf[:, 1])

    costs = get_dummy_dataset()
    costs[:, 0] *= -1
    costs[:, 1] *= -1
    kwargs = dict(larger_is_better_objectives=[0, 1])
    pareto_front = costs[is_pareto_front(costs, **kwargs)]
    modified_pf = pareto_front_to_surface(pareto_front, **kwargs)
    assert np.allclose(np.maximum.accumulate(modified_pf[:, 0]), modified_pf[:, 0])
    assert np.allclose(np.minimum.accumulate(modified_pf[:, 1]), modified_pf[:, 1])

    costs = get_dummy_dataset()
    costs[:, 0] *= -1
    kwargs = dict(larger_is_better_objectives=[0])
    pareto_front = costs[is_pareto_front(costs, **kwargs)]
    modified_pf = pareto_front_to_surface(pareto_front, **kwargs)
    assert np.allclose(np.maximum.accumulate(modified_pf[:, 0]), modified_pf[:, 0])
    assert np.allclose(np.maximum.accumulate(modified_pf[:, 1]), modified_pf[:, 1])

    costs = get_dummy_dataset()
    costs[:, 1] *= -1
    kwargs = dict(larger_is_better_objectives=[1])
    pareto_front = costs[is_pareto_front(costs, **kwargs)]
    modified_pf = pareto_front_to_surface(pareto_front, **kwargs)
    assert np.allclose(np.maximum.accumulate(modified_pf[:, 0]), modified_pf[:, 0])
    assert np.allclose(np.maximum.accumulate(modified_pf[:, 1]), modified_pf[:, 1])


def test_transform_surface_list() -> None:
    eaf_plot = EmpiricalAttainmentFuncPlot()
    surfs = np.array(
        [
            [-np.inf, 1, 2, 3, 4, np.inf],
            [np.inf, 8, 7, 6, 5, -np.inf],
        ]
    ).T
    sol = eaf_plot._transform_surface_list([surfs])
    ans = np.array([[0.7, 1, 2, 3, 4, 4.33], [8.33, 8, 7, 6, 5, 4.7]]).T
    assert np.allclose(sol, ans)

    eaf_plot = EmpiricalAttainmentFuncPlot(log_scale=[0, 1])
    surfs = np.array(
        [
            [LOGEPS, 1e-4, 1e-3, 1e-2, 1e-1, np.inf],
            [np.inf, 1e-5, 1e-6, 1e-7, 1e-8, LOGEPS],
        ]
    ).T
    ans = np.array(
        [
            [5.01187234e-05, 1e-4, 1e-3, 1e-2, 1e-1, 2.13796209e-01],
            [2.13796209e-05, 1e-5, 1e-6, 1e-7, 1e-8, 5.01187234e-09],
        ]
    ).T
    sol = eaf_plot._transform_surface_list([surfs])
    assert np.allclose(sol, ans)


def test_step_direction() -> None:
    larger_is_better_objectives = None
    assert _step_direction(larger_is_better_objectives) == "post"
    larger_is_better_objectives = []
    assert _step_direction(larger_is_better_objectives) == "post"
    larger_is_better_objectives = [1]
    assert _step_direction(larger_is_better_objectives) == "post"
    larger_is_better_objectives = [0]
    assert _step_direction(larger_is_better_objectives) == "pre"
    larger_is_better_objectives = [0, 1]
    assert _step_direction(larger_is_better_objectives) == "pre"


def test_check_surface():
    surf = np.arange(4 * 3 * 2).reshape((4, 3, 2))
    with pytest.raises(ValueError):
        _check_surface(surf)

    surf = np.arange(3 * 2).reshape((3, 2))
    _check_surface(surf)

    surf = np.random.random((30, 2))
    with pytest.raises(ValueError):
        _check_surface(surf)

    _, ax = plt.subplots()
    eaf_plot = EmpiricalAttainmentFuncPlot()
    with pytest.raises(ValueError):
        eaf_plot.plot_surface(ax, np.random.random((30, 2)), color="red", label="dummy")
    with pytest.raises(ValueError):
        eaf_plot.plot_surface_with_band(ax, np.random.random((3, 30, 2)), color="red", label="dummy")

    plt.close()


if __name__ == "__main__":
    unittest.main()
