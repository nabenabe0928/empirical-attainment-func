import itertools
from typing import List
import unittest

import numpy as np

import pytest

from eaf.eaf import (
    _compute_emp_att_surf,
    _get_pf_set_list,
    get_empirical_attainment_surface,
)


def func(X: np.ndarray) -> np.ndarray:
    f1 = np.sum(X**2, axis=-1)
    f2 = np.sum((X - 2) ** 2, axis=-1)
    return np.stack([f1, f2], axis=-1)


def get_dummy_dataset() -> np.ndarray:
    dim, n_samples, n_trials = 2, 20, 10
    X = np.random.random((n_trials, n_samples, dim)) * 10 - 5
    costs = func(X)
    return costs


def naive_sol1(sol: np.ndarray, level: int, pf_set_list: List[np.ndarray]) -> None:
    for p in sol:
        if p[-1] == np.inf:
            continue

        cnt = 0
        for pf_set in pf_set_list:
            attained = np.any(np.all(pf_set <= p, axis=-1))
            cnt += attained
        assert cnt >= level


def naive_sol2(costs: np.ndarray, level: int, pf_set_list: List[np.ndarray]) -> None:
    for p in itertools.product(costs[:, 0], costs[:, 1]):
        cnt = 0
        point = np.asarray(p)
        for pf_set in pf_set_list:
            attained = np.any(np.all(pf_set <= point, axis=-1))
            cnt += attained
        if cnt >= level:
            assert np.any(np.all(p == point, axis=-1))


def test_compute_emp_att_surf() -> None:
    costs = get_dummy_dataset()
    pf_set_list = _get_pf_set_list(costs)
    costs = costs.reshape(20 * 10, 2)
    order = np.argsort(costs[:, 0])
    costs = costs[order]
    levels = np.array([1, 5, 10])
    sols = _compute_emp_att_surf(X=costs[:, 0], pf_set_list=pf_set_list, levels=levels)
    for level, sol in zip(levels, sols):
        naive_sol1(sol, level, pf_set_list)
        naive_sol2(costs, level, pf_set_list)


def test_get_pf_set_list() -> None:
    costs = np.array(
        [
            [[0, 0], [0, 0], [0, 0]],
            [[1, 0], [0, 1], [2, 2]],
        ]
    )
    pf_set_list = _get_pf_set_list(costs)
    assert len(pf_set_list) == 2
    assert np.all(pf_set_list[0] == np.array([[0, 0]]))
    assert np.all(pf_set_list[1] == np.array([[0, 1], [1, 0]]))


def test_get_empirical_attainment_surface() -> None:
    costs = np.random.random((2, 3))
    with pytest.raises(ValueError):
        get_empirical_attainment_surface(costs, levels=[1])

    costs = np.random.random((2, 3, 3))
    with pytest.raises(NotImplementedError):
        get_empirical_attainment_surface(costs, levels=[1])

    costs = np.random.random((2, 3, 2))
    with pytest.raises(ValueError):
        get_empirical_attainment_surface(costs, levels=[2, 1])

    for level in range(-1, 5):
        if 1 <= level <= 2:
            get_empirical_attainment_surface(costs, levels=[level])
        else:
            with pytest.raises(ValueError):
                get_empirical_attainment_surface(costs, levels=[level])

    emp_att_surfs = get_empirical_attainment_surface(costs, levels=[1], larger_is_better_objectives=[0])
    assert np.all(emp_att_surfs[:, 1:, :] >= 0)
    costs = np.array([[[0, 1], [1, 0], [2, 2]]])
    sol = get_empirical_attainment_surface(costs, levels=[1], larger_is_better_objectives=[0])
    ans = np.array([[-np.inf, 0], [1, 0], [2, 2], [2, np.inf]])
    assert np.allclose(sol, ans)

    sol = get_empirical_attainment_surface(costs, levels=[1], larger_is_better_objectives=[0, 1])
    ans = np.array([[-np.inf, 2], [2, 2], [2, -np.inf]])
    assert np.allclose(sol, ans)


if __name__ == "__main__":
    unittest.main()
