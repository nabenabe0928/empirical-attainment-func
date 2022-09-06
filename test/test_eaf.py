import itertools
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


def test_compute_emp_att_surf() -> None:
    costs = get_dummy_dataset()
    pf_set_list = _get_pf_set_list(costs)
    costs = costs.reshape(20 * 10, 2)
    order = np.argsort(costs[:, 0])
    costs = costs[order]
    level = 5
    sol = _compute_emp_att_surf(X=costs[:, 0], pf_set_list=pf_set_list, level=level)
    for p in sol:
        if p[-1] == np.inf:
            continue

        cnt = 0
        for pf_set in pf_set_list:
            attained = np.any(np.all(pf_set <= p, axis=-1))
            print(pf_set, p, attained)
            cnt += attained
        assert cnt >= level

    for p in itertools.product(costs[:, 0], costs[:, 1]):
        cnt = 0
        point = np.asarray(p)
        for pf_set in pf_set_list:
            attained = np.any(np.all(pf_set <= point, axis=-1))
            cnt += attained
        if cnt >= level:
            assert np.any(np.all(p == point, axis=-1))


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
        get_empirical_attainment_surface(costs, level=1)

    costs = np.random.random((2, 3, 3))
    with pytest.raises(NotImplementedError):
        get_empirical_attainment_surface(costs, level=1)

    costs = np.random.random((2, 3, 2))
    for level in range(-1, 5):
        if 1 <= level <= 2:
            get_empirical_attainment_surface(costs, level=level)
        else:
            with pytest.raises(ValueError):
                get_empirical_attainment_surface(costs, level=level)


if __name__ == "__main__":
    unittest.main()
