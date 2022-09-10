import unittest

import matplotlib.pyplot as plt

import pytest

import numpy as np

from eaf import get_empirical_attainment_surface, plot_surface, plot_surface_with_band
from eaf.plot_surface import _check_surface, _step_direction


def func(X: np.ndarray) -> np.ndarray:
    f1 = np.sum(X**2, axis=-1)
    f2 = np.sum((X - 2) ** 2, axis=-1)
    return np.stack([f1, f2], axis=-1)


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
    with pytest.raises(ValueError):
        plot_surface(ax, np.random.random((10, 30, 2)), colors=["red"] * 10, labels=["dummy"] * 10)
    with pytest.raises(ValueError):
        plot_surface_with_band(ax, np.random.random((3, 30, 2)), color="red", label="dummy")


def test_plot() -> None:
    dim, n_samples, n_trials = 2, 200, 50
    X = np.random.random((n_trials, n_samples, dim)) * 10 - 5
    costs = func(X)

    labels = [f"the {feat} attainment" for feat in ["best", "median", "worst"]]
    levels = [1, n_trials // 2, n_trials]
    colors = ["red", "blue", "green"]
    emp_att_surfs = get_empirical_attainment_surface(costs=costs, levels=levels)

    _, ax = plt.subplots()
    plot_surface(ax, colors=colors, labels=labels, emp_att_surfs=emp_att_surfs)
    ax.legend()
    ax.grid()
    plt.close()


def test_plot_with_band() -> None:
    dim, n_samples, n_independent_runs = 2, 100, 50
    X = np.random.random((n_independent_runs, n_samples, dim)) * 10 - 5
    costs = func(X)

    levels = [n_independent_runs // 4, n_independent_runs // 2, 3 * n_independent_runs // 4]
    emp_att_surfs = get_empirical_attainment_surface(costs=costs, levels=levels)

    _, ax = plt.subplots()
    plot_surface_with_band(ax, color="red", label="random", emp_att_surfs=emp_att_surfs)
    ax.legend()
    ax.grid()
    plt.close()

    levels = [1, 2, 3, 4]
    emp_att_surfs = get_empirical_attainment_surface(costs=costs, levels=levels)

    _, ax = plt.subplots()
    with pytest.raises(ValueError):
        plot_surface_with_band(ax, color="red", label="random", emp_att_surfs=emp_att_surfs)

    plt.close()


if __name__ == "__main__":
    unittest.main()
