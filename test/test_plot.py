import unittest

import matplotlib.pyplot as plt

import pytest

import numpy as np

from eaf import (
    get_empirical_attainment_surface,
    EmpiricalAttainmentFuncPlot,
)


def func(X: np.ndarray) -> np.ndarray:
    f1 = np.sum(X**2, axis=-1)
    f2 = np.sum((X - 2) ** 2, axis=-1)
    return np.stack([f1, f2], axis=-1)


def test_plot() -> None:
    dim, n_samples, n_trials = 2, 200, 50
    X = np.random.random((n_trials, n_samples, dim)) * 10 - 5
    costs = func(X)

    labels = [f"the {feat} attainment" for feat in ["best", "median", "worst"]]
    levels = [1, n_trials // 2, n_trials]
    colors = ["red", "blue", "green"]
    surfs_list = get_empirical_attainment_surface(costs=costs, levels=levels)

    _, ax = plt.subplots()
    eaf_plot = EmpiricalAttainmentFuncPlot()
    eaf_plot.plot_multiple_surface(ax, colors=colors, labels=labels, surfs=surfs_list)
    eaf_plot.plot_surface(ax, color=colors[0], label=labels[0], surf=surfs_list[0])
    # test log scale
    eaf_plot = EmpiricalAttainmentFuncPlot(log_scale=[0, 1])
    eaf_plot.plot_surface(ax, color=colors[0], label=labels[0], surf=surfs_list[0])

    with pytest.raises(ValueError):
        # shape error
        eaf_plot.plot_surface(ax, color=colors[0], label=labels[0], surf=surfs_list)

    ax.legend()
    ax.grid()
    plt.close()


def test_plot_with_band() -> None:
    dim, n_samples, n_independent_runs = 2, 100, 50
    X = np.random.random((n_independent_runs, n_samples, dim)) * 10 - 5
    costs = func(X)

    levels = [n_independent_runs // 4, n_independent_runs // 2, 3 * n_independent_runs // 4]
    surfs = get_empirical_attainment_surface(costs=costs, levels=levels)

    _, ax = plt.subplots()
    eaf_plot = EmpiricalAttainmentFuncPlot()
    eaf_plot.plot_surface_with_band(ax, color="red", label="random", surfs=surfs)
    eaf_plot.plot_multiple_surface_with_band(ax, colors=["red"] * 2, labels=["random"] * 2, surfs_list=[surfs] * 2)
    ax.legend()
    ax.grid()
    plt.close()

    levels = [1, 2, 3, 4]
    surfs = get_empirical_attainment_surface(costs=costs, levels=levels)

    _, ax = plt.subplots()
    eaf_plot = EmpiricalAttainmentFuncPlot()
    with pytest.raises(ValueError):
        # Shape error ===> 4 lines are not allowed
        eaf_plot.plot_surface_with_band(ax, surfs=surfs, color="red", label="random")

    plt.close()


if __name__ == "__main__":
    unittest.main()
