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


def test_pareto_plot() -> None:
    eaf_plot = EmpiricalAttainmentFuncPlot()
    with pytest.raises(AttributeError):
        _, ax = plt.subplots()
        eaf_plot.plot_true_pareto_surface(ax, color="red", label="dummy")

    pareto_sols = np.random.random((30, 2))
    pareto_sols[:, 0] = pareto_sols[np.argsort(pareto_sols[:, 0]), 0]
    pareto_sols[:, 1] = pareto_sols[np.argsort(pareto_sols[:, 1]), 1]
    eaf_plot = EmpiricalAttainmentFuncPlot(true_pareto_sols=pareto_sols)

    _, ax = plt.subplots()
    x_min, x_max = eaf_plot.x_min, eaf_plot.x_max
    y_min, y_max = eaf_plot.y_min, eaf_plot.y_max
    assert np.isfinite(x_min) and np.isfinite(x_max)
    assert np.isfinite(y_min) and np.isfinite(y_max)
    assert x_min < pareto_sols[:, 0].min() and x_max > pareto_sols[:, 0].max()
    assert y_min < pareto_sols[:, 0].min() and y_max > pareto_sols[:, 0].max()

    eaf_plot.plot_true_pareto_surface(ax, color="red", label="dummy")
    assert x_min == eaf_plot.x_min and x_max == eaf_plot.x_max
    assert y_min == eaf_plot.y_min and y_max == eaf_plot.y_max
    plt.close()


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


def test_plot_hypervolume() -> None:
    dim, n_samples, n_independent_runs = 2, 100, 50
    X = np.random.random((n_independent_runs, n_samples, dim)) * 10 - 5
    costs = func(X)
    _, ax = plt.subplots()
    ref_point = np.array([100, 100])
    for log_scale in [None, [0, 1]]:
        eaf_plot = EmpiricalAttainmentFuncPlot(ref_point=ref_point, log_scale=log_scale)
        for log in [True, False]:
            for axis_label in [True, False]:
                eaf_plot.plot_hypervolume2d_with_band(
                    ax,
                    costs_array=costs,
                    color="red",
                    label="dummy",
                    log=log,
                    axis_label=axis_label,
                )
                eaf_plot.plot_multiple_hypervolume2d_with_band(
                    ax,
                    costs_array=np.stack([costs, costs]),
                    colors=["red"] * 2,
                    labels=["dummy"] * 2,
                    log=log,
                    axis_label=axis_label,
                )

    with pytest.raises(ValueError):
        # shape error
        eaf_plot.plot_hypervolume2d_with_band(ax, costs_array=costs[0], color="red", label="dummy")

    with pytest.raises(ValueError):
        # shape error
        eaf_plot.plot_multiple_hypervolume2d_with_band(ax, costs_array=costs, colors=["red"] * 2, labels=["dummy"] * 2)

    eaf_plot = EmpiricalAttainmentFuncPlot(ref_point=ref_point, larger_is_better_objectives=[0])
    eaf_plot.plot_hypervolume2d_with_band(ax, costs_array=costs, color="red", label="dummy")

    eaf_plot = EmpiricalAttainmentFuncPlot()
    with pytest.raises(AttributeError):
        # no ref point
        eaf_plot.plot_hypervolume2d_with_band(ax, costs_array=costs, color="red", label="dummy")

    plt.close()


if __name__ == "__main__":
    unittest.main()
