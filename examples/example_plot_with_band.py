import matplotlib.pyplot as plt

import numpy as np

from eaf import get_empirical_attainment_surface, EmpiricalAttainmentFuncPlot
from examples.toy_func import func


def plot_single(ax: plt.Axes) -> None:
    dim, n_samples, n_independent_runs = 2, 100, 50
    X = np.random.random((n_independent_runs, n_samples, dim)) * 10 - 5
    costs = func(X)

    levels = [n_independent_runs // 4, n_independent_runs // 2, 3 * n_independent_runs // 4]
    surfs = get_empirical_attainment_surface(costs=costs, levels=levels)
    eaf_plot = EmpiricalAttainmentFuncPlot()

    eaf_plot.plot_surface_with_band(ax, surfs=surfs, color="red", label="random")
    ax.legend()
    ax.grid()


def plot_multiple(ax: plt.Axes) -> None:
    dim, n_samples, n_independent_runs = 2, 100, 50
    X = np.random.random((3, n_independent_runs, n_samples, dim)) * 10 - 5
    costs_list = func(X)

    levels = [n_independent_runs // 4, n_independent_runs // 2, 3 * n_independent_runs // 4]
    surfs_list = [get_empirical_attainment_surface(costs=costs, levels=levels) for costs in costs_list]
    eaf_plot = EmpiricalAttainmentFuncPlot()

    colors = ["red", "blue", "green"]
    labels = ["Exp. 1", "Exp. 2", "Exp. 3"]
    markers = ["v", "^", "o"]
    kwargs = dict(
        colors=colors,
        labels=labels,
        markers=markers,
        markersize=3,
    )
    eaf_plot.plot_multiple_surface_with_band(ax, surfs_list=surfs_list, **kwargs)
    ax.legend()
    ax.grid()


if __name__ == "__main__":
    _, axes = plt.subplots(ncols=2)
    plot_single(axes[0])
    plot_multiple(axes[1])
    plt.show()
