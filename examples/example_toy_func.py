import matplotlib.pyplot as plt

import numpy as np

from eaf import get_empirical_attainment_surface, EmpiricalAttainmentFuncPlot
from examples.toy_func import func


def plot_single(ax: plt.Axes) -> None:
    dim, n_samples, n_independent_runs = 2, 100, 50
    X = np.random.random((n_independent_runs, n_samples, dim)) * 10 - 5
    costs = func(X)

    levels = [n_independent_runs // 2]
    surf = get_empirical_attainment_surface(costs=costs, levels=levels)[0]
    eaf_plot = EmpiricalAttainmentFuncPlot()

    eaf_plot.plot_surface(ax, color="red", label="median", surf=surf)
    ax.grid()
    ax.legend()


def plot_multiple(ax: plt.Axes) -> None:
    dim, n_samples, n_independent_runs = 2, 100, 50
    X = np.random.random((n_independent_runs, n_samples, dim)) * 10 - 5
    costs = func(X)

    levels = [1, n_independent_runs // 2, n_independent_runs]
    labels = ["best", "median", "worst"]
    colors = ["red", "blue", "green"]
    markers = ["v", "^", "o"]
    surfs = get_empirical_attainment_surface(costs=costs, levels=levels)
    eaf_plot = EmpiricalAttainmentFuncPlot()

    kwargs = dict(
        labels=labels,
        colors=colors,
        markers=markers,
    )
    eaf_plot.plot_multiple_surface(ax, surfs=surfs, **kwargs)
    ax.grid()
    ax.legend()


if __name__ == "__main__":
    _, axes = plt.subplots(ncols=2)
    plot_single(axes[0])
    plot_multiple(axes[1])
    plt.show()
