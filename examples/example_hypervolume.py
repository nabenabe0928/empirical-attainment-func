import matplotlib.pyplot as plt

import numpy as np

from eaf import EmpiricalAttainmentFuncPlot
from examples.toy_func import func


REF_POINTS = np.array([50, 98])
PARETO_SOLS = np.array([[0, 0]])  # this is not correct, but just defined it to be able to run


def plot_single(ax: plt.Axes) -> None:
    dim, n_samples, n_independent_runs = 2, 100, 50
    X = np.random.random((n_independent_runs, n_samples, dim)) * 10 - 5
    costs = func(X)

    eaf_plot = EmpiricalAttainmentFuncPlot(ref_point=REF_POINTS, true_pareto_sols=PARETO_SOLS)

    eaf_plot.plot_hypervolume2d_with_band(ax, costs_array=costs, color="red", label="test", marker="o")
    ax.grid()
    ax.legend()


def plot_multiple(ax: plt.Axes) -> None:
    n_methods, dim, n_samples, n_independent_runs = 3, 2, 100, 50
    X = np.random.random((n_methods, n_independent_runs, n_samples, dim)) * 10 - 5
    costs = func(X)

    labels = ["Exp. 1", "Exp. 2", "Exp. 3"]
    colors = ["red", "blue", "green"]
    markers = ["v", "^", "o"]
    eaf_plot = EmpiricalAttainmentFuncPlot(ref_point=REF_POINTS, true_pareto_sols=PARETO_SOLS)

    kwargs = dict(
        labels=labels,
        colors=colors,
        markers=markers,
        markersize=5,
    )
    eaf_plot.plot_multiple_hypervolume2d_with_band(ax, costs_array=costs, **kwargs)
    ax.grid()
    ax.legend()


if __name__ == "__main__":
    _, axes = plt.subplots(ncols=2)
    plot_single(axes[0])
    plot_multiple(axes[1])
    plt.show()
