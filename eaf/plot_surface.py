from typing import Any, List

import matplotlib.pyplot as plt

import numpy as np


def plot_surface(
    ax: plt.Axes,
    emp_att_surfs: np.ndarray,
    colors: List[str],
    labels: List[str],
    **kwargs: Any,
) -> None:
    """

    Args:
        ax (plt.Axes):
            The subplots axes.
        emp_att_surfs (np.ndarray):
            The vertices of the empirical attainment surfaces for each level.
            If emp_att_surf[i, j, 1] takes np.inf, this is not actually on the surface.
            The shape is (levels.size, X.size, 2).
        colors (List[str]):
            The colors of each plot
        labels (List[str]):
            The labels of each plot.
        kwargs:
            The kwargs for scatter.
    """
    X = emp_att_surfs[..., 0].flatten()
    Y = emp_att_surfs[..., 1].flatten()
    X = X[np.isfinite(X)]
    Y = Y[np.isfinite(Y)]
    (x_min, x_max) = X.min(), X.max()
    (y_min, y_max) = Y.min(), Y.max()
    x_min -= 0.1 * (x_max - x_min)
    x_max += 0.1 * (x_max - x_min)
    y_min -= 0.1 * (y_max - y_min)
    y_max += 0.1 * (y_max - y_min)

    emp_att_surfs[..., 0][emp_att_surfs[..., 0] == -np.inf] = x_min
    emp_att_surfs[..., 0][emp_att_surfs[..., 0] == np.inf] = x_max
    emp_att_surfs[..., 1][emp_att_surfs[..., 1] == -np.inf] = y_min
    emp_att_surfs[..., 1][emp_att_surfs[..., 1] == np.inf] = y_max

    for color, label, emp_att_surf in zip(colors, labels, emp_att_surfs):
        ax.plot(emp_att_surf[:, 0], emp_att_surf[:, 1], color=color, label=label, **kwargs)

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    ax.legend(loc="upper right")
