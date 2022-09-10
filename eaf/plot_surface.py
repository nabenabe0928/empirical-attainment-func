from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt

import numpy as np


def _transform_attainment_surface(
    emp_att_surfs: np.ndarray,
    log_scale: Optional[List[int]],
) -> Tuple[np.ndarray, float, float, float, float]:
    X = emp_att_surfs[..., 0].flatten()
    Y = emp_att_surfs[..., 1].flatten()
    X = X[np.isfinite(X)]
    Y = Y[np.isfinite(Y)]
    (x_min, x_max) = X.min(), X.max()
    (y_min, y_max) = Y.min(), Y.max()

    log_scale = log_scale if log_scale is not None else []
    if 0 in log_scale:
        x_min, x_max = np.log(x_min), np.log(x_max)
        x_min -= 0.1 * (x_max - x_min)
        x_max += 0.1 * (x_max - x_min)
        x_min, x_max = np.exp(x_min), np.exp(x_max)
    else:
        x_min -= 0.1 * (x_max - x_min)
        x_max += 0.1 * (x_max - x_min)
    if 1 in log_scale:
        y_min, y_max = np.log(y_min), np.log(y_max)
        y_min -= 0.1 * (y_max - y_min)
        y_max += 0.1 * (y_max - y_min)
        y_min, y_max = np.exp(y_min), np.exp(y_max)
    else:
        y_min -= 0.1 * (y_max - y_min)
        y_max += 0.1 * (y_max - y_min)

    emp_att_surfs[..., 0][emp_att_surfs[..., 0] == -np.inf] = x_min
    emp_att_surfs[..., 0][emp_att_surfs[..., 0] == np.inf] = x_max
    emp_att_surfs[..., 1][emp_att_surfs[..., 1] == -np.inf] = y_min
    emp_att_surfs[..., 1][emp_att_surfs[..., 1] == np.inf] = y_max

    return emp_att_surfs, x_min, x_max, y_min, y_max


def _step_direction(larger_is_better_objectives: Optional[List[int]]) -> str:
    """
    Check here:
        https://matplotlib.org/stable/gallery/lines_bars_and_markers/step_demo.html#sphx-glr-gallery-lines-bars-and-markers-step-demo-py

    min x min (post)
        o...       R
           :
           o...
              :
              o

    max x max (pre)
        o
        :
        ...o
           :
    R      ...o

    min x max (post)
              o
              :
           o...
           :
        o...       R

    max x min (pre)
    R      ...o
           :
        ...o
        :
        o
    """
    if larger_is_better_objectives is None:
        larger_is_better_objectives = []

    large_f1_is_better = bool(0 in larger_is_better_objectives)
    return "pre" if large_f1_is_better else "post"


def plot_surface(
    ax: plt.Axes,
    emp_att_surfs: np.ndarray,
    colors: List[str],
    labels: List[str],
    larger_is_better_objectives: Optional[List[int]] = None,
    log_scale: Optional[List[int]] = None,
    **kwargs: Any,
) -> List[Any]:
    """
    Plot multiple surfaces.

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
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.
        kwargs:
            The kwargs for scatter.
    """
    emp_att_surfs, x_min, x_max, y_min, y_max = _transform_attainment_surface(emp_att_surfs, log_scale)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    step_dir = _step_direction(larger_is_better_objectives)

    kwargs.update(drawstyle=f"steps-{step_dir}")
    lines = []
    for color, label, emp_att_surf in zip(colors, labels, emp_att_surfs):
        X = emp_att_surf[:, 0]
        Y = emp_att_surf[:, 1]
        asc = X[0] <= X[-1]
        X = X if asc else X[::-1]
        Y = Y if asc else Y[::-1]
        line, = ax.plot(X, Y, color=color, label=label, **kwargs)

        lines.append(line)

    return lines


def plot_surface_with_band(
    ax: plt.Axes,
    emp_att_surfs: np.ndarray,
    color: str,
    label: str,
    larger_is_better_objectives: Optional[List[int]] = None,
    log_scale: Optional[List[int]] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot the surface with a band.
    Typically, we would like to plot median with the band between
    25% -- 75% percentile attainment surfaces.

    Args:
        ax (plt.Axes):
            The subplots axes.
        emp_att_surfs (np.ndarray):
            The vertices of the empirical attainment surfaces for each level.
            If emp_att_surf[i, j, 1] takes np.inf, this is not actually on the surface.
            The shape is (3, X.size, 2).
        colors (str):
            The color of the plot
        label (str):
            The label of the plot.
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.
        kwargs:
            The kwargs for scatter.
    """
    if emp_att_surfs.shape[0] != 3:
        raise ValueError(f"plot_surface_with_band requires three levels, but got only {emp_att_surfs.shape[0]} levels")

    emp_att_surfs, x_min, x_max, y_min, y_max = _transform_attainment_surface(emp_att_surfs, log_scale)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    dx = emp_att_surfs[0, :, 0]
    asc = dx[0] <= dx[-1]

    q0 = emp_att_surfs[0, :, 1] if asc else emp_att_surfs[0, :, 1][::-1]
    m = emp_att_surfs[1, :, 1] if asc else emp_att_surfs[1, :, 1][::-1]
    q1 = emp_att_surfs[-1, :, 1] if asc else emp_att_surfs[-1, :, 1][::-1]
    step_dir = _step_direction(larger_is_better_objectives)

    line, = ax.plot(dx, m, color=color, label=label, drawstyle=f"steps-{step_dir}", **kwargs)
    ax.fill_between(dx, q0, q1, color=color, alpha=0.2, step=step_dir, **kwargs)
    return line
