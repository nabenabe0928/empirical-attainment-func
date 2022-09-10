from typing import Any, List, Optional

from eaf.utils import _check_surface, _step_direction, _transform_attainment_surface

import matplotlib.pyplot as plt

import numpy as np


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
            The shape is (n_surfaces, X.size, 2).
        colors (List[str]):
            The colors of each plot
        labels (List[str]):
            The labels of each plot.
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.
        log_scale (Optional[List[int]]):
            The indices of the log scale.
            For example, if you would like to plot the first objective in the log scale,
            you need to feed log_scale=[0].
            In principle, log_scale changes the minimum value of the axes
            from -np.inf to a small positive value.
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
        _check_surface(emp_att_surf)
        (line,) = ax.plot(emp_att_surf[..., 0], emp_att_surf[..., 1], color=color, label=label, **kwargs)

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
        log_scale (Optional[List[int]]):
            The indices of the log scale.
            For example, if you would like to plot the first objective in the log scale,
            you need to feed log_scale=[0].
            In principle, log_scale changes the minimum value of the axes
            from -np.inf to a small positive value.
        kwargs:
            The kwargs for scatter.
    """
    if emp_att_surfs.shape[0] != 3:
        raise ValueError(f"plot_surface_with_band requires three levels, but got only {emp_att_surfs.shape[0]} levels")

    emp_att_surfs, x_min, x_max, y_min, y_max = _transform_attainment_surface(emp_att_surfs, log_scale)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    _check_surface(emp_att_surfs[0])
    surf_lower = emp_att_surfs[0]
    _check_surface(emp_att_surfs[1])
    surf_med = emp_att_surfs[1]
    _check_surface(emp_att_surfs[2])
    surf_upper = emp_att_surfs[2]

    X = surf_lower[:, 0]
    step_dir = _step_direction(larger_is_better_objectives)

    (line,) = ax.plot(X, surf_med[:, 1], color=color, label=label, drawstyle=f"steps-{step_dir}", **kwargs)
    ax.fill_between(X, surf_lower[:, 1], surf_upper[:, 1], color=color, alpha=0.2, step=step_dir, **kwargs)
    return line
