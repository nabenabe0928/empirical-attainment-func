from typing import Any, List, Optional, Union

from eaf.utils import _check_surface, _step_direction, _transform_attainment_surface, _transform_surface_list

import matplotlib.pyplot as plt

import numpy as np


def _change_scale(ax: plt.Axes, log_scale: Optional[List[int]]) -> None:
    log_scale = [] if log_scale is None else log_scale
    if 0 in log_scale:
        ax.set_xscale("log")
    if 1 in log_scale:
        ax.set_yscale("log")


def plot_surface(
    ax: plt.Axes,
    emp_att_surf: np.ndarray,
    color: str,
    label: str,
    larger_is_better_objectives: Optional[List[int]] = None,
    log_scale: Optional[List[int]] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot multiple surfaces.

    Args:
        ax (plt.Axes):
            The subplots axes.
        emp_att_surf (np.ndarray):
            The vertices of the empirical attainment surface.
            The shape must be (X.size, 2).
        color (str):
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
    if len(emp_att_surf.shape) != 2 or emp_att_surf.shape[1] != 2:
        raise ValueError(f"The shape of emp_att_surf must be (n_points, 2), but got {emp_att_surf.shape}")

    emp_att_surf, x_min, x_max, y_min, y_max = _transform_attainment_surface(emp_att_surf, log_scale)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    step_dir = _step_direction(larger_is_better_objectives)

    kwargs.update(drawstyle=f"steps-{step_dir}")
    _check_surface(emp_att_surf)
    X, Y = emp_att_surf[:, 0], emp_att_surf[:, 1]
    line = ax.plot(X, Y, color=color, label=label, **kwargs)
    _change_scale(ax, log_scale)
    return line


def plot_multiple_surface(
    ax: plt.Axes,
    emp_att_surfs_list: Union[np.ndarray, List[np.ndarray]],
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
        emp_att_surfs_list (Union[np.ndarray, List[np.ndarray]]):
            The vertices of the empirical attainment surfaces for each plot.
            Each element should have the shape of (X.size, 2).
            If this is an array, then the shape must be (n_surf, X.size, 2).
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
    plot_kwargs = dict(
        larger_is_better_objectives=larger_is_better_objectives,
        log_scale=log_scale,
    )
    lines: List[Any] = []
    emp_att_surfs_list, X_min, X_max, Y_min, Y_max = _transform_surface_list(emp_att_surfs_list, log_scale)
    for surf, color, label in zip(emp_att_surfs_list, colors, labels):
        line = plot_surface(ax, surf, color, label, **plot_kwargs)
        lines.append(line)

    ax.set_xlim(X_min, X_max)
    ax.set_ylim(Y_min, Y_max)
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
    _change_scale(ax, log_scale)
    return line


def plot_multiple_surface_with_band(
    ax: plt.Axes,
    emp_att_surfs_list: Union[np.ndarray, List[np.ndarray]],
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
        emp_att_surfs_list (Union[np.ndarray, List[np.ndarray]]):
            The vertices of the empirical attainment surfaces for each plot.
            Each element should have the shape of (3, X.size, 2).
            If this is an array, then the shape must be (n_surf, 3, X.size, 2).
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
    plot_kwargs = dict(
        larger_is_better_objectives=larger_is_better_objectives,
        log_scale=log_scale,
    )
    lines: List[Any] = []
    emp_att_surfs_list, X_min, X_max, Y_min, Y_max = _transform_surface_list(emp_att_surfs_list, log_scale)
    for surf, color, label in zip(emp_att_surfs_list, colors, labels):
        line = plot_surface_with_band(ax, surf, color, label, **plot_kwargs)
        lines.append(line)

    ax.set_xlim(X_min, X_max)
    ax.set_ylim(Y_min, Y_max)
    return lines
