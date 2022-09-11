from typing import Any, List, Optional, Union

from eaf.utils import (
    LOGEPS,
    _change_scale,
    _check_surface,
    _get_slighly_expanded_value_range,
    _step_direction,
    pareto_front_to_surface,
)

import matplotlib.pyplot as plt

import numpy as np


class EmpiricalAttainmentFuncPlot:
    """
    The class to plot empirical attainment function.

    Args:
        true_pareto_sols (Optional[np.ndarray]):
            The true Pareto solutions if available.
            It is used to compute the best values of each objective.
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.
        log_scale (Optional[List[int]]):
            The indices of the log scale.
            For example, if you would like to plot the first objective in the log scale,
            you need to feed log_scale=[0].
            In principle, log_scale changes the minimum value of the axes
            from -np.inf to a small positive value.
        x_min, x_max, y_min, y_max (float):
            The lower/upper bounds for each objective if available.
            It is used to fix the value ranges of each objective in plots.
    """

    def __init__(
        self,
        true_pareto_sols: Optional[np.ndarray] = None,
        larger_is_better_objectives: Optional[List[int]] = None,
        log_scale: Optional[List[int]] = None,
        x_min: float = np.inf,
        x_max: float = -np.inf,
        y_min: float = np.inf,
        y_max: float = -np.inf,
    ):
        self.step_dir = _step_direction(larger_is_better_objectives)
        self.larger_is_better_objectives = (
            larger_is_better_objectives if larger_is_better_objectives is not None else []
        )
        self.log_scale = log_scale if log_scale is not None else []
        self.x_is_log, self.y_is_log = 0 in self.log_scale, 1 in self.log_scale
        self._plot_kwargs = dict(
            larger_is_better_objectives=larger_is_better_objectives,
            log_scale=log_scale,
        )

        if true_pareto_sols is not None:
            self.x_min, self.x_max, self.y_min, self.y_max = _get_slighly_expanded_value_range(
                true_pareto_sols, self.log_scale
            )
            self._true_pareto_sols = true_pareto_sols.copy()
        else:
            # We cannot plot until we call _transform_surface_list
            self._true_pareto_sols = None
            self.x_min = max(LOGEPS, x_min) if 0 in self.log_scale else x_min
            self.x_max = x_max
            self.y_min = max(LOGEPS, y_min) if 1 in self.log_scale else y_min
            self.y_max = y_max

    def _transform_surface_list(self, surfs_list: List[np.ndarray]) -> List[np.ndarray]:
        for surf in surfs_list:
            x_min, x_max, y_min, y_max = _get_slighly_expanded_value_range(surf, self.log_scale)
            self.x_min, self.x_max = min(self.x_min, x_min), max(self.x_max, x_max)
            self.y_min, self.y_max = min(self.y_min, y_min), max(self.y_max, y_max)

        for surf in surfs_list:
            lb = LOGEPS if self.x_is_log else -np.inf
            surf[..., 0][surf[..., 0] == lb] = self.x_min
            surf[..., 0][surf[..., 0] == np.inf] = self.x_max

            lb = LOGEPS if self.y_is_log else -np.inf
            surf[..., 1][surf[..., 1] == lb] = self.y_min
            surf[..., 1][surf[..., 1] == np.inf] = self.y_max

        return surfs_list

    def plot_surface(
        self, ax: plt.Axes, surf: np.ndarray, color: str, label: str, transform: bool = True, **kwargs: Any
    ) -> Any:
        """
        Plot multiple surfaces.

        Args:
            ax (plt.Axes):
                The subplots axes.
            surf (np.ndarray):
                The vertices of the empirical attainment surface.
                The shape must be (X.size, 2).
            color (str):
                The color of the plot
            label (str):
                The label of the plot.
            kwargs:
                The kwargs for scatter.
        """
        if len(surf.shape) != 2 or surf.shape[1] != 2:
            raise ValueError(f"The shape of surf must be (n_points, 2), but got {surf.shape}")

        if transform:
            surf = self._transform_surface_list(surfs_list=[surf])[0]
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)

        kwargs.update(drawstyle=f"steps-{self.step_dir}")
        _check_surface(surf)
        X, Y = surf[:, 0], surf[:, 1]
        line = ax.plot(X, Y, color=color, label=label, **kwargs)
        _change_scale(ax, self.log_scale)
        return line

    def plot_true_pareto_surface(self, ax: plt.Axes, color: str, label: str, **kwargs: Any) -> None:
        """
        Plot multiple surfaces.

        Args:
            ax (plt.Axes):
                The subplots axes.
            color (str):
                The color of the plot
            label (str):
                The label of the plot.
            kwargs:
                The kwargs for scatter.
        """
        if self._true_pareto_sols is None:
            raise AttributeError("true_pareto_sols is not provided at the instantiation")

        true_pareto_surf = pareto_front_to_surface(
            self._true_pareto_sols.copy(),
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            **self._plot_kwargs,
        )
        self.plot_surface(ax, surf=true_pareto_surf, color=color, label=label, transform=False, **kwargs)

    def plot_multiple_surface(
        self,
        ax: plt.Axes,
        surfs: Union[np.ndarray, List[np.ndarray]],
        colors: List[str],
        labels: List[str],
        **kwargs: Any,
    ) -> List[Any]:
        """
        Plot multiple surfaces.

        Args:
            ax (plt.Axes):
                The subplots axes.
            surfs (Union[np.ndarray, List[np.ndarray]]):
                The vertices of the empirical attainment surfaces for each plot.
                Each element should have the shape of (X.size, 2).
                If this is an array, then the shape must be (n_surf, X.size, 2).
            colors (List[str]):
                The colors of each plot
            labels (List[str]):
                The labels of each plot.
            kwargs:
                The kwargs for scatter.
        """
        lines: List[Any] = []
        surfs = self._transform_surface_list(surfs)
        for surf, color, label in zip(surfs, colors, labels):
            line = self.plot_surface(ax, surf, color, label, transform=False, **kwargs)
            lines.append(line)

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        return lines

    def plot_surface_with_band(
        self,
        ax: plt.Axes,
        surfs: np.ndarray,
        color: str,
        label: str,
        transform: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Plot the surface with a band.
        Typically, we would like to plot median with the band between
        25% -- 75% percentile attainment surfaces.

        Args:
            ax (plt.Axes):
                The subplots axes.
            surfs (np.ndarray):
                The vertices of the empirical attainment surfaces for each level.
                If surf[i, j, 1] takes np.inf, this is not actually on the surface.
                The shape is (3, X.size, 2).
            colors (str):
                The color of the plot
            label (str):
                The label of the plot.
            kwargs:
                The kwargs for scatter.
        """
        if surfs.shape[0] != 3:
            raise ValueError(f"plot_surface_with_band requires three levels, but got only {surfs.shape[0]} levels")
        if transform:
            surfs = self._transform_surface_list(surfs_list=surfs)
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)

        for surf in surfs:
            _check_surface(surf)

        X = surfs[0, :, 0]
        line = ax.plot(X, surfs[1, :, 1], color=color, label=label, drawstyle=f"steps-{self.step_dir}", **kwargs)
        ax.fill_between(X, surfs[0, :, 1], surfs[2, :, 1], color=color, alpha=0.2, step=self.step_dir, **kwargs)
        _change_scale(ax, self.log_scale)
        return line

    def plot_multiple_surface_with_band(
        self,
        ax: plt.Axes,
        surfs_list: Union[np.ndarray, List[np.ndarray]],
        colors: List[str],
        labels: List[str],
        **kwargs: Any,
    ) -> List[Any]:
        """
        Plot multiple surfaces.

        Args:
            ax (plt.Axes):
                The subplots axes.
            surfs_list (Union[np.ndarray, List[np.ndarray]]):
                The vertices of the empirical attainment surfaces for each plot.
                Each element should have the shape of (3, X.size, 2).
                If this is an array, then the shape must be (n_surf, 3, X.size, 2).
            colors (List[str]):
                The colors of each plot
            labels (List[str]):
                The labels of each plot.
            kwargs:
                The kwargs for scatter.
        """
        lines: List[Any] = []
        surfs_list = self._transform_surface_list(surfs_list)
        for surf, color, label in zip(surfs_list, colors, labels):
            line = self.plot_surface_with_band(ax, surf, color, label, transform=False, **kwargs)
            lines.append(line)

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        return lines
