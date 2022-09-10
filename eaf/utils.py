from typing import List, Optional, Tuple

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


def _check_surface(surf: np.ndarray) -> np.ndarray:
    if len(surf.shape) != 2:
        raise ValueError(f"The shape of surf must be (n_points, n_obj), but got {surf.shape}")

    X = surf[:, 0]
    if np.any(np.maximum.accumulate(X) != X):
        raise ValueError("The axis [:, 0] of surf must be an increasing sequence")


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
