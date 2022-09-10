from typing import List, Optional, Tuple

import numpy as np


LOGEPS = 1e-300


def _transform_attainment_surface(
    emp_att_surfs: np.ndarray,
    log_scale: Optional[List[int]],
) -> Tuple[np.ndarray, float, float, float, float]:
    X = emp_att_surfs[..., 0].flatten()
    Y = emp_att_surfs[..., 1].flatten()
    log_scale = log_scale if log_scale is not None else []
    x_is_log, y_is_log = 0 in log_scale, 1 in log_scale

    X = X[np.isfinite(X) & (X > LOGEPS) if x_is_log else np.isfinite(X)]
    Y = Y[np.isfinite(Y) & (Y > LOGEPS) if y_is_log else np.isfinite(Y)]
    (x_min, x_max) = (np.log(X.min()), np.log(X.max())) if x_is_log else (X.min(), X.max())
    (y_min, y_max) = (np.log(Y.min()), np.log(Y.max())) if y_is_log else (Y.min(), Y.max())

    x_min -= 0.1 * (x_max - x_min)
    x_max += 0.1 * (x_max - x_min)
    y_min -= 0.1 * (y_max - y_min)
    y_max += 0.1 * (y_max - y_min)
    (x_min, x_max) = (np.exp(x_min), np.exp(x_max)) if x_is_log else (x_min, x_max)
    (y_min, y_max) = (np.exp(y_min), np.exp(y_max)) if y_is_log else (y_min, y_max)

    lb = LOGEPS if x_is_log else -np.inf
    emp_att_surfs[..., 0][emp_att_surfs[..., 0] == lb] = x_min
    emp_att_surfs[..., 0][emp_att_surfs[..., 0] == np.inf] = x_max

    lb = LOGEPS if y_is_log else -np.inf
    emp_att_surfs[..., 1][emp_att_surfs[..., 1] == LOGEPS] = y_min
    emp_att_surfs[..., 1][emp_att_surfs[..., 1] == np.inf] = y_max

    return emp_att_surfs, x_min, x_max, y_min, y_max


def _transform_surface_list(
    emp_att_surfs_list: List[np.ndarray],
    log_scale: Optional[List[int]],
) -> Tuple[List[np.ndarray], float, float, float, float]:
    X_min, X_max, Y_min, Y_max = np.inf, -np.inf, np.inf, -np.inf
    for surf in emp_att_surfs_list:
        _, x_min, x_max, y_min, y_max = _transform_attainment_surface(surf.copy(), log_scale)
        X_min, X_max, Y_min, Y_max = min(X_min, x_min), max(X_max, x_max), min(Y_min, y_min), max(Y_max, y_max)

    log_scale = [] if log_scale is None else log_scale
    x_is_log, y_is_log = 0 in log_scale, 1 in log_scale
    for surf in emp_att_surfs_list:
        lb = LOGEPS if x_is_log else -np.inf
        surf[..., 0][surf[..., 0] == lb] = X_min
        surf[..., 0][surf[..., 0] == np.inf] = X_max

        lb = LOGEPS if y_is_log else -np.inf
        surf[..., 1][surf[..., 1] == LOGEPS] = Y_min
        surf[..., 1][surf[..., 1] == np.inf] = Y_max

    return emp_att_surfs_list, X_min, X_max, Y_min, Y_max


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
