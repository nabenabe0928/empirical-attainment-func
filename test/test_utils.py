import unittest

import matplotlib.pyplot as plt

import pytest

import numpy as np

from eaf import plot_surface, plot_surface_with_band
from eaf.utils import LOGEPS, _check_surface, _step_direction, _transform_attainment_surface


def test_transform_attainment_surface() -> None:
    emp_att_surfs = np.array(
        [
            [-np.inf, 1, 2, 3, 4, np.inf],
            [np.inf, 8, 7, 6, 5, -np.inf],
        ]
    ).T
    sol, _, _, _, _ = _transform_attainment_surface(emp_att_surfs, log_scale=None)
    ans = np.array([[0.7, 1, 2, 3, 4, 4.33], [8.33, 8, 7, 6, 5, 4.7]]).T

    emp_att_surfs = np.array(
        [
            [LOGEPS, 1e-4, 1e-3, 1e-2, 1e-1, np.inf],
            [np.inf, 1e-5, 1e-6, 1e-7, 1e-8, LOGEPS],
        ]
    ).T
    ans = np.array(
        [
            [5.01187234e-05, 1e-4, 1e-3, 1e-2, 1e-1, 2.13796209e-01],
            [2.13796209e-05, 1e-5, 1e-6, 1e-7, 1e-8, 5.01187234e-09],
        ]
    ).T
    sol, _, _, _, _ = _transform_attainment_surface(emp_att_surfs, log_scale=[0, 1])
    assert np.allclose(sol, ans)


def test_step_direction() -> None:
    larger_is_better_objectives = None
    assert _step_direction(larger_is_better_objectives) == "post"
    larger_is_better_objectives = []
    assert _step_direction(larger_is_better_objectives) == "post"
    larger_is_better_objectives = [1]
    assert _step_direction(larger_is_better_objectives) == "post"
    larger_is_better_objectives = [0]
    assert _step_direction(larger_is_better_objectives) == "pre"
    larger_is_better_objectives = [0, 1]
    assert _step_direction(larger_is_better_objectives) == "pre"


def test_check_surface():
    surf = np.arange(4 * 3 * 2).reshape((4, 3, 2))
    with pytest.raises(ValueError):
        _check_surface(surf)

    surf = np.arange(3 * 2).reshape((3, 2))
    _check_surface(surf)

    surf = np.random.random((30, 2))
    with pytest.raises(ValueError):
        _check_surface(surf)

    _, ax = plt.subplots()
    with pytest.raises(ValueError):
        plot_surface(ax, np.random.random((10, 30, 2)), colors=["red"] * 10, labels=["dummy"] * 10)
    with pytest.raises(ValueError):
        plot_surface_with_band(ax, np.random.random((3, 30, 2)), color="red", label="dummy")

    plt.close()


if __name__ == "__main__":
    unittest.main()
