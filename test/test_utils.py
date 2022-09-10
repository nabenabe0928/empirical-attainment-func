import unittest

import matplotlib.pyplot as plt

import pytest

import numpy as np

from eaf import plot_surface, plot_surface_with_band
from eaf.utils import _check_surface, _step_direction


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
