import unittest

import numpy as np

import pytest

from eaf.eaf import (
    _compute_emp_att_surf,
    _get_pf_set_list,
    get_empirical_attainment_surface,
)


def test_compute_emp_att_surf() -> None:
    pass


def test_get_pf_set_list() -> None:
    costs = np.array(
        [
            [[0, 0], [0, 0], [0, 0]],
            [[1, 0], [0, 1], [2, 2]],
        ]
    )
    pf_set_list = _get_pf_set_list(costs)
    assert len(pf_set_list) == 2
    assert np.all(pf_set_list[0] == np.array([[0, 0]]))
    assert np.all(pf_set_list[1] == np.array([[0, 1], [1, 0]]))


def test_get_empirical_attainment_surface() -> None:
    costs = np.random.random((2, 3))
    with pytest.raises(ValueError):
        get_empirical_attainment_surface(costs, level=1)

    costs = np.random.random((2, 3, 3))
    with pytest.raises(NotImplementedError):
        get_empirical_attainment_surface(costs, level=1)

    costs = np.random.random((2, 3, 2))
    for level in range(-1, 5):
        if 1 <= level <= 2:
            get_empirical_attainment_surface(costs, level=level)
        else:
            with pytest.raises(ValueError):
                get_empirical_attainment_surface(costs, level=level)


if __name__ == "__main__":
    unittest.main()
