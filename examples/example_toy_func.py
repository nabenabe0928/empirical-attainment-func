import matplotlib.pyplot as plt

import numpy as np

from eaf import get_empirical_attainment_surface
from examples.toy_func import func


if __name__ == "__main__":
    dim, n_samples, n_trials = 2, 300, 50
    X = np.random.random((n_trials, n_samples, dim)) * 10 - 5
    costs = func(X)

    label = "the best attainment"
    emp_att_surf = get_empirical_attainment_surface(costs=costs, level=1)
    plt.scatter(emp_att_surf[:, 0], emp_att_surf[:, 1], color="red", s=2, marker="*", label=label)

    label = "the median attainment"
    emp_att_surf = get_empirical_attainment_surface(costs=costs, level=n_trials//2)
    plt.scatter(emp_att_surf[:, 0], emp_att_surf[:, 1], color="blue", s=2, marker="*", label=label)

    label = "the worst attainment"
    emp_att_surf = get_empirical_attainment_surface(costs=costs, level=n_trials)
    plt.scatter(emp_att_surf[:, 0], emp_att_surf[:, 1], color="green", s=2, marker="*", label=label)
    plt.grid()
    plt.legend()
    plt.show()
