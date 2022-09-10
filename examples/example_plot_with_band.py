import matplotlib.pyplot as plt

import numpy as np

from eaf import get_empirical_attainment_surface, plot_surface, plot_surface_with_band
from examples.toy_func import func


if __name__ == "__main__":
    dim, n_samples, n_independent_runs = 2, 100, 50
    X = np.random.random((n_independent_runs, n_samples, dim)) * 10 - 5
    costs = func(X)

    levels = [n_independent_runs // 4, n_independent_runs // 2, 3 * n_independent_runs // 4]
    emp_att_surfs = get_empirical_attainment_surface(costs=costs, levels=levels)

    _, ax = plt.subplots()
    plot_surface_with_band(ax, color="red", label="random", emp_att_surfs=emp_att_surfs)
    plot_surface(ax, colors=["blue"]*3, labels=["low", "med", "high"], emp_att_surfs=emp_att_surfs)
    ax.legend()
    ax.grid()
    plt.show()
