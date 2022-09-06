import matplotlib.pyplot as plt

import numpy as np

from eaf import get_empirical_attainment_surface, plot_surface
from examples.toy_func import func


if __name__ == "__main__":
    dim, n_samples, n_trials = 2, 200, 50
    X = np.random.random((n_trials, n_samples, dim)) * 10 - 5
    costs = func(X)

    labels = [f"the {feat} attainment" for feat in ["best", "median", "worst"]]
    levels = [1, n_trials // 2, n_trials]
    colors = ["red", "blue", "green"]
    emp_att_surfs = get_empirical_attainment_surface(costs=costs, levels=[1, n_trials // 2, n_trials])

    _, ax = plt.subplots()
    plot_surface(ax, colors=colors, labels=labels, emp_att_surfs=emp_att_surfs)
    ax.grid()
    plt.show()
