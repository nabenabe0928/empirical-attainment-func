# Empirical attainment function for the multi-objective optimization visualization

[![Build Status](https://github.com/nabenabe0928/empirical-attainment-func/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/empirical-attainment-func)
[![codecov](https://codecov.io/gh/nabenabe0928/empirical-attainment-func/branch/main/graph/badge.svg?token=P3MJPKA8H7)](https://codecov.io/gh/nabenabe0928/empirical-attainment-func)

## Motivation

When we run single-objective optimization problems, comparisons between multiple methods over multiple seeds are not so hard.
Although we could make scatter plots for multi-objective optimization tasks, such plots do not allow comparisons using multiple seeds.
In this repository, we would like to give the solution to this issue.

We use $k$% attainment surface or empirical attainment function.
The original paper is available below.

[1] [On the Performance Assessment and Comparison of Stochastic Multi-objective Optimizers](https://eden.dei.uc.pt/~cmfonsec/fonseca-ppsn1996-reprint.pdf)


## Setup & test

1. Install the package

```shell
$ pip install empirical-attainment-func
```

2. Save the following file (`run_test.py`)

```python
import matplotlib.pyplot as plt

import numpy as np

from eaf import get_empirical_attainment_surface, plot_surface


def func(X: np.ndarray) -> np.ndarray:
    f1 = np.sum(X**2, axis=-1)
    f2 = np.sum((X - 2) ** 2, axis=-1)
    return np.stack([f1, f2], axis=-1)


if __name__ == "__main__":
    dim, n_samples, n_trials = 2, 300, 50
    X = np.random.random((n_trials, n_samples, dim)) * 10 - 5
    costs = func(X)

    labels = [f"the {feat} attainment" for feat in ["best", "median", "worst"]]
    levels = [1, n_trials // 2, n_trials]
    colors = ["red", "blue", "green"]
    emp_att_surfs = get_empirical_attainment_surface(costs=costs, levels=[1, n_trials // 2, n_trials])

    _, ax = plt.subplots()
    plot_surface(ax, colors=colors, labels=labels, emp_att_surfs=emp_att_surfs, s=2, marker="*")
    ax.grid()
    plt.show()

```

3. Run the Python file

```shell
$ python run_test.py
```

## Usage

All you need is to feed `costs` to `get_empirical_attainment_surface`.
The arguments for this function are as follows:
1. `costs` (np.ndarray): The costs obtained in the observations and he shape must be `(n_independent_runs, n_samples, n_obj)`.
2. `levels` (List[int]): A list of level: level controls the k in the k-% attainment surface and k = level / n_independent_runs. (For more details, see $k$% attainment surface Section below)
3. `larger_is_better_objectives` (Optional[List[int]]): The indices of the objectives that are better when the values are larger. If None, we consider all objectives are better when they are smaller.

Note that we currently support only `n_obj=2`.

## Preliminaries
1. Define a multi-output function as $f: \mathbb{R}^D \rightarrow \mathbb{R}^M$,
2. Assume we run $N$ independent optimization runs and obtain the Pareto sets for each run $\forall i \in \\{1,\dots,M\\}, \mathcal{F}_i \subseteq \mathbb{R}^M$,
3. Define the objective vector $\boldsymbol{f} \in \mathbb{R}^M$ weakly dominates a vector $\boldsymbol{y}$ in the objective space if and only if $\forall m \in \\{1,\dots,M\\}, f_m \leq y_m$ and notate it as $\boldsymbol{f} \preceq \boldsymbol{y}$, and
4. Define a set of objective vectors $F$ weakly dominates  a vector $\boldsymbol{y}$ in the objective space if and only if $\exists \boldsymbol{f} \in F, \boldsymbol{f} \leq \boldsymbol{y}$ and notate it as $F \preceq \boldsymbol{y}$


## Attainment surface

As seen in the figure below, the **attainment surface** is the surface in the objective space such that we can obtain by splitting the objective space like a step function by the Pareto front solutions yielded during the optimization.

It is simple to obtain the attainment surface if we have only one experiment;
however, it is hard to show the aggregated results from multiple experiments.
To address this issue, we use $k$% attainment surface.

![Conceptual visualization of the attainment surface](figs/attainment-surface.png)

Credit: Figure 4. in [Indicator-Based Evolutionary Algorithm with Hypervolume Approximation by Achievement Scalarizing Functions](https://dl.acm.org/doi/pdf/10.1145/1830483.1830578?casa_token=wAx-0-6HgLYAAAAA:LTZmyz4H20nnS9aaTJhQA84UejRISpWK_iCkl33LIT2ER6higBIahESB3x9-yZEq8jVkR9BzSjzMPQ).

## $k$% attainment surface
First, we define the following empirical attainment function:

$$
\alpha(\boldsymbol{y}) := \alpha ( \boldsymbol{y} |  \mathcal{F}_{1} , \dots , \mathcal{F}_{N})  = \frac{1}{N} \sum_{{n=1}}^{N} \mathbb{I} [ \mathcal{F}_{n} \preceq \boldsymbol{y} ] .
$$

The $k$% attainment surface is the attainment surface such that it is achieved by $k$% of independent runs and more formally, it is defined as:

$S = \biggl\\{\boldsymbol{y}\mid\alpha(\boldsymbol{y}) \geq \frac{k}{100}\biggr\\}.$

Note that as we only have $N$ independent runs, we define a control parameter **level** ($1 \leq L \leq N$) and obtain the following set:

$S_L = \biggl\\{\boldsymbol{y}\mid\alpha(\boldsymbol{y}) \geq \frac{L}{N}\biggr\\}.$

The best, median, worst attainment surfaces could be fetched by $\\{1/N,50,100\\}$% attainment surface, respectively.

Please check the following references for more details:

[2] [An Approach to Visualizing the 3D Empirical Attainment Function](https://dl.acm.org/doi/pdf/10.1145/2464576.2482716?casa_token=b9vWo8MI3i8AAAAA:4UaDmmM1YgQFVo-vEQdNKvk9-12RTT8sO7n16CQIvneP_J33w_eGo2wYhfphwufqY5OcYPYj_Gc3mA)

[3] [On the Computation of the Empirical Attainment Function](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.705.1929&rep=rep1&type=pdf)
