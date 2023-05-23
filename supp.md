# Supplementary information
## Preliminaries
1. Define a multi-output function as $f: \mathbb{R}^D \rightarrow \mathbb{R}^M$,
2. Assume we run $N$ independent optimization runs and obtain the Pareto sets for each run $\forall i \in \\{1,\dots,M\\}, \mathcal{F}_i \subseteq \mathbb{R}^M$,
3. Define the objective vector $\boldsymbol{f} \in \mathbb{R}^M$ weakly dominates a vector $\boldsymbol{y}$ in the objective space if and only if $\forall m \in \\{1,\dots,M\\}, f_m \leq y_m$ and notate it as $\boldsymbol{f} \preceq \boldsymbol{y}$, and
4. Define a set of objective vectors $F$ weakly dominates  a vector $\boldsymbol{y}$ in the objective space if and only if $\exists \boldsymbol{f} \in F, \boldsymbol{f} \leq \boldsymbol{y}$ and notate it as $F \preceq \boldsymbol{y}$


## Attainment surface

As seen in the figure below, the **attainment surface** is the surface in the objective space that we can obtain by splitting the objective space like a step function by the Pareto front solutions yielded during the optimization.

It is simple to obtain the attainment surface if we have only one experiment;
however, it is hard to show the aggregated results from multiple experiments.
To address this issue, we use the $k$% attainment surface.

![Conceptual visualization of the attainment surface](figs/attainment-surface.png)

Credit: Figure 4. in [Indicator-Based Evolutionary Algorithm with Hypervolume Approximation by Achievement Scalarizing Functions](https://dl.acm.org/doi/pdf/10.1145/1830483.1830578?casa_token=wAx-0-6HgLYAAAAA:LTZmyz4H20nnS9aaTJhQA84UejRISpWK_iCkl33LIT2ER6higBIahESB3x9-yZEq8jVkR9BzSjzMPQ).

## $k$% attainment surface
First, we define the following empirical attainment function:

$$
\alpha(\boldsymbol{y}) := \alpha ( \boldsymbol{y} |  \mathcal{F}\_{1} , \dots , \mathcal{F}\_{N})  = \frac{1}{N} \sum_{{n=1}}^{N} \mathbb{I} [ \mathcal{F}_{n} \preceq \boldsymbol{y} ] .
$$

The $k$% attainment surface is the attainment surface such that it is achieved by $k$% of independent runs and more formally, it is defined as:

$S = \biggl\\{\boldsymbol{y}\mid\alpha(\boldsymbol{y}) \geq \frac{k}{100}\biggr\\}.$

Note that as we only have $N$ independent runs, we define a control parameter **level** ( $1 \leq L \leq N$ ) and obtain the following set:

$S_L = \biggl\\{\boldsymbol{y}\mid\alpha(\boldsymbol{y}) \geq \frac{L}{N}\biggr\\}.$

The best, median, and worst attainment surfaces could be fetched by $\\{1/N,1/2,1\\}\times 100$% attainment surface, respectively.

Please check the following references for more details:

[2] [An Approach to Visualizing the 3D Empirical Attainment Function](https://dl.acm.org/doi/pdf/10.1145/2464576.2482716?casa_token=b9vWo8MI3i8AAAAA:4UaDmmM1YgQFVo-vEQdNKvk9-12RTT8sO7n16CQIvneP_J33w_eGo2wYhfphwufqY5OcYPYj_Gc3mA)

[3] [On the Computation of the Empirical Attainment Function](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.705.1929&rep=rep1&type=pdf)
