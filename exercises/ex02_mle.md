# Exercises — Phase 2: Maximum Likelihood Estimation

## Proofs (paper)

### Exercise 1

**Derive the MLE** for $\mu$ and $\sigma^2$ in the Normal($\mu, \sigma^2$)
model. Show that $\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum(x_i - \bar{x})^2$
is biased.

*Compute $E[\hat{\sigma}^2_{MLE}]$ and show it equals $\frac{n-1}{n}\sigma^2$.*

---

### Exercise 2

**Derive the MLE** for $\mu$ and $\sigma^2$ in the LogNormal($\mu, \sigma^2$)
model.

*Hint: if $X \sim \text{LogNormal}$, then $\log X \sim \text{Normal}$.
Transform the data first.*

---

### Exercise 3

**Derive the Fisher information** for the Exponential($\lambda$) model. Then
verify the Cramér-Rao lower bound:
$\text{Var}(\hat{\lambda}_{MLE}) \geq 1/(n \cdot I(\lambda))$.

*Show the MLE achieves this bound (it is efficient).*

---

### Exercise 4

**Prove** that $E[S(\theta_0)] = 0$ where
$S(\theta) = \partial \ell / \partial \theta$ is the score function.

*Differentiate under the integral sign in $\int f(x|\theta) dx = 1$.*

---

### Exercise 5

**Prove** the equivalence of the two forms of Fisher information:
$I(\theta) = \text{Var}(S(\theta)) = -E[\partial^2 \ell / \partial \theta^2]$.

*Differentiate the score identity from exercise 4 again.*

---

### Exercise 6

**Derive the MLE** for the Pareto($\alpha, x_m$) distribution where $x_m$ is
known. Compute $I(\alpha)$ and the asymptotic distribution of
$\hat{\alpha}_{MLE}$.

---

## Computations (paper)

### Exercise 7

Given salary data (in thousands):
$\{8.2, 9.1, 7.5, 11.3, 10.8, 8.7, 9.5, 12.1, 8.9, 10.2\}$.

Compute the MLE for LogNormal parameters $\hat{\mu}$ and $\hat{\sigma}^2$.
Then compute 95% CI for each parameter using observed Fisher information.

---

### Exercise 8

You fit a Gamma($\alpha, \beta$) to overtime cost data and obtain
$\hat{\alpha} = 2.3$, $\hat{\beta} = 150$. The observed Fisher information
matrix is:

$$\hat{I} = \begin{pmatrix} 0.82 & -0.003 \\ -0.003 & 0.0001 \end{pmatrix}$$

Compute standard errors and 95% CIs for both parameters.

*Invert the information matrix.*
