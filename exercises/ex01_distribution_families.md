# Exercises — Phase 1: Distribution Families

## Proofs (paper)

### Exercise 1

**Derive** $E[X]$ and $\text{Var}(X)$ for $X \sim \text{LogNormal}(\mu, \sigma^2)$
starting from the definition $X = e^Y$ where $Y \sim N(\mu, \sigma^2)$.

*Use the MGF of the Normal distribution.*

---

### Exercise 2

**Derive** the skewness of the LogNormal distribution:

$$\gamma_1 = (e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}$$

*You will need $E[X^3]$.*

---

### Exercise 3

**Prove** that the Gamma distribution with $\alpha = 1$ reduces to the
Exponential distribution. Then prove that the sum of $n$ i.i.d.
Exponential($\lambda$) random variables follows a Gamma($n, \lambda$)
distribution.

*Use MGFs.*

---

### Exercise 4

**Prove** that the Pareto distribution with shape parameter $\alpha \leq 2$
has infinite variance, and with $\alpha \leq 1$ has infinite mean.

*Evaluate the integrals directly and show divergence.*

---

### Exercise 5

**Derive** the kurtosis of the Normal distribution ($\kappa = 3$) and show
that excess kurtosis is zero. Then show that the LogNormal always has excess
kurtosis > 0.

*What does this mean for budget tail risk?*

---

## Computations (paper)

### Exercise 6

A company's salary data has sample mean R\$ 12,000 and sample variance
R\$ 9,000,000 (SD = R\$ 3,000). Assuming LogNormal: use the method of moments
to estimate $\mu$ and $\sigma^2$.

*Hint: solve $E[X] = e^{\mu + \sigma^2/2}$ and
$\text{Var}(X) = (e^{\sigma^2} - 1)e^{2\mu + \sigma^2}$.*

---

### Exercise 7

For $X \sim \text{Pareto}(\alpha = 3, x_m = 5000)$, compute $P(X > 20{,}000)$,
$P(X > 50{,}000)$, and $P(X > 100{,}000)$. Compare with the same tail
probabilities under $Y \sim N(7500, 2500^2)$ (matched mean).

*Which distribution assigns more weight to extreme severance costs?*

---

### Exercise 8

A salary distribution appears bimodal with peaks near R\$ 8,000 and R\$ 18,000.
If you fit a single Normal, what happens to the estimated variance? Why is this
misleading for budget risk?

*Think about what "average" means in a bimodal distribution.*
