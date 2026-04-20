# Exercises — Phase 5: Heavy Tails and Extreme Value Theory

## Proofs (paper)

### Exercise 1

**Prove** that the Pareto($\alpha$, $x_m$) distribution is heavy-tailed by
showing that $M_X(t) = E[e^{tX}] = \infty$ for all $t > 0$.

*Contrast with the Normal, where $M_X(t) < \infty$ for all $t$.*

---

### Exercise 2

**Prove** that the tail of the Normal distribution satisfies the bound:

$$\frac{1}{\sqrt{2\pi}} \cdot \frac{x}{x^2 + 1} \cdot e^{-x^2/2} \leq P(X > x) \leq \frac{1}{\sqrt{2\pi}} \cdot \frac{1}{x} \cdot e^{-x^2/2}$$

*Use integration by parts on $\int_x^\infty e^{-t^2/2} dt$.* What does this say
about how fast Normal tails decay?

---

### Exercise 3

**Show** that for $X \sim \text{Pareto}(\alpha, x_m)$ with $\alpha > 2$:

$$\frac{P(X > 2c)}{P(X > c)} = 2^{-\alpha}$$

and for $X \sim N(\mu, \sigma^2)$ the same ratio decays exponentially.

*This ratio captures "how quickly the tail thins." Compute both for $c = 3\sigma$
and compare.*

---

### Exercise 4

**Derive** the Hill estimator
$\hat{\alpha}_{Hill} = \left[\frac{1}{k} \sum_{i=1}^k \log \frac{X_{(n-i+1)}}{X_{(n-k)}}\right]^{-1}$
as the MLE of the tail index for the Pareto model fitted to the $k$ largest
order statistics.

*State the assumption: the tail is approximately Pareto above a threshold.*

---

## Computations (paper)

### Exercise 5

Severance costs follow $\text{Pareto}(\alpha = 2.5, x_m = 10{,}000)$. An analyst
fits $N(\mu, \sigma^2)$ matching the first two moments:
$\mu = E[X] = 16{,}667$, $\sigma^2 = \text{Var}(X)$.

Compare $P(X > 50{,}000)$ and $P(X > 100{,}000)$ under both models.

*By what factor does the Normal underestimate tail risk?*

---

### Exercise 6

For the same Pareto model, compute the Value at Risk at 95% and 99%:
$\text{VaR}_p = x_m \cdot (1 - p)^{-1/\alpha}$.

Compare with the Normal VaR at the same levels.

*What is the ratio of Pareto VaR to Normal VaR?*

---

### Exercise 7

You have 500 overtime cost observations. The 20 largest values are (in R\$):
{8200, 8500, 9100, 9800, 10200, 10900, 11500, 12800, 14200, 15100, 16500,
18200, 19800, 22000, 25100, 29000, 34500, 42000, 58000, 91000}.

Compute the Hill estimator for $k = 5, 10, 15, 20$. Is there evidence of a
heavy tail?

*Plot $\hat{\alpha}$ vs $k$ (on paper) — a stable region suggests a valid tail index.*
