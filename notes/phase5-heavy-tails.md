# Phase 5 — Heavy Tails and Extreme Value Theory: Theory Notes

## Overview

This is the article's practical climax. We formalize what "heavy-tailed" means,
derive tail behaviour for our candidate distributions, introduce EVT basics,
and quantify the budget impact of underestimating tail risk.

**Key message:** If you use a Normal distribution for severance costs, you are
systematically under-reserving.

---

## 1. Formal Definitions of Tail Weight

### Light-Tailed Distributions

A distribution is **light-tailed** if its moment generating function exists
for some $t > 0$:

$$M_X(t) = E[e^{tX}] < \infty \quad \text{for some } t > 0$$

Equivalently: the survival function $\bar{F}(x) = P(X > x)$ decays at least
exponentially:

$$\bar{F}(x) \leq C \cdot e^{-\lambda x} \quad \text{for some } C, \lambda > 0$$

**Examples:** Normal, Gamma, Weibull (with $k \geq 1$), Exponential.

### Heavy-Tailed Distributions

A distribution is **heavy-tailed** if:

$$M_X(t) = E[e^{tX}] = \infty \quad \text{for all } t > 0$$

Equivalently: the survival function decays slower than any exponential:

$$\lim_{x \to \infty} e^{\lambda x} \bar{F}(x) = \infty \quad \text{for all } \lambda > 0$$

**Examples:** Pareto, Student-t, Cauchy.

### Sub-Exponential Distributions

A distribution is **sub-exponential** if it's heavy-tailed AND:

$$\lim_{x \to \infty} \frac{P(X_1 + X_2 > x)}{P(X_1 > x)} = 2$$

where $X_1, X_2$ are i.i.d. copies.

**Interpretation:** The sum exceeds a threshold primarily because ONE of the
two variables is large (not because both are moderately large). This is the
"one big jump" principle.

**Examples:** LogNormal, Pareto, Weibull with $k < 1$.

### Classification of Our Candidate Distributions

| Distribution | Tail Class | Decay Rate |
|-------------|-----------|------------|
| Normal | Light | Super-exponential: $e^{-x^2/(2\sigma^2)}$ |
| Gamma | Light | Exponential: $x^{\alpha-1}e^{-\beta x}$ |
| Weibull ($k \geq 1$) | Light | Stretched exponential: $e^{-(x/\lambda)^k}$ |
| LogNormal | Sub-exponential | $\frac{1}{x}e^{-(\log x)^2/(2\sigma^2)}$ |
| Pareto | Heavy (power-law) | Polynomial: $x^{-\alpha}$ |

---

## 2. Tail Behaviour Derivations

### 2.1 Normal Tail

For $X \sim N(0, 1)$, the Mills ratio bound:

$$\frac{1}{\sqrt{2\pi}} \cdot \frac{x}{x^2 + 1} \cdot e^{-x^2/2} \leq P(X > x) \leq \frac{1}{\sqrt{2\pi}} \cdot \frac{1}{x} \cdot e^{-x^2/2}$$

**Proof of upper bound** (integration by parts):

$$P(X > x) = \int_x^\infty \frac{1}{\sqrt{2\pi}} e^{-t^2/2} dt$$

Write $e^{-t^2/2} = (-1/t) \cdot d(e^{-t^2/2})/dt \cdot (-1/t)$... Actually, integrate by parts with $u = 1/t$, $dv = te^{-t^2/2}dt$:

$$\int_x^\infty e^{-t^2/2} dt = \int_x^\infty \frac{1}{t} \cdot t \cdot e^{-t^2/2} dt = \left[-\frac{1}{t}e^{-t^2/2}\right]_x^\infty - \int_x^\infty \frac{1}{t^2}e^{-t^2/2}dt$$

$$= \frac{1}{x}e^{-x^2/2} - \int_x^\infty \frac{1}{t^2}e^{-t^2/2}dt \leq \frac{1}{x}e^{-x^2/2}$$

Therefore: $P(X > x) \leq \frac{1}{\sqrt{2\pi}} \cdot \frac{1}{x} \cdot e^{-x^2/2}$. $\square$

**Key insight:** Normal tails decay as $e^{-x^2/2}$ — faster than ANY exponential.

### 2.2 Pareto Tail

For $X \sim \text{Pareto}(\alpha, x_m)$:

$$P(X > x) = \left(\frac{x_m}{x}\right)^\alpha, \quad x \geq x_m$$

This is an **exact** power law — no approximation needed.

**Tail ratio:**

$$\frac{P(X > 2c)}{P(X > c)} = \frac{(x_m/(2c))^\alpha}{(x_m/c)^\alpha} = 2^{-\alpha}$$

For $\alpha = 2.5$: ratio $= 2^{-2.5} \approx 0.177$.
Doubling the threshold only reduces probability by a factor of ~5.7.

**Compare with Normal** (for large $c$):

$$\frac{P(X > 2c)}{P(X > c)} \approx \frac{(1/2c)e^{-2c^2}}{(1/c)e^{-c^2/2}} = \frac{1}{2} e^{-3c^2/2}$$

At $c = 3\sigma$: this ratio $\approx e^{-13.5}/2 \approx 7 \times 10^{-7}$.

**Conclusion:** Pareto tails thin polynomially; Normal tails thin super-exponentially.

### 2.3 LogNormal Tail

For $X \sim \text{LogNormal}(\mu, \sigma^2)$:

$$P(X > x) = 1 - \Phi\left(\frac{\log x - \mu}{\sigma}\right) \sim \frac{\sigma}{(\log x - \mu)\sqrt{2\pi}} \exp\left(-\frac{(\log x - \mu)^2}{2\sigma^2}\right)$$

The LogNormal is between Normal and Pareto:
- Heavier than Normal (sub-exponential)
- Lighter than Pareto (not a power law)

---

## 3. The Tail Index

### Definition

For a heavy-tailed distribution with power-law tail:

$$P(X > x) \sim L(x) \cdot x^{-\alpha} \quad \text{as } x \to \infty$$

where $L(x)$ is a slowly varying function ($L(tx)/L(x) \to 1$ for all $t > 0$).

The parameter $\alpha > 0$ is the **tail index**:
- Smaller $\alpha$ = heavier tail = more extreme events
- $\alpha \leq 1$: infinite mean
- $\alpha \leq 2$: infinite variance
- Pareto has exact power law with $L(x) = x_m^\alpha$

### The Hill Estimator

**Setup:** Given order statistics $X_{(1)} \leq X_{(2)} \leq \ldots \leq X_{(n)}$,
estimate the tail index using the $k$ largest observations.

**Derivation:** If the tail above threshold $u = X_{(n-k)}$ follows a Pareto:

$$P(X > x \mid X > u) = (u/x)^\alpha$$

The MLE for $\alpha$ from the exceedances $X_{(n-k+1)}, \ldots, X_{(n)}$:

$$\hat{\alpha}_{Hill} = \left[\frac{1}{k}\sum_{i=1}^k \log\frac{X_{(n-i+1)}}{X_{(n-k)}}\right]^{-1}$$

**Interpretation:**
- Choose $k$ too small: high variance (few data points)
- Choose $k$ too large: bias (non-tail data included)
- Plot $\hat{\alpha}$ vs $k$ and look for a stable region

---

## 4. Extreme Value Theory Basics

### The Problem

Given $n$ i.i.d. observations $X_1, \ldots, X_n$, what is the distribution of
$M_n = \max(X_1, \ldots, X_n)$ as $n \to \infty$?

### Fisher-Tippett-Gnedenko Theorem (Statement)

If there exist normalizing sequences $a_n > 0$, $b_n$ such that:

$$\frac{M_n - b_n}{a_n} \xrightarrow{d} G$$

for some non-degenerate distribution $G$, then $G$ must be one of three types:

**Type I (Gumbel):** Light tails (Normal, Gamma, Exponential, LogNormal)

$$G(x) = \exp(-e^{-x}), \quad x \in \mathbb{R}$$

**Type II (Frechet):** Heavy tails (Pareto, Student-t)

$$G(x) = \begin{cases} 0 & x \leq 0 \\ \exp(-x^{-\alpha}) & x > 0 \end{cases}$$

**Type III (Weibull):** Bounded upper tail

$$G(x) = \begin{cases} \exp(-(-x)^\alpha) & x \leq 0 \\ 1 & x > 0 \end{cases}$$

### Generalized Extreme Value (GEV) Distribution

Unifies all three types:

$$G(x; \xi) = \exp\left(-(1 + \xi x)^{-1/\xi}\right)$$

- $\xi > 0$: Frechet domain (heavy tails) — this is where Pareto lives
- $\xi = 0$: Gumbel domain (light tails) — Normal, Gamma
- $\xi < 0$: Weibull domain (bounded)

### Relevance to Budget Modelling

If cost extremes follow the Frechet domain ($\xi > 0$), then:
- Block maxima grow without bound as sample grows
- The "worst case" is much worse than Normal models suggest
- Budget reserves based on Normal quantiles are inadequate

---

## 5. Budget Impact Quantification

### Value at Risk (VaR)

The VaR at confidence level $p$ is the $p$-th quantile:

$$\text{VaR}_p = F^{-1}(p)$$

For the Pareto: $\text{VaR}_p = x_m \cdot (1-p)^{-1/\alpha}$

For the Normal: $\text{VaR}_p = \mu + \sigma \cdot z_p$

### Expected Shortfall (ES / CVaR)

The expected loss given that it exceeds VaR:

$$\text{ES}_p = E[X \mid X > \text{VaR}_p]$$

For the Pareto (with $\alpha > 1$):

$$\text{ES}_p = \frac{\alpha}{\alpha - 1} \cdot \text{VaR}_p$$

For the Normal:

$$\text{ES}_p = \mu + \sigma \cdot \frac{\phi(z_p)}{1 - p}$$

### The Key Comparison

Pareto($\alpha = 2.5$, $x_m = 10000$) vs Normal (moment-matched):
- True mean: R\$16,667, true std: R\$14,907

| Metric | Pareto | Normal | Ratio |
|--------|--------|--------|-------|
| $P(X > 50000)$ | 1.79% | 0.013% | **138x** |
| $P(X > 100000)$ | 0.032% | $\approx 0$ | **>1000x** |
| VaR(95%) | R\$35,566 | R\$41,190 | 0.86 |
| VaR(99%) | R\$63,096 | R\$51,363 | 1.23 |
| ES(95%) | R\$59,277 | R\$46,447 | 1.28 |
| ES(99%) | R\$105,160 | R\$55,297 | **1.90** |

The Normal **catastrophically** underestimates the probability of extreme events
and underestimates ES at high confidence levels.

---

## Numerical Example: The 10x Underestimation

Severance costs: Pareto($\alpha = 2.5$, $x_m = 10000$)

An analyst incorrectly assumes Normal($\mu = 16667$, $\sigma = 14907$).

$$\frac{P_{\text{Pareto}}(X > 50000)}{P_{\text{Normal}}(X > 50000)} = \frac{(10000/50000)^{2.5}}{1 - \Phi((50000 - 16667)/14907)}$$

$$= \frac{0.0179}{0.000013} \approx 138$$

The Normal model says a R\$50,000 severance event happens once in 77,000 cases.
The Pareto model says it happens once in 56 cases. The budget reserve
implications are radically different.
