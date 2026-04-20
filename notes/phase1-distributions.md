# Phase 1 — Distribution Families: Theory Notes

## Overview

This document derives the mathematical properties of the five candidate
distribution families for people cost modelling. For each family we derive
the PDF, CDF, expected value, variance, MGF (where tractable), skewness,
kurtosis, and characterize the tail behaviour.

---

## 1. Normal Distribution — $X \sim N(\mu, \sigma^2)$

### PDF

$$f(x \mid \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \quad x \in (-\infty, \infty)$$

### CDF

$$F(x) = \Phi\left(\frac{x - \mu}{\sigma}\right) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x - \mu}{\sigma\sqrt{2}}\right)\right]$$

No closed form; defined via the standard Normal CDF $\Phi$.

### Derivation of $E[X]$ and $\text{Var}(X)$

**Expected value:**

$$E[X] = \int_{-\infty}^{\infty} x \cdot \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)} dx$$

Substitute $z = (x - \mu)/\sigma$, so $x = \mu + \sigma z$, $dx = \sigma \, dz$:

$$E[X] = \int_{-\infty}^{\infty} (\mu + \sigma z) \cdot \frac{1}{\sqrt{2\pi}} e^{-z^2/2} dz = \mu \cdot 1 + \sigma \cdot 0 = \mu$$

The second integral vanishes because $z \cdot e^{-z^2/2}$ is an odd function.

**Variance:**

$$\text{Var}(X) = E[(X - \mu)^2] = \int_{-\infty}^{\infty} (x - \mu)^2 \cdot \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)} dx$$

With the same substitution:

$$\text{Var}(X) = \sigma^2 \int_{-\infty}^{\infty} z^2 \cdot \frac{1}{\sqrt{2\pi}} e^{-z^2/2} dz = \sigma^2 \cdot 1 = \sigma^2$$

where $\int z^2 \phi(z) dz = 1$ by integration by parts.

### MGF

$$M_X(t) = E[e^{tX}] = \exp\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$$

**Derivation:** Complete the square in the exponent of the integral:

$$M_X(t) = \int_{-\infty}^{\infty} e^{tx} \cdot \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)} dx$$

Combine exponents: $tx - (x-\mu)^2/(2\sigma^2) = -\frac{1}{2\sigma^2}[(x - (\mu + \sigma^2 t))^2 - 2\mu\sigma^2 t - \sigma^4 t^2]$

Factor out constant terms: $M_X(t) = e^{\mu t + \sigma^2 t^2/2} \cdot 1$

### Shape Properties

| Property | Value |
|----------|-------|
| Skewness | $\gamma_1 = 0$ (symmetric) |
| Kurtosis | $\kappa = 3$ (mesokurtic) |
| Excess kurtosis | $\kappa - 3 = 0$ |
| Tail behaviour | Light-tailed: $P(X > x) \sim \frac{\sigma}{x\sqrt{2\pi}} e^{-x^2/(2\sigma^2)}$ |
| Support | $(-\infty, \infty)$ |

### Why It's Often Wrong for Cost Data

- Symmetric: cost data is typically right-skewed
- Light tails: underestimates extreme costs
- Unbounded below: assigns positive probability to negative costs
- Appropriate only when the Central Limit Theorem applies to aggregates

---

## 2. LogNormal Distribution — $X \sim \text{LogNormal}(\mu, \sigma^2)$

### Definition

If $Y \sim N(\mu, \sigma^2)$, then $X = e^Y \sim \text{LogNormal}(\mu, \sigma^2)$.

### PDF

$$f(x \mid \mu, \sigma^2) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0$$

**Derivation:** Apply the change-of-variables formula. If $Y = \ln X$, then
$f_X(x) = f_Y(\ln x) \cdot |dy/dx| = f_Y(\ln x) \cdot (1/x)$.

### CDF

$$F(x) = \Phi\left(\frac{\ln x - \mu}{\sigma}\right)$$

### Derivation of $E[X]$ and $\text{Var}(X)$

**Expected value:** Using the MGF of the Normal distribution for $Y$:

$$E[X] = E[e^Y] = M_Y(1) = e^{\mu + \sigma^2/2}$$

**Second moment:**

$$E[X^2] = E[e^{2Y}] = M_Y(2) = e^{2\mu + 2\sigma^2}$$

**Variance:**

$$\text{Var}(X) = E[X^2] - (E[X])^2 = e^{2\mu + 2\sigma^2} - e^{2\mu + \sigma^2} = e^{2\mu + \sigma^2}(e^{\sigma^2} - 1)$$

### MGF

The MGF of the LogNormal does **not** exist in closed form (the integral
$E[e^{tX}] = E[e^{te^Y}]$ diverges for all $t > 0$).

### Shape Properties

**Skewness:**

$$\gamma_1 = (e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}$$

**Derivation:** We need $E[X^3] = M_Y(3) = e^{3\mu + 9\sigma^2/2}$. Then:

$$\gamma_1 = \frac{E[(X - \mu_X)^3]}{[\text{Var}(X)]^{3/2}} = \frac{E[X^3] - 3E[X]E[X^2] + 2(E[X])^3}{[\text{Var}(X)]^{3/2}}$$

After algebraic simplification (substituting the moments):

$$\gamma_1 = (e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}$$

Always positive: the LogNormal is always right-skewed.

**Kurtosis:**

$$\kappa = e^{4\sigma^2} + 2e^{3\sigma^2} + 3e^{2\sigma^2} - 3$$

(excess kurtosis, always > 0: heavier tails than Normal)

| Property | Value |
|----------|-------|
| Skewness | $(e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1} > 0$ |
| Excess kurtosis | Always > 0 (leptokurtic) |
| Tail behaviour | Sub-exponential (between light and heavy) |
| Support | $(0, \infty)$ |

### Why It's Good for Salary Data

- Always positive (salaries > 0)
- Right-skewed (few high earners pull the mean above median)
- Multiplicative processes (salary = base × multipliers) produce LogNormal by CLT on log scale
- Median = $e^\mu$ (more interpretable than mean for skewed data)

---

## 3. Gamma Distribution — $X \sim \text{Gamma}(\alpha, \beta)$

### Parametrization

We use the shape-rate parametrization: $\alpha > 0$ (shape), $\beta > 0$ (rate),
so that the mean is $\alpha/\beta$.

### PDF

$$f(x \mid \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}, \quad x > 0$$

where $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t} dt$ is the Gamma function.

### CDF

$$F(x) = \frac{\gamma(\alpha, \beta x)}{\Gamma(\alpha)}$$

where $\gamma(\alpha, z) = \int_0^z t^{\alpha-1} e^{-t} dt$ is the lower incomplete Gamma function. No elementary closed form.

### Derivation of $E[X]$ and $\text{Var}(X)$

**Expected value:**

$$E[X] = \int_0^\infty x \cdot \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x} dx = \frac{\beta^\alpha}{\Gamma(\alpha)} \int_0^\infty x^\alpha e^{-\beta x} dx$$

Substitute $u = \beta x$:

$$E[X] = \frac{\beta^\alpha}{\Gamma(\alpha)} \cdot \frac{\Gamma(\alpha + 1)}{\beta^{\alpha+1}} = \frac{\Gamma(\alpha + 1)}{\beta \cdot \Gamma(\alpha)} = \frac{\alpha}{\beta}$$

using $\Gamma(\alpha + 1) = \alpha \cdot \Gamma(\alpha)$.

**Second moment:**

$$E[X^2] = \frac{\beta^\alpha}{\Gamma(\alpha)} \cdot \frac{\Gamma(\alpha + 2)}{\beta^{\alpha+2}} = \frac{\alpha(\alpha+1)}{\beta^2}$$

**Variance:**

$$\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{\alpha(\alpha+1)}{\beta^2} - \frac{\alpha^2}{\beta^2} = \frac{\alpha}{\beta^2}$$

### MGF

$$M_X(t) = \left(\frac{\beta}{\beta - t}\right)^\alpha, \quad t < \beta$$

**Derivation:**

$$M_X(t) = \frac{\beta^\alpha}{\Gamma(\alpha)} \int_0^\infty x^{\alpha-1} e^{-(\beta - t)x} dx = \frac{\beta^\alpha}{(\beta - t)^\alpha}$$

(the integral is $\Gamma(\alpha)/(\beta - t)^\alpha$ by definition of the Gamma function).

### Shape Properties

| Property | Value |
|----------|-------|
| Skewness | $\gamma_1 = 2/\sqrt{\alpha}$ |
| Excess kurtosis | $6/\alpha$ |
| Tail behaviour | Light-tailed (exponential decay) |
| Support | $(0, \infty)$ |

**Special cases:**
- $\alpha = 1$: Exponential($\beta$)
- $\alpha = n/2$, $\beta = 1/2$: Chi-squared($n$)
- $\alpha \to \infty$: approaches Normal (by CLT)

### Why It's Useful for Cost Components

- Always positive
- Flexible shape: from highly skewed ($\alpha < 1$) to nearly symmetric ($\alpha \gg 1$)
- Two parameters give independent control over mean and variance
- Natural model for "waiting time" costs (overtime accumulation)

---

## 4. Pareto Distribution — $X \sim \text{Pareto}(\alpha, x_m)$

### PDF

$$f(x \mid \alpha, x_m) = \frac{\alpha \, x_m^\alpha}{x^{\alpha + 1}}, \quad x \geq x_m$$

### CDF

$$F(x) = 1 - \left(\frac{x_m}{x}\right)^\alpha, \quad x \geq x_m$$

### Derivation of $E[X]$ and $\text{Var}(X)$

**Expected value** (requires $\alpha > 1$):

$$E[X] = \int_{x_m}^\infty x \cdot \frac{\alpha x_m^\alpha}{x^{\alpha+1}} dx = \alpha x_m^\alpha \int_{x_m}^\infty x^{-\alpha} dx$$

$$= \alpha x_m^\alpha \left[\frac{x^{-\alpha+1}}{-\alpha+1}\right]_{x_m}^\infty = \alpha x_m^\alpha \cdot \frac{x_m^{-\alpha+1}}{\alpha - 1} = \frac{\alpha \, x_m}{\alpha - 1}$$

For $\alpha \leq 1$, the integral diverges: **infinite mean**.

**Second moment** (requires $\alpha > 2$):

$$E[X^2] = \alpha x_m^\alpha \int_{x_m}^\infty x^{-\alpha+1} dx = \frac{\alpha \, x_m^2}{\alpha - 2}$$

**Variance** (requires $\alpha > 2$):

$$\text{Var}(X) = \frac{\alpha \, x_m^2}{\alpha - 2} - \left(\frac{\alpha \, x_m}{\alpha - 1}\right)^2 = \frac{\alpha \, x_m^2}{(\alpha-1)^2(\alpha-2)}$$

For $\alpha \leq 2$: **infinite variance** (finite mean if $\alpha > 1$).

### MGF

The MGF does **not** exist for the Pareto distribution ($E[e^{tX}] = \infty$
for all $t > 0$). This is a signature of heavy tails.

### Shape Properties

| Property | Value |
|----------|-------|
| Skewness | $\frac{2(1+\alpha)}{\alpha-3}\sqrt{\frac{\alpha-2}{\alpha}}$ (requires $\alpha > 3$) |
| Excess kurtosis | $\frac{6(\alpha^3 + \alpha^2 - 6\alpha - 2)}{\alpha(\alpha-3)(\alpha-4)}$ (requires $\alpha > 4$) |
| Tail behaviour | **Heavy-tailed**: $P(X > x) = (x_m/x)^\alpha$ (polynomial decay) |
| Support | $[x_m, \infty)$ |

### Key Property: Power-Law Tail

$$P(X > x) = \left(\frac{x_m}{x}\right)^\alpha$$

This decays **polynomially**, not exponentially. For $\alpha = 2.5$:
- $P(X > 2x_m) = 2^{-2.5} \approx 0.177$ (17.7%)
- $P(X > 10x_m) = 10^{-2.5} \approx 0.003$ (0.3%)

Compare with Normal: $P(X > \mu + 3\sigma) \approx 0.13\%$ — but Normal tails
drop off much faster beyond that point.

### Why It's Critical for Severance/Extreme Costs

- Models the "80/20 rule": most costs are moderate, but a few are enormous
- Captures the real-world phenomenon of unbounded maximum losses
- The tail index $\alpha$ directly controls how "wild" the extremes are
- Budget reserves must be much larger under Pareto than Normal assumptions

---

## 5. Weibull Distribution — $X \sim \text{Weibull}(k, \lambda)$

### Parametrization

$k > 0$ (shape), $\lambda > 0$ (scale).

### PDF

$$f(x \mid k, \lambda) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} \exp\left(-\left(\frac{x}{\lambda}\right)^k\right), \quad x \geq 0$$

### CDF

$$F(x) = 1 - \exp\left(-\left(\frac{x}{\lambda}\right)^k\right)$$

This is one of few distributions with a closed-form CDF.

### Derivation of $E[X]$ and $\text{Var}(X)$

**Expected value:** Substitute $u = (x/\lambda)^k$, so $x = \lambda u^{1/k}$,
$dx = \frac{\lambda}{k} u^{1/k - 1} du$:

$$E[X] = \int_0^\infty x \cdot \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k} dx = \lambda \int_0^\infty u^{1/k} e^{-u} du = \lambda \, \Gamma\left(1 + \frac{1}{k}\right)$$

**Second moment:**

$$E[X^2] = \lambda^2 \, \Gamma\left(1 + \frac{2}{k}\right)$$

**Variance:**

$$\text{Var}(X) = \lambda^2 \left[\Gamma\left(1 + \frac{2}{k}\right) - \Gamma^2\left(1 + \frac{1}{k}\right)\right]$$

### MGF

No closed form in general. Expressed as a series:

$$M_X(t) = \sum_{n=0}^\infty \frac{(t\lambda)^n}{n!} \Gamma\left(1 + \frac{n}{k}\right)$$

### Shape Properties

| Property | Value |
|----------|-------|
| Skewness | Depends on $k$; positive for $k < 3.6$ |
| Tail behaviour | Light-tailed (stretched exponential: $e^{-(x/\lambda)^k}$) |
| Support | $[0, \infty)$ |

**Special cases:**
- $k = 1$: Exponential($1/\lambda$)
- $k = 2$: Rayleigh distribution
- $k \to \infty$: approaches a point mass at $\lambda$

### Hazard Function

$$h(x) = \frac{f(x)}{1 - F(x)} = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}$$

- $k < 1$: decreasing hazard (infant mortality)
- $k = 1$: constant hazard (memoryless = Exponential)
- $k > 1$: increasing hazard (aging/wear-out)

### Why It's Useful for Cost Modelling

- Always positive
- Very flexible shape (can be right-skewed or nearly symmetric)
- Models "time to event" costs well (time to hire, project overruns)
- Closed-form CDF and survival function

---

## Comparison Table

| Property | Normal | LogNormal | Gamma | Pareto | Weibull |
|----------|--------|-----------|-------|--------|---------|
| Support | $(-\infty, \infty)$ | $(0, \infty)$ | $(0, \infty)$ | $[x_m, \infty)$ | $[0, \infty)$ |
| Parameters | $\mu, \sigma^2$ | $\mu, \sigma^2$ | $\alpha, \beta$ | $\alpha, x_m$ | $k, \lambda$ |
| $E[X]$ | $\mu$ | $e^{\mu+\sigma^2/2}$ | $\alpha/\beta$ | $\frac{\alpha x_m}{\alpha-1}$ | $\lambda\Gamma(1+1/k)$ |
| $\text{Var}(X)$ | $\sigma^2$ | $e^{2\mu+\sigma^2}(e^{\sigma^2}-1)$ | $\alpha/\beta^2$ | $\frac{\alpha x_m^2}{(\alpha-1)^2(\alpha-2)}$ | (see above) |
| Skewness | 0 | $(e^{\sigma^2}+2)\sqrt{e^{\sigma^2}-1}$ | $2/\sqrt{\alpha}$ | heavy | depends on $k$ |
| Tail | Light | Sub-exponential | Light | **Heavy** (power) | Light |
| MGF exists? | Yes | No | Yes ($t < \beta$) | No | Series only |
| Closed CDF? | No ($\Phi$) | No ($\Phi$) | No ($\gamma$) | **Yes** | **Yes** |

---

## Cost Component to Distribution Mapping

| Component | Best Candidates | Rationale |
|-----------|----------------|-----------|
| Base salary ($S_i$) | LogNormal, Mixture(Normal) | Right-skewed, multiplicative growth, possible bimodality |
| Overtime cost ($C_{ot}$) | LogNormal, Gamma | Right-skewed, always positive, moderate tails |
| Severance cost ($C_{sev}$) | **Pareto**, LogNormal | Heavy-tailed: a few very expensive cases dominate |
| Hiring cost ($C_h$) | LogNormal, Gamma | Variable but bounded extremes |
| Benefits multiplier ($\beta_i$) | Uniform, Beta | Bounded on both sides |

The Normal distribution is **not** a primary candidate for any individual cost
component. It may be appropriate for the **total** budget (by CLT) if aggregating
many independent components, but not for individual cost shapes.

---

## Numerical Examples

### Example 1: LogNormal salary

Parameters: $\mu = 9.1$, $\sigma = 0.4$

- Mean: $e^{9.1 + 0.08} = e^{9.18} \approx$ R$ 9,726
- Median: $e^{9.1} \approx$ R$ 8,955
- Std dev: $9726 \cdot \sqrt{e^{0.16} - 1} \approx$ R$ 3,969
- Skewness: $(e^{0.16} + 2)\sqrt{e^{0.16} - 1} \approx 1.26$

### Example 2: Pareto severance

Parameters: $\alpha = 2.5$, $x_m = 10{,}000$

- Mean: $\frac{2.5 \times 10000}{1.5} \approx$ R$ 16,667
- Std dev: $\sqrt{\frac{2.5 \times 10000^2}{1.5^2 \times 0.5}} \approx$ R$ 14,907
- $P(X > 50000) = (10000/50000)^{2.5} \approx 1.8\%$
- $P(X > 100000) = (10000/100000)^{2.5} \approx 0.03\%$

### Example 3: Gamma overtime

Parameters: $\alpha = 4$, $\beta = 1/30$ (scale = 30)

- Mean: $4 \times 30 =$ R$ 120/hour
- Std dev: $\sqrt{4} \times 30 =$ R$ 60
- Skewness: $2/\sqrt{4} = 1.0$
