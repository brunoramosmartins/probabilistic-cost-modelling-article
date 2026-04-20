# Phase 3 — Model Comparison: Theory Notes

## Overview

Given data and multiple candidate distributions (each fitted via MLE), how do
we choose the best model? This document derives the information-theoretic and
hypothesis-testing frameworks for model selection: KL divergence, AIC, BIC,
likelihood ratio tests, and goodness-of-fit tests.

---

## 1. Kullback-Leibler Divergence

### Definition

The KL divergence from distribution $q$ to distribution $p$ (both continuous):

$$D_{KL}(p \| q) = \int_{-\infty}^{\infty} p(x) \log\frac{p(x)}{q(x)} \, dx$$

**Interpretation:** The expected "surprise penalty" of using $q$ to encode data
that actually follows $p$. It measures how much information is lost when $q$ is
used to approximate $p$.

### Theorem: Non-Negativity (Gibbs' Inequality)

**Statement:** $D_{KL}(p \| q) \geq 0$, with equality if and only if $p = q$
almost everywhere.

**Proof:**

$$D_{KL}(p \| q) = \int p(x) \log\frac{p(x)}{q(x)} \, dx = -\int p(x) \log\frac{q(x)}{p(x)} \, dx$$

Apply Jensen's inequality to the concave function $\log$:

$$-\int p(x) \log\frac{q(x)}{p(x)} \, dx \geq -\log\left(\int p(x) \cdot \frac{q(x)}{p(x)} \, dx\right)$$

$$= -\log\left(\int q(x) \, dx\right) = -\log(1) = 0$$

Since $\log$ is strictly concave, equality holds in Jensen's inequality if and
only if $q(x)/p(x)$ is constant $p$-a.e., which (given both integrate to 1)
means $p = q$ a.e. $\square$

### Key Properties

- $D_{KL}(p \| q) \neq D_{KL}(q \| p)$ in general (not symmetric)
- Not a metric (doesn't satisfy triangle inequality)
- $D_{KL}(p \| q) = 0 \iff p = q$ a.e.
- Connects to maximum likelihood: minimizing KL from true $p$ to model $q_\theta$ is equivalent to maximizing expected log-likelihood

---

## 2. Akaike Information Criterion (AIC)

### Motivation

We want to estimate the expected KL divergence from the true distribution $g$
to the fitted model $f(\cdot \mid \hat{\theta})$, where $\hat{\theta}$ is the
MLE from a sample of size $n$.

### Derivation

**Step 1:** The expected log-likelihood on new data (the quantity we want to maximize):

$$E_g\left[\log f(X \mid \hat{\theta})\right]$$

**Step 2:** The maximized log-likelihood on the training data is an optimistic
(biased) estimate of this:

$$\frac{1}{n}\ell(\hat{\theta}) = \frac{1}{n}\sum_{i=1}^n \log f(x_i \mid \hat{\theta})$$

**Step 3:** Akaike (1973) showed that the bias of using the training
log-likelihood as an estimate of the expected log-likelihood is approximately
$k/n$, where $k$ is the number of estimated parameters:

$$E\left[\frac{1}{n}\ell(\hat{\theta})\right] - E_g\left[\log f(X \mid \hat{\theta})\right] \approx \frac{k}{n}$$

**Step 4:** Correcting for this bias and multiplying by $-2n$ (convention):

$$\text{AIC} = -2\ell(\hat{\theta}) + 2k$$

**Rule:** Lower AIC = better model (less expected information loss).

### AICc (Small-Sample Correction)

For small $n$ relative to $k$, Hurvich and Tsai (1989) derived:

$$\text{AIC}_c = \text{AIC} + \frac{2k(k+1)}{n - k - 1}$$

Use AICc when $n/k < 40$.

### Akaike Weights

Given $M$ candidate models with AIC values $\text{AIC}_1, \ldots, \text{AIC}_M$:

$$\Delta_i = \text{AIC}_i - \min_j \text{AIC}_j$$

$$w_i = \frac{e^{-\Delta_i / 2}}{\sum_{j=1}^M e^{-\Delta_j / 2}}$$

$w_i$ is interpreted as the approximate probability that model $i$ is the best
model (in the KL sense) among the candidates.

---

## 3. Bayesian Information Criterion (BIC)

### Motivation

BIC approximates the Bayesian model evidence (marginal likelihood):

$$p(\mathbf{x} \mid M) = \int L(\theta) \pi(\theta) \, d\theta$$

### Derivation (Laplace Approximation)

**Step 1:** Expand $\log L(\theta)$ around the MLE $\hat{\theta}$:

$$\log L(\theta) \approx \log L(\hat{\theta}) - \frac{1}{2}(\theta - \hat{\theta})^T \hat{I}_n (\theta - \hat{\theta})$$

where $\hat{I}_n = -\nabla^2 \ell(\hat{\theta})$ is the observed Fisher information matrix.

**Step 2:** Substitute into the integral (assume flat prior $\pi(\theta) \propto 1$):

$$p(\mathbf{x} \mid M) \approx L(\hat{\theta}) \cdot (2\pi)^{k/2} \cdot |\hat{I}_n|^{-1/2}$$

**Step 3:** Take $-2\log$ of both sides:

$$-2\log p(\mathbf{x} \mid M) \approx -2\ell(\hat{\theta}) + k\log n - k\log(2\pi) + \log|\hat{I}_n|$$

**Step 4:** For large $n$, $\hat{I}_n \approx n \cdot I_1(\theta)$, so
$\log|\hat{I}_n| \approx k\log n + O(1)$. The remaining terms are $O(1)$ and
dominated by $k\log n$:

$$\text{BIC} = -2\ell(\hat{\theta}) + k\log n$$

**Rule:** Lower BIC = better model.

### AIC vs BIC

| Aspect | AIC | BIC |
|--------|-----|-----|
| Derived from | Expected KL divergence | Bayesian model evidence |
| Penalty | $2k$ | $k\log n$ |
| For $n > 7$ | BIC penalizes more | BIC penalizes more |
| Consistency | No (can overfit asymptotically) | Yes (selects true model as $n \to \infty$) |
| Efficiency | Yes (minimizes prediction risk) | No |
| Best for | Prediction | Identifying the true model |

---

## 4. Likelihood Ratio Test (LRT)

### Setup: Nested Models

Model $M_0$ (restricted) is nested within $M_1$ (full) if $M_0$ is a special
case obtained by fixing some parameters of $M_1$.

Example: Exponential($\beta$) is nested within Gamma($\alpha, \beta$) via $\alpha = 1$.

### Test Statistic

$$\Lambda = -2\left[\ell(\hat{\theta}_0) - \ell(\hat{\theta}_1)\right] = 2\left[\ell(\hat{\theta}_1) - \ell(\hat{\theta}_0)\right]$$

where $\hat{\theta}_0$ maximizes under $M_0$ and $\hat{\theta}_1$ under $M_1$.

Note: $\Lambda \geq 0$ always (the full model can only fit better or equal).

### Wilks' Theorem

**Statement:** Under $H_0$ (the restricted model is correct), as $n \to \infty$:

$$\Lambda \xrightarrow{d} \chi^2(\Delta k)$$

where $\Delta k = k_1 - k_0$ is the difference in number of parameters.

**Proof sketch:**

Expand $\ell(\theta)$ to second order around $\hat{\theta}_1$:

$$\ell(\hat{\theta}_0) \approx \ell(\hat{\theta}_1) - \frac{1}{2}(\hat{\theta}_0 - \hat{\theta}_1)^T \hat{I}_n (\hat{\theta}_0 - \hat{\theta}_1)$$

Under $H_0$, $\hat{\theta}_1 - \hat{\theta}_0$ is asymptotically Normal with
covariance governed by Fisher information in the constrained directions.
The resulting quadratic form in Normal variables follows a $\chi^2$ distribution
with degrees of freedom equal to the number of constraints ($\Delta k$). $\square$

### Decision Rule

At significance level $\alpha$:
- Reject $H_0$ if $\Lambda > \chi^2_{\alpha}(\Delta k)$
- Equivalently: reject if p-value $= P(\chi^2(\Delta k) > \Lambda) < \alpha$

### Important Limitation

The LRT only applies to **nested** models. Normal vs LogNormal is NOT nested —
use AIC/BIC or the Vuong test instead.

---

## 5. Goodness-of-Fit Tests

### 5.1 Kolmogorov-Smirnov (KS) Test

**Idea:** Compare the empirical CDF $F_n(x)$ with the theoretical CDF $F_0(x)$.

**Test statistic:**

$$D_n = \sup_x |F_n(x) - F_0(x)|$$

**Properties:**
- Distribution-free under $H_0$ (when parameters are known)
- When parameters are estimated from data, the standard KS critical values are
  conservative (Lilliefors correction needed)
- Sensitive to differences in the center of the distribution
- Less sensitive to tail differences

**Decision:** Reject $H_0$ (data follows $F_0$) if $D_n$ exceeds the critical value.

### 5.2 Anderson-Darling (AD) Test

**Idea:** Like KS but with more weight on the tails.

**Test statistic:**

$$A^2 = -n - \sum_{i=1}^n \frac{2i - 1}{n}\left[\log F_0(x_{(i)}) + \log(1 - F_0(x_{(n+1-i)}))\right]$$

where $x_{(1)} \leq x_{(2)} \leq \ldots \leq x_{(n)}$ are the order statistics.

**Properties:**
- More powerful than KS for detecting tail deviations
- Critical values depend on the specific distribution family
- Better for detecting heavy-tail misspecification

### When to Use Each

| Scenario | Recommended Test |
|----------|-----------------|
| General distributional check | KS |
| Tail behaviour matters (cost modelling!) | **AD** |
| Large sample, any deviation matters | KS (lower power but simpler) |
| Small sample, specific alternative | AD |

---

## 6. Non-Nested Model Comparison: Vuong Test (Brief)

For comparing two non-nested models $f$ and $g$:

$$V_n = \frac{\ell_f(\hat{\theta}_f) - \ell_g(\hat{\theta}_g)}{\hat{\omega}\sqrt{n}}$$

where $\hat{\omega}^2 = \frac{1}{n}\sum_{i=1}^n \left[\log\frac{f(x_i|\hat{\theta}_f)}{g(x_i|\hat{\theta}_g)}\right]^2 - \left[\frac{1}{n}\sum_{i=1}^n \log\frac{f(x_i|\hat{\theta}_f)}{g(x_i|\hat{\theta}_g)}\right]^2$

Under $H_0$ (both models equally close to truth): $V_n \xrightarrow{d} N(0, 1)$.

---

## 7. Practical Decision Framework

Given fitted candidates with FitResults:

1. **Compute AIC and BIC** for all candidates
2. **Compute Akaike weights** — is there a clear winner or are models close?
3. **Run GoF tests** — does the winning model actually fit well?
4. **If models are nested**, use LRT for formal hypothesis test
5. **Consider the use case:**
   - For prediction/budget: prefer AIC (minimizes prediction error)
   - For identifying the "true" model: prefer BIC (consistent)
   - For tail risk: ensure GoF in the tails (AD test)

---

## Numerical Examples

### Example 1: AIC/BIC Comparison

Three models fitted to $n = 200$ salary observations:

| Model | $k$ | $\ell(\hat{\theta})$ | AIC | BIC |
|-------|------|---------------------|-----|-----|
| Normal | 2 | -1842.3 | 3688.6 | 3695.2 |
| LogNormal | 2 | -1831.7 | 3667.4 | 3674.0 |
| Gamma | 2 | -1835.1 | 3674.2 | 3680.8 |

Winner (both criteria): **LogNormal** (lowest AIC and BIC).

Akaike weights:
- $\Delta_{\text{LN}} = 0$, $\Delta_{\text{Gamma}} = 6.8$, $\Delta_{\text{Normal}} = 21.2$
- $w_{\text{LN}} = 0.967$, $w_{\text{Gamma}} = 0.032$, $w_{\text{Normal}} \approx 0$

Interpretation: 96.7% probability that LogNormal is the best model.

### Example 2: Likelihood Ratio Test

Testing Exponential vs Gamma (nested: $H_0: \alpha = 1$):
- $\ell_{\text{Exp}} = -4521.3$, $\ell_{\text{Gamma}} = -4518.1$
- $\Lambda = 2 \times (4521.3 - 4518.1) = 6.4$
- $\Delta k = 1$, critical value $\chi^2_{0.05}(1) = 3.841$
- $6.4 > 3.841$ → **Reject $H_0$**: Gamma fits significantly better than Exponential.
