# Phase 2 — Maximum Likelihood Estimation: Theory Notes

## Overview

This document derives MLE theory from first principles. We build the complete
chain: likelihood → log-likelihood → score function → Fisher information →
Cramér-Rao bound → asymptotic normality. Then we derive the MLE for each of
our candidate distributions.

---

## 1. The Likelihood Function

### Setup

We observe data $x_1, x_2, \ldots, x_n$ drawn i.i.d. from a distribution
with PDF $f(x \mid \theta)$, where $\theta \in \Theta$ is an unknown parameter
(possibly a vector).

### Definition

The **likelihood function** treats the observed data as fixed and the parameter
as variable:

$$L(\theta \mid \mathbf{x}) = \prod_{i=1}^n f(x_i \mid \theta)$$

**Key insight:** $L(\theta)$ is NOT a probability distribution over $\theta$.
It is a function that measures how "compatible" each parameter value is with
the observed data.

### Log-Likelihood

Taking logarithms converts the product into a sum (numerically stable,
analytically convenient):

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i \mid \theta)$$

Since $\log$ is monotonically increasing, maximizing $\ell$ is equivalent to
maximizing $L$.

---

## 2. The Maximum Likelihood Estimator

### Definition

The **MLE** is the parameter value that maximizes the likelihood:

$$\hat{\theta}_{MLE} = \arg\max_{\theta \in \Theta} \ell(\theta)$$

### Finding the MLE

For smooth, differentiable likelihoods:

1. Compute the score: $S(\theta) = \frac{\partial \ell}{\partial \theta}$
2. Solve the score equation: $S(\hat{\theta}) = 0$
3. Verify it's a maximum: $\frac{\partial^2 \ell}{\partial \theta^2}\bigg|_{\hat{\theta}} < 0$

For vector parameters $\theta = (\theta_1, \ldots, \theta_k)$, the score is a
gradient and the second-order condition requires the Hessian to be negative
definite.

---

## 3. The Score Function

### Definition

$$S(\theta) = \frac{\partial}{\partial \theta} \ell(\theta) = \sum_{i=1}^n \frac{\partial}{\partial \theta} \log f(x_i \mid \theta)$$

### Theorem: The Score Has Zero Mean at the True Parameter

**Statement:** If $\theta_0$ is the true parameter, then $E_{\theta_0}[S(\theta_0)] = 0$.

**Proof:**

Start from the normalization condition:

$$\int f(x \mid \theta) \, dx = 1$$

Differentiate both sides with respect to $\theta$ (assuming we can exchange
differentiation and integration — regularity conditions):

$$\int \frac{\partial}{\partial \theta} f(x \mid \theta) \, dx = 0$$

Now use the identity $\frac{\partial}{\partial \theta} f = f \cdot \frac{\partial}{\partial \theta} \log f$:

$$\int f(x \mid \theta) \cdot \frac{\partial}{\partial \theta} \log f(x \mid \theta) \, dx = 0$$

This is exactly $E_\theta\left[\frac{\partial}{\partial \theta} \log f(X \mid \theta)\right] = 0$.

For a single observation, $E[s_i(\theta_0)] = 0$ where $s_i = \frac{\partial}{\partial \theta} \log f(x_i \mid \theta)$.

For the full score: $E[S(\theta_0)] = \sum_{i=1}^n E[s_i(\theta_0)] = 0$. $\square$

**Interpretation:** At the true parameter, the score is centered at zero.
It "points" uphill on average, and when you're at the top, there's nowhere to go.

---

## 4. Fisher Information

### Definition (Variance Form)

$$I(\theta) = \text{Var}_\theta[S(\theta)] = E_\theta[S(\theta)^2]$$

(The second equality follows because $E[S] = 0$, so $\text{Var}(S) = E[S^2]$.)

For a single observation:

$$I_1(\theta) = E_\theta\left[\left(\frac{\partial}{\partial \theta} \log f(X \mid \theta)\right)^2\right]$$

For $n$ i.i.d. observations: $I_n(\theta) = n \cdot I_1(\theta)$.

### Theorem: Equivalent Form (Hessian)

**Statement:** Under regularity conditions:

$$I_1(\theta) = -E_\theta\left[\frac{\partial^2}{\partial \theta^2} \log f(X \mid \theta)\right]$$

**Proof:**

Start from the score identity $E\left[\frac{\partial}{\partial \theta} \log f(X \mid \theta)\right] = 0$.

Differentiate both sides with respect to $\theta$:

$$E\left[\frac{\partial^2}{\partial \theta^2} \log f(X \mid \theta)\right] + E\left[\left(\frac{\partial}{\partial \theta} \log f(X \mid \theta)\right)^2\right] = 0$$

(The first term comes from differentiating the integrand; the second comes from
the product rule applied to $\frac{\partial}{\partial \theta}[f \cdot (\partial \log f / \partial \theta)] = f \cdot \partial^2 \log f / \partial \theta^2 + f \cdot (\partial \log f / \partial \theta)^2$.)

Therefore:

$$E\left[\left(\frac{\partial}{\partial \theta} \log f\right)^2\right] = -E\left[\frac{\partial^2}{\partial \theta^2} \log f\right]$$

$$I_1(\theta) = -E\left[\frac{\partial^2}{\partial \theta^2} \log f(X \mid \theta)\right] \quad \square$$

**Interpretation:** Fisher information measures the curvature of the
log-likelihood at the true parameter. More curvature = more information = more
precise estimates.

### Observed Fisher Information

In practice, we often use the **observed** Fisher information:

$$\hat{I}(\hat{\theta}) = -\frac{\partial^2 \ell}{\partial \theta^2}\bigg|_{\theta = \hat{\theta}}$$

This is the negative Hessian of the log-likelihood evaluated at the MLE.

---

## 5. The Cramér-Rao Lower Bound

### Theorem

For any unbiased estimator $\hat{\theta}$ of $\theta$:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{n \cdot I_1(\theta)}$$

**Proof sketch:** Apply the Cauchy-Schwarz inequality to
$\text{Cov}(\hat{\theta}, S(\theta))$:

$$[\text{Cov}(\hat{\theta}, S)]^2 \leq \text{Var}(\hat{\theta}) \cdot \text{Var}(S)$$

Since $\hat{\theta}$ is unbiased, one can show $\text{Cov}(\hat{\theta}, S) = 1$.
And $\text{Var}(S) = I_n(\theta) = n \cdot I_1(\theta)$.

Therefore: $1 \leq \text{Var}(\hat{\theta}) \cdot n \cdot I_1(\theta)$, which gives
$\text{Var}(\hat{\theta}) \geq 1/(n \cdot I_1(\theta))$. $\square$

### Efficiency

An estimator that achieves the Cramér-Rao bound is called **efficient**.
The MLE is asymptotically efficient: as $n \to \infty$, it achieves the bound.

---

## 6. Asymptotic Normality of the MLE

### Theorem

Under regularity conditions, as $n \to \infty$:

$$\sqrt{n}(\hat{\theta}_{MLE} - \theta_0) \xrightarrow{d} N(0, I_1(\theta_0)^{-1})$$

Equivalently:

$$\hat{\theta}_{MLE} \dot{\sim} N\left(\theta_0, \frac{1}{n \cdot I_1(\theta_0)}\right)$$

**Proof sketch (Taylor expansion approach):**

Expand the score around $\theta_0$:

$$0 = S(\hat{\theta}) \approx S(\theta_0) + (\hat{\theta} - \theta_0) \cdot \frac{\partial S}{\partial \theta}\bigg|_{\theta_0}$$

Solving:

$$\hat{\theta} - \theta_0 \approx -\frac{S(\theta_0)}{\partial S / \partial \theta |_{\theta_0}}$$

By the CLT: $S(\theta_0) / \sqrt{n} \xrightarrow{d} N(0, I_1(\theta_0))$

By the LLN: $\frac{1}{n} \frac{\partial S}{\partial \theta}\bigg|_{\theta_0} \xrightarrow{p} -I_1(\theta_0)$

Combining via Slutsky's theorem:

$$\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} N(0, I_1(\theta_0)^{-1}) \quad \square$$

### Practical Consequence: Confidence Intervals

An approximate $(1-\alpha)$ confidence interval for $\theta$:

$$\hat{\theta} \pm z_{\alpha/2} \cdot \frac{1}{\sqrt{n \cdot I_1(\hat{\theta})}}$$

Or using observed Fisher information:

$$\hat{\theta} \pm z_{\alpha/2} \cdot \frac{1}{\sqrt{\hat{I}_n(\hat{\theta})}}$$

---

## 7. MLE for Specific Distributions

### 7.1 Normal($\mu, \sigma^2$) — Closed Form

**Log-likelihood:**

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

**Score equations:**

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n(x_i - \mu) = 0 \implies \hat{\mu} = \bar{x}$$

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n(x_i - \mu)^2 = 0 \implies \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^2$$

**Note:** $\hat{\sigma}^2_{MLE}$ is biased: $E[\hat{\sigma}^2_{MLE}] = \frac{n-1}{n}\sigma^2$.

**Fisher information matrix:**

$$I(\mu, \sigma^2) = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4) \end{pmatrix}$$

The off-diagonal zeros mean $\hat{\mu}$ and $\hat{\sigma}^2$ are asymptotically independent.

### 7.2 LogNormal($\mu, \sigma^2$) — Closed Form

**Key insight:** If $X_i \sim \text{LogNormal}(\mu, \sigma^2)$, then $Y_i = \log X_i \sim N(\mu, \sigma^2)$.

**MLE:** Transform the data and apply Normal MLE:

$$\hat{\mu} = \frac{1}{n}\sum_{i=1}^n \log x_i = \overline{\log x}$$

$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (\log x_i - \hat{\mu})^2$$

**Fisher information:** Same as Normal, computed on the log-transformed data.

### 7.3 Gamma($\alpha, \beta$) — No Closed Form for $\alpha$

**Log-likelihood:**

$$\ell(\alpha, \beta) = n\alpha\log\beta - n\log\Gamma(\alpha) + (\alpha - 1)\sum_{i=1}^n \log x_i - \beta\sum_{i=1}^n x_i$$

**Score equations:**

$$\frac{\partial \ell}{\partial \beta} = \frac{n\alpha}{\beta} - \sum_{i=1}^n x_i = 0 \implies \hat{\beta} = \frac{\hat{\alpha}}{\bar{x}}$$

$$\frac{\partial \ell}{\partial \alpha} = n\log\hat{\beta} - n\psi(\hat{\alpha}) + \sum_{i=1}^n \log x_i = 0$$

where $\psi(\alpha) = \Gamma'(\alpha)/\Gamma(\alpha)$ is the digamma function.

The second equation has no closed form — solve numerically.

**Fisher information:**

$$I(\alpha, \beta) = n \begin{pmatrix} \psi'(\alpha) & -1/\beta \\ -1/\beta & \alpha/\beta^2 \end{pmatrix}$$

where $\psi'(\alpha)$ is the trigamma function.

### 7.4 Pareto($\alpha, x_m$) — Closed Form (with $x_m$ known)

**Log-likelihood** (assuming $x_m$ is known):

$$\ell(\alpha) = n\log\alpha + n\alpha\log x_m - (\alpha + 1)\sum_{i=1}^n \log x_i$$

**Score:**

$$\frac{\partial \ell}{\partial \alpha} = \frac{n}{\alpha} + n\log x_m - \sum_{i=1}^n \log x_i = 0$$

$$\hat{\alpha} = \frac{n}{\sum_{i=1}^n \log(x_i / x_m)} = \frac{n}{\sum_{i=1}^n \log x_i - n\log x_m}$$

**Fisher information:**

$$I_1(\alpha) = \frac{1}{\alpha^2}$$

**Asymptotic distribution:**

$$\hat{\alpha} \dot{\sim} N\left(\alpha, \frac{\alpha^2}{n}\right)$$

### 7.5 Weibull($k, \lambda$) — Numerical

**Log-likelihood:**

$$\ell(k, \lambda) = n\log k - nk\log\lambda + (k-1)\sum_{i=1}^n \log x_i - \sum_{i=1}^n \left(\frac{x_i}{\lambda}\right)^k$$

No closed form for either parameter — solve the score equations numerically.

---

## 8. Multivariate Generalization

For a parameter vector $\theta = (\theta_1, \ldots, \theta_k)$:

- The **score** is a gradient: $S(\theta) = \nabla_\theta \ell(\theta) \in \mathbb{R}^k$
- The **Fisher information** is a $k \times k$ matrix:
  $[I(\theta)]_{jl} = -E\left[\frac{\partial^2 \ell}{\partial \theta_j \partial \theta_l}\right]$
- The **observed Fisher information** is the negative Hessian at the MLE
- The **asymptotic covariance** of the MLE is $I_n(\theta_0)^{-1}$
- **Standard errors** are the square roots of the diagonal of $I_n(\hat{\theta})^{-1}$

---

## Numerical Examples

### Example 1: Normal MLE

Data: $\{8.2, 9.1, 7.5, 11.3, 10.8, 8.7, 9.5, 12.1, 8.9, 10.2\}$ (in thousands)

- $\hat{\mu} = \bar{x} = 9.63$
- $\hat{\sigma}^2 = \frac{1}{10}\sum(x_i - 9.63)^2 = 1.945$
- $\hat{\sigma} = 1.395$
- SE($\hat{\mu}$) = $\hat{\sigma}/\sqrt{n} = 1.395/\sqrt{10} = 0.441$
- 95% CI for $\mu$: $9.63 \pm 1.96 \times 0.441 = [8.77, 10.49]$

### Example 2: Pareto MLE

Data: 20 severance costs with $x_m = 10000$, $\sum \log(x_i/x_m) = 12.8$

- $\hat{\alpha} = 20/12.8 = 1.5625$
- SE($\hat{\alpha}$) = $\hat{\alpha}/\sqrt{n} = 1.5625/\sqrt{20} = 0.349$
- 95% CI: $1.5625 \pm 1.96 \times 0.349 = [0.878, 2.247]$

### Example 3: Fisher Information Interpretation

For Normal($\mu$, $\sigma^2$ known):
- $I_1(\mu) = 1/\sigma^2$
- High variance $\sigma^2$ → low information → imprecise $\hat{\mu}$
- 10x more data → 10x more information → SE shrinks by $\sqrt{10}$
