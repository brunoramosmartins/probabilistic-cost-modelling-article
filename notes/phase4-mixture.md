# Phase 4 — Mixture Models: Theory Notes

## Overview

Salary data is often multimodal: junior, senior, and director clusters form
distinct peaks. A single distribution cannot capture this structure. This
document derives the EM algorithm for Gaussian Mixture Models from first
principles, proves monotone convergence, and connects to budget modelling.

---

## 1. Gaussian Mixture Model (GMM)

### Definition

A $K$-component Gaussian Mixture Model:

$$f(x \mid \theta) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(x \mid \mu_k, \sigma_k^2)$$

where:
- $\pi_k \geq 0$, $\sum_{k=1}^K \pi_k = 1$ (mixing weights)
- $\mu_k$ = mean of component $k$
- $\sigma_k^2$ = variance of component $k$
- $\theta = \{\pi_1, \ldots, \pi_K, \mu_1, \ldots, \mu_K, \sigma_1^2, \ldots, \sigma_K^2\}$

### Log-Likelihood

$$\ell(\theta) = \sum_{i=1}^n \log\left[\sum_{k=1}^K \pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)\right]$$

**Critical problem:** The $\log$ of a sum cannot be decomposed into a sum of
per-component terms. Direct differentiation leads to coupled equations with
no closed-form solution.

---

## 2. Why Direct MLE Fails

Setting $\partial \ell / \partial \mu_k = 0$:

$$\sum_{i=1}^n \frac{\pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \cdot \mathcal{N}(x_i \mid \mu_j, \sigma_j^2)} \cdot \frac{x_i - \mu_k}{\sigma_k^2} = 0$$

Each equation involves ALL parameters (through the denominator). The equations
are coupled and nonlinear — no closed-form solution exists.

**Solution:** The EM algorithm iteratively decouples the problem.

---

## 3. Latent Variables and Complete Data

### Introducing Latent Variables

For each observation $x_i$, introduce a latent indicator $z_i \in \{1, \ldots, K\}$
indicating which component generated it:

$$P(z_i = k) = \pi_k$$
$$x_i \mid z_i = k \sim \mathcal{N}(\mu_k, \sigma_k^2)$$

### Complete-Data Log-Likelihood

If we knew $z_i$ for all $i$, the log-likelihood would decompose:

$$\ell_c(\theta) = \sum_{i=1}^n \sum_{k=1}^K \mathbb{1}[z_i = k] \left[\log \pi_k + \log \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)\right]$$

This IS separable by component — easy to maximize! But we don't observe $z_i$.

---

## 4. The EM Algorithm

### Overview

EM alternates between:
- **E-step:** Compute the expected value of the latent variables given current parameters
- **M-step:** Maximize the expected complete-data log-likelihood

### E-Step: Compute Responsibilities

The **responsibility** of component $k$ for observation $i$:

$$\gamma_{ik} = P(z_i = k \mid x_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \cdot \mathcal{N}(x_i \mid \mu_k^{(t)}, \sigma_k^{2(t)})}{\sum_{j=1}^K \pi_j^{(t)} \cdot \mathcal{N}(x_i \mid \mu_j^{(t)}, \sigma_j^{2(t)})}$$

**Derivation:** By Bayes' theorem:

$$P(z_i = k \mid x_i) = \frac{P(x_i \mid z_i = k) \cdot P(z_i = k)}{P(x_i)} = \frac{\pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)}{\sum_j \pi_j \cdot \mathcal{N}(x_i \mid \mu_j, \sigma_j^2)}$$

**Properties:**
- $0 \leq \gamma_{ik} \leq 1$
- $\sum_{k=1}^K \gamma_{ik} = 1$ for each $i$
- $\gamma_{ik} \approx 1$ when $x_i$ is "clearly" from component $k$

### Expected Complete-Data Log-Likelihood (Q function)

$$Q(\theta \mid \theta^{(t)}) = \sum_{i=1}^n \sum_{k=1}^K \gamma_{ik} \left[\log \pi_k + \log \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)\right]$$

### M-Step: Maximize Q

Maximize $Q$ with respect to $\theta$ subject to $\sum_k \pi_k = 1$.

**Define:** $N_k = \sum_{i=1}^n \gamma_{ik}$ (effective number of points in component $k$).

**Update for $\pi_k$** (using Lagrange multipliers for $\sum_k \pi_k = 1$):

$$\frac{\partial}{\partial \pi_k}\left[Q + \lambda(1 - \sum_k \pi_k)\right] = \frac{N_k}{\pi_k} - \lambda = 0$$

$$\pi_k = \frac{N_k}{\lambda}$$

Summing over $k$: $\sum_k \pi_k = \sum_k N_k / \lambda = n / \lambda = 1$, so $\lambda = n$.

$$\boxed{\pi_k^{(t+1)} = \frac{N_k}{n}}$$

**Update for $\mu_k$:**

$$\frac{\partial Q}{\partial \mu_k} = \sum_{i=1}^n \gamma_{ik} \cdot \frac{x_i - \mu_k}{\sigma_k^2} = 0$$

$$\boxed{\mu_k^{(t+1)} = \frac{\sum_{i=1}^n \gamma_{ik} \cdot x_i}{N_k} = \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik} \cdot x_i}$$

**Update for $\sigma_k^2$:**

$$\frac{\partial Q}{\partial \sigma_k^2} = \sum_{i=1}^n \gamma_{ik}\left[-\frac{1}{2\sigma_k^2} + \frac{(x_i - \mu_k)^2}{2\sigma_k^4}\right] = 0$$

$$\boxed{\sigma_k^{2(t+1)} = \frac{\sum_{i=1}^n \gamma_{ik}(x_i - \mu_k^{(t+1)})^2}{N_k}}$$

**Interpretation:** Each update is a "weighted" version of the standard MLE,
where the weights are the responsibilities.

---

## 5. Monotone Convergence of EM

### Theorem

The EM algorithm monotonically increases the observed-data log-likelihood:

$$\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$$

### Proof

**Step 1: Define the ELBO (Evidence Lower Bound)**

For any distribution $q(z)$ over the latent variables:

$$\ell(\theta) = \log p(\mathbf{x} \mid \theta) = \log \sum_z p(\mathbf{x}, z \mid \theta)$$

$$= \log \sum_z q(z) \frac{p(\mathbf{x}, z \mid \theta)}{q(z)}$$

**Step 2: Apply Jensen's inequality** ($\log$ is concave):

$$\ell(\theta) \geq \sum_z q(z) \log \frac{p(\mathbf{x}, z \mid \theta)}{q(z)} \equiv \mathcal{L}(q, \theta)$$

This is the ELBO: $\ell(\theta) \geq \mathcal{L}(q, \theta)$.

**Step 3: E-step makes the bound tight**

The gap is $\ell(\theta) - \mathcal{L}(q, \theta) = D_{KL}(q(z) \| p(z \mid \mathbf{x}, \theta))$.

Setting $q(z) = p(z \mid \mathbf{x}, \theta^{(t)})$ makes $D_{KL} = 0$, so
$\mathcal{L}(q^{(t)}, \theta^{(t)}) = \ell(\theta^{(t)})$.

**Step 4: M-step increases the ELBO**

$$\theta^{(t+1)} = \arg\max_\theta \mathcal{L}(q^{(t)}, \theta)$$

Therefore: $\mathcal{L}(q^{(t)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t)}, \theta^{(t)}) = \ell(\theta^{(t)})$

**Step 5: Combine**

$$\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t)}, \theta^{(t+1)}) \geq \ell(\theta^{(t)}) \quad \square$$

### Convergence Properties

- EM converges to a **local** maximum or saddle point (not necessarily global)
- Convergence rate is linear (can be slow near the optimum)
- **Remedy:** Multiple random restarts, choose the one with highest $\ell$

---

## 6. Choosing the Number of Components K

### BIC for K Selection

Fit GMMs with $K = 1, 2, 3, \ldots$ and choose the $K$ that minimizes BIC:

$$\text{BIC}(K) = -2\ell(\hat{\theta}_K) + p_K \cdot \log n$$

where $p_K = 3K - 1$ (number of free parameters: $K$ means + $K$ variances + $K-1$ weights).

**Why BIC and not AIC?** BIC's stronger penalty helps avoid overfitting with
too many components. AIC tends to select too many components.

### Alternative: Likelihood Ratio Test

Test $K$ vs $K+1$ components via LRT. However, regularity conditions for
Wilks' theorem fail at the boundary of the parameter space (a component with
$\pi_k = 0$ is on the boundary). Use parametric bootstrap for the null
distribution.

---

## 7. Connection to Budget Modelling

### Multimodal Salary → Wider Budget Uncertainty

If salaries are bimodal (juniors at R\$8,000, seniors at R\$18,000):
- A single Normal fit gives mean R\$12,200 with large variance
- The "average" employee at R\$12,200 doesn't exist in either cluster
- Budget predictions based on the single Normal are misleading

The GMM correctly identifies two distinct cost pools, each with its own
uncertainty profile. Budget planning should account for each separately.

---

## Numerical Example

### 2-Component GMM

Parameters: $\pi_1 = 0.6$, $\mu_1 = 8$, $\sigma_1 = 1.5$, $\pi_2 = 0.4$, $\mu_2 = 18$, $\sigma_2 = 2$

**For $x = 10$:**

$$\gamma_{i1} = \frac{0.6 \cdot \mathcal{N}(10 \mid 8, 1.5^2)}{0.6 \cdot \mathcal{N}(10 \mid 8, 1.5^2) + 0.4 \cdot \mathcal{N}(10 \mid 18, 2^2)}$$

- $\mathcal{N}(10 \mid 8, 2.25) = \frac{1}{\sqrt{2\pi \cdot 2.25}} e^{-(10-8)^2/(2 \cdot 2.25)} = 0.1094$
- $\mathcal{N}(10 \mid 18, 4) = \frac{1}{\sqrt{2\pi \cdot 4}} e^{-(10-18)^2/8} = 0.000017$

$$\gamma_{i1} = \frac{0.6 \times 0.1094}{0.6 \times 0.1094 + 0.4 \times 0.000017} = \frac{0.06564}{0.06565} \approx 0.9999$$

Point $x = 10$ almost certainly belongs to component 1.

**For $x = 16$:**

- $\mathcal{N}(16 \mid 8, 2.25) = 6.7 \times 10^{-8}$
- $\mathcal{N}(16 \mid 18, 4) = 0.1210$

$$\gamma_{i1} = \frac{0.6 \times 6.7 \times 10^{-8}}{0.6 \times 6.7 \times 10^{-8} + 0.4 \times 0.1210} \approx 0.0000008$$

Point $x = 16$ almost certainly belongs to component 2.
