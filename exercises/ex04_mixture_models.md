# Exercises — Phase 4: Mixture Models

## Proofs (paper)

### Exercise 1

**Show** that the log-likelihood of a mixture model,
$\ell(\theta) = \sum_{i=1}^n \log \left[\sum_{k=1}^K \pi_k f_k(x_i | \theta_k)\right]$,
cannot be decomposed into a sum of per-component terms.

*This is why direct MLE is intractable for mixtures.*

---

### Exercise 2

**Derive the E-step** of the EM algorithm for a 2-component Gaussian mixture.
Starting from the complete-data log-likelihood, show that the responsibilities
are:

$$\gamma_{ik} = \frac{\pi_k \cdot \mathcal{N}(x_i | \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \cdot \mathcal{N}(x_i | \mu_j, \sigma_j^2)}$$

---

### Exercise 3

**Derive the M-step** updates: given responsibilities $\gamma_{ik}$, derive
the update formulas for $\pi_k$, $\mu_k$, and $\sigma_k^2$ by maximizing the
expected complete-data log-likelihood.

*Use Lagrange multipliers for the constraint $\sum_k \pi_k = 1$.*

---

### Exercise 4

**Prove** that the EM algorithm monotonically increases the observed-data
log-likelihood: $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$.

*Use Jensen's inequality on the concave function $\log$. Define the ELBO and
show that EM maximizes it.*

---

### Exercise 5

**Prove** that EM converges to a stationary point of the likelihood (not
necessarily a global maximum).

*Discuss the implications: multiple random restarts.*

---

## Computations (paper)

### Exercise 6

Given a 2-component mixture with $\pi_1 = 0.6$, $\mu_1 = 8$, $\sigma_1 = 1.5$,
$\pi_2 = 0.4$, $\mu_2 = 18$, $\sigma_2 = 2$:

- Compute the responsibility $\gamma_{i1}$ for data point $x_i = 10$.
- Compute the responsibility for $x_i = 16$.
- Interpret: which component "owns" each data point?

---

### Exercise 7

After one E-step, you have responsibilities for 6 data points:

| $x_i$ | $\gamma_{i1}$ | $\gamma_{i2}$ |
|--------|-------------|-------------|
| 7.5 | 0.95 | 0.05 |
| 8.2 | 0.92 | 0.08 |
| 9.1 | 0.80 | 0.20 |
| 15.3 | 0.10 | 0.90 |
| 17.8 | 0.02 | 0.98 |
| 19.5 | 0.01 | 0.99 |

Perform one M-step: compute updated $\pi_k$, $\mu_k$, and $\sigma_k^2$.
