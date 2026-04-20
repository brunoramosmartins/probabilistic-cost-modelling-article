# Exercises — Phase 3: Model Comparison

## Proofs (paper)

### Exercise 1

**Prove Gibbs' inequality:** $D_{KL}(p \| q) \geq 0$ with equality if and only
if $p = q$ almost everywhere.

*Use Jensen's inequality on the concave function $\log$.*

---

### Exercise 2

**Derive AIC** as an approximately unbiased estimator of the expected KL
divergence between the true distribution and the fitted model.

*Start from $E_{\text{true}}[\log f(X | \hat{\theta})]$ and show the bias is
approximately $k/n$ where $k$ is the number of parameters.*

---

### Exercise 3

**Derive BIC** from the Laplace approximation to the Bayesian marginal
likelihood: $p(x | M) = \int L(\theta) \pi(\theta) d\theta$.

*Show that $-2 \log p(x|M) \approx -2\ell(\hat{\theta}) + k \log n$.*

---

### Exercise 4

**Prove Wilks' theorem** (sketch): For nested models $M_0 \subset M_1$, the
likelihood ratio statistic
$\Lambda = -2[\ell(\hat{\theta}_0) - \ell(\hat{\theta}_1)] \xrightarrow{d} \chi^2(\Delta k)$
under $H_0$.

*Use the quadratic approximation of the log-likelihood around $\hat{\theta}_1$.*

---

### Exercise 5

**Prove** that AIC and BIC can select different models, and explain when each
is preferable.

*Construct a concrete 2-model example where $n$ determines which criterion
picks which model.*

---

## Computations (paper)

### Exercise 6

You fit three distributions to salary data ($n = 200$):

| Model | $k$ (params) | $\ell(\hat{\theta})$ |
|-------|-------------|---------------------|
| Normal | 2 | $-1842.3$ |
| LogNormal | 2 | $-1831.7$ |
| Gamma | 2 | $-1835.1$ |

Compute AIC, AICc, and BIC for each. Which model wins under each criterion?
Compute Akaike weights. What is the probability that LogNormal is the best model?

---

### Exercise 7

You want to test whether a Gamma fits better than an Exponential (nested:
Exponential is Gamma with $\alpha = 1$).
$\ell(\hat{\theta}_{\text{Exp}}) = -4521.3$,
$\ell(\hat{\theta}_{\text{Gamma}}) = -4518.1$.

Perform the LRT at $\alpha = 0.05$.

*What is $\Delta k$? What is the critical value?*
