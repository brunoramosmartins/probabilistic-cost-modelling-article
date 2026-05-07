# TIL — The EM Algorithm Is Coordinate Ascent on a Lower Bound

**Phase:** 4 · **Topic:** EM algorithm, ELBO, variational inference · **Domain:** machine learning, latent-variable models

## Hook

The EM algorithm's monotonic convergence feels almost magical: each
iteration is guaranteed to improve the log-likelihood. The magic
dissolves once you see EM as alternating optimization over a single
objective — the ELBO.

## Insight

Define the ELBO (Evidence Lower Bound) as a function of two arguments:
a distribution $q$ over latent variables, and the model parameters
$\theta$.

$$\text{ELBO}(q, \theta) = E_q[\log p(x, z \mid \theta)] - E_q[\log q(z)]$$

The ELBO is always $\leq \ell(\theta)$, with equality when
$q(z) = p(z \mid x, \theta)$. The gap is exactly $D_{KL}(q \| p(z|x,\theta))$.

The E-step picks the optimal $q$ for the current $\theta$, making the
ELBO equal to $\ell$ — closing the gap. The M-step picks the optimal
$\theta$ for the current $q$, raising the ELBO. Since ELBO $\leq \ell$
always, raising the ELBO can only raise $\ell$ — proving monotone
convergence "for free."

This perspective generalizes immediately. Replace "find optimal $q$"
(E-step) with "find approximately optimal $q$ in some restricted family"
and you get **variational inference** — the engine behind Variational
Autoencoders, BERT pretraining, and most modern probabilistic deep
learning. The same lower-bound-and-coordinate-ascent structure scales
from a 2-component Gaussian mixture to billion-parameter neural networks.

## Example

For a 2-component GMM, the E-step computes responsibilities:

$$\gamma_{ik} = \frac{\pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)}{\sum_j \pi_j \cdot \mathcal{N}(x_i \mid \mu_j, \sigma_j^2)}$$

These are exactly $P(z_i = k \mid x_i, \theta_{\text{old}})$ — the
posterior over the latent cluster assignment. The M-step then re-fits
$\pi_k$, $\mu_k$, $\sigma_k^2$ as weighted averages with $\gamma$ as
weights. The log-likelihood $\ell(\theta_{\text{new}}) \geq \ell(\theta_{\text{old}})$
— guaranteed.

## Takeaway

EM is not a special-purpose algorithm. It is the simplest instance of
"improve a lower bound by alternation," one of the most reusable ideas
in modern machine learning. Once you see EM this way, half of variational
methods stop looking strange.
