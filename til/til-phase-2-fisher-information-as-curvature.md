# TIL — Fisher Information Is Just the Curvature of the Log-Likelihood

**Phase:** 2 · **Topic:** MLE, Fisher information · **Domain:** statistical inference

## Hook

Fisher information sounds like an abstract concept from a graduate
statistics course. It is actually one of the most concrete ideas in
inference: how sharply does the data point at one specific parameter
value?

## Insight

When you maximize a log-likelihood, you find a peak. That peak can be a
sharp spike (the data identifies the parameter well) or a flat plateau
(many parameter values fit roughly equally well). Curvature — formally,
the second derivative of $\ell(\theta)$ — quantifies which one you have.
Fisher information is exactly that curvature, evaluated at the true
parameter and averaged over the data:

$$I(\theta) = -E\left[\frac{\partial^2 \ell}{\partial \theta^2}\right]$$

High curvature = high information = small standard error. Low curvature
= low information = wide confidence intervals. The asymptotic standard
error of the MLE is $1/\sqrt{n \cdot I(\theta)}$ — exactly what falls
out of the Cramér-Rao lower bound. In practice, we use the *observed*
Fisher information (the negative Hessian of $\ell$ evaluated at $\hat{\theta}$)
to compute confidence intervals — that is what `scipy` and every fitting
library do under the hood.

## Example

For a Normal with known $\sigma^2$, Fisher information for $\mu$ is
$n/\sigma^2$. If $\sigma = 10$ and $n = 100$, $I = 1$. The standard
error of $\hat{\mu}$ is $1/\sqrt{I} = 1$. Double the sample to $n = 200$
and $I = 2$, so SE drops to $1/\sqrt{2} \approx 0.71$ — exactly the
$\sqrt{n}$ shrinkage we expect. If $\sigma$ doubles to 20, $I$ drops to
0.25 and SE jumps to 2 — half the precision.

## Takeaway

Curvature is precision. Whenever you compute a Hessian at the MLE to
get standard errors, you are computing observed Fisher information.
The asymptotic theory promises you valid confidence intervals as long
as the log-likelihood is well-behaved at its peak.
