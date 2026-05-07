# TIL — The MLE for Variance Is Biased, and That Is a Feature

**Phase:** 2 · **Topic:** MLE, bias, sample size · **Domain:** statistical inference

## Hook

The Maximum Likelihood Estimator for $\sigma^2$ in a Normal model is
$\frac{1}{n}\sum(x_i - \bar{x})^2$. The "unbiased" version uses
$\frac{1}{n-1}$. Why does the textbook prefer one and the calculator
the other?

## Insight

$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum(x_i - \bar{x})^2$ systematically
underestimates the true variance because it uses $\bar{x}$ — itself
estimated from the same data. Each observation has been "used up" once
to estimate the mean, leaving only $n-1$ effective degrees of freedom.
The expected value of $\hat{\sigma}^2_{MLE}$ is $\frac{n-1}{n} \sigma^2$,
not $\sigma^2$. The unbiased estimator scales by $\frac{1}{n-1}$ to
correct this.

But the MLE has a property the unbiased estimator lacks: it minimizes
the expected KL divergence to the true distribution among all
estimators of that form. In other words, $\hat{\sigma}^2_{MLE}$ is
biased downward but produces the model whose predictions track the
true distribution most closely. For prediction (the budget question),
this matters more than unbiasedness.

The deeper point: the bias is $O(1/n)$ and disappears asymptotically.
The /n vs /(n-1) debate is a small-sample artifact that vanishes when
data is plentiful. With $n = 10$, the bias is 10%. With $n = 100$, it
is 1%. With $n = 1000$, it is 0.1% — invisible.

## Example

Synthetic data drawn from $N(0, 1)$ with $n = 10$. The MLE estimate of
$\sigma^2$ averages around 0.9 across many replications — biased
downward by 10%, exactly as $\frac{n-1}{n} = 0.9$ predicts. The
unbiased estimator averages around 1.0. With $n = 1000$, both produce
~1.0 to three decimal places.

## Takeaway

$\hat{\sigma}^2_{MLE}$ is biased, but the bias is $O(1/n)$ and
disappears asymptotically. The /n vs /(n-1) debate is a small-sample
artifact — and for predictive purposes, MLE still wins. Use MLE for
modelling and prediction; switch to unbiased only if your sample is
small and you specifically need an unbiased estimate.
