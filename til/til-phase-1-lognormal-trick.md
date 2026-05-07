# TIL — If $Y$ Is Normal, $e^Y$ Inherits Everything You Know About Normals

**Phase:** 1 · **Topic:** LogNormal distribution, log-transform · **Domain:** people-cost modelling

## Hook

The LogNormal looks intimidating until you realize it is just a Normal
in disguise. Once you spot the disguise, every result you already know
about Normals — MLE formulas, confidence intervals, hypothesis tests —
extends "for free."

## Insight

The LogNormal is defined exactly so that $\log(X) \sim N(\mu, \sigma^2)$.
This is more than a definition — it is a workflow. If your data is
right-skewed and positive, take $\log$ of every observation, fit a
Normal to those, and convert back. The MLE for $\mu$ becomes the mean
of the log-data. The MLE for $\sigma^2$ becomes the variance of the
log-data. Confidence intervals work on the log scale. The whole
apparatus of Normal-based statistics ports over without re-derivation.

This works because so many cost components have a multiplicative
generative process — a salary is a base × performance × tenure × market
adjustment, all positive multipliers. On the log scale, multiplicative
becomes additive, and the Central Limit Theorem turns the sum of those
log-multipliers into something approximately Normal.

## Example

Salary data (in thousands): {8.2, 9.1, 7.5, 11.3, 10.8, 8.7, 9.5, 12.1,
8.9, 10.2}. Take $\log$ of each: {2.10, 2.21, 2.01, 2.42, 2.38, 2.16,
2.25, 2.49, 2.19, 2.32}. Mean = 2.25, variance = 0.024. So the LogNormal
MLE is $\hat{\mu} = 2.25$, $\hat{\sigma}^2 = 0.024$. No optimization, no
special software — just a log-transform and standard arithmetic.

To convert back: median salary = $e^{2.25} \approx$ R\$ 9,500. The 95th
percentile of the log-distribution is at $2.25 + 1.645 \cdot \sqrt{0.024} = 2.51$,
which on the original scale is $e^{2.51} \approx$ R\$ 12,300.

## Takeaway

The LogNormal trick — log-transform, do Normal stuff, exponentiate back
— is the cheapest way to handle right-skewed positive data. Whenever a
cost component arises from a multiplicative process (and most do), this
is the model to try first.
