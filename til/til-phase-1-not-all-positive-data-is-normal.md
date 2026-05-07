# TIL — Positive Skewed Data Is Not Normal Just Because the Mean Is Comfortable

**Phase:** 1 · **Topic:** Distribution families, distribution mismatch · **Domain:** people-cost modelling

## Hook

When data is strictly positive and you have a sample mean and standard
deviation, it is tempting to wrap a Normal around it and move on. The
Normal is convenient, ubiquitous, and built into every spreadsheet.
It is also frequently wrong for cost data.

## Insight

The Normal distribution has support on $(-\infty, \infty)$ — it assigns
positive probability to negative values. For salaries, severance, or
hiring costs, this is a category error: those quantities cannot be
negative. More importantly, positive cost data is almost always
right-skewed (the mean is higher than the median because of a long right
tail), while the Normal is symmetric. Forcing a symmetric model on
asymmetric data produces three failures: it misestimates the location
of the typical value (the mean of a skewed distribution is not its
typical value), it severely misestimates tail probabilities, and it
underestimates the budget reserve needed at high confidence levels.

The right default for strictly positive, right-skewed cost data is the
LogNormal. It earns its place because it has the same multiplicative
generative story as most cost components — base value × multiplier ×
adjustment — and because its tail decay matches what real cost data
exhibits.

## Example

A team with median salary R\$ 9,000 and a few executives at R\$ 30,000.
A Normal fit gives mean ≈ R\$ 12,000 and σ ≈ R\$ 5,000. Under that
Normal, $P(\text{salary} > \text{R\$ 25,000}) \approx 0.5\%$. The data
is actually LogNormal — the true probability is closer to 4%. An 8x
underestimation, just from defaulting to Normal.

## Takeaway

For strictly positive, right-skewed data, the default candidate is
LogNormal — not Normal. The Normal earns its place by passing AIC/BIC
against alternatives, never by being the comfortable choice on a
spreadsheet.
