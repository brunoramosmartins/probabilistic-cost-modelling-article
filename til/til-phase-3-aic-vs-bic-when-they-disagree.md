# TIL — AIC and BIC Answer Different Questions, Which Is Why They Disagree

**Phase:** 3 · **Topic:** Model selection, AIC, BIC · **Domain:** information theory, applied statistics

## Hook

AIC and BIC look almost identical: both subtract twice the
log-likelihood and add a complexity penalty. The penalty is $2k$ vs
$k \log n$. When $n > 7$, BIC penalizes more. When they pick different
models, which one is right?

## Insight

Both criteria are correct — for different questions. AIC estimates the
expected KL divergence between the true distribution and the fitted
model, so it is optimal for **prediction** (minimizing how surprised
your model is by future data). BIC approximates the Bayesian marginal
likelihood, so it is optimal for **identification** (selecting the model
that actually generated the data, asymptotically).

When data is plentiful, BIC tends to pick simpler models because the
$\log n$ penalty grows with sample size and discourages extra
parameters. AIC, with its fixed $2k$ penalty, tolerates extra parameters
if they provide tiny prediction gains. The two criteria embody different
notions of "best": AIC is asymptotically efficient (minimizes prediction
risk), BIC is asymptotically consistent (recovers the true model when
$n \to \infty$).

In practice this means: AIC may overfit slightly when the candidate set
contains a model close to the truth but with extra parameters. BIC may
underfit when the true model has many small effects. Neither is wrong;
they optimize different things.

## Example

200 salary observations from a slightly-skewed-but-mostly-Normal
distribution. AIC might prefer LogNormal (better tail fit, marginal
predictive gain). BIC might prefer Normal (the extra structural
complexity is not justified by the evidence). Both are right in their
own frame. With $n = 2{,}000$, both will likely converge on the same
answer because the LogNormal advantage becomes statistically robust.

## Takeaway

For budget prediction (where your VaR will be tested against future
data), use AIC. For "what is the true cost-generation mechanism?" (e.g.,
to explain to the CFO why you are switching models), use BIC. When in
doubt, report both and let the disagreement become part of the analysis.
