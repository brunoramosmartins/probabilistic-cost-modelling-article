# TIL — AIC Asymptotically Picks the Right Model — Asymptotically

**Phase:** 6 · **Topic:** Model selection, sample size, asymptotic theory · **Domain:** experimental statistics

## Hook

AIC and BIC are both presented as principled selection criteria. Both
come with consistency theorems. But "consistent" in statistics means
"as $n \to \infty$" — and your real sample is always finite.

## Insight

At small $n$, AIC has non-trivial probability of selecting the wrong
model — even when the true model is among the candidates. With $n = 30$,
in our experiments AIC selects LogNormal correctly only ~70% of the
time over the true Normal alternative. With $n = 100$, it is ~85%. With
$n = 1{,}000$, it is ~99%. BIC is more conservative — at small $n$ it
may favor a too-simple model, but as $n$ grows, it converges to the
truth more reliably than AIC. Both are guaranteed in the limit; the
small-sample behavior is what actually shows up in practice.

The practical implication: for budget data with $n = 50$ or $n = 100$,
do not bet a quarter on the AIC winner without validation. Bootstrap
the selection, check robustness across multiple datasets, and report
the Akaike weight — if the second-place model has weight > 0.1, the
choice is genuinely contested. The asymptotic theorems are real, but
they describe behavior at sample sizes you may not have.

## Example

A simulation with 100 replications of $n = 50$ data drawn from
LogNormal, with Normal and LogNormal as candidates. AIC picks LogNormal
in 78 cases, Normal in 22 cases. BIC picks LogNormal in 72 cases. Both
regularly fail at this sample size. At $n = 5{,}000$, both reach 100%
correct selection.

## Takeaway

"AIC selects the right model as $n \to \infty$" is true and largely
irrelevant in practice. What matters is selection probability at *your*
$n$. Run a small simulation with synthetic data matching your true
scale to estimate this — and beware whenever your real sample is so
small that the selection criterion itself is noisy.
