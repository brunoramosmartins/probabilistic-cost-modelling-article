# TIL — 138x: How Much Tail Risk a Normal Hides for Severance Costs

**Phase:** 5 · **Topic:** Heavy tails, distribution mismatch, budget impact · **Domain:** people-cost modelling

## Hook

One number tells the entire story of why distribution choice matters
in budget modelling: **138**.

## Insight

Take severance costs. Fit a Pareto with shape $\alpha = 2.5$ and scale
$x_m = $ R\$ 10,000 — typical for processes where most events are
moderate but a few are extreme. Now match the first two moments with a
Normal: $\mu = $ R\$ 16,667, $\sigma \approx$ R\$ 14,907. Both
distributions have the same mean and the same variance. Under the
Pareto, the probability of a single severance event exceeding R\$ 50,000
is **1.79%**. Under the moment-matched Normal, it is **0.013%**. The
ratio is 138x.

The Normal is not "a little off." It assigns essentially zero
probability to events the Pareto considers routine. For a 50-person
team with about 3 severance events per year, the Normal predicts that
R\$ 50,000+ events happen maybe once every 25 years. The Pareto
predicts roughly one every two years. Capital reserved on the Normal
forecast will not cover the Pareto reality.

The disparity grows further into the tail because Normal decays
super-exponentially while Pareto decays polynomially. Beyond R\$ 100,000,
the ratio becomes effectively infinite — the Normal predicts something
"once in a billion years," the Pareto predicts roughly once a decade.

## Example

Same model. $P(X > $ R\$ 100,000$)$: Pareto = 0.032%, Normal $\approx$ 0%.
$P(X > $ R\$ 200,000$)$: Pareto = 0.0057%, Normal on the order of
$10^{-15}$. At every threshold past the first standard deviation of the
Normal, the gap explodes.

## Takeaway

When a single distributional choice changes a tail probability by 100x,
that choice is not "a methodological detail." It is the difference
between a budget that holds and a budget that breaks. The 138x is not a
worst case — for thinner Pareto tails (smaller $\alpha$, closer to 1)
the gap can exceed 1000x.
