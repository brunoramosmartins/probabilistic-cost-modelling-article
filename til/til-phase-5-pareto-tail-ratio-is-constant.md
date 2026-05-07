# TIL — Pareto Halves Its Tail Probability by a Constant Factor

**Phase:** 5 · **Topic:** Heavy tails, Pareto, tail ratios · **Domain:** risk modelling

## Hook

Under a Normal distribution, doubling the threshold makes events
astronomically rarer. Under a Pareto, doubling the threshold reduces
probability by a constant factor — independent of where you started.
This single difference explains why heavy tails break Normal-based
intuition.

## Insight

For $X \sim \text{Pareto}(\alpha, x_m)$:

$$\frac{P(X > 2c)}{P(X > c)} = \frac{(x_m / 2c)^\alpha}{(x_m / c)^\alpha} = 2^{-\alpha}$$

The ratio depends only on $\alpha$, not on $c$. The tail "thins" by the
same multiplier every time you double the threshold. For $\alpha = 2.5$,
that multiplier is $2^{-2.5} \approx 0.177$ — about a 5.7-fold drop.
For Normal, the same ratio depends on $c$ and decays exponentially: at
$c = 3\sigma$, the ratio is roughly $e^{-13.5}/2 \approx 7 \times 10^{-7}$.
Astronomically smaller.

This single fact explains why heavy-tailed distributions break
Normal-based budget intuition. The "1-in-77,000" event under Normal
becomes "1-in-56" under a fitted Pareto with $\alpha = 2.5$ — a 138x
gap. And because the ratio is constant, doubling again produces another
1-in-56 reduction, not a 1-in-77,000² collapse.

## Example

Severance costs $\sim \text{Pareto}(2.5, 10000)$:

- $P(X > 20{,}000) \approx 0.177$
- $P(X > 40{,}000) \approx 0.177^2 \approx 0.031$
- $P(X > 80{,}000) \approx 0.177^3 \approx 0.0055$

Each doubling reduces probability by the same multiplier — the tail
never "runs out of mass" quickly the way a Normal does.

## Takeaway

When extreme costs follow a power law, "doubling the threshold" is not
a way to escape risk — it is just a slow taper. Capital reserves and
stop-loss provisions calibrated to Normal tail decay will be repeatedly
inadequate.
