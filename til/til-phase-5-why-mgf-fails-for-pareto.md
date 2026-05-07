# TIL — A Distribution Is "Heavy-Tailed" Iff Its MGF Diverges

**Phase:** 5 · **Topic:** Heavy tails, Moment Generating Function · **Domain:** probability theory, risk modelling

## Hook

The everyday phrase "heavy tails" sounds informal. There is a precise
mathematical definition that separates the well-behaved Normal from the
wild Pareto — and it gates which statistical tools are even applicable.

## Insight

The Moment Generating Function $M_X(t) = E[e^{tX}]$ either exists for
some $t > 0$ or diverges for all $t > 0$. If it exists, the
distribution is **light-tailed** in the technical sense: every moment
is finite, and the survival probability $P(X > x)$ decays at least
exponentially. If it diverges, the distribution is **heavy-tailed**:
some moments may be infinite, and the survival decays only polynomially
or slower.

The Pareto's MGF diverges because the integral
$\int e^{tx} \cdot x^{-\alpha-1} \, dx$ has integrand growing
exponentially while the density only decays polynomially. Exponential
always beats polynomial.

This is not a curiosity. Many statistical tools — Chernoff bounds,
Cramér's large-deviations theorem, MGF-based hypothesis tests — require
the MGF to exist. They simply do not apply to heavy-tailed data, and
using them anyway produces meaningless results. Knowing whether your
MGF exists is the gatekeeper between "Normal-style statistics" and
"heavy-tail-aware statistics."

## Example

- **Normal**: $M_X(t) = \exp(\mu t + \sigma^2 t^2 / 2)$ — exists for all
  $t \in \mathbb{R}$. All moments finite. Concentration inequalities
  (Chernoff, Hoeffding) apply.
- **Pareto** $(\alpha = 2.5, x_m = 1)$: $E[e^X]$ diverges. There is no
  exponential moment, no Chernoff bound, no central limit refinement
  involving exponential moments.

For severance cost data following Pareto, you cannot legitimately use
"Hoeffding-style" probability bounds — the assumptions fail, and any
"bound" computed under them is a fiction.

## Takeaway

Before reaching for any tool that assumes "exponential decay" —
including most of classical concentration inequalities — verify the MGF
exists. For heavy-tailed cost data (severance, claim sizes, file sizes),
you need methods explicitly built for the regime: extreme value theory,
Hill estimators, ES-based risk measures.
