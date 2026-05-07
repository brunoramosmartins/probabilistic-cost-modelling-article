# TIL — Today I Learned

Short, portfolio-ready notes capturing one non-obvious insight from each phase
of the article *The Shape of What You'll Spend — Probabilistic Modelling of
People Costs*. Each TIL is a **skeleton**: hook → insight → example → takeaway.
The author writes the final text in his own voice.

## Why TILs?

The article publishes once. The TILs publish continuously — one per phase on
LinkedIn, the personal blog, or Medium. They keep the work visible while the
article matures, and let me go deeper on points the article only touches.

## Format

Each file:

- Title that states the **insight**, not the topic
- 200–350 words
- One concrete example (numbers or a small derivation)
- One-line takeaway
- Tags: phase, math branch, applied domain

## Index

Each phase has at least one TIL. Phase 5 (Heavy Tails) has three because it
is the article's central insight and warrants both a foundational entry and
two payoff entries. Pick whichever fits the audience of the moment, or run
them as a series.

| Phase | TIL file                                                                                  | Insight                                                  |
|-------|-------------------------------------------------------------------------------------------|----------------------------------------------------------|
| 1     | [til-phase-1-not-all-positive-data-is-normal.md](til-phase-1-not-all-positive-data-is-normal.md) | Positive skewed data is not Normal — it just looks like it |
| 1     | [til-phase-1-lognormal-trick.md](til-phase-1-lognormal-trick.md)                         | If $Y$ is Normal, $e^Y$ inherits everything you know     |
| 2     | [til-phase-2-fisher-information-as-curvature.md](til-phase-2-fisher-information-as-curvature.md) | Fisher information = curvature of the log-likelihood     |
| 2     | [til-phase-2-why-mle-variance-is-biased.md](til-phase-2-why-mle-variance-is-biased.md)   | MLE variance is biased — and that's a feature            |
| 3     | [til-phase-3-aic-vs-bic-when-they-disagree.md](til-phase-3-aic-vs-bic-when-they-disagree.md) | AIC for prediction, BIC for identification               |
| 3     | [til-phase-3-akaike-weights-as-probability.md](til-phase-3-akaike-weights-as-probability.md) | Akaike weights are a Bayesian posterior in disguise      |
| 4     | [til-phase-4-the-average-employee-does-not-exist.md](til-phase-4-the-average-employee-does-not-exist.md) | In a bimodal distribution, the mean represents no one    |
| 4     | [til-phase-4-em-as-coordinate-ascent-on-elbo.md](til-phase-4-em-as-coordinate-ascent-on-elbo.md) | EM is coordinate ascent on a lower bound (the ELBO)      |
| 5     | [til-phase-5-pareto-tail-ratio-is-constant.md](til-phase-5-pareto-tail-ratio-is-constant.md) | Pareto halves its tail by a constant factor              |
| 5     | [til-phase-5-why-mgf-fails-for-pareto.md](til-phase-5-why-mgf-fails-for-pareto.md)       | A distribution is heavy-tailed iff its MGF diverges      |
| 5     | [til-phase-5-138x-what-normal-hides.md](til-phase-5-138x-what-normal-hides.md)           | 138x: how much tail risk a Normal hides for severance    |
| 6     | [til-phase-6-fixed-seeds-and-reproducibility.md](til-phase-6-fixed-seeds-and-reproducibility.md) | A fixed seed is the cheapest insurance policy            |
| 6     | [til-phase-6-aic-consistency-vs-sample-size.md](til-phase-6-aic-consistency-vs-sample-size.md)   | AIC is consistent — asymptotically                       |

## Publishing order

Draft in this folder during each phase, publish after the phase PR is merged.
Recommended LinkedIn cadence:

1. **Hook week:** `til-phase-5-138x-what-normal-hides.md`
2. **Foundation week:** `til-phase-1-not-all-positive-data-is-normal.md`
3. **Accessible technique:** `til-phase-4-the-average-employee-does-not-exist.md`
4. **Tooling:** `til-phase-1-lognormal-trick.md`
5. **Selection:** `til-phase-3-akaike-weights-as-probability.md`
6. **Tail intuition:** `til-phase-5-pareto-tail-ratio-is-constant.md`
7. **Theory:** `til-phase-2-fisher-information-as-curvature.md`
8. **Selection nuance:** `til-phase-3-aic-vs-bic-when-they-disagree.md`
9. **Heavy-tail formality:** `til-phase-5-why-mgf-fails-for-pareto.md`
10. **MLE detail:** `til-phase-2-why-mle-variance-is-biased.md`
11. **ML bridge:** `til-phase-4-em-as-coordinate-ascent-on-elbo.md`
12. **Reproducibility:** `til-phase-6-fixed-seeds-and-reproducibility.md`
13. **Selection caveat:** `til-phase-6-aic-consistency-vs-sample-size.md`

Each post links to the GitHub folder so the audience can follow progress and
trace back to the article and code.
