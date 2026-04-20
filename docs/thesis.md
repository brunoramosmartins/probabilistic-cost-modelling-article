# Thesis — The Shape of What You'll Spend

## Central Claim

A budget built on Normal distributions underestimates tail risk by construction.
People costs — salaries, overtime, severance — are structurally non-Normal:
right-skewed, heavy-tailed, and often multimodal. Choosing the wrong
distributional family is not a modelling nuance — it is a systematic bias that
propagates through every downstream calculation. Maximum Likelihood Estimation
provides the principled framework for fitting, and information-theoretic criteria
(AIC, BIC, likelihood ratio tests) provide the principled framework for choosing.

## Central Axis

The distribution you assume **is** your model. Everything else — mean, variance,
confidence intervals, risk estimates — flows from that choice.

```
Wrong distribution → wrong parameters → wrong budget → wrong decisions
```

## Scope

### In Scope

- **Distribution families:** Normal, LogNormal, Gamma, Pareto, Weibull
- **Estimation:** Maximum Likelihood Estimation (MLE), Fisher information,
  Cramer-Rao bound, asymptotic properties
- **Model comparison:** AIC, BIC, KL divergence, likelihood ratio tests,
  goodness-of-fit tests (KS, Anderson-Darling)
- **Mixture models:** Gaussian Mixture Models, EM algorithm derivation
- **Heavy tails:** Tail index, Extreme Value Theory basics, budget impact
- **Budget impact analysis:** VaR, Expected Shortfall, distribution choice
  consequences
- **Synthetic data:** Configurable data generators with known ground truth

### Anti-Scope (explicitly excluded)

- Bayesian estimation and posterior inference
- Time-series models (ARIMA, state-space)
- Markov Chain Monte Carlo (MCMC)
- Copulas and multivariate dependence
- Real company data (privacy, NDA constraints)
- Production-grade software or deployment
- Monte Carlo simulation (covered in Article 1)

## Target Audience

- Data scientists and analysts working with cost or financial data
- Budget analysts who want to move beyond point estimates
- ML engineers interested in statistical foundations
- Anyone who uses `scipy.stats.norm.fit()` and wants to understand when
  that's the wrong choice

### Prerequisites

- Calculus (differentiation, integration, Taylor series)
- Basic probability (PDF, CDF, expectation, variance)
- Linear algebra (matrix inversion for Fisher information)
- Familiarity with Python and NumPy

## Abstract

People costs — salaries, overtime, severance, hiring — are the largest and most
uncertain component of operational budgets. Standard practice models these costs
with Normal distributions, implicitly assuming symmetric, light-tailed behaviour.
This article demonstrates that people costs are structurally non-Normal:
right-skewed (salaries), heavy-tailed (severance), and often multimodal (salary
bands). We derive Maximum Likelihood Estimation from first principles, fit five
candidate distribution families to synthetic cost data, and use
information-theoretic criteria (AIC, BIC) and goodness-of-fit tests to select
the best model. We show that choosing a Normal distribution when costs follow a
LogNormal or Pareto leads to systematic underestimation of tail risk — in one
experiment, underestimating the probability of exceeding twice the expected cost
by a factor of 10x. The article provides a practical framework for distribution
selection in budget modelling, with reproducible code and publication-quality
visualizations.
