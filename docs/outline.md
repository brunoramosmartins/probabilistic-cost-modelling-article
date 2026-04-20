# Article Outline — The Shape of What You'll Spend

## Probabilistic Modelling of People Costs: from Distribution Selection to Budget Impact

---

## Section 1: Introduction (~600 words)

**Hook:** A budget is a probability statement disguised as a spreadsheet.

- The problem: budgets assume Normal distributions by default
- Why this matters: tail risk, multimodality, asymmetry
- What the article delivers: a framework for choosing distributions
- Connection to the Monte Carlo article (brief)
- Road map of the article

## Section 2: The Cost Components (~500 words)

- Define the cost model: salary, overtime, severance, hiring, benefits
- Each component as a random variable
- Why each has distinct distributional properties
- The concrete example: 50-person team
- Table: component → candidate distributions

## Section 3: Distribution Families (~700 words)

- The five candidates: Normal, LogNormal, Gamma, Pareto, Weibull
- For each: PDF, key properties, when it's appropriate
- Shape characterization: skewness, kurtosis, tail weight
- Comparison table
- Visual: the "distribution zoo" figure
- Why Normal is the wrong default for most cost components

## Section 4: Maximum Likelihood Estimation (~800 words)

- The estimation problem: given data and a family, find the best parameters
- Likelihood and log-likelihood
- The MLE principle: maximize ℓ(θ)
- Score function and Fisher information
- Asymptotic normality: confidence intervals for parameters
- MLE for specific distributions (closed-form and numerical)
- Visual: MLE convergence figure

## Section 5: Model Comparison (~700 words)

- The selection problem: given data and multiple fitted models, choose the best
- KL divergence: measuring distance between distributions
- AIC: balancing fit and complexity (information-theoretic)
- BIC: the Bayesian perspective on model selection
- Goodness-of-fit tests: KS, Anderson-Darling
- Likelihood ratio test for nested models
- Visual: model comparison table figure
- Decision guide: when to use AIC vs BIC vs GoF

## Section 6: Mixture Models and Multimodality (~700 words)

- The problem: salary data is often bimodal (junior vs senior)
- Why a single distribution fails
- Gaussian Mixture Models: the model
- The EM algorithm: E-step, M-step, convergence
- Choosing K: BIC for number of components
- Visual: mixture detection figure
- Connection to budget: multimodal salary → wider budget uncertainty

## Section 7: Heavy Tails and Extreme Costs (~800 words)

- The climax: this is where getting the distribution wrong hurts most
- Light vs heavy tails: formal definition
- The Pareto distribution and power-law decay
- Tail index and the Hill estimator
- EVT basics: what the theory guarantees
- Budget impact: VaR and Expected Shortfall under different models
- Visual: tail risk comparison figure
- The key result: Normal underestimates P(cost > 2x mean) by 10x

## Section 8: Experiments and Results (~1,200 words)

- Experiment A: Distribution Zoo — all families on same data
- Experiment B: MLE Convergence — consistency demonstration
- Experiment C: Wrong Distribution Impact — the cost of being wrong
- Experiment D: Mixture Detection — finding hidden structure
- Experiment E: Heavy Tail Risk — quantifying underestimation
- Experiment F: Model Comparison — systematic ranking
- Experiment G: End-to-End Pipeline — full workflow demonstration
- All figures embedded with captions

## Section 9: Practical Framework (~400 words)

- Decision tree: how to choose a distribution for your cost data
- Step 1: Visualize (histogram, Q-Q plot)
- Step 2: Fit candidates (MLE)
- Step 3: Compare (AIC/BIC)
- Step 4: Validate (GoF tests)
- Step 5: Quantify impact (budget consequences)
- Common pitfalls and how to avoid them

## Section 10: Conclusion (~300 words)

- Recap: the distribution is the model
- Key takeaways (3 bullet points)
- Connection to the Monte Carlo article
- Call to action: stop using Normal by default
- Links to code repository

---

## Total: ~6,700 words

## Notation (consistent throughout)

| Symbol | Meaning |
|--------|---------|
| $X$ | Random variable (cost component) |
| $f(x \mid \theta)$ | PDF of the model |
| $\theta$ | Parameter vector |
| $\hat{\theta}$ | MLE estimate |
| $\ell(\theta)$ | Log-likelihood |
| $S(\theta)$ | Score function |
| $I(\theta)$ | Fisher information |
| $n$ | Sample size |
| $k$ | Number of parameters |
| $D_{KL}(p \| q)$ | KL divergence from $q$ to $p$ |
