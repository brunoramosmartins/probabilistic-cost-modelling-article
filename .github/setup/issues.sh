#!/bin/bash
# Creates all project issues with full bodies. Run after labels and milestones.
# Usage: bash .github/setup/issues.sh owner/repo
#
# IMPORTANT: This script assumes milestones were created in order (IDs 1–9).
# If milestone IDs differ, adjust the --milestone flags accordingly.
# Run `gh api repos/OWNER/REPO/milestones` to check IDs.

set -euo pipefail

REPO="${1:?Usage: bash issues.sh owner/repo}"

echo "Creating issues for $REPO..."

# ──────────────────────────────────────────────
# PHASE 0 — Foundation
# ──────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "[Phase 0] Write thesis and define scope" \
  --label "phase:0,type:documentation,priority:high" \
  --milestone "Phase 0 — Foundation" \
  --body "## Context
The thesis anchors the article. It must be a falsifiable claim about why
distributional choice matters for budget accuracy, not just a tutorial overview.

## Tasks
- [ ] Draft central claim (v0.1)
- [ ] Define scope: distribution families, MLE, GoF, model comparison, mixture models, heavy tails, budget impact
- [ ] Define anti-scope: Bayesian estimation, time-series, MCMC, copulas, real company data
- [ ] Identify target audience and prerequisites
- [ ] Write 1-paragraph abstract

## Definition of Done
- [ ] \`docs/thesis.md\` exists with thesis, scope, anti-scope, audience, abstract
- [ ] Thesis is a single falsifiable sentence
- [ ] Scope clearly separates theory from application phases

## References
- Project motivation (IT headcount budgeting context)"

gh issue create --repo "$REPO" \
  --title "[Phase 0] Design cost model and document expansion points" \
  --label "phase:0,type:documentation,priority:high" \
  --milestone "Phase 0 — Foundation" \
  --body "## Context
The cost model defines which cost components are modelled probabilistically.
It must be generic, minimal, and expandable.

## Tasks
- [ ] Define cost components: base salary, overtime, benefits, severance, hiring costs
- [ ] For each: suggest candidate distributions with rationale
- [ ] Document default parameters for synthetic data generation
- [ ] List expansion points (marked as v2)
- [ ] Write a concrete example: 50-person team with specific parameters

## Definition of Done
- [ ] \`docs/model-design.md\` exists with full specification
- [ ] Each component has at least 2 candidate distributions
- [ ] A reader could generate synthetic data from this document alone

## References
- Budget Model Design section of this roadmap"

gh issue create --repo "$REPO" \
  --title "[Phase 0] Configure repository, GitHub templates, and Claude Code rules" \
  --label "phase:0,type:infrastructure,priority:high" \
  --milestone "Phase 0 — Foundation" \
  --body "## Context
Professional repository setup from day one. GitHub configuration ensures
consistent tracking. Claude Code rules enforce project conventions.

## Tasks
- [ ] Initialize all directories with .gitkeep where needed
- [ ] Create \`.claude/CLAUDE.md\` with all project rules
- [ ] Create \`.github/ISSUE_TEMPLATE/task.md\` and \`bug.md\`
- [ ] Create \`.github/PULL_REQUEST_TEMPLATE.md\`
- [ ] Create \`.github/setup/labels.sh\`
- [ ] Create \`.github/setup/milestones.sh\`
- [ ] Create \`.github/setup/issues.sh\`
- [ ] Write \`requirements.txt\` with pinned versions
- [ ] Write \`pyproject.toml\` with ruff config
- [ ] Write initial \`README.md\`

## Definition of Done
- [ ] All directories exist
- [ ] \`bash .github/setup/labels.sh owner/repo\` creates all labels
- [ ] \`bash .github/setup/milestones.sh owner/repo\` creates all milestones
- [ ] Claude Code loads \`.claude/CLAUDE.md\` and follows rules
- [ ] \`pip install -r requirements.txt\` succeeds

## References
- Repository structure in roadmap
- Claude Code configuration in roadmap"

# ──────────────────────────────────────────────
# PHASE 1 — Distribution Families
# ──────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "[Phase 1] Derive properties of candidate distribution families" \
  --label "phase:1,type:theory,priority:critical" \
  --milestone "Phase 1 — Distribution Families" \
  --body "## Context
Before fitting distributions, we must understand their mathematical properties.
Each candidate family has distinct shape characteristics that make it more or
less suitable for specific cost components.

## Tasks
- [ ] For each distribution (Normal, LogNormal, Gamma, Pareto, Weibull):
      - Write the PDF and CDF
      - Derive E[X] and Var(X) from the PDF
      - Derive the MGF (where tractable)
      - Characterize the shape: symmetry, skewness, kurtosis, tail behaviour
      - State the support (domain)
- [ ] Create comparison table: family x property
- [ ] For each cost component, justify which families are plausible candidates
- [ ] Document in \`notes/phase1-distributions.md\`

## Definition of Done
- [ ] All 5 distributions fully characterized with step-by-step derivations
- [ ] Comparison table is complete
- [ ] Cost-component-to-distribution mapping is justified, not arbitrary
- [ ] Each derivation includes a numerical example

## References
- Casella & Berger, Ch. 3 (common distributions)
- Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions"

gh issue create --repo "$REPO" \
  --title "[Phase 1] Implement distribution catalogue and synthetic data generator" \
  --label "phase:1,type:code,priority:high" \
  --milestone "Phase 1 — Distribution Families" \
  --body "## Context
The code infrastructure for the entire article: a catalogue of distribution
helpers and a generator for synthetic cost data with known ground truth.

## Tasks
- [ ] Implement \`src/distributions.py\`:
      - Wrapper class for each distribution (PDF, CDF, sample, moments)
      - Unified interface: \`dist.pdf(x)\`, \`dist.cdf(x)\`, \`dist.sample(n)\`
      - Analytical moments: \`dist.mean()\`, \`dist.var()\`
- [ ] Implement \`src/data_gen.py\`:
      - \`generate_salary_data(n, dist, params, seed)\` -> array
      - \`generate_overtime_data(n, params, seed)\` -> array
      - \`generate_mixed_data(n, components, weights, seed)\` -> array (for mixture)
      - \`inject_outliers(data, fraction, multiplier)\` -> array
- [ ] Create tests in \`tests/\` (DO NOT RUN)

## Definition of Done
- [ ] Both modules implemented with type hints and docstrings
- [ ] Tests created
- [ ] Reminded author to run \`pytest tests/\` and \`ruff check .\`

## References
- Theory from Issue #4
- \`scipy.stats\` distribution API"

# ──────────────────────────────────────────────
# PHASE 2 — MLE
# ──────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "[Phase 2] Derive MLE theory: likelihood, score, Fisher information" \
  --label "phase:2,type:theory,priority:critical" \
  --milestone "Phase 2 — Maximum Likelihood Estimation" \
  --body "## Context
MLE is THE estimation framework for this article. Every distribution fit
uses MLE, so the theory must be derived rigorously: from likelihood to
score to Fisher information to asymptotic normality.

## Tasks
- [ ] Define the likelihood function: L(theta|x) = prod f(x_i|theta)
- [ ] Define log-likelihood: l(theta) = sum log f(x_i|theta)
- [ ] Derive the score function: S(theta) = dl/dtheta
- [ ] Prove: E[S(theta_0)] = 0 (score has zero mean at true parameter)
- [ ] Define Fisher information: I(theta) = Var(S(theta)) = -E[d2l/dtheta2]
- [ ] Prove the two forms of Fisher information are equivalent
- [ ] State and prove (or prove sketch) asymptotic normality of MLE:
      sqrt(n)(theta_hat - theta_0) ->_d N(0, I(theta_0)^-1)
- [ ] Derive MLE for specific distributions:
      - Normal(mu, sigma2): closed-form
      - LogNormal(mu, sigma2): closed-form
      - Gamma(alpha, beta): score equations (no closed form for alpha)
      - Pareto(alpha, x_m): closed-form
- [ ] Prove: MLE is consistent and asymptotically efficient (Cramer-Rao bound)

## Definition of Done
- [ ] Complete derivation chain: likelihood -> score -> Fisher -> asymptotics
- [ ] MLE derived for all 4 distributions
- [ ] Cramer-Rao bound stated and connected to MLE efficiency
- [ ] Each derivation has a worked numerical example

## References
- Casella & Berger, Ch. 7 (point estimation)
- Lehmann & Casella, Theory of Point Estimation"

gh issue create --repo "$REPO" \
  --title "[Phase 2] Implement MLE fitting pipeline" \
  --label "phase:2,type:code,priority:high" \
  --milestone "Phase 2 — Maximum Likelihood Estimation" \
  --body "## Context
The fitting pipeline wraps scipy.optimize to find MLE for each candidate
distribution given data. Must handle edge cases and report Fisher-information-
based confidence intervals for parameters.

## Tasks
- [ ] Implement \`src/fitting.py\`:
      - \`fit_mle(data, distribution)\` -> FitResult(params, loglik, se, ci)
      - Use analytical solutions where available (Normal, LogNormal, Pareto)
      - Use \`scipy.optimize.minimize\` for Gamma, Weibull
      - Compute standard errors from observed Fisher information
- [ ] Implement \`fit_all(data, candidates)\` -> list of FitResult, sorted by loglik
- [ ] Create tests (DO NOT RUN):
      - Recover known parameters from large samples
      - SE shrinks with sqrt(n)
      - CI covers true parameter in repeated trials

## Definition of Done
- [ ] Pipeline fits all 5 distributions
- [ ] SE and CI computed from Fisher information
- [ ] Tests created and author reminded to run them

## References
- Theory from Issue #6
- \`scipy.optimize.minimize\` documentation"

# ──────────────────────────────────────────────
# PHASE 3 — Model Comparison
# ──────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "[Phase 3] Derive AIC, BIC, KL divergence, and likelihood ratio test" \
  --label "phase:3,type:theory,priority:critical" \
  --milestone "Phase 3 — Model Comparison" \
  --body "## Context
Fitting multiple distributions to the same data requires principled model
selection. This issue derives the information-theoretic and hypothesis-testing
frameworks for choosing between models.

## Tasks
- [ ] Define KL divergence: D_KL(p||q) = integral p(x) log(p(x)/q(x)) dx
- [ ] Prove: D_KL >= 0 (Gibbs' inequality) and D_KL = 0 iff p = q a.e.
- [ ] Derive AIC from KL divergence:
      - AIC ~ 2k - 2l(theta_hat) as estimator of expected KL
      - Explain the bias-correction term 2k
      - AICc for small samples
- [ ] Derive BIC from Bayesian model evidence:
      - BIC ~ k*log(n) - 2l(theta_hat)
      - Explain why BIC penalizes complexity more than AIC
- [ ] Derive the likelihood ratio test:
      - Lambda = -2[l(theta_hat_0) - l(theta_hat_1)]
      - Wilks' theorem: Lambda ->_d chi2(Delta_k) under H0
      - Apply to nested models
- [ ] Discuss non-nested comparison: Vuong test (brief)
- [ ] Goodness-of-fit tests:
      - Kolmogorov-Smirnov: definition, distribution-free property
      - Anderson-Darling: tail-sensitive variant
      - When to use each

## Definition of Done
- [ ] KL divergence defined and non-negativity proved
- [ ] AIC derived from KL (not just stated as a formula)
- [ ] BIC derived from Bayesian marginal likelihood
- [ ] LRT derived with Wilks' theorem
- [ ] GoF tests defined with interpretation guidelines
- [ ] Each criterion includes a worked example comparing 2 distributions

## References
- Burnham & Anderson, Model Selection and Multimodel Inference
- Claeskens & Hjort, Model Selection and Model Averaging
- Lehmann & Romano, Testing Statistical Hypotheses"

gh issue create --repo "$REPO" \
  --title "[Phase 3] Implement model comparison tools" \
  --label "phase:3,type:code,priority:high" \
  --milestone "Phase 3 — Model Comparison" \
  --body "## Context
The model comparison module takes FitResult objects from the fitting pipeline
and ranks models using AIC, BIC, and GoF tests.

## Tasks
- [ ] Implement \`src/model_selection.py\`:
      - \`compute_aic(loglik, k)\` and \`compute_bic(loglik, k, n)\`
      - \`aic_weights(aic_values)\` -> Akaike weights for model averaging
      - \`likelihood_ratio_test(loglik_0, loglik_1, df)\` -> (statistic, p_value)
      - \`ks_test(data, distribution, params)\` -> (statistic, p_value)
      - \`ad_test(data, distribution, params)\` -> (statistic, p_value)
      - \`compare_models(fit_results)\` -> comparison table (DataFrame)
- [ ] Create tests (DO NOT RUN):
      - AIC selects true model in large samples
      - BIC selects true model (stronger consistency)
      - KS test rejects wrong distribution at alpha = 0.05

## Definition of Done
- [ ] All comparison tools implemented
- [ ] \`compare_models\` produces a clean summary table
- [ ] Tests created, author reminded to run

## References
- Theory from Issue #8"

# ──────────────────────────────────────────────
# PHASE 4 — Mixture Models
# ──────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "[Phase 4] Derive EM algorithm for Gaussian Mixture Models" \
  --label "phase:4,type:theory,priority:critical" \
  --milestone "Phase 4 — Mixture Models" \
  --body "## Context
Salary data is often multimodal: juniors, seniors, and directors form distinct
clusters. A single distribution cannot capture this structure. The EM algorithm
is the standard approach for fitting mixture models.

## Tasks
- [ ] Define the GMM: f(x) = sum_k pi_k * N(x|mu_k, sigma2_k)
- [ ] Explain why direct MLE fails (log of sums != sum of logs)
- [ ] Derive the EM algorithm:
      E-step: compute responsibilities
      M-step: update pi_k, mu_k, sigma2_k from responsibilities
- [ ] Prove: EM monotonically increases the log-likelihood (Jensen's inequality)
- [ ] Discuss convergence: to local maximum, not necessarily global
- [ ] Derive BIC for choosing number of components K
- [ ] Connect to budget: multimodal salary -> multimodal total cost

## Definition of Done
- [ ] EM derivation is complete (E-step, M-step, monotonicity proof)
- [ ] Jensen's inequality argument is explicit
- [ ] BIC for K selection is derived
- [ ] A 2-component example is worked through numerically

## References
- Bishop, Pattern Recognition and Machine Learning, Ch. 9
- McLachlan & Peel, Finite Mixture Models"

gh issue create --repo "$REPO" \
  --title "[Phase 4] Implement GMM fitting and multimodality detection" \
  --label "phase:4,type:code,priority:high" \
  --milestone "Phase 4 — Mixture Models" \
  --body "## Context
The mixture module fits GMMs to salary data and detects multimodality,
answering: is the salary distribution really unimodal, or are there
distinct population clusters?

## Tasks
- [ ] Implement \`src/mixture.py\`:
      - \`fit_gmm(data, K, max_iter, tol, seed)\` -> GMMResult
      - \`select_K(data, K_range, criterion='bic')\` -> optimal K
      - \`detect_multimodality(data)\` -> bool + evidence
      - Use own EM implementation (for the article), with scipy fallback for validation
- [ ] Create tests (DO NOT RUN):
      - Recovers parameters of known 2-component mixture
      - BIC selects correct K for 1, 2, 3 component data
      - Convergence within max_iter for well-separated components

## Definition of Done
- [ ] Own EM implementation matches scipy/sklearn within tolerance
- [ ] K selection works correctly on synthetic data
- [ ] Tests created, author reminded to run

## References
- Theory from Issue #10
- sklearn.mixture.GaussianMixture for validation"

# ──────────────────────────────────────────────
# PHASE 5 — Heavy Tails
# ──────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "[Phase 5] Derive heavy-tail theory: tail index, Pareto, EVT basics" \
  --label "phase:5,type:theory,priority:high" \
  --milestone "Phase 5 — Heavy Tails" \
  --body "## Context
The key practical message of the article: Normal distributions underestimate
tail risk. Heavy-tailed distributions (Pareto, LogNormal tails) assign much
higher probability to extreme costs. This is where getting the distribution
wrong has the biggest budget impact.

## Tasks
- [ ] Define heavy tails formally:
      - Light-tailed: P(X > x) decays exponentially (Normal, Gamma)
      - Heavy-tailed: P(X > x) decays polynomially (Pareto)
      - Sub-exponential: LogNormal (between light and heavy)
- [ ] Define the tail index alpha: P(X > x) ~ x^{-alpha} as x -> infinity
- [ ] Derive tail behaviour of key distributions
- [ ] Introduce Extreme Value Theory basics:
      - Block maxima: Generalized Extreme Value (GEV) distribution
      - Fisher-Tippett-Gnedenko theorem (statement)
      - Three domains of attraction: Gumbel, Frechet, Weibull
- [ ] Hill estimator for tail index
- [ ] Quantify budget impact:
      - P(cost > 2x expected) under Normal vs Pareto
      - Value at Risk (VaR) and Expected Shortfall (ES) comparison

## Definition of Done
- [ ] Heavy vs light tail formally defined with examples
- [ ] Tail behaviour derived for Normal, LogNormal, Pareto
- [ ] EVT basics stated (not fully proved)
- [ ] Budget impact quantified with concrete numbers
- [ ] Hill estimator described with interpretation

## References
- Embrechts, Kluppelberg & Mikosch, Modelling Extremal Events
- Resnick, Heavy-Tail Phenomena
- Coles, An Introduction to Statistical Modeling of Extreme Values"

gh issue create --repo "$REPO" \
  --title "[Phase 5] Implement tail analysis tools and budget impact comparison" \
  --label "phase:5,type:code,priority:high" \
  --milestone "Phase 5 — Heavy Tails" \
  --body "## Context
The tail analysis tools quantify the practical difference between light-
and heavy-tailed models: how much more budget reserve do you need when
costs follow a Pareto instead of a Normal?

## Tasks
- [ ] Implement \`src/heavy_tails.py\`:
      - \`hill_estimator(data, k)\` -> tail index estimate
      - \`tail_probability(distribution, params, threshold)\` -> P(X > threshold)
      - \`qq_plot_data(data, distribution)\` -> (theoretical, empirical) quantiles
      - \`compare_tail_risk(data, distributions, threshold)\` -> comparison table
- [ ] Implement \`src/budget_impact.py\`:
      - \`var_at_level(samples, alpha)\` -> Value at Risk
      - \`expected_shortfall(samples, alpha)\` -> ES (CVaR)
      - \`budget_reserve(samples, confidence)\` -> reserve needed above mean
      - \`compare_distributions_impact(data, fits, budget_ceiling)\` ->
        table with P(overbudget), VaR, ES for each distribution
- [ ] Create tests (DO NOT RUN)

## Definition of Done
- [ ] Hill estimator validated on Pareto samples (known alpha)
- [ ] VaR and ES match analytical values for Normal
- [ ] Comparison table clearly shows impact of distribution choice
- [ ] Tests created, author reminded to run

## References
- Theory from Issue #12"

# ──────────────────────────────────────────────
# PHASE 6 — Experiments
# ──────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "[Phase 6] Run all experiments and create publication-quality figures" \
  --label "phase:6,type:experiment,priority:critical" \
  --milestone "Phase 6 — Experiments & Visualizations" \
  --body "## Context
All experiments that produce figures for the article. Each experiment
isolates one concept and validates one theoretical claim.

## Tasks
- [ ] Experiment A — Distribution Zoo:
      Visualize all 5 candidate distributions fitted to same synthetic salary data
- [ ] Experiment B — MLE Convergence:
      Show MLE estimates converging to true params as n grows
- [ ] Experiment C — Wrong Distribution Impact:
      Fit Normal to LogNormal data; compare P(overbudget) predictions
- [ ] Experiment D — Mixture Detection:
      Generate bimodal salary data; show GMM detects 2 components
- [ ] Experiment E — Heavy Tail Risk:
      Compare P(cost > 2x mean) under Normal vs Pareto vs LogNormal
- [ ] Experiment F — Model Comparison:
      Full AIC/BIC/KS comparison table on synthetic data
- [ ] Experiment G — End-to-End Pipeline:
      Data -> fit all -> select best -> compute budget impact
- [ ] All figures: 300 DPI, clean labels, consistent style
- [ ] All scripts use fixed seeds

## Definition of Done
- [ ] All 7 experiments produce figures in \`figures/\`
- [ ] Each script runs standalone
- [ ] Figures are publication-quality
- [ ] Author has run all scripts and verified outputs

## References
- All theory from Phases 1-5"

# ──────────────────────────────────────────────
# PHASE 7 — Article Writing
# ──────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "[Phase 7] Write article sections 1-5 (foundations through model comparison)" \
  --label "phase:7,type:writing,priority:high" \
  --milestone "Phase 7 — Article Writing" \
  --body "## Context
The first half builds the problem and the tools: from why distributions
matter, through MLE, to model comparison.

## Tasks
- [ ] Section 1: Introduction (~600 words) — hook + why distributions matter
- [ ] Section 2: The cost components (~500 words) — salary, overtime, incidents
- [ ] Section 3: Distribution families (~700 words) — catalogue with properties
- [ ] Section 4: Maximum Likelihood Estimation (~800 words) — full derivation
- [ ] Section 5: Model comparison (~700 words) — AIC, BIC, GoF
- [ ] Embed relevant figures
- [ ] Consistent notation throughout

## Definition of Done
- [ ] Sections 1-5 complete in \`article/probabilistic-cost-modelling.md\`
- [ ] All derivations self-contained
- [ ] Notation defined on first use

## References
- All theory notes from Phases 1-3"

gh issue create --repo "$REPO" \
  --title "[Phase 7] Write article sections 6-10 (mixture through conclusion)" \
  --label "phase:7,type:writing,priority:high" \
  --milestone "Phase 7 — Article Writing" \
  --body "## Context
The second half delivers the advanced topics and practical guidance.

## Tasks
- [ ] Section 6: Mixture models (~700 words) — EM derivation, multimodality
- [ ] Section 7: Heavy tails (~800 words) — tail risk, EVT basics
- [ ] Section 8: Experiments (~1,200 words) — all experiments with figures
- [ ] Section 9: Practical framework (~400 words) — decision guide
- [ ] Section 10: Conclusion (~300 words)
- [ ] Polish: consistent notation, figure captions, references

## Definition of Done
- [ ] Sections 6-10 complete
- [ ] Article reads as cohesive narrative
- [ ] All figures referenced with captions
- [ ] MD compatible with author's HTML pipeline

## References
- Theory notes from Phases 4-5
- All experiment figures"

# ──────────────────────────────────────────────
# PHASE 8 — Review & Publish
# ──────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "[Phase 8] Mathematical validation and code reproducibility" \
  --label "phase:8,type:review,priority:critical" \
  --milestone "Phase 8 — Review & Publish" \
  --body "## Context
Final quality gate. Portfolio article with math errors = credibility loss.

## Tasks
- [ ] Review every derivation in the article
- [ ] Cross-check numerical examples with code output
- [ ] Author runs: pip install -> all scripts -> all figures match
- [ ] Author runs: pytest tests/ -> all pass
- [ ] Author runs: ruff check . -> clean
- [ ] Fix any discrepancies

## Definition of Done
- [ ] Zero mathematical errors
- [ ] All scripts reproduce figures with fixed seeds
- [ ] All tests pass, ruff clean
- [ ] Author has personally verified

## References
- Full article"

gh issue create --repo "$REPO" \
  --title "[Phase 8] Publish to GitHub Pages and Medium" \
  --label "phase:8,type:writing,type:infrastructure,priority:high" \
  --milestone "Phase 8 — Review & Publish" \
  --body "## Context
Make the article publicly accessible and shareable.

## Tasks
- [ ] Copy article MD to github.io repo
- [ ] Run MD -> HTML pipeline
- [ ] Verify rendering (desktop + mobile)
- [ ] Publish Medium cross-post with canonical link
- [ ] Write LinkedIn post
- [ ] Update README with live links and badges

## Definition of Done
- [ ] Article live on GitHub Pages
- [ ] Medium published with canonical link
- [ ] LinkedIn post drafted
- [ ] README has all live links

## References
- Author's existing github.io repo and pipeline"

echo "All issues created successfully."
echo ""
echo "IMPORTANT: Verify milestone IDs match by running:"
echo "  gh api repos/\$REPO/milestones --jq '.[] | {number, title}'"
