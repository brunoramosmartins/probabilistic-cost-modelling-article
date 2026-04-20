# Roadmap: The Shape of What You'll Spend

## Probabilistic Modelling of People Costs — from Distribution Selection to Budget Impact

---

## Project Context

Build a **portfolio-grade technical article** that tackles a problem most budget analysts ignore: the shape of cost distributions matters as much as their mean. Choosing a Normal distribution when costs are heavy-tailed, or ignoring multimodality in salary data, leads to systematically wrong budgets. This article derives the mathematics of distribution selection, parameter estimation, and model comparison from first principles, then demonstrates the budget impact of getting it right — or wrong.

### How This Relates to the Monte Carlo Article

| Aspect | Article 1 (Monte Carlo) | Article 2 (This one) |
|--------|------------------------|---------------------|
| Core question | How to simulate the total budget | Which distributions model each cost component |
| Mathematical focus | LLN, CLT, variance reduction | MLE, GoF tests, KL divergence, mixture models |
| Takes as input | Distributions (assumed) | Raw cost data (observed) |
| Produces as output | Budget distribution + CI | Fitted models + model comparison |
| Key skill demonstrated | Simulation & convergence theory | Statistical modelling & inference |

The articles are designed to be read independently but form a natural pair: this article answers "which distributions?" and the Monte Carlo article answers "now what do I do with them?"

### Tech Stack

- Python 3.x
- numpy / scipy (distribution fitting, MLE, GoF tests)
- matplotlib / seaborn (publication-quality figures)
- pandas
- statsmodels (optional: additional fitting tools)
- ruff (linter)
- pytest (testing — author runs manually)

### Author Background

Analytics Engineer transitioning to Data Science / Machine Learning. Background in Mathematics (formal proofs, calculus, linear algebra). Currently working in IT headcount budgeting. Portfolio oriented toward statistical thinking, probabilistic modelling, and applied ML. Publication targets: GitHub Pages (via existing MD → HTML pipeline) and Medium.

### What This Project Is

This is a **technical article for portfolio and personal technical development**. It is not production software. The code supports the article's arguments and must be correct and reproducible, but the primary deliverable is the written article with rigorous mathematical content. The author wants to study these topics deeply, including paper exercises between implementation phases.

---

## Thesis (v0.1)

> "A budget built on Normal distributions underestimates tail risk by construction. People costs — salaries, overtime, severance — are structurally non-Normal: right-skewed, heavy-tailed, and often multimodal. Choosing the wrong distributional family is not a modelling nuance — it is a systematic bias that propagates through every downstream calculation. Maximum Likelihood Estimation provides the principled framework for fitting, and information-theoretic criteria (AIC, BIC, likelihood ratio tests) provide the principled framework for choosing."

### Central Axis

The distribution you assume **is** your model. Everything else — mean, variance, confidence intervals, risk estimates — flows from that choice.

```
Wrong distribution → wrong parameters → wrong budget → wrong decisions
```

---

## GitHub Semantic Guide

### Tags

Immutable snapshots marking the end of each phase.

**Convention:** `v0.x-phase-name` for internal milestones, `v1.0.0` for the public portfolio release.

```bash
# After Phase 0
git tag -a v0.1-foundation -m "Phase 0: thesis, model design, project scaffold"
git push origin v0.1-foundation

# Public release
git tag -a v1.0.0 -m "v1.0.0: full article with distribution fitting, model comparison, and experiments"
git push origin v1.0.0
```

**When to create a tag:** After the phase's PR is merged into `main`.

### Releases

**Rule:** Create a release only when there is external value.

| Tag | Release? | Reasoning |
|-----|----------|-----------|
| `v0.1-foundation` | No | Internal scaffolding |
| `v0.2-distribution-families` | No | Theory notes only |
| `v0.3-mle-estimation` | No | Proofs and derivations |
| `v0.4-model-comparison` | No | Information theory — internal |
| `v0.5-mixture-models` | No | Advanced topic — internal |
| `v0.6-heavy-tails` | No | Theory and implementation |
| `v0.7-experiments` | Yes (pre-release) | Working simulations + figures for peer review |
| `v0.8-article-draft` | Yes (pre-release) | Full draft article for feedback |
| `v1.0.0` | **Yes (stable)** | Published article |

### Milestones

Each phase = one milestone. All issues within a phase belong to its milestone.

### Issues

Full body: Context, Tasks, Definition of Done, References. Title: `[Phase X] Short description`.

### Relationship

```
Issue → belongs to → Milestone (1 per phase)
Milestone completion → triggers → Tag
Tag (when externally valuable) → triggers → Release
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       ARTICLE PIPELINE                           │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Theory     │───▶│    Code     │───▶│   Figures   │          │
│  │   Notes      │    │   (src/)    │    │ (figures/)  │          │
│  │  (notes/)    │    │             │    │             │          │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘          │
│                            │                  │                  │
│                ┌───────────┴──────────────────┘                  │
│                ▼                                                  │
│  ┌──────────────────────────────────┐                            │
│  │     Article (article/)           │                            │
│  │  probabilistic-cost-modelling.md │                            │
│  └──────────────┬───────────────────┘                            │
│                 │                                                 │
│                 ▼                                                 │
│  ┌──────────────────────────────────┐                            │
│  │  Author's MD → HTML pipeline     │                            │
│  │  (external repo: github.io)      │                            │
│  │  + Medium cross-post             │                            │
│  └──────────────────────────────────┘                            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                 │
│                                                                  │
│  ┌────────────────────┐                                          │
│  │ Synthetic Cost Data │  Configurable:                          │
│  │  src/data_gen.py    │  - sample size, noise, outliers         │
│  └─────────┬──────────┘                                          │
│            │                                                     │
│            ▼                                                     │
│  ┌────────────────────┐                                          │
│  │  Distribution       │  Candidates:                            │
│  │  Fitting Pipeline   │  - Normal, LogNormal, Gamma, Pareto,   │
│  │  src/fitting.py     │    Weibull, Mixture (GMM)              │
│  └─────────┬──────────┘                                          │
│            │                                                     │
│      ┌─────┼───────────────┐                                     │
│      ▼     ▼               ▼                                     │
│  ┌──────┐ ┌─────────┐ ┌──────────┐                               │
│  │ MLE  │ │ GoF     │ │ AIC/BIC  │                               │
│  │ θ̂    │ │ KS, AD  │ │ LRT     │                               │
│  └──────┘ └─────────┘ └──────────┘                               │
│      │         │            │                                     │
│      ▼         ▼            ▼                                     │
│  ┌──────────────────────────────────┐                             │
│  │  Model Selection + Budget Impact │                             │
│  │  scripts/                        │                             │
│  └──────────────────────────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
probabilistic-cost-modelling-article/
│
├── .claude/
│   └── CLAUDE.md                          # Claude Code project rules
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── task.md                        # Task issue template
│   │   └── bug.md                         # Bug issue template
│   ├── PULL_REQUEST_TEMPLATE.md           # PR template
│   └── setup/
│       ├── labels.sh                      # Bash script: create all labels
│       ├── milestones.sh                  # Bash script: create all milestones
│       └── issues.sh                      # Bash script: create all issues
│
├── article/                               # Final article source
│   └── probabilistic-cost-modelling.md    # Main article in Markdown
│
├── docs/                                  # Project planning documents
│   ├── thesis.md                          # Thesis statement and scope
│   ├── model-design.md                    # Cost component specification
│   └── outline.md                         # Article section outline
│
├── src/                                   # Reusable source code
│   ├── __init__.py
│   ├── data_gen.py                        # Synthetic cost data generator
│   ├── distributions.py                   # Distribution catalogue + PDF/CDF helpers
│   ├── fitting.py                         # MLE fitting pipeline
│   ├── model_selection.py                 # AIC, BIC, LRT, KS test, AD test
│   ├── mixture.py                         # Gaussian Mixture Model (GMM) fitting
│   ├── heavy_tails.py                     # Tail index estimation, EVT basics
│   └── budget_impact.py                   # Downstream impact analysis
│
├── scripts/                               # Standalone experiment scripts
│   ├── exp_distribution_zoo.py            # Visualize all candidate distributions
│   ├── exp_mle_convergence.py             # MLE consistency and efficiency demo
│   ├── exp_wrong_distribution.py          # Budget impact of wrong choice
│   ├── exp_mixture_detection.py           # Detecting multimodality in salary data
│   ├── exp_heavy_tail_risk.py             # Tail risk: Normal vs heavy-tailed
│   ├── exp_model_comparison.py            # AIC/BIC/LRT on synthetic cost data
│   └── exp_full_pipeline.py              # End-to-end: data → fit → select → budget
│
├── notebooks/                             # Jupyter notebooks (exploration)
│   ├── 01_distribution_families.ipynb
│   ├── 02_mle_estimation.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_mixture_models.ipynb
│   ├── 05_heavy_tails.ipynb
│   └── 06_budget_impact.ipynb
│
├── exercises/                             # Paper exercises (LaTeX-compatible MD)
│   ├── ex01_distribution_families.md
│   ├── ex02_mle.md
│   ├── ex03_model_comparison.md
│   ├── ex04_mixture_models.md
│   └── ex05_heavy_tails.md
│
├── figures/                               # Generated plots and diagrams
│   └── .gitkeep
│
├── notes/                                 # Phase-by-phase theory notes
│   ├── phase1-distributions.md
│   ├── phase2-mle.md
│   ├── phase3-model-comparison.md
│   ├── phase4-mixture.md
│   └── phase5-heavy-tails.md
│
├── tests/                                 # Unit tests (author runs manually)
│   ├── test_data_gen.py
│   ├── test_fitting.py
│   ├── test_model_selection.py
│   ├── test_mixture.py
│   ├── test_heavy_tails.py
│   └── test_budget_impact.py
│
├── requirements.txt                       # Pinned Python dependencies
├── pyproject.toml                         # Project metadata + ruff config
├── LICENSE                                # MIT
└── README.md                              # Project overview + quick start
```

---

## Claude Code Configuration

### File: `.claude/CLAUDE.md`

```markdown
# Project Rules — Probabilistic Cost Modelling Article

## What This Project Is

This is a **technical article for portfolio and personal technical development**.
It is NOT production software. The primary deliverable is a written article with
rigorous mathematical content, supported by correct and reproducible code.

## Development Rules

### Git & GitHub — CRITICAL RULES

1. **NEVER commit directly.** After any implementation, present the commit
   message and changed files in chat for the author's validation.

2. **NEVER create branches.** The author creates all branches manually.
   When suggesting a branch, only mention the name in the commit/PR proposal.

3. **NEVER create PRs automatically.** Present the PR details in chat.
   The author will create the PR manually on GitHub.

4. **NEVER push to any branch.** All git operations are done by the author.

5. Follow Conventional Commits: `<type>(<scope>): <short description>`

### Output Format for Commits

After every implementation, present the commit proposal in a **fenced code
block ready to copy**:

~~~
```
git add <files>
git commit -m "<type>(<scope>): <short description>"
```
~~~

If multiple logical commits are needed, present each separately:

~~~
```
# Commit 1: theory notes
git add notes/phase1-distributions.md
git commit -m "docs(theory): Phase 1 — distribution families theory notes"

# Commit 2: implementation
git add src/distributions.py src/data_gen.py
git commit -m "feat(core): implement distribution catalogue and data generator"

# Commit 3: tests
git add tests/test_distributions.py tests/test_data_gen.py
git commit -m "test(core): add distribution and data generator tests"
```
~~~

### Output Format for PRs

Present the PR proposal in a **fenced code block ready to copy**:

~~~
```
gh pr create \
  --base main \
  --head phase-1/distribution-families \
  --title "feat(theory): Phase 1 — distribution families and data generator" \
  --body "## Summary

Introduces the distribution catalogue (Normal, LogNormal, Gamma, Pareto,
Weibull) with PDF/CDF helpers and the synthetic cost data generator.

### Deliverables
- \`notes/phase1-distributions.md\`: theory notes with derivations
- \`src/distributions.py\`: distribution catalogue
- \`src/data_gen.py\`: synthetic data generator
- \`tests/test_distributions.py\`, \`tests/test_data_gen.py\`

### Checklist
- [x] Code runs without errors
- [x] Tests created (author will run)
- [ ] Author ran \`ruff check .\`
- [ ] Author ran \`pytest tests/\`

Closes #4, #5" \
  --milestone "Phase 1 — Distribution Families"
```
~~~

### Output Format for Tags and Releases

When a phase is complete and merged, present:

~~~
```
# Tag
git tag -a v0.2-distribution-families -m "Phase 1: distribution catalogue and theory notes"
git push origin v0.2-distribution-families
```
~~~

When a release is warranted:

~~~
```
# Release (only for phases with external value)
gh release create v0.7-experiments \
  --title "v0.7 — Experiments and Visualizations" \
  --notes "All experiments complete with publication-quality figures." \
  --prerelease
```
~~~

### Testing & Linting — CRITICAL

- **Create tests** in `tests/` but **NEVER run them.**
- **NEVER run `ruff`.**
- After creating tests, say:
  "Tests created. Please run `pytest tests/` and `ruff check .` and share
  any failures so we can debug together."
- If the author shares failures, help debug in chat.

### Code Style

- Python 3.10+ syntax
- Type hints on all function signatures
- Google-style docstrings on all public functions
- numpy-style docstrings for mathematical functions (Parameters, Returns,
  Notes with formulas, Examples)
- No Makefile (author doesn't have make on Windows)
- Document all commands in README.md

### Mathematical Content

- All derivations must be step-by-step with no skipped algebra
- LaTeX-compatible syntax: `$$...$$` for display, `$...$` for inline
- Every theorem/proposition: statement → proof → example
- Exercises go in `exercises/`, one file per phase
- When a derivation is used in both notes and article, derive in notes
  first, then condense for article

### Article Output

- Final deliverable: `article/probabilistic-cost-modelling.md`
- Processed by author's existing MD → HTML pipeline (separate repo)
- Standard Markdown with LaTeX math blocks
- Figures referenced with relative paths: `../figures/filename.png`
```

---

## GitHub Setup Scripts

### File: `.github/setup/labels.sh`

```bash
#!/bin/bash
# Creates all project labels. Run once after repo creation.
# Usage: bash .github/setup/labels.sh

set -euo pipefail

REPO="${1:?Usage: bash labels.sh owner/repo}"

echo "Creating labels for $REPO..."

# Delete default labels (optional — uncomment if desired)
# for label in "bug" "documentation" "duplicate" "enhancement" "good first issue" \
#   "help wanted" "invalid" "question" "wontfix"; do
#   gh label delete "$label" --repo "$REPO" --yes 2>/dev/null || true
# done

# --- Phase labels ---
gh label create "phase:0" --color "0E8A16" --description "Phase 0 — Foundation" --repo "$REPO" --force
gh label create "phase:1" --color "1D76DB" --description "Phase 1 — Distribution Families" --repo "$REPO" --force
gh label create "phase:2" --color "5319E7" --description "Phase 2 — Maximum Likelihood Estimation" --repo "$REPO" --force
gh label create "phase:3" --color "D93F0B" --description "Phase 3 — Model Comparison" --repo "$REPO" --force
gh label create "phase:4" --color "FBCA04" --description "Phase 4 — Mixture Models" --repo "$REPO" --force
gh label create "phase:5" --color "B60205" --description "Phase 5 — Heavy Tails" --repo "$REPO" --force
gh label create "phase:6" --color "006B75" --description "Phase 6 — Experiments & Visualizations" --repo "$REPO" --force
gh label create "phase:7" --color "0E8A16" --description "Phase 7 — Article Writing" --repo "$REPO" --force
gh label create "phase:8" --color "5319E7" --description "Phase 8 — Review & Publish" --repo "$REPO" --force

# --- Type labels ---
gh label create "type:theory" --color "C5DEF5" --description "Mathematical derivation or proof" --repo "$REPO" --force
gh label create "type:code" --color "BFD4F2" --description "Implementation task" --repo "$REPO" --force
gh label create "type:experiment" --color "D4C5F9" --description "Experimental validation or simulation" --repo "$REPO" --force
gh label create "type:writing" --color "FEF2C0" --description "Article writing task" --repo "$REPO" --force
gh label create "type:documentation" --color "0075CA" --description "Planning or project docs" --repo "$REPO" --force
gh label create "type:infrastructure" --color "E4E669" --description "Repo setup, CI, tooling" --repo "$REPO" --force
gh label create "type:review" --color "F9D0C4" --description "Review or validation task" --repo "$REPO" --force
gh label create "type:bug" --color "D73A4A" --description "Something is broken" --repo "$REPO" --force
gh label create "type:content" --color "BFDADC" --description "LinkedIn, Medium, or social content" --repo "$REPO" --force

# --- Priority labels ---
gh label create "priority:critical" --color "B60205" --description "Must be done, blocks other work" --repo "$REPO" --force
gh label create "priority:high" --color "D93F0B" --description "Important, do soon" --repo "$REPO" --force
gh label create "priority:medium" --color "FBCA04" --description "Can wait but should be done" --repo "$REPO" --force
gh label create "priority:low" --color "0E8A16" --description "Nice to have" --repo "$REPO" --force

echo "All labels created successfully."
```

### File: `.github/setup/milestones.sh`

```bash
#!/bin/bash
# Creates all project milestones. Run once after repo creation.
# Usage: bash .github/setup/milestones.sh owner/repo

set -euo pipefail

REPO="${1:?Usage: bash milestones.sh owner/repo}"

echo "Creating milestones for $REPO..."

gh api "repos/$REPO/milestones" -f title="Phase 0 — Foundation" \
  -f description="Thesis, model design, project scaffold, GitHub configuration." \
  -f state="open" --silent

gh api "repos/$REPO/milestones" -f title="Phase 1 — Distribution Families" \
  -f description="Catalogue of candidate distributions, properties, and synthetic data generator." \
  -f state="open" --silent

gh api "repos/$REPO/milestones" -f title="Phase 2 — Maximum Likelihood Estimation" \
  -f description="MLE derivation, Fisher information, asymptotic theory." \
  -f state="open" --silent

gh api "repos/$REPO/milestones" -f title="Phase 3 — Model Comparison" \
  -f description="AIC, BIC, likelihood ratio tests, KL divergence." \
  -f state="open" --silent

gh api "repos/$REPO/milestones" -f title="Phase 4 — Mixture Models" \
  -f description="GMM, EM algorithm derivation, multimodality detection." \
  -f state="open" --silent

gh api "repos/$REPO/milestones" -f title="Phase 5 — Heavy Tails" \
  -f description="Fat-tailed distributions, tail index, EVT basics, budget risk." \
  -f state="open" --silent

gh api "repos/$REPO/milestones" -f title="Phase 6 — Experiments & Visualizations" \
  -f description="All experiments, publication-quality figures, budget impact analysis." \
  -f state="open" --silent

gh api "repos/$REPO/milestones" -f title="Phase 7 — Article Writing" \
  -f description="Full article assembly from theory notes and experiments." \
  -f state="open" --silent

gh api "repos/$REPO/milestones" -f title="Phase 8 — Review & Publish" \
  -f description="Mathematical validation, code reproducibility, publication." \
  -f state="open" --silent

echo "All milestones created successfully."
```

### File: `.github/setup/issues.sh`

```bash
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
- [ ] Create comparison table: family × property
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
      - \`generate_salary_data(n, dist, params, seed)\` → array
      - \`generate_overtime_data(n, params, seed)\` → array
      - \`generate_mixed_data(n, components, weights, seed)\` → array (for mixture)
      - \`inject_outliers(data, fraction, multiplier)\` → array
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
- [ ] Define the likelihood function: L(θ|x) = Π f(x_i|θ)
- [ ] Define log-likelihood: ℓ(θ) = Σ log f(x_i|θ)
- [ ] Derive the score function: S(θ) = ∂ℓ/∂θ
- [ ] Prove: E[S(θ₀)] = 0 (score has zero mean at true parameter)
- [ ] Define Fisher information: I(θ) = Var(S(θ)) = -E[∂²ℓ/∂θ²]
- [ ] Prove the two forms of Fisher information are equivalent
- [ ] State and prove (or prove sketch) asymptotic normality of MLE:
      √n(θ̂ − θ₀) →_d N(0, I(θ₀)⁻¹)
- [ ] Derive MLE for specific distributions:
      - Normal(μ, σ²): closed-form
      - LogNormal(μ, σ²): closed-form
      - Gamma(α, β): score equations (no closed form for α)
      - Pareto(α, x_m): closed-form
- [ ] Prove: MLE is consistent and asymptotically efficient (Cramér-Rao bound)

## Definition of Done
- [ ] Complete derivation chain: likelihood → score → Fisher → asymptotics
- [ ] MLE derived for all 4 distributions
- [ ] Cramér-Rao bound stated and connected to MLE efficiency
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
      - \`fit_mle(data, distribution)\` → FitResult(params, loglik, se, ci)
      - Use analytical solutions where available (Normal, LogNormal, Pareto)
      - Use \`scipy.optimize.minimize\` for Gamma, Weibull
      - Compute standard errors from observed Fisher information
- [ ] Implement \`fit_all(data, candidates)\` → list of FitResult, sorted by loglik
- [ ] Create tests (DO NOT RUN):
      - Recover known parameters from large samples
      - SE shrinks with √n
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
- [ ] Define KL divergence: D_KL(p||q) = ∫ p(x) log(p(x)/q(x)) dx
- [ ] Prove: D_KL ≥ 0 (Gibbs' inequality) and D_KL = 0 iff p = q a.e.
- [ ] Derive AIC from KL divergence:
      - AIC ≈ 2k − 2ℓ(θ̂) as estimator of expected KL
      - Explain the bias-correction term 2k
      - AICc for small samples
- [ ] Derive BIC from Bayesian model evidence:
      - BIC ≈ k·log(n) − 2ℓ(θ̂)
      - Explain why BIC penalizes complexity more than AIC
- [ ] Derive the likelihood ratio test:
      - Λ = −2[ℓ(θ̂₀) − ℓ(θ̂₁)]
      - Wilks' theorem: Λ →_d χ²(Δk) under H₀
      - Apply to nested models (e.g., Normal vs LogNormal is NOT nested)
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
      - \`aic_weights(aic_values)\` → Akaike weights for model averaging
      - \`likelihood_ratio_test(loglik_0, loglik_1, df)\` → (statistic, p_value)
      - \`ks_test(data, distribution, params)\` → (statistic, p_value)
      - \`ad_test(data, distribution, params)\` → (statistic, p_value)
      - \`compare_models(fit_results)\` → comparison table (DataFrame)
- [ ] Create tests (DO NOT RUN):
      - AIC selects true model in large samples
      - BIC selects true model (stronger consistency)
      - KS test rejects wrong distribution at α = 0.05

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
- [ ] Define the GMM: f(x) = Σ_k π_k · N(x|μ_k, σ²_k)
- [ ] Explain why direct MLE fails (log of sums ≠ sum of logs)
- [ ] Derive the EM algorithm:
      E-step: compute responsibilities γ_ik = π_k·N(x_i|μ_k,σ²_k) / Σ_j π_j·N(x_i|μ_j,σ²_j)
      M-step: update π_k, μ_k, σ²_k from responsibilities
- [ ] Prove: EM monotonically increases the log-likelihood (Jensen's inequality)
- [ ] Discuss convergence: to local maximum, not necessarily global
- [ ] Derive BIC for choosing number of components K
- [ ] Connect to budget: multimodal salary → multimodal total cost

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
      - \`fit_gmm(data, K, max_iter, tol, seed)\` → GMMResult
      - \`select_K(data, K_range, criterion='bic')\` → optimal K
      - \`detect_multimodality(data)\` → bool + evidence
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
higher probability to extreme costs. This is where 'getting the distribution
wrong' has the biggest budget impact.

## Tasks
- [ ] Define heavy tails formally:
      - Light-tailed: P(X > x) decays exponentially (Normal, Gamma)
      - Heavy-tailed: P(X > x) decays polynomially (Pareto)
      - Sub-exponential: LogNormal (between light and heavy)
- [ ] Define the tail index α: P(X > x) ~ x^{−α} as x → ∞
- [ ] Derive tail behaviour of key distributions:
      - Normal: P(X > x) ~ (1/x)·φ(x) (super-exponential)
      - LogNormal: P(X > x) ~ ... (sub-exponential)
      - Pareto: P(X > x) = (x_m/x)^α (polynomial)
- [ ] Introduce Extreme Value Theory basics:
      - Block maxima: Generalized Extreme Value (GEV) distribution
      - Fisher-Tippett-Gnedenko theorem (statement)
      - Three domains of attraction: Gumbel, Fréchet, Weibull
- [ ] Hill estimator for tail index
- [ ] Quantify budget impact:
      - P(cost > 2× expected) under Normal vs Pareto
      - Value at Risk (VaR) and Expected Shortfall (ES) comparison

## Definition of Done
- [ ] Heavy vs light tail formally defined with examples
- [ ] Tail behaviour derived for Normal, LogNormal, Pareto
- [ ] EVT basics stated (not fully proved — measure theory heavy)
- [ ] Budget impact quantified with concrete numbers
- [ ] Hill estimator described with interpretation

## References
- Embrechts, Klüppelberg & Mikosch, Modelling Extremal Events
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
      - \`hill_estimator(data, k)\` → tail index estimate
      - \`tail_probability(distribution, params, threshold)\` → P(X > threshold)
      - \`qq_plot_data(data, distribution)\` → (theoretical, empirical) quantiles
      - \`compare_tail_risk(data, distributions, threshold)\` → comparison table
- [ ] Implement \`src/budget_impact.py\`:
      - \`var_at_level(samples, alpha)\` → Value at Risk
      - \`expected_shortfall(samples, alpha)\` → ES (CVaR)
      - \`budget_reserve(samples, confidence)\` → reserve needed above mean
      - \`compare_distributions_impact(data, fits, budget_ceiling)\` →
        table with P(overbudget), VaR, ES for each distribution
- [ ] Create tests (DO NOT RUN)

## Definition of Done
- [ ] Hill estimator validated on Pareto samples (known α)
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
      Compare P(cost > 2× mean) under Normal vs Pareto vs LogNormal
- [ ] Experiment F — Model Comparison:
      Full AIC/BIC/KS comparison table on synthetic data
- [ ] Experiment G — End-to-End Pipeline:
      Data → fit all → select best → compute budget impact
- [ ] All figures: 300 DPI, clean labels, consistent style
- [ ] All scripts use fixed seeds

## Definition of Done
- [ ] All 7 experiments produce figures in \`figures/\`
- [ ] Each script runs standalone
- [ ] Figures are publication-quality
- [ ] Author has run all scripts and verified outputs

## References
- All theory from Phases 1–5"

# ──────────────────────────────────────────────
# PHASE 7 — Article Writing
# ──────────────────────────────────────────────

gh issue create --repo "$REPO" \
  --title "[Phase 7] Write article sections 1–5 (foundations through model comparison)" \
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
- [ ] Sections 1–5 complete in \`article/probabilistic-cost-modelling.md\`
- [ ] All derivations self-contained
- [ ] Notation defined on first use

## References
- All theory notes from Phases 1–3"

gh issue create --repo "$REPO" \
  --title "[Phase 7] Write article sections 6–10 (mixture through conclusion)" \
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
- [ ] Sections 6–10 complete
- [ ] Article reads as cohesive narrative
- [ ] All figures referenced with captions
- [ ] MD compatible with author's HTML pipeline

## References
- Theory notes from Phases 4–5
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
- [ ] Author runs: pip install → all scripts → all figures match
- [ ] Author runs: pytest tests/ → all pass
- [ ] Author runs: ruff check . → clean
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
- [ ] Run MD → HTML pipeline
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
echo "  gh api repos/$REPO/milestones --jq '.[] | {number, title}'"
```

---

## Cost Model Design

### Philosophy

The model represents **individual cost components** as random variables with distinct distributional properties. Unlike the Monte Carlo article (which focuses on total cost simulation), this article is about understanding the **shape** of each component.

### Cost Components and Candidate Distributions

| Component | Symbol | Candidates | Rationale |
|-----------|--------|------------|-----------|
| Base salary | $S_i$ | LogNormal, Gamma, Mixture(Normal) | Right-skewed; multimodal if junior/senior clusters |
| Monthly overtime hours | $H_i$ | Poisson, NegBin, Zero-inflated Poisson | Discrete, often zero-inflated |
| Overtime cost per event | $C_{ot}$ | LogNormal, Gamma | Right-skewed, always positive |
| Annual severance events | $N_{sev}$ | Poisson, Binomial | Rare events |
| Severance cost per event | $C_{sev}$ | Pareto, LogNormal | Heavy-tailed: a few very expensive cases |
| Hiring cost per position | $C_h$ | LogNormal, Gamma | Variable (recruiter fees, relocation) |
| Benefits multiplier | $\beta_i$ | Uniform, Beta | Bounded: typically 1.5–2.0× salary |

### Key Questions the Article Answers

1. **Which family?** — Is salary LogNormal, Gamma, or a mixture? How to tell.
2. **Which parameters?** — Given the family, what are the MLE estimates and their uncertainty?
3. **Does it matter?** — If I pick Normal instead of LogNormal, how wrong is my budget?
4. **What about the tails?** — Severance costs might follow a Pareto. How much reserve does that imply?
5. **What about clusters?** — If salaries are bimodal (junior vs senior), a single distribution is wrong.

---

## Phase 0 — Foundation

### Objective

Define the thesis, scope, cost model, and repository skeleton. Produce all planning documents. Configure GitHub (labels, milestones, issues via bash scripts) and Claude Code rules. No code, no theory — only structure.

### Tasks

- [ ] **Write thesis document** (Issue #1)
- [ ] **Design cost model** (Issue #2)
- [ ] **Configure repository and GitHub** (Issue #3)

### Deliverables

- [ ] `docs/thesis.md`
- [ ] `docs/model-design.md`
- [ ] `docs/outline.md`
- [ ] `.claude/CLAUDE.md`
- [ ] `.github/ISSUE_TEMPLATE/task.md`, `bug.md`
- [ ] `.github/PULL_REQUEST_TEMPLATE.md`
- [ ] `.github/setup/labels.sh`, `milestones.sh`, `issues.sh`
- [ ] `requirements.txt`, `pyproject.toml`
- [ ] `README.md`

### GitHub

| Item | Value |
|------|-------|
| Branch | `phase-0/foundation` |
| Merge strategy | Squash merge |
| PR title | `feat(docs): Phase 0 — foundation documents and project scaffold` |
| Milestone | `Phase 0 — Foundation` |
| Tag | `v0.1-foundation` |
| Release | **No** — internal scaffolding |

---

## Phase 1 — Distribution Families

### Objective

Build a rigorous catalogue of probability distributions relevant to people costs. For each candidate family, derive the PDF, CDF, expected value, variance, skewness, and kurtosis from first principles. Characterize the shape properties that make each family suitable (or unsuitable) for specific cost components. Implement the distribution catalogue and synthetic data generator.

### Topics

- Normal: the baseline assumption (and why it's often wrong)
- LogNormal: the natural model for right-skewed positive quantities
- Gamma: flexible shape via $\alpha$, always positive
- Pareto: the power-law tail for extreme costs
- Weibull: flexible hazard rate, common in reliability
- Shape characterization: skewness, kurtosis, tail weight

### Tasks

- [ ] **Derive properties of all 5 distribution families** (Issue #4)
- [ ] **Implement distribution catalogue and data generator** (Issue #5)
- [ ] **Write theory notes**

### Deliverables

- [ ] `notes/phase1-distributions.md`
- [ ] `exercises/ex01_distribution_families.md`
- [ ] `src/distributions.py`
- [ ] `src/data_gen.py`
- [ ] `tests/test_distributions.py`
- [ ] `tests/test_data_gen.py`
- [ ] `notebooks/01_distribution_families.ipynb`
- [ ] `figures/distribution_zoo.png`

### GitHub

| Item | Value |
|------|-------|
| Branch | `phase-1/distribution-families` |
| Merge strategy | Squash merge |
| PR title | `feat(theory): Phase 1 — distribution families and data generator` |
| Milestone | `Phase 1 — Distribution Families` |
| Tag | `v0.2-distribution-families` |
| Release | **No** — internal theory |

### 📝 Exercises — After Phase 1

**File: `exercises/ex01_distribution_families.md`**

#### Proofs (paper)

1. **Derive** $E[X]$ and $\text{Var}(X)$ for $X \sim \text{LogNormal}(\mu, \sigma^2)$ starting from the definition $X = e^Y$ where $Y \sim N(\mu, \sigma^2)$. *Use the MGF of the Normal distribution.*

2. **Derive** the skewness of the LogNormal distribution: $\gamma_1 = (e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}$. *You will need $E[X^3]$.*

3. **Prove** that the Gamma distribution with $\alpha = 1$ reduces to the Exponential distribution. Then prove that the sum of $n$ i.i.d. Exponential($\lambda$) random variables follows a Gamma($n, \lambda$) distribution. *Use MGFs.*

4. **Prove** that the Pareto distribution with shape parameter $\alpha \leq 2$ has infinite variance, and with $\alpha \leq 1$ has infinite mean. *Evaluate the integrals directly and show divergence.*

5. **Derive** the kurtosis of the Normal distribution ($\kappa = 3$) and show that excess kurtosis is zero. Then show that the LogNormal always has excess kurtosis > 0. *What does this mean for budget tail risk?*

#### Computations (paper)

6. A company's salary data has sample mean R$ 12,000 and sample variance R$ 9,000,000 (SD = R$ 3,000). Assuming LogNormal: use the method of moments to estimate $\mu$ and $\sigma^2$. *Hint: solve $E[X] = e^{\mu + \sigma^2/2}$ and $\text{Var}(X) = (e^{\sigma^2} - 1)e^{2\mu + \sigma^2}$.*

7. For $X \sim \text{Pareto}(\alpha = 3, x_m = 5000)$, compute $P(X > 20{,}000)$, $P(X > 50{,}000)$, and $P(X > 100{,}000)$. Compare with the same tail probabilities under $Y \sim N(7500, 2500^2)$ (matched mean). *Which distribution assigns more weight to extreme severance costs?*

8. A salary distribution appears bimodal with peaks near R$ 8,000 and R$ 18,000. If you fit a single Normal, what happens to the estimated variance? Why is this misleading for budget risk? *Think about what "average" means in a bimodal distribution.*

---

## Phase 2 — Maximum Likelihood Estimation

### Objective

Derive MLE theory from first principles: likelihood, score function, Fisher information, Cramér-Rao bound, and asymptotic normality of the MLE. Derive MLE for each candidate distribution. Implement the fitting pipeline. This phase answers: "given data, how do I find the best parameters for a chosen distribution?"

### Tasks

- [ ] **Derive MLE theory** (Issue #6)
- [ ] **Implement fitting pipeline** (Issue #7)
- [ ] **Write theory notes**

### Deliverables

- [ ] `notes/phase2-mle.md`
- [ ] `exercises/ex02_mle.md`
- [ ] `src/fitting.py`
- [ ] `tests/test_fitting.py`
- [ ] `notebooks/02_mle_estimation.ipynb`
- [ ] `figures/mle_convergence.png`

### GitHub

| Item | Value |
|------|-------|
| Branch | `phase-2/mle-estimation` |
| Merge strategy | Squash merge |
| PR title | `feat(core): Phase 2 — MLE theory and fitting pipeline` |
| Milestone | `Phase 2 — Maximum Likelihood Estimation` |
| Tag | `v0.3-mle-estimation` |
| Release | **No** — theory and implementation |

### 📝 Exercises — After Phase 2

**File: `exercises/ex02_mle.md`**

#### Proofs (paper)

1. **Derive the MLE** for $\mu$ and $\sigma^2$ in the Normal($\mu, \sigma^2$) model. Show that $\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum(x_i - \bar{x})^2$ is biased. *Compute $E[\hat{\sigma}^2_{MLE}]$ and show it equals $\frac{n-1}{n}\sigma^2$.*

2. **Derive the MLE** for $\mu$ and $\sigma^2$ in the LogNormal($\mu, \sigma^2$) model. *Hint: if $X \sim \text{LogNormal}$, then $\log X \sim \text{Normal}$. Transform the data first.*

3. **Derive the Fisher information** for the Exponential($\lambda$) model. Then verify the Cramér-Rao lower bound: $\text{Var}(\hat{\lambda}_{MLE}) \geq 1/(n \cdot I(\lambda))$. *Show the MLE achieves this bound (it is efficient).*

4. **Prove** that $E[S(\theta_0)] = 0$ where $S(\theta) = \partial \ell / \partial \theta$ is the score function. *Differentiate under the integral sign in $\int f(x|\theta) dx = 1$.*

5. **Prove** the equivalence of the two forms of Fisher information: $I(\theta) = \text{Var}(S(\theta)) = -E[\partial^2 \ell / \partial \theta^2]$. *Differentiate the score identity from exercise 4 again.*

6. **Derive the MLE** for the Pareto($\alpha, x_m$) distribution where $x_m$ is known. Compute $I(\alpha)$ and the asymptotic distribution of $\hat{\alpha}_{MLE}$.

#### Computations (paper)

7. Given salary data (in thousands): {8.2, 9.1, 7.5, 11.3, 10.8, 8.7, 9.5, 12.1, 8.9, 10.2}. Compute the MLE for LogNormal parameters $\hat{\mu}$ and $\hat{\sigma}^2$. Then compute 95% CI for each parameter using observed Fisher information.

8. You fit a Gamma($\alpha, \beta$) to overtime cost data and obtain $\hat{\alpha} = 2.3$, $\hat{\beta} = 150$. The observed Fisher information matrix is $\hat{I} = \begin{pmatrix} 0.82 & -0.003 \\ -0.003 & 0.0001 \end{pmatrix}$. Compute standard errors and 95% CIs for both parameters. *Invert the information matrix.*

---

## Phase 3 — Model Comparison

### Objective

Derive the information-theoretic and hypothesis-testing frameworks for choosing between distribution models. Prove that KL divergence is non-negative, derive AIC as an estimator of expected KL, derive BIC from Bayesian model evidence, and implement goodness-of-fit tests. This phase answers: "given data and multiple fitted distributions, which one should I use?"

### Tasks

- [ ] **Derive AIC, BIC, KL divergence, and LRT** (Issue #8)
- [ ] **Implement model comparison tools** (Issue #9)
- [ ] **Write theory notes**

### Deliverables

- [ ] `notes/phase3-model-comparison.md`
- [ ] `exercises/ex03_model_comparison.md`
- [ ] `src/model_selection.py`
- [ ] `tests/test_model_selection.py`
- [ ] `notebooks/03_model_comparison.ipynb`
- [ ] `figures/model_comparison_table.png`

### GitHub

| Item | Value |
|------|-------|
| Branch | `phase-3/model-comparison` |
| Merge strategy | Squash merge |
| PR title | `feat(core): Phase 3 — model comparison framework` |
| Milestone | `Phase 3 — Model Comparison` |
| Tag | `v0.4-model-comparison` |
| Release | **No** — theory and tools |

### 📝 Exercises — After Phase 3

**File: `exercises/ex03_model_comparison.md`**

#### Proofs (paper)

1. **Prove Gibbs' inequality:** $D_{KL}(p \| q) \geq 0$ with equality if and only if $p = q$ almost everywhere. *Use Jensen's inequality on the concave function $\log$.*

2. **Derive AIC** as an approximately unbiased estimator of the expected KL divergence between the true distribution and the fitted model. *Start from $E_{\text{true}}[\log f(X | \hat{\theta})]$ and show the bias is approximately $k/n$ where $k$ is the number of parameters.*

3. **Derive BIC** from the Laplace approximation to the Bayesian marginal likelihood: $p(x | M) = \int L(\theta) \pi(\theta) d\theta$. *Show that $-2 \log p(x|M) \approx -2\ell(\hat{\theta}) + k \log n$.*

4. **Prove Wilks' theorem** (sketch): For nested models $M_0 \subset M_1$, the likelihood ratio statistic $\Lambda = -2[\ell(\hat{\theta}_0) - \ell(\hat{\theta}_1)] \xrightarrow{d} \chi^2(\Delta k)$ under $H_0$. *Use the quadratic approximation of the log-likelihood around $\hat{\theta}_1$.*

5. **Prove** that AIC and BIC can select different models, and explain when each is preferable. *Construct a concrete 2-model example where $n$ determines which criterion picks which model.*

#### Computations (paper)

6. You fit three distributions to salary data ($n = 200$):

   | Model | $k$ (params) | $\ell(\hat{\theta})$ |
   |-------|-------------|---------------------|
   | Normal | 2 | −1842.3 |
   | LogNormal | 2 | −1831.7 |
   | Gamma | 2 | −1835.1 |

   Compute AIC, AICc, and BIC for each. Which model wins under each criterion? Compute Akaike weights. What is the probability that LogNormal is the best model?

7. You want to test whether a Gamma fits better than an Exponential (nested: Exponential is Gamma with $\alpha = 1$). $\ell(\hat{\theta}_{\text{Exp}}) = -4521.3$, $\ell(\hat{\theta}_{\text{Gamma}}) = -4518.1$. Perform the LRT at $\alpha = 0.05$. *What is $\Delta k$? What is the critical value?*

---

## Phase 4 — Mixture Models

### Objective

Derive the Expectation-Maximization (EM) algorithm for Gaussian Mixture Models from first principles. Prove that EM monotonically increases the log-likelihood (via Jensen's inequality). Implement GMM fitting and multimodality detection. This phase answers: "what if the salary distribution isn't unimodal?"

### Tasks

- [ ] **Derive EM algorithm for GMM** (Issue #10)
- [ ] **Implement GMM and multimodality detection** (Issue #11)
- [ ] **Write theory notes**

### Deliverables

- [ ] `notes/phase4-mixture.md`
- [ ] `exercises/ex04_mixture_models.md`
- [ ] `src/mixture.py`
- [ ] `tests/test_mixture.py`
- [ ] `notebooks/04_mixture_models.ipynb`
- [ ] `figures/mixture_detection.png`

### GitHub

| Item | Value |
|------|-------|
| Branch | `phase-4/mixture-models` |
| Merge strategy | Squash merge |
| PR title | `feat(core): Phase 4 — EM algorithm and mixture model fitting` |
| Milestone | `Phase 4 — Mixture Models` |
| Tag | `v0.5-mixture-models` |
| Release | **No** — advanced topic, internal |

### 📝 Exercises — After Phase 4

**File: `exercises/ex04_mixture_models.md`**

#### Proofs (paper)

1. **Show** that the log-likelihood of a mixture model, $\ell(\theta) = \sum_{i=1}^n \log \left[\sum_{k=1}^K \pi_k f_k(x_i | \theta_k)\right]$, cannot be decomposed into a sum of per-component terms. *This is why direct MLE is intractable for mixtures.*

2. **Derive the E-step** of the EM algorithm for a 2-component Gaussian mixture. Starting from the complete-data log-likelihood, show that the responsibilities are:

   $$\gamma_{ik} = \frac{\pi_k \cdot \mathcal{N}(x_i | \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \cdot \mathcal{N}(x_i | \mu_j, \sigma_j^2)}$$

3. **Derive the M-step** updates: given responsibilities $\gamma_{ik}$, derive the update formulas for $\pi_k$, $\mu_k$, and $\sigma_k^2$ by maximizing the expected complete-data log-likelihood. *Use Lagrange multipliers for the constraint $\sum_k \pi_k = 1$.*

4. **Prove** that the EM algorithm monotonically increases the observed-data log-likelihood: $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$. *Use Jensen's inequality on the concave function $\log$. Define the ELBO and show that EM maximizes it.*

5. **Prove** that EM converges to a stationary point of the likelihood (not necessarily a global maximum). *Discuss the implications: multiple random restarts.*

#### Computations (paper)

6. Given a 2-component mixture with $\pi_1 = 0.6$, $\mu_1 = 8$, $\sigma_1 = 1.5$, $\pi_2 = 0.4$, $\mu_2 = 18$, $\sigma_2 = 2$:
   - Compute the responsibility $\gamma_{i1}$ for data point $x_i = 10$.
   - Compute the responsibility for $x_i = 16$.
   - Interpret: which component "owns" each data point?

7. After one E-step, you have responsibilities for 6 data points:

   | $x_i$ | $\gamma_{i1}$ | $\gamma_{i2}$ |
   |--------|-------------|-------------|
   | 7.5 | 0.95 | 0.05 |
   | 8.2 | 0.92 | 0.08 |
   | 9.1 | 0.80 | 0.20 |
   | 15.3 | 0.10 | 0.90 |
   | 17.8 | 0.02 | 0.98 |
   | 19.5 | 0.01 | 0.99 |

   Perform one M-step: compute updated $\pi_k$, $\mu_k$, and $\sigma_k^2$.

---

## Phase 5 — Heavy Tails and Extreme Value Theory

### Objective

Formalize the distinction between light-tailed and heavy-tailed distributions. Derive tail behaviour for key distributions. Introduce EVT basics (Fisher-Tippett-Gnedenko theorem). Implement tail analysis tools. Quantify the budget impact of underestimating tail risk. This is the article's practical climax: "if you use a Normal distribution for severance costs, you are systematically under-reserving."

### Tasks

- [ ] **Derive heavy-tail theory** (Issue #12)
- [ ] **Implement tail analysis and budget impact tools** (Issue #13)
- [ ] **Write theory notes**

### Deliverables

- [ ] `notes/phase5-heavy-tails.md`
- [ ] `exercises/ex05_heavy_tails.md`
- [ ] `src/heavy_tails.py`
- [ ] `src/budget_impact.py`
- [ ] `tests/test_heavy_tails.py`
- [ ] `tests/test_budget_impact.py`
- [ ] `notebooks/05_heavy_tails.ipynb`
- [ ] `figures/tail_comparison.png`
- [ ] `figures/budget_impact_comparison.png`

### GitHub

| Item | Value |
|------|-------|
| Branch | `phase-5/heavy-tails` |
| Merge strategy | Squash merge |
| PR title | `feat(core): Phase 5 — heavy-tail analysis and budget impact` |
| Milestone | `Phase 5 — Heavy Tails` |
| Tag | `v0.6-heavy-tails` |
| Release | **No** — theory and tools |

### 📝 Exercises — After Phase 5

**File: `exercises/ex05_heavy_tails.md`**

#### Proofs (paper)

1. **Prove** that the Pareto($\alpha$, $x_m$) distribution is heavy-tailed by showing that $M_X(t) = E[e^{tX}] = \infty$ for all $t > 0$. *Contrast with the Normal, where $M_X(t) < \infty$ for all $t$.*

2. **Prove** that the tail of the Normal distribution satisfies the bound:

   $$\frac{1}{\sqrt{2\pi}} \cdot \frac{x}{x^2 + 1} \cdot e^{-x^2/2} \leq P(X > x) \leq \frac{1}{\sqrt{2\pi}} \cdot \frac{1}{x} \cdot e^{-x^2/2}$$

   *Use integration by parts on $\int_x^\infty e^{-t^2/2} dt$.* What does this say about how fast Normal tails decay?

3. **Show** that for $X \sim \text{Pareto}(\alpha, x_m)$ with $\alpha > 2$:

   $$\frac{P(X > 2c)}{P(X > c)} = 2^{-\alpha}$$

   and for $X \sim N(\mu, \sigma^2)$ the same ratio decays exponentially. *This ratio captures "how quickly the tail thins." Compute both for $c = 3\sigma$ and compare.*

4. **Derive** the Hill estimator $\hat{\alpha}_{Hill} = \left[\frac{1}{k} \sum_{i=1}^k \log \frac{X_{(n-i+1)}}{X_{(n-k)}}\right]^{-1}$ as the MLE of the tail index for the Pareto model fitted to the $k$ largest order statistics. *State the assumption: the tail is approximately Pareto above a threshold.*

#### Computations (paper)

5. Severance costs follow $\text{Pareto}(\alpha = 2.5, x_m = 10{,}000)$. An analyst fits $N(\mu, \sigma^2)$ matching the first two moments: $\mu = E[X] = 16{,}667$, $\sigma^2 = \text{Var}(X)$. Compare $P(X > 50{,}000)$ and $P(X > 100{,}000)$ under both models. *By what factor does the Normal underestimate tail risk?*

6. For the same Pareto model, compute the Value at Risk at 95% and 99%: $\text{VaR}_\alpha = x_m \cdot (1 - \alpha)^{-1/\alpha_{\text{pareto}}}$. Compare with the Normal VaR at the same levels. *What is the ratio of Pareto VaR to Normal VaR?*

7. You have 500 overtime cost observations. The 20 largest values are (in R$): {8200, 8500, 9100, 9800, 10200, 10900, 11500, 12800, 14200, 15100, 16500, 18200, 19800, 22000, 25100, 29000, 34500, 42000, 58000, 91000}. Compute the Hill estimator for $k = 5, 10, 15, 20$. Is there evidence of a heavy tail? *Plot $\hat{\alpha}$ vs $k$ (on paper) — a stable region suggests a valid tail index.*

---

## Phase 6 — Experiments and Visualizations

### Objective

Run all definitive experiments and create publication-quality figures. Each experiment isolates one concept and produces a figure for the article.

### Experiments

| ID | Name | Purpose | Key Figure |
|----|------|---------|------------|
| A | Distribution Zoo | All 5 families fitted to same data | `distribution_zoo.png` |
| B | MLE Convergence | MLE → true params as n grows | `mle_convergence.png` |
| C | Wrong Distribution | Budget error from wrong choice | `wrong_distribution_impact.png` |
| D | Mixture Detection | GMM finds bimodal structure | `mixture_detection.png` |
| E | Heavy Tail Risk | Normal vs Pareto tail probabilities | `tail_risk_comparison.png` |
| F | Model Comparison | AIC/BIC/KS ranking table | `model_comparison.png` |
| G | End-to-End | Full pipeline on synthetic team | `full_pipeline.png` |

### Tasks

- [ ] **Run all 7 experiments** (Issue #14)
- [ ] All figures: 300 DPI, consistent style, clean labels
- [ ] All scripts use fixed seeds

### Deliverables

- [ ] All scripts in `scripts/`
- [ ] All figures in `figures/`
- [ ] `notebooks/06_budget_impact.ipynb`

### GitHub

| Item | Value |
|------|-------|
| Branch | `phase-6/experiments` |
| Merge strategy | Squash merge |
| PR title | `feat(experiments): Phase 6 — all experiments and publication figures` |
| Milestone | `Phase 6 — Experiments & Visualizations` |
| Tag | `v0.7-experiments` |
| Release | **Yes (pre-release)** — figures and notebooks for peer review |

---

## Phase 7 — Article Writing

### Objective

Assemble all theory notes, experiment results, and figures into a cohesive, publication-ready article.

### Article Sections

| # | Section | Source Phase | Target Words |
|---|---------|-------------|-------------|
| 1 | Introduction: why distributions matter | — | 600 |
| 2 | The cost components | Phase 0 | 500 |
| 3 | Distribution families | Phase 1 | 700 |
| 4 | Maximum Likelihood Estimation | Phase 2 | 800 |
| 5 | Model comparison | Phase 3 | 700 |
| 6 | Mixture models and multimodality | Phase 4 | 700 |
| 7 | Heavy tails and extreme costs | Phase 5 | 800 |
| 8 | Experiments and results | Phase 6 | 1,200 |
| 9 | Practical framework | — | 400 |
| 10 | Conclusion | — | 300 |
| | **Total** | | **~6,700** |

### Tasks

- [ ] **Write sections 1–5** (Issue #15)
- [ ] **Write sections 6–10** (Issue #16)
- [ ] Polish: notation, figures, references

### Deliverables

- [ ] `article/probabilistic-cost-modelling.md`

### GitHub

| Item | Value |
|------|-------|
| Branch | `phase-7/article` |
| Merge strategy | Squash merge |
| PR title | `feat(article): Phase 7 — full article draft` |
| Milestone | `Phase 7 — Article Writing` |
| Tag | `v0.8-article-draft` |
| Release | **Yes (pre-release)** — full draft for feedback |

---

## Phase 8 — Review, Polish & Publish

### Objective

Final mathematical validation, code reproducibility check, and publication.

### Tasks

- [ ] **Mathematical validation and reproducibility** (Issue #17)
- [ ] **Publish to GitHub Pages and Medium** (Issue #18)

### Deliverables

- [ ] Published article on GitHub Pages
- [ ] Medium cross-post
- [ ] LinkedIn post draft
- [ ] Final `README.md`

### GitHub

| Item | Value |
|------|-------|
| Branch | `phase-8/publish` |
| Merge strategy | Squash merge |
| PR title | `chore(publish): Phase 8 — final review and publication` |
| Milestone | `Phase 8 — Review & Publish` |
| Tag | `v1.0.0` |
| Release | **Yes (stable)** — public portfolio release |

---

## GitHub Workflow Standards

### Branch Naming Convention

```
phase-N/short-description     # phase work
fix/short-description          # bug fixes
docs/short-description         # documentation only
```

### Conventional Commits

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(core): implement MLE fitting pipeline` |
| `fix` | Bug fix | `fix(fitting): handle zero-variance edge case` |
| `docs` | Documentation | `docs(thesis): refine scope boundaries` |
| `test` | Tests | `test(mixture): add EM convergence test` |
| `refactor` | Restructuring | `refactor(distributions): unify API interface` |
| `chore` | Maintenance | `chore(github): add issue templates` |
| `style` | Formatting | `style(article): fix LaTeX alignment` |

### Pull Request Template

```markdown
## Summary

Closes #

## Type of Change

- [ ] New feature (`feat`)
- [ ] Bug fix (`fix`)
- [ ] Documentation (`docs`)
- [ ] Refactor (`refactor`)
- [ ] Test (`test`)

## Checklist

- [ ] Code runs without errors
- [ ] Tests created (author will run `pytest tests/`)
- [ ] Author ran `ruff check .`
- [ ] Documentation updated (if applicable)
- [ ] Figures regenerated (if applicable)
- [ ] No hardcoded paths or secrets

## Mathematical Validation (if applicable)

- [ ] Derivations reviewed for correctness
- [ ] Numerical examples match code output
```

### Issue Templates

#### Task (`.github/ISSUE_TEMPLATE/task.md`)

```markdown
---
name: Task
about: A specific piece of work
labels: ''
---

## Context

## Tasks
- [ ] Task 1

## Definition of Done
- [ ] Criterion 1

## References
```

#### Bug (`.github/ISSUE_TEMPLATE/bug.md`)

```markdown
---
name: Bug
about: Something is not working as expected
labels: 'type:bug'
---

## Description

## Steps to Reproduce
1.

## Expected Behaviour

## Actual Behaviour

## Environment
- Python version:
- OS:
```

---

## Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 0 — Foundation | 2–3 days | Week 1 |
| Phase 1 — Distribution Families | 5–7 days | Week 1–2 |
| Phase 2 — MLE | 7–10 days | Week 2–4 |
| Phase 3 — Model Comparison | 5–7 days | Week 4–5 |
| Phase 4 — Mixture Models | 7–10 days | Week 5–6 |
| Phase 5 — Heavy Tails | 5–7 days | Week 6–7 |
| Phase 6 — Experiments | 5–7 days | Week 7–8 |
| Phase 7 — Article Writing | 7–10 days | Week 8–10 |
| Phase 8 — Review & Publish | 3–5 days | Week 10 |

**Total: 8–10 weeks**

*Exercises are done on paper between phases. Budget 2–4 hours per exercise set.*

---

## Skills This Article Develops

| Skill | How It's Demonstrated |
|-------|----------------------|
| Statistical inference | MLE derivation, Fisher information, Cramér-Rao |
| Model selection | AIC/BIC derivation, GoF tests, KL divergence |
| Unsupervised learning | EM algorithm for GMM (from scratch) |
| Risk modelling | Heavy tails, VaR, Expected Shortfall |
| Scientific communication | Technical article with proofs and reproducible code |
| Software engineering | Modular code, tests, CI/CD-ready repo |
| Mathematical rigour | 25+ paper exercises with formal proofs |

---

## References

### Core Textbooks

- Casella, G. & Berger, R. (2002). *Statistical Inference*. Duxbury.
- Lehmann, E. & Casella, G. (1998). *Theory of Point Estimation*. Springer.
- Burnham, K. & Anderson, D. (2002). *Model Selection and Multimodel Inference*. Springer.
- Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Springer.

### Supplementary

- McLachlan, G. & Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Embrechts, P., Klüppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events*. Springer.
- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
- Johnson, N., Kotz, S. & Balakrishnan, N. (1994). *Continuous Univariate Distributions*. Wiley.

### Python Libraries

- NumPy: array operations, random sampling
- SciPy: `scipy.stats` (distributions), `scipy.optimize` (MLE)
- Matplotlib/Seaborn: publication-quality figures
- Pandas: data manipulation
- Statsmodels: additional fitting and diagnostic tools
