# The Shape of What You'll Spend

**Probabilistic Modelling of People Costs — from Distribution Selection to Budget Impact**

---

## About

A budget built on Normal distributions underestimates tail risk by construction.
People costs — salaries, overtime, severance — are structurally non-Normal:
right-skewed, heavy-tailed, and often multimodal.

This article derives the mathematics of distribution selection, parameter
estimation, and model comparison from first principles, then demonstrates the
budget impact of getting it right — or wrong.

## Article Thesis

> Choosing the wrong distributional family is not a modelling nuance — it is a
> systematic bias that propagates through every downstream calculation.

## Repository Structure

```
probabilistic-cost-modelling-article/
├── article/                    # Final article (Markdown + LaTeX math)
├── docs/                       # Planning: thesis, model design, outline
├── src/                        # Reusable source code (Python)
├── scripts/                    # Standalone experiment scripts
├── notebooks/                  # Jupyter notebooks (exploration)
├── exercises/                  # Paper exercises (LaTeX-compatible MD)
├── figures/                    # Generated plots (300 DPI)
├── notes/                      # Phase-by-phase theory notes
├── tests/                      # Unit tests (pytest)
├── .claude/                    # Claude Code configuration
├── .github/                    # Issue templates, PR template, setup scripts
├── requirements.txt            # Pinned Python dependencies
├── pyproject.toml              # Project metadata + ruff config
└── LICENSE                     # MIT
```

## Quick Start

```bash
# Clone
git clone https://github.com/brunoramosmartins/probabilistic-cost-modelling-article.git
cd probabilistic-cost-modelling-article

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run linter
ruff check .

# Run tests
pytest tests/
```

## GitHub Setup (first time only)

```bash
# Create labels
bash .github/setup/labels.sh owner/repo

# Create milestones
bash .github/setup/milestones.sh owner/repo

# Create issues (run after labels and milestones)
bash .github/setup/issues.sh owner/repo
```

## Tech Stack

- **Python 3.10+**
- numpy / scipy — distribution fitting, MLE, GoF tests
- matplotlib / seaborn — publication-quality figures
- pandas — data manipulation
- statsmodels — additional fitting tools
- ruff — linter and formatter
- pytest — testing

## Phases

| Phase | Topic | Status |
|-------|-------|--------|
| 0 | Foundation (thesis, model, scaffold) | In progress |
| 1 | Distribution Families | Planned |
| 2 | Maximum Likelihood Estimation | Planned |
| 3 | Model Comparison (AIC, BIC, GoF) | Planned |
| 4 | Mixture Models (EM algorithm) | Planned |
| 5 | Heavy Tails (EVT, budget impact) | Planned |
| 6 | Experiments & Visualizations | Planned |
| 7 | Article Writing | Planned |
| 8 | Review & Publish | Planned |

## Related Work

This article is the second in a pair:

1. **Monte Carlo Simulation for Budget Estimation** — focuses on *simulating*
   total cost given assumed distributions.
2. **This article** — focuses on *choosing* which distributions to assume in the
   first place.

## License

MIT
