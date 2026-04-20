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
