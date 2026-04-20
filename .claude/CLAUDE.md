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
