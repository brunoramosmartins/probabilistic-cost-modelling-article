# Cost Model Design

## Philosophy

The model represents **individual cost components** as random variables with
distinct distributional properties. Unlike the Monte Carlo article (which focuses
on total cost simulation), this article is about understanding the **shape** of
each component.

Each cost component is modelled independently. The article does not model
dependence between components (anti-scope: copulas). The goal is to demonstrate
that choosing the right marginal distribution for each component is the critical
first step.

## Cost Components

### 1. Base Salary ($S_i$)

| Property | Value |
|----------|-------|
| Support | $(0, \infty)$ |
| Candidate distributions | LogNormal, Gamma, Mixture(Normal) |
| Rationale | Right-skewed; multimodal if junior/senior clusters exist |
| Typical range | R$ 5,000 – R$ 40,000/month |

**Default synthetic parameters:**
- LogNormal: $\mu = 9.1$, $\sigma = 0.4$ (median ~ R$ 9,000)
- Mixture(Normal): $\pi = (0.6, 0.4)$, $\mu = (8000, 18000)$, $\sigma = (1500, 2500)$

### 2. Monthly Overtime Hours ($H_i$)

| Property | Value |
|----------|-------|
| Support | $\{0, 1, 2, ...\}$ |
| Candidate distributions | Poisson, Negative Binomial, Zero-inflated Poisson |
| Rationale | Discrete, often zero-inflated (many employees have zero overtime) |
| Typical range | 0 – 60 hours/month |

**Default synthetic parameters:**
- Poisson: $\lambda = 8$
- Zero-inflated Poisson: $p_0 = 0.3$, $\lambda = 12$

*Note: This component is discrete. The article focuses on continuous
distributions but mentions discrete components for completeness.*

### 3. Overtime Cost per Event ($C_{ot}$)

| Property | Value |
|----------|-------|
| Support | $(0, \infty)$ |
| Candidate distributions | LogNormal, Gamma |
| Rationale | Right-skewed, always positive, variable by seniority |
| Typical range | R$ 50 – R$ 500/hour |

**Default synthetic parameters:**
- LogNormal: $\mu = 4.6$, $\sigma = 0.5$ (median ~ R$ 100/hour)
- Gamma: $\alpha = 4$, $\beta = 30$

### 4. Annual Severance Events ($N_{sev}$)

| Property | Value |
|----------|-------|
| Support | $\{0, 1, 2, ...\}$ |
| Candidate distributions | Poisson, Binomial |
| Rationale | Rare events, count per year |
| Typical range | 0 – 10 events/year for a 50-person team |

**Default synthetic parameters:**
- Poisson: $\lambda = 3$ (for a 50-person team)
- Binomial: $n = 50$, $p = 0.06$

### 5. Severance Cost per Event ($C_{sev}$)

| Property | Value |
|----------|-------|
| Support | $(x_m, \infty)$ with $x_m > 0$ |
| Candidate distributions | Pareto, LogNormal |
| Rationale | Heavy-tailed: most cases are moderate, a few are extremely expensive |
| Typical range | R$ 10,000 – R$ 500,000+ |

**Default synthetic parameters:**
- Pareto: $\alpha = 2.5$, $x_m = 10000$
- LogNormal: $\mu = 10.5$, $\sigma = 1.0$

### 6. Hiring Cost per Position ($C_h$)

| Property | Value |
|----------|-------|
| Support | $(0, \infty)$ |
| Candidate distributions | LogNormal, Gamma |
| Rationale | Variable (recruiter fees, relocation, signing bonus) |
| Typical range | R$ 5,000 – R$ 80,000 |

**Default synthetic parameters:**
- LogNormal: $\mu = 9.5$, $\sigma = 0.7$
- Gamma: $\alpha = 3$, $\beta = 5000$

### 7. Benefits Multiplier ($\beta_i$)

| Property | Value |
|----------|-------|
| Support | $[a, b]$ typically $[1.3, 2.2]$ |
| Candidate distributions | Uniform, Beta |
| Rationale | Bounded: benefits are a multiplier of salary |
| Typical range | 1.3x – 2.2x of base salary |

**Default synthetic parameters:**
- Uniform: $a = 1.4$, $b = 2.0$
- Beta: $\alpha = 3$, $\beta = 3$ (scaled to $[1.3, 2.2]$)

## Concrete Example: 50-Person Team

| Component | Distribution | Parameters | Annual Budget Impact |
|-----------|-------------|------------|---------------------|
| Base salary | LogNormal($9.1, 0.4$) | 50 employees | ~R$ 5.4M/year |
| Overtime | LogNormal($4.6, 0.5$) | ~8 hrs/month avg | ~R$ 480K/year |
| Severance | Pareto($2.5, 10000$) | ~3 events/year | ~R$ 50K–500K/year |
| Hiring | LogNormal($9.5, 0.7$) | ~5 hires/year | ~R$ 75K/year |
| Benefits | Uniform($1.4, 2.0$) | multiplier | included in salary |

**Total expected annual people cost: ~R$ 6.0M–6.5M**

The key insight: the *variance* of this total depends critically on which
distributions you assume. Normal assumptions produce a narrow CI; correct
heavy-tailed assumptions produce a much wider one.

## Expansion Points (v2 — not in this article)

- [ ] Dependence between salary and overtime (higher salary → less overtime?)
- [ ] Copula-based multivariate modelling
- [ ] Time-varying distributions (salary growth, inflation)
- [ ] Bayesian hierarchical model for team-level effects
- [ ] Real data calibration (requires NDA-free dataset)

## Key Questions This Model Answers

1. **Which family?** Is salary LogNormal, Gamma, or a mixture? How to tell.
2. **Which parameters?** Given the family, what are the MLE estimates and their
   uncertainty?
3. **Does it matter?** If I pick Normal instead of LogNormal, how wrong is my
   budget?
4. **What about the tails?** Severance costs might follow a Pareto. How much
   reserve does that imply?
5. **What about clusters?** If salaries are bimodal (junior vs senior), a single
   distribution is wrong.
