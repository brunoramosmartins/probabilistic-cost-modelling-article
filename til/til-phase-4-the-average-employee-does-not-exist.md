# TIL — In a Bimodal Distribution, the Mean Represents No One

**Phase:** 4 · **Topic:** Mixture models, multimodality, summary statistics · **Domain:** workforce analytics

## Hook

The "average employee" is a comforting concept. In a team with juniors
at R\$ 8,000 and seniors at R\$ 18,000, the average is R\$ 12,200 —
a salary nobody actually receives.

## Insight

The mean is a single-number summary that assumes a unimodal
distribution. When the actual distribution is bimodal — two clusters
with a gap between them — the mean falls in the empty space. The
standard deviation around that mean is inflated because it is measuring
distance to a point that does not exist in the data. Confidence
intervals built on this inflated SD give the right number for the wrong
question: they describe spread around a fictional center, not the
spread within either real cluster.

For budget purposes, this matters in three ways. First, variance is
overestimated, so reserves look bigger than the true model would
require. Second, tail probabilities are misestimated in either
direction (depending on which cluster the threshold falls between).
Third, policy decisions framed in terms of "the typical employee"
address a phantom — neither juniors nor seniors are well-served by a
policy targeting the gap between them.

The diagnostic is simple: a histogram. If you see two peaks, do not
report a single mean. Fit a Gaussian Mixture Model with $K = 2$, report
the mean and standard deviation of each cluster, and let downstream
decisions act on the actual structure.

## Example

60% juniors at R\$ 8,000 ± R\$ 1,500, 40% seniors at R\$ 18,000 ±
R\$ 2,500. The pooled mean is $0.6 \cdot 8000 + 0.4 \cdot 18000 = $
R\$ 12,000. The pooled SD is around R\$ 5,200 — about 3x larger than
within either cluster. $P(\text{salary} < \text{R\$ 5,000})$ under the
single Normal: ~8%. Under the true mixture: ~2.5%. Each cluster's
actual variability is being washed out.

## Takeaway

Before reporting a mean, verify it represents an actual mode. Histograms
are five-second insurance against bimodality. When clusters exist,
model them as a Gaussian Mixture, not a single Normal — and report
cluster-specific means instead of a global one.
