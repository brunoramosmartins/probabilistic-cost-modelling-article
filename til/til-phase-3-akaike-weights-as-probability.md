# TIL — Akaike Weights Are a Bayesian Posterior in Disguise

**Phase:** 3 · **Topic:** Model selection, Akaike weights · **Domain:** information theory

## Hook

AIC values themselves do not have a natural interpretation — they live
on an arbitrary scale. But you can transform a set of AICs into Akaike
weights, and those weights look suspiciously like probabilities.

## Insight

Given AIC values for $K$ candidate models, the Akaike weight of model
$i$ is:

$$w_i = \frac{e^{-\Delta_i / 2}}{\sum_j e^{-\Delta_j / 2}}$$

where $\Delta_i = AIC_i - \min(AIC)$. The weights sum to 1 and are
non-negative — they look like a probability distribution. They are:
under a uniform prior over models, $w_i$ approximates $P(\text{model } i$
$\text{is the K-L best one, given the data})$. It is not a traditional
Bayesian posterior (BIC weights would be closer to that), but it is a
coherent summary of evidence strength across competing models.

The practical value is communicative. AIC differences alone are hard to
interpret ("Model A's AIC is 5 lower than Model B's — so what?"). Akaike
weights translate that gap into a probability: $\Delta = 5$ corresponds
to $w_A / w_B \approx 12$, so Model A has about 12x the evidential
support of Model B. $\Delta > 10$ means the inferior model is essentially
ruled out.

## Example

Three models with AIC values 100, 105, 120. $\Delta = 0, 5, 20$. Raw
weights: $e^0 = 1$, $e^{-2.5} \approx 0.082$, $e^{-10} \approx 0.000045$.
Normalized: 0.924, 0.076, $\approx 0$. Conclusion: Model 1 has roughly
92% of the evidence, Model 2 has roughly 8%, Model 3 is essentially
ruled out.

## Takeaway

When stakeholders ask "how confident are you in your distribution
choice?", Akaike weights give a direct answer. Below 0.5, the choice is
contested. Above 0.9, you have a clear winner. Between 0.5 and 0.9,
model averaging may produce better predictions than committing to one.
