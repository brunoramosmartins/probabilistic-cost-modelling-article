# TIL — A Fixed Seed Is the Cheapest Insurance Policy in Computational Work

**Phase:** 6 · **Topic:** Reproducibility, seeds, scientific rigor · **Domain:** experimental design, software engineering

## Hook

Every figure in a serious technical article should be reproducible to
the pixel. The cost of guaranteeing this is approximately one line of
code: `rng = np.random.default_rng(42)`.

## Insight

Random number generators in NumPy, SciPy, and most scientific libraries
are deterministic given a seed. Set the seed, and any sequence of
random operations produces identical results across runs. Don't set it,
and every plot is a fresh face — invalidating cross-references,
breaking captions, making bug reports impossible to pin down. Fixing
the seed costs nothing and prevents an entire class of "I cannot
reproduce your number" headaches.

The deeper discipline is to make every script self-contained: take the
seed as a constant at the top of the file, pass it explicitly to every
function that draws random numbers, and document the exact NumPy
version in your environment. Implicit global RNG state — `np.random`
without an explicit `Generator` — is a hidden source of irreproducibility,
because library updates can change the default RNG. Use `np.random.default_rng(seed)`
and pass the generator explicitly.

This is what separates "I ran this once and it worked" from "anyone can
run this and verify the same result."

## Example

In `scripts/exp_mle_convergence.py`, `SEED = 42` is defined at the top.
The 200 replications at each sample size, the 9 sample sizes, the
figure layout — every output is bit-for-bit reproducible. Re-running
the script tomorrow, or on a different machine, or six months from now,
produces an identical figure. The only thing that changes the output is
an explicit change to the seed or to the code itself.

## Takeaway

Reproducibility is not a research-grade luxury. For technical writing,
it is table stakes. Set seeds, pin versions, document the workflow.
The cost is negligible; the debugging time saved is hours per paper —
and the credibility gained is permanent.
