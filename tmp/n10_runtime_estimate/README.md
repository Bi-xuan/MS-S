# N=10 runtime estimate (2026-10-07)

Scope: 40 trials, upper support, sample sizes 100, 1000, 10000,
1000000, ten seeds each; `lambda_support_recovery.select_support`
with default statistical settings. Assume the existing fixed-support
population construction extended to 10 variables (nine true off-diagonal
edges, population omega 1, edge magnitudes clipped to [0.2, 0.6]).

Local runtime: Python 3.10.18, NumPy 2.2.6, SciPy 1.15.3, x86_64,
eight logical CPUs visible. BLAS/OpenMP thread limits all set to one.
These measurements do not benchmark the target 100-CPU machine.

## Measurements

- 24 sampled support fits, each with ten Halton starts and up to 800
  iterations: 0.086–0.864 seconds per support, including all starts.
- Generating 1,000,000 ten-variable observations and their covariance:
  0.275 seconds locally.
- One complete API call, num_samples=100, observation RNG seed 126,
  `n_jobs=5`, all statistical defaults, API random_seed left at 42:
  254.884 seconds (4.248 minutes).
- Full nested curve: 132.009 seconds.
- Bootstrap selection: approximately 122.875 seconds; two comparisons,
  199 replicates each. Final selected dimension: 2.

Detailed measurements are in fit_timings.json and trial_timing.json;
the two benchmark scripts reproduce the pilots.

## Work and extrapolation

There are 45 allowed upper edges and 46 sequential path dimensions.
The nested curve fits 1 + 45 + 44 + ... + 1 = 1036 supports per trial.
Bootstrap adds at most 2 comparisons * 199 replicates * 2 supports = 796
support fits, plus at most three observed candidate refits. Each support
fit has ten starts.

Twenty concurrent trials with five workers each execute the 40 trials in
two waves. Two waves at the measured pilot time give 8.50 minutes, assuming
comparable per-core throughput without significant contention. A practical
planning range is 5–20 minutes; reserve a one-hour allocation for unmeasured
hardware performance, seed variation, process startup and shared-node load.
This is an extrapolation from one full trial and 24 sampled fits, not a
measured completion time for the full study or a guaranteed upper bound.

If n_jobs stays at its literal default of one, the 40 independent trials
can use at most 40 CPU cores with one BLAS thread each. A rough planning
range is 10–30 minutes under comparable CPU performance; this configuration
was not measured end to end.

Bootstrap draws covariance matrices using Wishart sampling when sample size
is at least the matrix dimension, so it does not generate one million rows
per bootstrap replicate. API fitting receives only a 10-by-10 covariance.

## Producing the existing summary

Use return_result=True and save the selected mask, full curve masks and
objectives, sample size, seed, and true support. Those data allow the same
precision, exact recovery, best-on-path selection, and true-support-on-path
summary, plus existing plot types. The API itself has no file or plotting
side effects. A small harness must serialize the result into the formats
expected by the existing summary and plotting scripts.

The default shell runner currently calls the older experiment CLI, rather
than the package API. Changing N alone in that runner does not implement
the requested API-based study. No production experiment outputs or estimator
code were modified; only these timing-pilot files were added.
