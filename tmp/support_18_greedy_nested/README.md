# Forward-nested support reconstruction: support 18

The reconstructed path passes through the exact true support in **8/10 seeds**.
The true off-diagonal edges are **(1,4), (2,4), (3,4)** in one-based indexing.

| Seed | Exact true support at D_m=4 | First three edges added, in order |
|---|---|---|
| 3 | Yes | (3,4), (2,4), (1,4) |
| 4 | Yes | (3,4), (2,4), (1,4) |
| 6 | No | (3,4), (1,4), (1,2) |
| 8 | Yes | (3,4), (1,4), (2,4) |
| 12 | No | (3,4), (1,4), (1,2) |
| 13 | Yes | (3,4), (2,4), (1,4) |
| 18 | Yes | (3,4), (1,4), (2,4) |
| 21 | Yes | (3,4), (1,4), (2,4) |
| 22 | Yes | (3,4), (1,4), (2,4) |
| 27 | Yes | (3,4), (2,4), (1,4) |

The saved unrestricted paths hit the true support in 5/10 seeds. Nesting adds
successes for seeds 3, 4, and 21 and preserves the original five successes.
This measures support reconstruction across dimensions, not the outcome of a
subsequent penalty/scale-based dimension selector.

## Method

Input: `experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18`.
Use each saved empirical covariance (N=10000), without resampling. Start with
the diagonal-only support at D_m=1. At every subsequent dimension through 7,
enumerate all one-edge upper-triangular extensions of the preceding selected
support, fit each candidate, and select the strict numerical minimum objective.
There are 22 candidate support fits per seed, each with 10 starts.
No true support is forced; truth is used only to evaluate the resulting path.

The objective is ||Sigma_hat - Lambda.T Sigma_hat Lambda - I||_F^2, with
omega fixed at 1. Fits call the existing `solve_support_with_restarts`, with
beta=1, max_iter=800, tol=1e-7, zero_tol=1e-5, max_restarts=10, Halton
initialization, min_omega=0, and omega_upper=lambda_min(Sigma_hat)-1e-6.
These settings follow the current experiment driver and study runner; the
saved NPZ files do not record all historical solver settings. Support-level
minima are numerical ADMM fits, not certificates of global optimality.

Exact ties retain candidate enumeration order. The smallest winner/runner-up
gap over D_m=2,3,4 across all seeds is 5.53978906e-5, so the recovery outcomes
are unaffected by using the production support comparison tolerance of 1e-8.

## Failed seeds

At D_m=3 both failed seeds have edges (1,4), (3,4), so the true support is
an eligible extension at D_m=4. Nevertheless, adding (1,2) has a lower objective:

| Seed | Selected incorrect support objective | True support objective |
|---|---:|---:|
| 6 | 3.32576848e-5 | 2.99073504e-4 |
| 12 | 1.40941853e-4 | 3.05833930e-4 |

Both paths first contain all true edges at D_m=6, alongside two extra edges.
They therefore never equal the true support.

## Reproduction and checks

Run from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python tmp/support_18_greedy_nested/analyze.py
```

`summary.csv` contains per-seed results and complete edge-addition orders.
`results.json` and `seed_*.json` include every candidate objective and fitted
coefficient matrix, the complete selected paths, and solver settings.

Checks passed for candidate counts, nesting, dimensions, fitted coefficients
outside the support, recomputed objectives, and nonincreasing objective paths.
The first two dimensions reproduce saved exhaustive-search objectives within
rtol=1e-6 / atol=1e-8; selected D_m=2 supports match exactly. A separate
inspection confirms every selected edge has a nonzero fitted coefficient.
SHA-256 comparisons confirm all original experiment files and production
Python files are unchanged. All new files reside in this diagnostic directory.
