# Two-plateau bootstrap selection on nested support paths

Source: `experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18`.

The observed support paths and fitted coefficients are taken from `tmp/support_18_greedy_nested/results.json`. At L_m=1, select the two bounded plateaus with greatest absolute log-width and order their supports by dimension. The unbounded end plateaus are excluded, as in the existing selector. Plateaus are screened with objective floor 1e-8; the test uses raw objective differences. There is no subsequent factor-of-two scale adjustment because this procedure directly selects a candidate support.

For each seed, simulate B=199 independent, mean-zero Gaussian datasets of N=10000 observations from the smaller fitted model's stationary covariance. Omega is fixed at 1. Refit the ORIGINAL two candidate supports on each replicate; do not reconstruct or re-screen supports inside the bootstrap. This follows the user's clarified fixed-support test.

Compute p=(1 + number of bootstrap gains >= observed gain)/(B+1). Retain the larger support when p<0.05; otherwise choose the smaller support. Precision is TP divided by the number of selected off-diagonal edges. The true dimension is 4 and the true edges are (1,4), (2,4), (3,4).

| Seed | Candidate dimensions (small, large) | Observed gain | Exceedances / 199 | p | Selected dimension | Precision |
|---|---|---|---|---|---|---|
| 3 | 2, 5 | 0.0067752844 | 0 / 199 | 0.005 | 5 | 3/4 = 75.00% |
| 4 | 2, 4 | 0.023217514 | 0 / 199 | 0.005 | 4 | 3/3 = 100.00% |
| 6 | 4, 6 | 3.3237373e-05 | 176 / 199 | 0.885 | 4 | 2/3 = 66.67% |
| 8 | 4, 6 | 0.00028232098 | 120 / 199 | 0.605 | 4 | 3/3 = 100.00% |
| 12 | 4, 5 | 0.00011104051 | 94 / 199 | 0.475 | 4 | 2/3 = 66.67% |
| 13 | 2, 4 | 0.01469633 | 0 / 199 | 0.005 | 4 | 3/3 = 100.00% |
| 18 | 4, 6 | 0.00024688558 | 108 / 199 | 0.545 | 4 | 3/3 = 100.00% |
| 21 | 5, 6 | 4.4285916e-05 | 136 / 199 | 0.685 | 5 | 3/4 = 75.00% |
| 22 | 4, 5 | 0.00018310547 | 76 / 199 | 0.385 | 4 | 3/3 = 100.00% |
| 27 | 4, 5 | 0.00069567572 | 19 / 199 | 0.100 | 4 | 3/3 = 100.00% |

Mean per-seed precision: 88.33%. Correct dimension: 8/10. Exact support recovery: 6/10.

Dimension 4 is absent from the two-candidate shortlist for seeds 3 and 21. Precision alone does not establish complete support recovery.

## Numerical and statistical details

Fits call the unmodified production `solve_support_with_restarts` with beta=1, max_iter=800, tol=1e-7, zero_tol=1e-5, 10 Halton starts, min_omega=0, omega_fixed=1. These are numerical fits, not certified global minima.

For bootstrap refits, omega_upper=None: sample covariance eigenvalues can fall below the known omega even when sampling from a valid stationary model. The original empirical eigenvalue cap would reject those samples entirely. No generated samples are discarded or redrawn. Original observed fits are unchanged; every generating covariance is positive definite and its fitted Lambda has spectral radius below 1.

The larger numerical fit was worse than the smaller fit in 66 replicates. Since the models are nested, the smaller fitted matrix is a feasible candidate for the larger model. Including it changes these gains to zero. All observed gains are positive, so every exceedance count, p-value, and selected dimension is unchanged by this numerical fallback. Raw and verified records are both retained.

These are plug-in parametric-bootstrap tail estimates for the fixed, data-selected candidate pair; they are not a guarantee of post-selection type-I error control. B=199 gives p-values on a 0.005 grid. Monte Carlo binomial intervals describe uncertainty in the bootstrap exceedance probability, not uncertainty in model validity.

| Seed | 95% Monte Carlo interval | Contains alpha=0.05? |
|---|---|---|
| 3 | [0.0000, 0.0184] | No |
| 4 | [0.0000, 0.0184] | No |
| 6 | [0.8316, 0.9253] | No |
| 8 | [0.5314, 0.6715] | No |
| 12 | [0.4014, 0.5442] | No |
| 13 | [0.0000, 0.0184] | No |
| 18 | [0.4708, 0.6133] | No |
| 21 | [0.6139, 0.7474] | No |
| 22 | [0.3141, 0.4533] | No |
| 27 | [0.0585, 0.1451] | No |

## Reproduction

From the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python tmp/support_18_plateau_bootstrap/analyze.py
.venv/bin/python tmp/support_18_plateau_bootstrap/report.py
```

The simulation uses independent deterministic NumPy SeedSequence([20260913, original_seed, replicate_index]) streams. The runner resumes from completed JSONL records and checks that its configuration matches. All 3980 stored fit objectives and support constraints are independently recomputed and verified. SHA-256 checks confirm original experiment files, nested reconstruction results, and production Python files were unchanged during the run.

`run_b199/config.json` records candidates, fitted null parameters, generating covariances, and settings. `run_b199/replicates.jsonl` records all raw fits and empirical covariance matrices. `run_b199/replicates_verified.jsonl` additionally records the feasible fallback. `run_b199/results_verified.json` and `run_b199/summary_verified.csv` contain final results.
