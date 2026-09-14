# Top-three plateau candidates: descending bootstrap selection

Use the previously reconstructed nested support paths for support_18, N=10000. Settings remain L_m=1, B=199 per pair, alpha=0.05 per comparison, and objective floor 1e-8 for plateau screening. Retain the three bounded plateaus with largest log-width and sort their dimensions descending (d1>d2>d3).

Test the original support at d1 against the original support at d2, generating bootstrap samples under the fitted d2 model. If p<0.05, select d1 and stop. Otherwise test d2 against d3, using a new null covariance fitted at d3; select d2 if p<0.05, otherwise select d3. Each replicate refits both fixed supports. No support reconstruction or plateau screening occurs inside the bootstrap. No multiplicity adjustment is applied, retaining the previous per-comparison threshold.

| Seed | Candidates, descending | First p | Second p | Selected dimension | Precision |
|---|---|---|---|---|---|
| 3 | 5, 4, 2 | 0.075 | 0.005 | 4 | 3/3 = 100.00% |
| 4 | 5, 4, 2 | 0.075 | 0.005 | 4 | 3/3 = 100.00% |
| 6 | 6, 4, 2 | 0.885 | 0.005 | 4 | 2/3 = 66.67% |
| 8 | 6, 4, 2 | 0.605 | 0.005 | 4 | 3/3 = 100.00% |
| 12 | 5, 4, 2 | 0.475 | 0.005 | 4 | 2/3 = 66.67% |
| 13 | 4, 3, 2 | 0.010 | Not reached | 4 | 3/3 = 100.00% |
| 18 | 6, 4, 2 | 0.545 | 0.005 | 4 | 3/3 = 100.00% |
| 21 | 6, 5, 2 | 0.685 | 0.005 | 5 | 3/4 = 75.00% |
| 22 | 6, 5, 4 | 0.785 | 0.385 | 4 | 3/3 = 100.00% |
| 27 | 5, 4, 2 | 0.100 | 0.005 | 4 | 3/3 = 100.00% |

Mean precision: 90.83%. Correct dimension: 9/10. Exact support recovery: 7/10.

Precision excludes diagonal entries: TP / selected off-diagonal edges. The true support has three edges, (1,4), (2,4), (3,4), and true dimension 4. Seed 21's top-three shortlist still excludes dimension 4. Seeds 6 and 12 have the wrong support at dimension 4.

## Bootstrap and verification

The unmodified ADMM solver uses beta=1, 800 iterations, tol=1e-7, 10 Halton starts, zero_tol=1e-5, omega_fixed=1, min_omega=0, omega_upper=None. The empirical eigenvalue cap is omitted exactly as in the top-two experiment, retaining all generated samples. Each null covariance solves Sigma=Lambda.T Sigma Lambda+I; positive definiteness and stability are checked. No new optimization of the observed paths is performed.

The statistic is the smaller fit's raw objective minus the larger fit's raw objective, with p=(1+number of bootstrap gains >= observed gain)/200. When the larger numerical fit is worse, the smaller fitted matrix is retained as a feasible candidate on the larger support. Clipping such negative gains to zero leaves every observed tail count unchanged because the observed gains are positive. Fits remain numerical approximations, not global-optimality certificates.

Identical pairs reuse the previous 199 bootstrap replicates. When the fitted smaller model is identical but the larger support changes, the same samples and smaller fits are reused and the new larger support is refitted. Independent replicate streams use SeedSequence([20260913, original_seed, replicate_index]); changing the null covariance reuses the random stream as common random numbers across comparisons. This does not make successive test statistics independent. The fixed-candidate bootstrap and unadjusted sequential rule do not assert familywise or post-selection error control.

Computed 19 reached pairwise comparisons. Reuse counts: {'whole_pair_draws': 1592, 'smaller_fit_draws': 199, 'newly_generated_draws': 1990}. Feasible larger-fit fallbacks: 206. All stored fit objectives and support constraints were independently recomputed. Hash checks confirm original experiments, production Python files, nested-path input, and reused bootstrap inputs are unchanged.

## Monte Carlo uncertainty

95% binomial intervals for the bootstrap exceedance probability (not model-validity confidence intervals):

| Seed | Stage | Larger vs smaller | p | Interval |
|---|---|---|---|---|
| 3 | 1 | 5 vs 4 | 0.075 | [0.0390, 0.1152] |
| 3 | 2 | 4 vs 2 | 0.005 | [0.0000, 0.0184] |
| 4 | 1 | 5 vs 4 | 0.075 | [0.0390, 0.1152] |
| 4 | 2 | 4 vs 2 | 0.005 | [0.0000, 0.0184] |
| 6 | 1 | 6 vs 4 | 0.885 | [0.8316, 0.9253] |
| 6 | 2 | 4 vs 2 | 0.005 | [0.0000, 0.0184] |
| 8 | 1 | 6 vs 4 | 0.605 | [0.5314, 0.6715] |
| 8 | 2 | 4 vs 2 | 0.005 | [0.0000, 0.0184] |
| 12 | 1 | 5 vs 4 | 0.475 | [0.4014, 0.5442] |
| 12 | 2 | 4 vs 2 | 0.005 | [0.0000, 0.0184] |
| 13 | 1 | 4 vs 3 | 0.010 | [0.0001, 0.0277] |
| 18 | 1 | 6 vs 4 | 0.545 | [0.4708, 0.6133] |
| 18 | 2 | 4 vs 2 | 0.005 | [0.0000, 0.0184] |
| 21 | 1 | 6 vs 5 | 0.685 | [0.6139, 0.7474] |
| 21 | 2 | 5 vs 2 | 0.005 | [0.0000, 0.0184] |
| 22 | 1 | 6 vs 5 | 0.785 | [0.7202, 0.8390] |
| 22 | 2 | 5 vs 4 | 0.385 | [0.3141, 0.4533] |
| 27 | 1 | 5 vs 4 | 0.100 | [0.0585, 0.1451] |
| 27 | 2 | 4 vs 2 | 0.005 | [0.0000, 0.0184] |

## Reproduction

From the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python tmp/support_18_plateau_bootstrap_top3/analyze.py
```

The script resumes from `replicates.jsonl`. `config.json` records all candidate pairs and parameters; `results.json` records reached tests, tail counts, Monte Carlo intervals, selections, and verification hashes; `summary.csv` contains the per-seed comparison with top-two selection.
