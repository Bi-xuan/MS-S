# Nested support recovery and top-three plateau bootstrap across sample sizes

Sources are the three explicitly requested support_18 folders, including the folder named nsm1e6 (not the separately named nsm1000000 folder). Each saved covariance is reused without resampling for the observed reconstruction.

Settings: L_m=1, B=199 per pair, alpha=0.05 per comparison, objective floor=1e-8 for screening. Start with the diagonal-only support at D=1, enumerate every upper-triangular one-edge extension at each dimension through 7, fit all candidates with 10 Halton starts, and retain the strict minimum objective. Rank bounded plateaus by absolute log-width and keep the top three. All 30 curves have at least three bounded plateaus.

Order candidates d1>d2>d3. Test d1 against d2 under the fitted d2 null. If p<0.05, select d1 and stop. Otherwise test d2 against d3 under the fitted d3 null; select d2 if p<0.05, else d3. The original pair of supports is fixed and both are refitted in every bootstrap replicate. No bootstrap support reconstruction, no bootstrap screening, and no multiplicity adjustment are performed. At p=0.050 exactly, the smaller support is selected.

The statistic uses raw objective gains and p=(1 + number of bootstrap gains >= observed gain)/200. Precision is the fraction of selected off-diagonal edges that belong to the true support. True edges are (1,4), (2,4), (3,4); true dimension is 4. Means below are arithmetic means across the ten seeds.

## N=1000000

| Seed | Candidates descending | First p | Second p | Selected dimension | Precision |
|---|---|---|---|---|---|
| 2 | 6, 4, 2 | 0.260 | 0.005 | 4 | 3/3 = 100.00% |
| 3 | 5, 4, 2 | 0.030 | Not reached | 5 | 3/4 = 75.00% |
| 124 | 5, 4, 2 | 0.230 | 0.005 | 4 | 3/3 = 100.00% |
| 139 | 6, 4, 2 | 0.630 | 0.005 | 4 | 3/3 = 100.00% |
| 147 | 6, 4, 2 | 0.975 | 0.005 | 4 | 3/3 = 100.00% |
| 153 | 6, 4, 2 | 0.445 | 0.005 | 4 | 3/3 = 100.00% |
| 156 | 6, 4, 2 | 0.555 | 0.005 | 4 | 3/3 = 100.00% |
| 173 | 5, 4, 2 | 0.105 | 0.005 | 4 | 3/3 = 100.00% |
| 192 | 6, 4, 2 | 0.180 | 0.005 | 4 | 3/3 = 100.00% |
| 237 | 5, 4, 2 | 0.100 | 0.005 | 4 | 3/3 = 100.00% |

Mean precision: 97.50%. Correct dimension: 9/10. Exact support recovery: 9/10. Observed nested path passes through the true support: 10/10.

## N=100

| Seed | Candidates descending | First p | Second p | Selected dimension | Precision |
|---|---|---|---|---|---|
| 124 | 6, 5, 2 | 0.830 | 0.710 | 2 | 1/1 = 100.00% |
| 139 | 4, 3, 2 | 0.610 | 0.250 | 2 | 1/1 = 100.00% |
| 141 | 6, 5, 4 | 0.845 | 0.755 | 4 | 2/3 = 66.67% |
| 147 | 6, 5, 4 | 0.855 | 0.685 | 4 | 2/3 = 66.67% |
| 153 | 6, 5, 3 | 0.880 | 0.950 | 3 | 0/2 = 0.00% |
| 156 | 6, 5, 4 | 0.785 | 0.670 | 4 | 2/3 = 66.67% |
| 173 | 6, 4, 2 | 0.890 | 0.605 | 2 | 1/1 = 100.00% |
| 192 | 6, 4, 3 | 0.925 | 0.830 | 3 | 1/2 = 50.00% |
| 237 | 5, 3, 2 | 0.430 | 0.285 | 2 | 1/1 = 100.00% |
| 243 | 6, 5, 4 | 0.570 | 0.705 | 4 | 1/3 = 33.33% |

Mean precision: 68.33%. Correct dimension: 4/10. Exact support recovery: 0/10. Observed nested path passes through the true support: 2/10.

## N=1000

| Seed | Candidates descending | First p | Second p | Selected dimension | Precision |
|---|---|---|---|---|---|
| 6 | 6, 5, 3 | 0.665 | 0.150 | 3 | 2/2 = 100.00% |
| 15 | 6, 4, 2 | 0.755 | 0.400 | 2 | 1/1 = 100.00% |
| 23 | 6, 5, 4 | 0.350 | 0.050 | 4 | 3/3 = 100.00% |
| 46 | 5, 4, 2 | 0.535 | 0.050 | 2 | 1/1 = 100.00% |
| 51 | 5, 3, 2 | 0.610 | 0.065 | 2 | 1/1 = 100.00% |
| 53 | 6, 4, 2 | 0.520 | 0.090 | 2 | 1/1 = 100.00% |
| 59 | 6, 5, 4 | 0.895 | 0.625 | 4 | 3/3 = 100.00% |
| 61 | 6, 5, 2 | 0.845 | 0.985 | 2 | 1/1 = 100.00% |
| 74 | 6, 3, 2 | 0.570 | 0.160 | 2 | 1/1 = 100.00% |
| 85 | 6, 4, 3 | 0.615 | 0.095 | 3 | 2/2 = 100.00% |

Mean precision: 100.00%. Correct dimension: 2/10. Exact support recovery: 2/10. Observed nested path passes through the true support: 4/10.

## Interpretation and Monte Carlo uncertainty

At N=1000, precision is 100% for every seed, but most selected models omit true edges; exact support recovery occurs in only 2/10 seeds. At N=100, four selections have the true dimension but all four have incorrect supports. At N=1000000, seed 3 retains dimension 5 rather than 4 (first-stage p=0.030).

With B=199, p-values have resolution 0.005. The following reached tests have 95% binomial Monte Carlo intervals for their bootstrap exceedance probability that include 0.05:

| N | Seed | Stage | Larger vs smaller | p | Monte Carlo interval |
|---|---|---|---|---|---|
| 1000000 | 3 | 1 | 5 vs 4 | 0.030 | [0.0082, 0.0577] |
| 1000 | 23 | 2 | 5 vs 4 | 0.050 | [0.0209, 0.0841] |
| 1000 | 46 | 2 | 4 vs 2 | 0.050 | [0.0209, 0.0841] |
| 1000 | 51 | 2 | 3 vs 2 | 0.065 | [0.0315, 0.1030] |

These are finite-bootstrap outcomes. The intervals describe simulation uncertainty, not model-validity confidence intervals. The fixed, data-selected candidate pairs and unadjusted sequential testing do not claim post-selection or familywise error control.

## Computation and verification

Under the mean-zero Gaussian model, X.T @ X has the Wishart(N, Sigma_null) distribution. Each bootstrap covariance is drawn as Wishart(df=N, scale=Sigma_null)/N. This is the exact finite-sample covariance law of the original sampling scheme, not an asymptotic covariance approximation. It avoids allocating a million-row dataset for each large-N replicate. The original code uses the uncentered covariance X.T X/N, hence df=N rather than N-1. [SciPy Wishart sampler documentation](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.stats.wishart.html).

Random streams use SeedSequence([20260913, N, seed, replicate_index]). Generating covariances solve Sigma_null=Lambda.T Sigma_null Lambda+I; stability, positive definiteness, and the Wishart mean are verified. Omega is fixed at 1. As in the earlier bootstrap runs, the empirical eigenvalue cap is omitted to retain all generated samples.

The diagnostic batches the same ADMM matrix updates, per-start convergence checks, 800-iteration cap, 10 Halton starts, beta=1, tol=1e-7, and zero_tol=1e-5. A preflight comparison of 82 fits against the unmodified native solver agreed to maximum objective error 1.0408340855860843e-17. All 30 observed D=1,2 objectives reproduce the saved unrestricted objectives to less than 7e-16, and all 30 reconstructed objective curves are nonincreasing within 1e-8.

Post-run verification refitted the bootstrap sample nearest the observed cutoff for each of 59 reached tests using the original solver (118 native fits). Maximum objective difference: 1.388e-17; every tail indicator matched. All 11741 bootstrap sample pairs have checked support constraints and independently recomputed objective values. No generated sample is discarded.

There were 542 negative raw numerical gains. The smaller fitted coefficient matrix is feasible on the larger nested support, so it is retained as a fallback and the gain becomes zero. Every observed gain is positive; the fallback leaves all tail counts and selections unchanged. Numerical ADMM fits are not certificates of global minima.

SHA-256 checks confirm the three original experiment directories and production Python files are unchanged. All new artifacts are in this diagnostic directory.

## Reproduction

From the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python tmp/support_18_multisample_bootstrap/analyze.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python tmp/support_18_multisample_bootstrap/report.py
```

The runner resumes from nested_paths.json and bootstrap.jsonl. results.json contains all selected dimensions, precisions, test parameters, p-values, and Monte Carlo intervals; summary.csv provides a compact per-seed table. The two validation JSON files record equivalence checks against the original solver.
