# Support 18, N=10000, Lm=D_m

Source: `experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18`.

Run `.venv/bin/python tmp/support_18_lm_dimension_n10000/analyze.py` from the project root. This reuses the earlier replay runner, invokes the scaling selector for both Lm=1 and Lm=D_m, and computes support precision from the saved selected support mask at the recommended dimension. Original experiment files and production code remain unchanged.

Settings: objective floor 1e-8, recommendation factor 2, automatic window width, r=L=1, xi=10, N=10000 and covariance constants from the NPZ. The dimension-specific penalty is `K*sqrt(D_m)+sqrt(2*v)*D_m`.

Precision is TP / selected off-diagonal edges. Diagonal entries and the dimension offset are excluded. The true support has three edges and true dimension 4. A failed selection has undefined precision. Means are arithmetic averages across seeds with a defined result, not pooled edge counts.

| Seed | Plateau dimension | Plateau precision | Window dimension | Window precision |
|---|---:|---:|---:|---:|
| 3 | 4 | 2/3 | 4 | 2/3 |
| 4 | 4 | 2/3 | 2 | 1/1 |
| 6 | 6 | 3/5 | 2 | 1/1 |
| 8 | 6 | 3/5 | 2 | 1/1 |
| 12 | 4 | 2/3 | Failed | N/A |
| 13 | 4 | 3/3 | 4 | 3/3 |
| 18 | 6 | 3/5 | Failed | N/A |
| 21 | 4 | 2/3 | 2 | 1/1 |
| 22 | 4 | 3/3 | Failed | N/A |
| 27 | 5 | 3/4 | Failed | N/A |
| Mean | 4.70 | 72.17% | 2.67 | 94.44% |

Plateau means use 10 seeds; window means use 6 successful seeds. Window's 100% precision at dimension 2 means it selected one true edge out of the three true edges; it does not imply full support recovery.

Compared with Lm=1, all plateau outcomes are unchanged. Only window seed 8 changes, from failure to dimension 2 with precision 100%. Baseline window mean precision is 93.33% over 5 successful seeds. Dimension accuracy remains 6/10 for plateau and 2/10 for window, counting failures as unsuccessful.

Verification: all 20 baseline dimensions/failures reproduce the saved results, and all successful baseline scales match the saved logs to relative tolerance 1e-10. The replay verifies the penalty formula and directly minimizes each successful penalized criterion. Additional assertions verify selected-support validity, selected edge count equals dimension minus one, and the true edge count is three.

`comparison.csv` and `results.json` contain all baseline and modified scales, dimensions, TP counts, selected edge counts, and precision values. `summary.json` contains averages and denominators. Per-seed logs contain selection diagnostics.
