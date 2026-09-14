# Dimension selection on the forward-nested support paths

Source paths and objectives: `../results.json`, reconstructed from the support_18 N=10000 experiment.

Each table entry is selected dimension / support precision. Precision is TP divided by selected off-diagonal edges; diagonal entries and the dimension offset are excluded. Failed selections have undefined precision. The true dimension is 4 with three true edges.

Settings: objective floor 1e-8, recommendation factor 2, automatic window width, r=L=1, xi=10, and covariance-derived constants from each original NPZ. The penalty is sqrt(D_m) * (K + sqrt(2*v*L_m)), evaluated with L_m=1, D_m, or D_m^8. The existing selectors are called without modification.

## Plateau

| Seed | L_m=1 | L_m=D_m | L_m=D_m^8 |
|---|---|---|---|
| 3 | 2 / 100.00% | 2 / 100.00% | 2 / 100.00% |
| 4 | 4 / 100.00% | 4 / 100.00% | 4 / 100.00% |
| 6 | 6 / 60.00% | 6 / 60.00% | 6 / 60.00% |
| 8 | 6 / 60.00% | 6 / 60.00% | 6 / 60.00% |
| 12 | 4 / 66.67% | 4 / 66.67% | 4 / 66.67% |
| 13 | 4 / 100.00% | 4 / 100.00% | 4 / 100.00% |
| 18 | 6 / 60.00% | 6 / 60.00% | 6 / 60.00% |
| 21 | 5 / 75.00% | 5 / 75.00% | 2 / 100.00% |
| 22 | 4 / 100.00% | 4 / 100.00% | 4 / 100.00% |
| 27 | 5 / 75.00% | 5 / 75.00% | 5 / 75.00% |

## Window

| Seed | L_m=1 | L_m=D_m | L_m=D_m^8 |
|---|---|---|---|
| 3 | Failed / — | Failed / — | Failed / — |
| 4 | Failed / — | Failed / — | Failed / — |
| 6 | 2 / 100.00% | 2 / 100.00% | Failed / — |
| 8 | Failed / — | 2 / 100.00% | Failed / — |
| 12 | Failed / — | Failed / — | 5 / 50.00% |
| 13 | 4 / 100.00% | 4 / 100.00% | 5 / 75.00% |
| 18 | 4 / 100.00% | 4 / 100.00% | 4 / 100.00% |
| 21 | 2 / 100.00% | 2 / 100.00% | Failed / — |
| 22 | Failed / — | Failed / — | Failed / — |
| 27 | Failed / — | Failed / — | 5 / 75.00% |

## Aggregate results

Mean precision is the arithmetic mean over defined results, not pooled edge counts. Failures count as unsuccessful for dimension and exact-support recovery.

| Method | L_m | Returned | Mean precision | Correct dimension | Exact support |
|---|---|---|---|---|---|
| plateau | 1 | 10/10 | 79.67% (10 seeds) | 4/10 | 3/10 |
| plateau | D_m | 10/10 | 79.67% (10 seeds) | 4/10 | 3/10 |
| plateau | D_m^8 | 10/10 | 82.17% (10 seeds) | 4/10 | 3/10 |
| window | 1 | 4/10 | 100.00% (4 seeds) | 2/10 | 2/10 |
| window | D_m | 5/10 | 100.00% (5 seeds) | 2/10 | 2/10 |
| window | D_m^8 | 4/10 | 75.00% (4 seeds) | 1/10 | 1/10 |

Precision of 100% at dimension 2 or 3 still omits true edges; it does not imply exact recovery.

## Reproduction and verification

Run `.venv/bin/python tmp/support_18_greedy_nested/select_dimensions.py` from the repository root.

The script checks all three penalty formulas, directly minimizes the penalized objective for every returned dimension, and verifies support edge counts. `comparison.csv` records dimensions, TP counts, precisions, scales, and failure reasons; `results.json` additionally records all selection diagnostics and objective/penalty arrays. Hash checks confirm the nested reconstruction results, original experiment files, and production Python files are unchanged.
