# Support 18: empirical support ranking versus scale calibration

Source: `experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18`.

All ten available seeds were evaluated: 3, 4, 6, 8, 12, 13, 18, 21, 22, 27. The true dimension is 4, corresponding to three off-diagonal edges plus the project's dimension offset. The true edges, in one-based notation, are (1,4), (2,4), (3,4). Population separation is outside this audit.

## Conclusion

For the saved plateau procedure, support ranking and scale calibration are comparable bottlenecks: ranking fails in 5/10 runs and scale calibration fails in 4/10, with one overlapping seed. Ranking is slightly more frequent, but these ten observations do not establish a clear dominant mechanism. For window, scale selection is the larger operational bottleneck: 3 wrong dimensions and 5 failures to return a dimension, versus 5 empirical-ranking failures. Counts describe this support and these seeds, not the whole experiment family.

The distinction matters for the final goal: plateau selects dimension 4 in 6/10 runs but recovers the true support and dimension jointly in only 2/10. Window selects dimension 4 in 2/10 but succeeds jointly in only 1/10.

## Saved outcomes

“Correct support” refers to the selected mask at dimension 4. “No result” denotes the logged window exception, not a selected dimension.

| Seed | Correct support at 4 | Plateau dimension | Window dimension | Plateau joint recovery | Window joint recovery |
|---|---|---:|---:|---|---|
| 3 | No | 4 | 4 | No | No |
| 4 | No | 4 | 2 | No | No |
| 6 | No | 6 | 2 | No | No |
| 8 | Yes | 6 | No result | No | No |
| 12 | No | 4 | No result | No | No |
| 13 | Yes | 4 | 4 | Yes | Yes |
| 18 | Yes | 6 | No result | No | No |
| 21 | No | 4 | 2 | No | No |
| 22 | Yes | 4 | No result | Yes | No |
| 27 | Yes | 5 | No result | No | No |

| Failure category | Plateau seeds | Window seeds |
|---|---|---|
| Ranking only | 3, 4, 12, 21 | 3 |
| Calibration only, including no result | 8, 18, 27 | 8, 18, 22, 27 |
| Both | 6 | 4, 6, 12, 21 |
| Neither | 13, 22 | 13 |

Conditioned on already having the correct support at dimension 4, plateau still fails dimension selection in 3/5 cases and window in 4/5. These conditional counts are descriptive; the two errors are not independent.

## The five wrong supports are genuine empirical-ranking failures

For seeds 3, 4, 6, and 12 the selected edges are (1,2), (1,4), (3,4). For seed 21 they are (1,3), (1,4), (2,4).

A lower-bound certificate establishes that no coefficients on the true support can attain an objective as small as the saved competitor, with omega fixed at 1. This conclusion does not depend on ADMM finding the global optimum on the true support.

Let S be the empirical covariance, c the saved competing objective, and Q the squared Frobenius residual. On the true star support, the leading 3x3 block of Lambda is diagonal, so R_ii = S_ii(1-Lambda_ii^2)-1 and R_ij = S_ij(1-Lambda_ii Lambda_jj) for i,j <= 3. If a true-support fit had Q <= c, it would satisfy |Lambda_ii| <= b_i, where b_i^2 = (S_ii-1+sqrt(c))/S_ii. Consequently Q >= B(c), where B(c) = 2 sum_{i<j<=3} S_ij^2 max(0,1-b_i b_j)^2. If B(c)>c this is a contradiction. B(c) is a conditional lower bound under Q<=c, not an unconditional lower bound on every true-support fit.

| Seed | Saved competing objective c | Conditional bound B(c) | True-support ADMM refit |
|---|---:|---:|---:|
| 3 | 2.93022419e-4 | 8.28224693e-4 | 8.62207466e-4 |
| 4 | 2.21948372e-4 | 5.00068679e-4 | 6.35536409e-4 |
| 6 | 3.32576848e-5 | 2.94181947e-4 | 2.99073504e-4 |
| 12 | 1.40941853e-4 | 2.62920146e-4 | 3.05833930e-4 |
| 21 | 1.83461292e-4 | 9.66435433e-4 | 1.48138196e-3 |

Every B(c) exceeds c by much more than the 1e-8 comparison tolerance. More accurate optimization of the existing objective cannot make the true support win these comparisons. Refitted objectives are feasible numerical fits, not claimed global optima. For the five originally correct supports, refitting reproduces the saved dimension-4 objective to numerical precision; this is not a proof of global optimality across all supports.

## Calibration diagnosis

Dimension 4 occupies a nonempty interval of penalty scales in every original curve. Thus no original case is a failure of the penalty shape to make dimension 4 selectable. Exact intervals and recommended scales are in `results.json`.

All four plateau errors are over-selection: seeds 6, 8, 18 select 6 and seed 27 selects 5. In all ten runs, doubling the chosen plateau center stays within that same plateau. Therefore the original plateau failures arise from choosing the wrong plateau by largest absolute log-width, not from the recommendation factor moving the scale out of a correct plateau.

Window returns dimension 2 for seeds 4, 6, 21. It raises “No stable multi-transition cluster satisfied all jump criteria” for seeds 8, 12, 18, 22, 27. These five are calibration-rule nonreturns and must remain in the denominator when evaluating an end-to-end procedure.

Sensitivity check: rerunning original curves at objective floors 0, 1e-10, 1e-8 gives identical dimensions/nonreturns. At 1e-7, plateau seed 6 changes from 6 to 4; at 1e-6 seed 8 also changes from 6 to 4. Seeds 18 and 27 remain wrong, and window outcomes do not change at any of these floors. This identifies sensitivity in the plateau tail; it does not justify tuning the floor to these labels. Seed 6 still has the wrong support after its dimension changes to 4.

## Explicit true-support interventions

Two artificial counterfactual curves were built for every seed, retaining its covariance and original dimension-1-to-3 objectives:

1. True4-only: replace dimension 4 with a refit on the true support, retaining every other original objective.
2. True-nested: additionally rebuild dimensions 5, 6, 7 by enumerating every one-edge upper-triangular extension of the preceding support, refitting, and selecting with the existing 1e-8 comparison tolerance. Every support at dimensions 4 through 7 then contains the true edges.

| Seed | Original plateau/window | True4-only plateau/window | True-nested plateau/window |
|---|---|---|---|
| 3 | 4 / 4 | 2 / 5 | 2 / No result |
| 4 | 4 / 2 | 6 / No result | 4 / No result |
| 6 | 6 / 2 | 6 / 2 | 6 / 2 |
| 8 | 6 / No result | 6 / No result | 6 / No result |
| 12 | 4 / No result | 4 / No result | 4 / No result |
| 13 | 4 / 4 | 4 / 4 | 4 / 4 |
| 18 | 6 / No result | 6 / No result | 6 / 4 |
| 21 | 4 / 2 | 5 / 3 | 5 / 3 |
| 22 | 4 / No result | 4 / No result | 4 / No result |
| 27 | 5 / No result | 5 / No result | 5 / No result |

True-nested leaves dimension 4 selectable in every seed, but plateau selects it in only 4/10 and window in only 2/10 (six window nonreturns). Because the dimension-4 support is forced correct here, these are also the joint-recovery counts for this constructed curve. True4-only yields 3/10 and 1/10 respectively.

These interventions demonstrate persistent calibration difficulty and interaction, not the performance of an improved objective. They force a support that the original objective sometimes disfavors, preserve the original smaller-dimension candidates, and use greedy extensions rather than exhaustive optimization of every supermodel. Their counts must not be interpreted as causal predictions for a future support-reconstruction method. Numerical refitting can also change larger-dimension candidates, as illustrated by window seed 18.

## Priority implied by this audit

For plateau, treat the two mechanisms as comparable. Give empirical ranking a slight priority because four of six apparent dimension successes have incorrect supports, and all five ranking failures are certified. Develop calibration diagnostics alongside it: three of five runs with the right support still fail selection. For window, improve its ability to return a calibrated scale before relying on it as the main selector. Evaluate every change using joint support-and-dimension recovery.

## Reproducibility and verification

Run `.venv/bin/python tmp/support_18_audit/analyze.py` from the project root. The script writes `results.json` and `summary.csv` in this directory. Original experiment files and production code are untouched.

Original selectors were replayed with objective floor 1e-8, recommendation factor 2, automatic window width, and penalty constants inferred from the saved covariance and N=10000 with r=Lm=L=1 and xi=10. All 20 saved selection dimensions/nonreturns were reproduced; all 15 successful recommended scales agree with saved logs at relative tolerance 1e-10.

New refits use fixed omega 1, beta 1, 800 iterations, tolerance 1e-7, 10 Halton starts, coefficient threshold 1e-5, minimum omega 0, and omega upper bound lambda_min(Sigma_hat)-1e-6. These are the previously established seed-6 diagnostic settings; the original NPZ files do not fully record their solver settings. Assertions check all stored support-valid flags, the true support and dimension grid, the ranking certificates against saved support outcomes, and nonincreasing rebuilt curves within tolerance.
