# Forced true support and nested additions: support 18, seed 6

Neither selection method returns the true dimension 4. Plateau returns 6 and window returns 2, the same dimensions as on the original curve.

## Procedure

- Keep the original covariance estimate and the saved objectives/supports at dimensions 1, 2, and 3.
- At dimension 4, force the support of Lambda_star and refit its coefficients against Sigma_hat. Do not substitute the objective obtained by simply evaluating the unchanged Lambda_star.
- At each dimension 5, 6, and 7, enumerate every one-edge upper-triangular extension of the previously selected support, refit all allowed coefficients, and select the extension using the existing objective comparison with tolerance 1e-8.
- Use fixed omega 1, ADMM beta 1, 800 iterations, convergence tolerance 1e-7, 10 Halton restarts, and coefficient threshold 1e-5. These are the settings that reproduced the saved dimension-4 result in the preceding diagnostic. Solver settings are not fully recorded in the source NPZ.
- Use the existing plateau and window implementations, objective floor 1e-8, recommendation factor 2, automatic window width, and default theorem penalty constants with the saved sample count 10000.

## Objective curve

| Dimension | Original objective | Forced nested objective | Change to support |
| --- | ---: | ---: | --- |
| 1 | 7.98118620e-2 | 7.98118620e-2 | Preserve original |
| 2 | 1.37116945e-2 | 1.37116945e-2 | Preserve original |
| 3 | 5.37957237e-3 | 5.37957237e-3 | Preserve original |
| 4 | 3.32576848e-5 | 2.99073504e-4 | Force true edges (1,4), (2,4), (3,4) |
| 5 | 5.85855132e-6 | 2.32909450e-5 | Add (1,2) |
| 6 | 2.03116654e-8 | 2.03116654e-8 | Add (1,3) |
| 7 | 8.55031151e-9 | 8.55031151e-9 | Add (2,3) |

Edge labels are one-based. At dimension 5 the alternative additions (1,3) and (2,3) scored 2.93564944e-4 and 2.89258480e-4. At dimension 6 the alternative addition (2,3) scored 9.90176652e-6. Thus the chosen additions are separated by much more than the objective comparison tolerance.

## Selection results

| Method | Original selected dimension | Forced nested selected dimension | Forced nested recommended scale |
| --- | ---: | ---: | ---: |
| Plateau | 6 | 6 | 1.57598602788e-12 |
| Window | 2 | 2 | 1.04595946222e-8 |

The plateau for dimension 6 is [2.42768734538e-14, 2.55771399557e-11), with natural-log width 6.95993. Dimension 4 is selected on [2.74038892435e-10, 4.44770927706e-9), with log-width 2.78687. The dimension-6 plateau is wider, so it wins. The original objective floor treats dimension 7 as zero while retaining dimension 6's 2.03117e-8; the result is conditional on this numerical-floor setting.

Window groups the transitions 4 -> 3 at scale 4.44770927706e-9 and 3 -> 2 at scale 6.14940820353e-9 into a jump of size 2. Its estimated minimal scale is 5.22979731110e-9, and doubling it gives 1.04595946222e-8, where dimension 2 is selected.

Forcing the correct support at dimension 4 therefore does not make either scale-selection rule choose dimension 4. That dimension remains available on the penalized path, but the rules choose scales outside its interval.

## Reproduction and verification

Run from the repository root:

```sh
.venv/bin/python tmp/support_18_seed_6_analysis/nested_experiment.py
```

The script asserts that dimensions 1–3 are unchanged, dimension 4 has the true support, each later support contains the preceding support plus exactly one edge, and the resulting objective curve is nonincreasing. Reapplying both selectors to the original NPZ reproduces the original selected dimensions and recommended scales.

Outputs: `objective_curve_sigma_hat.npz`, `objective_comparison.csv`, `results.json` (including all candidate scores, fitted coefficients, and selection diagnostics), and four original/nested selection logs. Original experiment files and production code are unchanged.
