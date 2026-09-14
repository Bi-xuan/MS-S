# Support 18, N=1,000,000: Lm = D_m

Reproduction: `.venv/bin/python tmp/support_18_lm_dimension/analyze.py` from the project root.

The runner invokes `experiments/select_scaling_parameter.py`'s `run` function for each of the ten saved objective curves, for both plateau and window. It temporarily replaces the imported penalty evaluator with `pen_n(D_m, constants, Lm=D_m)`. Thus the new penalty is `K*sqrt(D_m) + sqrt(2*v)*D_m`. Production code and original experiment outputs are unchanged.

Settings: objective floor 1e-8, recommendation factor 2, automatic window width, N from the saved NPZ (1,000,000), r=L=1, xi=10, covariance constants from saved Sigma. No objective refitting is needed for this penalty-only change.

All 20 baseline dimensions/failures reproduce the saved results; all successful baseline recommended scales reproduce the saved logs within relative tolerance 1e-10. Assertions also check the new penalty formula and directly minimize the penalized objective at every successful recommended scale.

| Seed | Plateau Lm=1 | Plateau Lm=D_m | Window Lm=1 | Window Lm=D_m |
|---|---:|---:|---:|---:|
| 2 | 4 | 4 | Failed | 2 |
| 3 | 4 | 4 | 2 | 2 |
| 124 | 4 | 4 | 2 | 2 |
| 139 | 4 | 4 | Failed | 4 |
| 147 | 4 | 4 | Failed | 2 |
| 153 | 4 | 4 | 2 | 2 |
| 156 | 4 | 4 | 2 | 2 |
| 173 | 4 | 4 | Failed | Failed |
| 192 | 4 | 4 | 4 | 4 |
| 237 | 4 | 4 | 2 | 2 |

The true dimension is 4. Plateau remains correct in 10/10 runs. Window changes from one correct, five selecting dimension 2, and four failures to two correct, seven selecting dimension 2, and one failure. The remaining failure is “No stable multi-transition cluster satisfied all jump criteria.”

`comparison.csv` and `results.json` contain both minimal and recommended scales, dimensions and failure reasons for all 40 selections. Per-seed logs contain full selector diagnostics.
