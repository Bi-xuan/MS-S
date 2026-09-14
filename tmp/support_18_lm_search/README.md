# Choosing Lm for precision on support 18, N=10000

The power family `Lm(D) = D^p` improves plateau support precision after reselecting the scale. A moderate candidate is p=8; an aggressive candidate exceeding 90% mean precision on these ten seeds is p=32. These are empirically tuned candidates, not theoretically optimal or independently validated choices. Stronger powers harm the current window selector.

The source is `experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18`. All evaluations reuse its saved objective curves and selected support masks. Settings stay fixed: objective floor 1e-8, recommendation factor 2, automatic window width, r=L=1, xi=10, and N and covariance constants from the NPZ. Every form gets newly selected scales for each seed and method.

Support precision is the number of true selected off-diagonal edges divided by the number of selected off-diagonal edges. Recall divides the same numerator by the three true edges. All averages are arithmetic means across successful runs; undefined precision and selection failures are excluded and their denominators are shown. The model dimension is edge count plus one.

## Candidate comparison

| Lm | Plateau precision | Plateau recall | Plateau mean D | Plateau returned | Window precision | Window returned |
|---|---:|---:|---:|---:|---:|---:|
| D | 72.17% | 86.67% | 4.70 | 10/10 | 94.44% | 6/10 |
| D^2 | 75.50% | 83.33% | 4.50 | 10/10 | 95.00% | 5/10 |
| D^8 | 82.83% | 73.33% | 3.90 | 10/10 | 68.75% | 4/10 |
| D^32 | 93.33% | 40.00% | 2.40 | 10/10 | 62.22% | 3/10 |
| D^48 | 100.00% | 33.33% | 2.00 | 10/10 | 50.00% | 1/10 |

p=48 selects one true edge for every seed. It maximizes observed precision by discarding two-thirds of the true edges. p=32 selects one edge in eight seeds and three edges in two seeds. p=8 retains substantially more true edges, while reducing three seeds' selected dimensions compared with Lm=D.

## Per-seed results for p=32

| Seed | Plateau D | Plateau precision | Window D | Window precision |
|---|---:|---:|---:|---:|
| 3 | 2 | 100% | Failed | N/A |
| 4 | 4 | 66.67% | 4 | 66.67% |
| 6 | 4 | 66.67% | Failed | N/A |
| 8 | 2 | 100% | Failed | N/A |
| 12 | 2 | 100% | 6 | 60% |
| 13 | 2 | 100% | Failed | N/A |
| 18 | 2 | 100% | Failed | N/A |
| 21 | 2 | 100% | Failed | N/A |
| 22 | 2 | 100% | Failed | N/A |
| 27 | 2 | 100% | 6 | 60% |

For p=8 the plateau dimensions, in the same seed order, are [2,4,6,6,4,4,2,2,4,5], and precisions are [1,2/3,3/5,3/5,2/3,1,1,1,1,3/4]. Its four successful window runs are seeds 3,12,13,27, all selecting D=5, with precisions 3/4,1/2,3/4,3/4.

## Why Lm=D did little, and why growth alone is insufficient

Here K/sqrt(2v) is approximately 5 in every seed, so Lm=D produces a penalty proportional to `sqrt(D)*(5+sqrt(D))`. The K term still makes a substantial contribution. With Lm=D^p, the exact penalty is `K*sqrt(D)+sqrt(2v)*D^((p+1)/2)`; for p=32 the second term grows as D^16.5.

Scale reselection is essential. Increasing all penalties by one common multiplier can be absorbed by the selected scale. What changes the choice is their shape across dimensions. Plateau selects the largest log-width among bounded dimension intervals. For consecutive active dimensions, a plateau width depends on the ratio of adjacent penalty differences, as well as on the objective improvements. Faster pointwise growth of Lm does not therefore guarantee selection of a smaller model after recalibration.

This is visible in the sweep: even Lm=exp(4*(D-1)) leaves plateau outcomes unchanged, while Lm=exp(D^2-1) worsens its mean precision to 64.83%. Powers work better here. None of the tested forms improved plateau precision while retaining at least the six window successes of Lm=D; a common strong power cannot be recommended as an improvement to both selectors on this evidence.

## Sensitivity and transfer check

Plateau precision is 78.83%, 82.83%, 82.83% at p=7,8,9, with recall 80%,73.33%,66.67%, respectively. At p=30,32,34 precision is 89.33%,93.33%,96.67%, with recall 46.67%,40%,36.67%. The tradeoff changes in discrete jumps; there is no uniquely optimal exponent without a recall or model-size preference.

On the separately saved N=1e6 support_18 curves, plateau at p=8 and p=32 still selects D=4 with 100% support precision and recall in all ten seeds. At p=64 it begins underselecting there too, with mean recall 53.33%. This transfer check involves the same true support and does not establish generalization to other supports.

## Reproduction and implementation

Run these scripts from the repository root using `.venv/bin/python`:

- `tmp/support_18_lm_search/analyze.py`: powers, exponential and squared-dimension exponential forms.
- `tmp/support_18_lm_search/refine.py`: coefficient-scaled powers and `1+c*(D-1)^p` forms.
- `tmp/support_18_lm_search/check_candidates.py`: exponent sensitivity and N=1e6 transfer check.
- `tmp/support_18_lm_search/verify_finalists.py`: replay p=1,2,8,32,48 through the actual experiment script's `run` function and save full logs.

For a selected power, the per-dimension evaluation is `pen_n(float(d_m), constants, Lm=float(d_m)**p)`. Supplying a scalar CLI `--Lm 32` is not equivalent to Lm=D^32. No production code or original experiment outputs have been modified.

`finalists.csv` holds all selected dimensions, scales, TP counts, edge counts, precision, recall and errors for the five reported forms. `finalist_logs/` contains 100 full replay logs. `sweep.csv`, `refined.csv`, and `power_sensitivity.csv` contain the broader search. Each evaluated penalty is checked against `pen_n`, and each successful selected dimension is checked by direct minimization of the penalized objective. The finalist script confirms agreement with the experiment entry point and reproduces the previous Lm=D baseline.
