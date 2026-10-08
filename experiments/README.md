# Objective-curve support recovery

## Fixed-support study with the package API

Run the 40-trial study through `lambda_support_recovery.select_support()`:

```sh
# Use an environment with Python 3.10+ and requirements.txt installed.
TOTAL_CPUS=100 CPUS_PER_TRIAL=5 bash experiments/run_fixed_support_scaling_package.sh
```

The new runner defaults to `N=10`, `support_scope="upper"`, sample sizes
100, 1000, 10000, and 1000000, and the same ten seeds per sample size as
`run_fixed_support_scaling.sh`. It runs up to 20 concurrent trials with five
package workers each, with BLAS/OpenMP threads limited to one per worker.
`TOTAL_CPUS` defaults to `SLURM_CPUS_PER_TASK`, then `NSLOTS`, then 100.
All workers for a trial run on its local machine; this is a single-node CPU
budget. Set `PYTHON=/path/to/python` to select an environment. The installed
package is used; a local checkout can instead be selected with
`PYTHONPATH=/path/to/lambda-support-recovery/src`.

The fixed population matrix uses the original construction: diagonal values
from 0.10 to 0.55 and edges `(i, N-1)` for `i=0,...,N-2`, with magnitudes
bounded by 0.20 and 0.60 and population noise 1. Change these data-generation
settings with `N`, `LAMBDA_STAR_OFFDIAG_ABS_MIN`,
`LAMBDA_STAR_OFFDIAG_ABS_MAX`, and `OMEGA_STAR`. `N` must be at least 3 and
no greater than the smallest sample size (100).

Trial seeds generate observations using `seed + 2`, matching the original
experiment. The package's `random_seed` and `bootstrap_seed` stay at their
defaults. The API receives only `num_samples`, `support_scope="upper"`,
`n_jobs`, `return_result=True`, and a progress callback. Every statistical
option retains the installed package default, including estimated fixed
omega (`fit_omega_ref=True`, `kappa=0.93`), nested supports, ten restarts,
800 iterations, and 199 bootstrap replicates. Original-runner environment
variables such as `MAX_RESTARTS`, `OMEGA_REF`, and `BOOTSTRAP_REPLICATES`
are not estimator overrides in this runner. Installed API defaults, package
version/path, fitting settings, seeds, and timings are recorded for audit.

The default output folder is
`experiments/output/fixed_support_scaling_n10_package_fit_omega_ref_true`.
Each seed has `objective_curve_sigma_hat.npz`,
`selection_plateau_bootstrap.json`, `selection_plateau_bootstrap.log`, and
`result.csv`. Once all trials succeed, the existing analysis scripts produce
`selection_trials.csv`, `selection_summary.md`, `edge_recovery_heatmap.png`,
`roc_curve.png`, and `final_support_operating_points.png`. A failed trial
stops its worker and prevents summary/plot generation; its log retains the
package error, with no change to the estimator settings.

`run-all` recomputes trials in the chosen folder. To regenerate just the
summary and plots from a completed study:

```sh
bash experiments/run_fixed_support_scaling_package.sh summarize path/to/output
```

This runner leaves the original shell script available. Its commands are
`run-all` and `summarize`; it does not implement the original script's plain
Plateau `reselect` command.

`compute_objective_curve.py` uses forward-nested support recovery by default
(`--nested-supports true`). Starting from the diagonal-only model at `D_m=1`,
each subsequent dimension fits every permitted one-edge extension of the
previously selected support and selects the smallest fitted objective. Exact
ties follow candidate enumeration order, including parallel runs. Nesting uses
the selected support mask, even if a permitted coefficient fits to zero.

This works with `--support-scope all`, `--support-scope upper`, and preselected
edges. Existing fixed/free omega fitting and optional post-selection refinement
retain their behavior. If no valid fit is found at a dimension, it and all later
dimensions receive `Inf` objectives and invalid-support flags; no unconstrained
search replaces the missing predecessor.

To derive a fixed reference from the empirical covariance, run with
`--omega-ref none --fit-omega-ref true`. The reference is
`kappa * lambda_min(Sigma_hat)`, with `--kappa 0.93` by default. When both
curves are requested, the same reference is used to fit each curve;
`omega_star` still generates the population covariance and its samples.
`--refine-after-fixed-omega` continues to control the final free-omega refit.
Fixed references may equal the smallest eigenvalue of the fitted covariance.

To restore independent support searches at every dimension:

```sh
python experiments/compute_objective_curve.py --nested-supports false
```

Both `run_experiment.sh` and `run_upper_support_scaling_study.sh` expose the
same option through `NESTED_SUPPORTS`, defaulting to `true`:

```sh
NESTED_SUPPORTS=false bash experiments/run_upper_support_scaling_study.sh trial 0
```

New NPZ files record `nested_supports`. Resuming checks the mode as well as the
existing experiment metadata. Older files without this field are treated as
unrestricted: use `--nested-supports false` to resume them, or choose a new
output path for a nested run. Nested checkpoints must contain a contiguous
dimension prefix with valid nested masks for every finite objective. Original
experiment files are not automatically converted or overwritten when modes
differ.

# Plateau plus bootstrap dimension selection

Run the new method on a saved nested objective curve:

```sh
python experiments/select_scaling_parameter.py path/to/objective_curve_sigma_hat.npz \
  --method plateau-bootstrap --top-plateaus 3 \
  --bootstrap-replicates 199 --bootstrap-alpha 0.05 --n-jobs 4 \
  --output-json selection_bootstrap.json
```

`--top-plateaus` defaults to **3**, `--bootstrap-replicates` to **199**, and
`--bootstrap-alpha` to **0.05**. `--bootstrap-seed` defaults to `20260913`.
The existing `window` and `plateau` methods remain available; the standalone
CLI still defaults to `window`, so specify `--method plateau-bootstrap`.

The procedure ranks bounded plateaus by `log(right) - log(left)`, using the
existing penalty (`Lm=1` by default) and objective floor (`1e-8` by default).
The intervals starting at zero and extending to infinity are excluded. Exact
width ties favor the smaller dimension. If fewer than the requested number
exist, it uses all available candidates; a single candidate requires no test.
No bounded plateau, missing model masks, or nonnested candidates cause an
explicit failure.

Candidates are tested in decreasing dimension. For each adjacent larger/smaller
pair, the null is the smaller fitted model's stationary Gaussian covariance.
Every bootstrap replicate refits **both original support masks**; support
reconstruction and plateau screening are not repeated. The gain uses raw
objectives, with `p = (1 + count(T_boot >= T_observed)) / (B + 1)`.
Only `p < alpha` retains the larger model and stops. Otherwise the procedure
moves to the next pair, or selects the smallest candidate after all tests fail
to reject. There is no multiple-testing correction or recommended-scale
multiplier in this procedure. The smaller fit is feasible on the larger mask,
so negative gains from local optimization are clipped to zero.

Bootstrap sample size is the original positive `num_samples` in the NPZ;
`--num-samples` changes only the penalty. Sampling uses the exact distribution
of `X.T @ X / N` for independent zero-mean Gaussian rows (Wishart for `N >= n`).
Fixed omega fits do not apply the empirical eigenvalue cap within bootstrap
replicates, so legitimate samples with minimum eigenvalue below omega are
included. Free omega final fits retain that cap. Unstable null fits or failed
refits abort selection; replicates are never discarded. Pair-specific RNG
streams make results independent of the refit worker count.

New curve files save final fitting settings, including whether omega is fixed
or free after refinement. The selector refits observed candidates and verifies
agreement with their saved objectives before calibration. Legacy files without
these settings use their numeric `omega_ref`, 800 iterations, tolerance `1e-7`,
and 10 Halton restarts. `--fit-max-restarts` can override the restart count;
other mismatches require regenerating the curve with recorded settings.
Bootstrap refitting currently requires Halton initialization.

The console reports selected dimension, zero-based model edges, and precision
when `Lambda_star` is available. Precision excludes diagonals and uses the
selected model mask, including coefficients fitted to zero. Truth is used
only for reporting. Optional JSON includes plateau intervals, pairwise
p-values, all bootstrap gains, settings, and precision as a fraction.

The upper-support study runs the same sample-size study as
`run_fixed_support_scaling.sh` for each of the 20 strict-upper true supports
with `n=4` and true `D_m=4`. It uses sample sizes 100, 1000, 10000, and 1000000,
with the same ten seeds per sample size (800 trials total). Each
`OUTPUT_ROOT/support_XX/num_samples_N/seed_S/` directory contains the curve,
compute log, `selection_plateau_bootstrap.log` and `.json`, and `result.csv`.
Each support has its own `selection_trials.csv` and `selection_summary.md`
with average precision, exact support recovery, MC recovery, average score,
best-on-path selection, and true-support-on-path metrics. MC recovery means
the selected and true supports have the same full maximal-class family.
Jaccard distances exclude diagonal edges. The score is
`a + (1-a) exp(-d(selected, truth))` for MC recovery, and
`a exp(-min d(selected, member))` otherwise, minimizing over all directed
members of the true MC (including members with lower-triangular edges).
Set `SCORE_A` to configure `a` (default `0.5`, strictly between 0 and 1).
Best-on-path selection continues to mean minimum `FP + FN`, including ties.
The study also writes `OUTPUT_ROOT/selection_summary.md` with only sample
size and average score, pooling individual trials across all supports.
Configure bootstrap through:

```sh
TOP_PLATEAUS=2 BOOTSTRAP_REPLICATES=399 BOOTSTRAP_ALPHA=0.05 \
  bash experiments/run_upper_support_scaling_study.sh trial 0
```

`BOOTSTRAP_SEED` is also configurable. All environment settings from
`run_fixed_support_scaling.sh` are available, including coefficient bounds,
fitting settings, `SUPPORT_SCOPE`, `NESTED_SUPPORTS`, and `OBJECTIVE_FLOOR`.
`TOTAL_CPUS=100 CPUS_PER_TRIAL=5` defaults to 20 concurrent trials with five
CPUs each. Set `NUM_SUPPORTS` to run fewer supports (default 20). The default
output root is `experiments/output/upper_support_scaling_n4_dm4`.

Use `trial TASK_ID` (0 through 799 by default) for Slurm arrays, then
`aggregate` after all trials finish. Task order is support, sample size, then
seed. With no command, a set `SLURM_ARRAY_TASK_ID` selects one trial;
otherwise all trials run.

`reselect [OUTPUT_FOLDER]` discovers saved support/sample-size/seed curves,
runs curve-only `plateau` selections with `--lm-mode support-count` and
`--Lm "${LM_WEIGHT}"` (default 0.1), and uses up to `N_JOBS` concurrent trials
with one CPU each. New selections and summaries go under each support's
`reselect_lm_support_count_plateau/` directory; the pooled score table goes
under `OUTPUT_ROOT/reselect_lm_support_count_plateau/selection_summary.md`.
The earlier
`support_XX/seed_S/` layout and window-versus-plateau coverage plots are
replaced by the per-support sample-size summaries.

# Reselect fixed-support curves without fitting

```sh
N_JOBS=8 bash experiments/run_fixed_support_scaling.sh reselect
```

This reads existing `num_samples_*/seed_*/objective_curve_sigma_hat.npz` files
from `experiments/output/fixed_support_scaling_n4_omega_ref_eq_star` by default.
Pass another folder after `reselect`, or set `OUTPUT_ROOT`, to change the input.
It runs up to `N_JOBS` curve-only `plateau` selections concurrently. All new
files go under `OUTPUT_ROOT/reselect_lm_support_count_plateau/`: each trial has
`selection_plateau.log`, `selection_plateau.json`, and `result.csv`, and the
subfolder root has `selection_trials.csv` and `selection_summary.md`. The
original curves, trial results, summaries, and bootstrap reports remain intact.

For each candidate dimension `D_m`, this reselect uses `Lm` equal to
`LM_WEIGHT` times the number of available supports with `D_m - 1` edges.
`LM_WEIGHT` defaults to `0.1` and can be set in the environment. For an
`n=4` upper-triangular curve, the support counts are `C(6, D_m - 1)`, or
`1, 6, 15, 20, 15, 6, 1` for dimensions 1 through 7. The same rule is
available directly with `select_scaling_parameter.py --lm-mode support-count
--Lm 0.1`.
