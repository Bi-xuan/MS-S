# Objective-curve support recovery

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

The upper-support study now runs `window`, `plateau`, and `plateau-bootstrap`.
It writes `selection_bootstrap.log` and `.json` per trial, and appends bootstrap
status, dimension, correctness, and precision to its CSVs. Its existing coverage
plots still compare window and ordinary plateau. Configure bootstrap through:

```sh
TOP_PLATEAUS=2 BOOTSTRAP_REPLICATES=399 BOOTSTRAP_ALPHA=0.05 \
  bash experiments/run_upper_support_scaling_study.sh trial 0
```

`BOOTSTRAP_SEED` is also configurable. `N_JOBS` controls refit workers for an
individual trial; `reselect` parallelizes trials with one refit worker each.
After upgrading an existing study, rerun `reselect` across its saved nested
curves to refresh all CSV headers before aggregation. Selection does not
reconstruct older unrestricted curves automatically.
