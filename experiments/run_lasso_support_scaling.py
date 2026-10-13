#!/usr/bin/env python3
"""Add five-fold-CV lasso fits to saved scaling trials and refresh summaries."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

# Each worker fits one trial; avoid BLAS oversubscription in batch jobs.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.compute_objective_curve import atomic_savez_compressed, covariance_from_lambda_star
from experiments.run_oracle_support_scaling import study_layout, summary_targets, refresh_study_summaries
from experiments.summarize_fixed_support_scaling import collect_trials
from optimizers.lasso import gamma_grid, select_lasso, validate_settings


def load_lasso_inputs(curve_path):
    with np.load(curve_path, allow_pickle=False) as data:
        saved = dict(data)
    required = {"Sigma", "n", "num_samples", "random_seed", "Lambda_star", "omega_star"}
    missing = required - saved.keys()
    if missing:
        raise ValueError(f"{curve_path} is missing lasso inputs: {', '.join(sorted(missing))}")
    n, count = int(saved["n"]), int(saved["num_samples"])
    sigma = np.asarray(saved["Sigma"], dtype=float)
    if (n < 2 or count - (count + 4) // 5 < n or sigma.shape != (n, n)
            or not np.all(np.isfinite(sigma)) or not np.allclose(sigma, sigma.T)
            or np.linalg.eigvalsh(sigma)[0] <= 0):
        raise ValueError(f"{curve_path} requires a positive-definite covariance and at least n training rows per fold.")
    if (int(saved["random_seed"]) < 0 or np.asarray(saved["Lambda_star"]).shape != (n, n)
            or not np.all(np.isfinite(saved["Lambda_star"]))
            or not np.isfinite(saved["omega_star"]) or float(saved["omega_star"]) <= 0):
        raise ValueError(f"{curve_path} has invalid simulation provenance.")
    return saved


def fit_lasso_trial(curve_path, gammas, options):
    saved = load_lasso_inputs(curve_path)
    # Existing studies store covariances, not observation rows. Reconstruct
    # their original simulation; truth is never used for fitting or tuning.
    population = covariance_from_lambda_star(saved["Lambda_star"], float(saved["omega_star"]))
    sample_seed = int(saved.get("sample_seed", int(saved["random_seed"]) + 2))
    samples = np.random.default_rng(sample_seed).multivariate_normal(
        np.zeros(int(saved["n"])), population, size=int(saved["num_samples"]),
    )
    if not np.allclose(samples.T @ samples / len(samples), saved["Sigma"], rtol=2e-12, atol=2e-12):
        raise ValueError(f"{curve_path}: reconstructed observations do not match the saved covariance.")
    result = select_lasso(samples, gammas, **options)
    saved.update({f"lasso_{name}": value for name, value in result.items()})
    saved["lasso_fit_options_json"] = json.dumps(dict(
        options, folds=5, sample_seed=sample_seed, covariance="X.T @ X / N",
        penalty="strict-upper L1", gamma_units="absolute", cv_loss="unsquared Frobenius",
        tie_rule="largest gamma among exact minima", unpenalized_refit=False,
    ), sort_keys=True)
    atomic_savez_compressed(curve_path, **saved)
    return curve_path


def run_lasso(root, *, upper_study=False, total_cpus=100, cpus_per_trial=5,
              gammas=None, **options):
    root = Path(root)
    if total_cpus < cpus_per_trial or cpus_per_trial < 1:
        raise ValueError("Require TOTAL_CPUS >= CPUS_PER_TRIAL >= 1.")
    options = dict(dict(fold_seed=20261013, kappa=.93, zero_tol=1e-5,
                        diagonal_bound=.999999, max_restarts=10, max_iter=1500, tol=1e-6), **options)
    validate_settings(**{name: value for name, value in options.items() if name != "fold_seed"})
    if options["fold_seed"] < 0:
        raise ValueError("Lasso fold_seed must be nonnegative.")
    gammas = gamma_grid() if gammas is None else np.asarray(gammas, dtype=float)
    if (gammas.ndim != 1 or not len(gammas) or gammas[0] != 0
            or not np.all(np.isfinite(gammas)) or np.any(np.diff(gammas) <= 0)):
        raise ValueError("Lasso gamma grid must start at zero and increase strictly.")
    supports = sorted(path for path in root.glob("support_*") if path.is_dir()) if upper_study else [root]
    if not supports:
        raise ValueError(f"No saved support directories under {root}.")
    layouts = {support: study_layout(support) for support in supports}
    curves = [curve for support in supports
              for curve in sorted(support.glob("num_samples_*/seed_*/objective_curve_sigma_hat.npz"))]
    for curve in curves:
        load_lasso_inputs(curve)
    for support, (sizes, seeds) in layouts.items():
        for selection_root, selection_file in summary_targets(support):
            collect_trials(support, sizes, seeds, selection_file, selection_root)
    workers = min(total_cpus // cpus_per_trial, len(curves))
    print(f"Fitting {len(curves)} lasso trials with {workers} single-CPU workers.", flush=True)
    if workers == 1:
        for curve in curves:
            print(f"Saved lasso: {fit_lasso_trial(curve, gammas, options)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fit_lasso_trial, curve, gammas, options) for curve in curves]
            for future in as_completed(futures):
                print(f"Saved lasso: {future.result()}", flush=True)
    refresh_study_summaries(root, layouts, upper_study=upper_study)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--upper-study", action="store_true")
    parser.add_argument("--total-cpus", type=int, default=100)
    parser.add_argument("--cpus-per-trial", type=int, default=5,
                        help="CPU allocation per trial; caps concurrent trials at total_cpus / cpus_per_trial.")
    parser.add_argument("--gamma-min", type=float, default=1e-7)
    parser.add_argument("--gamma-max", type=float, default=1.)
    parser.add_argument("--gamma-count", type=int, default=29, help="Positive grid values; zero is added.")
    parser.add_argument("--gamma-spacing", choices=("log", "linear"), default="log")
    parser.add_argument("--fold-seed", type=int, default=20261013)
    parser.add_argument("--kappa", type=float, default=.93)
    parser.add_argument("--zero-tol", type=float, default=1e-5)
    parser.add_argument("--diagonal-bound", type=float, default=.999999)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = vars(parser.parse_args())
    try:
        gammas = gamma_grid(args.pop("gamma_min"), args.pop("gamma_max"),
                            args.pop("gamma_count"), args.pop("gamma_spacing"))
        run_lasso(args.pop("input_dir"), gammas=gammas, **args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
