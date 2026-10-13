#!/usr/bin/env python3
"""Fit independent supports at the known dimension in a saved scaling study."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.compute_objective_curve import atomic_savez_compressed
from experiments.summarize_fixed_support_scaling import (
    collect_trials, write_markdown_summary, write_overall_score_summary, write_trials_csv,
)
from optimizers.support_search import optimize_lambda
from supports.common import validate_support_mask


def load_oracle_inputs(curve_path: Path) -> tuple[dict, dict]:
    """Resolve the original fitting settings without using its recovered path."""
    with np.load(curve_path, allow_pickle=False) as data:
        saved = dict(data)
    required = {"Sigma", "n", "lambda_star_dimension", "support_scope",
                "omega_ref", "solve_seed", "fit_settings_json"}
    missing = required - saved.keys()
    if missing:
        raise ValueError(f"{curve_path} is missing oracle inputs: {', '.join(sorted(missing))}")
    n = int(saved["n"])
    dimension = int(saved["lambda_star_dimension"])
    scope = str(saved["support_scope"])
    edge_count = n * (n - 1) // (2 if scope == "upper" else 1)
    if scope not in ("all", "upper") or not 1 <= dimension <= edge_count + 1:
        raise ValueError(f"{curve_path} has an invalid scope or true dimension.")
    sigma = np.asarray(saved["Sigma"], dtype=float)
    if sigma.shape != (n, n) or not np.all(np.isfinite(sigma)):
        raise ValueError(f"{curve_path} has an invalid covariance.")
    settings = json.loads(str(saved["fit_settings_json"].item()))
    required_settings = {"omega_fixed", "beta", "max_iter", "tol", "zero_tol",
                         "max_restarts", "init_strategy"}
    if not required_settings <= settings.keys():
        raise ValueError(f"{curve_path} has incomplete fitting settings.")
    omega_ref = float(saved["omega_ref"])
    # A free final fit with a numeric reference denotes two-stage refinement.
    options = {key: settings[key] for key in required_settings - {"omega_fixed"}}
    options.update(
        min_omega=settings.get("min_omega", 0.0),
        omega_ref=omega_ref,
        refine_after_fixed_omega=omega_ref is not None and settings["omega_fixed"] is None,
        random_seed=int(saved["solve_seed"]), support_scope=scope,
        return_metadata=True,
    )
    return saved, options


def fit_oracle_trial(curve_path: Path, n_jobs: int) -> Path:
    saved, options = load_oracle_inputs(curve_path)
    dimension = int(saved["lambda_star_dimension"])
    fitted_lambda, omega, objective, metadata = optimize_lambda(
        saved["Sigma"], dimension, n_jobs=n_jobs, **options,
    )
    if (fitted_lambda is None or omega is None or not np.isfinite(objective)
            or not np.all(np.isfinite(fitted_lambda)) or not np.isfinite(omega)):
        raise ValueError(f"{curve_path}: no finite oracle fit at dimension {dimension}.")
    n = int(saved["n"])
    allowed_edges = [(i, j) for i in range(n) for j in range(n)
                     if i != j and (options["support_scope"] == "all" or i < j)]
    mask = validate_support_mask(metadata["selected_support_mask"], n, dimension - 1, allowed_edges)
    saved.update(
        oracle_dimension=dimension, oracle_support_mask=mask,
        oracle_support_edges=np.array(metadata["selected_support_edges"], dtype=int).reshape(-1, 2),
        oracle_fitted_lambda=fitted_lambda, oracle_fitted_omega=omega,
        oracle_objective_value=objective,
        oracle_fit_options_json=json.dumps(options, sort_keys=True),
    )
    atomic_savez_compressed(curve_path, **saved)
    return curve_path


def study_layout(root: Path):
    sample_sizes = sorted(int(path.name.removeprefix("num_samples_"))
                          for path in root.glob("num_samples_*") if path.is_dir())
    if not sample_sizes:
        raise ValueError(f"No saved sample-size directories under {root}.")
    counts = [len(list((root / f"num_samples_{size}").glob("seed_*/objective_curve_sigma_hat.npz")))
              for size in sample_sizes]
    if not counts[0] or len(set(counts)) != 1:
        raise ValueError(f"{root} must have the same positive trial count at each sample size.")
    return tuple(sample_sizes), counts[0]


def summary_targets(root: Path):
    """Refresh the original summary and any existing reselect summaries."""
    yield root, "selection_plateau_bootstrap.json"
    for summary in sorted(root.glob("*/selection_summary.md")):
        selection_root = summary.parent
        if list(selection_root.glob("num_samples_*/seed_*/selection_plateau.json")):
            yield selection_root, "selection_plateau.json"


def run_oracle(root: Path, *, upper_study=False, total_cpus=100, cpus_per_trial=5):
    root = Path(root)
    if total_cpus < cpus_per_trial or cpus_per_trial < 1:
        raise ValueError("Require TOTAL_CPUS >= CPUS_PER_TRIAL >= 1.")
    support_roots = sorted(path for path in root.glob("support_*") if path.is_dir()) if upper_study else [root]
    if not support_roots:
        raise ValueError(f"No saved support directories under {root}.")
    layouts = {support: study_layout(support) for support in support_roots}
    curves = [curve for support in support_roots
              for curve in sorted(support.glob("num_samples_*/seed_*/objective_curve_sigma_hat.npz"))]
    # Check all input files and reports before replacing the first NPZ.
    for curve in curves:
        load_oracle_inputs(curve)
    for support, (sizes, seeds) in layouts.items():
        for selection_root, selection_file in summary_targets(support):
            collect_trials(support, sizes, seeds, selection_file, selection_root)
    workers = min(total_cpus // cpus_per_trial, len(curves))
    print(f"Fitting {len(curves)} oracle trials: {workers} concurrent trials x {cpus_per_trial} CPUs.", flush=True)
    if workers == 1:
        for curve in curves:
            print(f"Saved oracle: {fit_oracle_trial(curve, cpus_per_trial)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fit_oracle_trial, curve, cpus_per_trial) for curve in curves]
            for future in futures:
                print(f"Saved oracle: {future.result()}", flush=True)
    refresh_study_summaries(root, layouts, upper_study=upper_study)


def refresh_study_summaries(root, layouts, *, upper_study=False):
    """Refresh original/reselect tables, preserving all saved comparisons."""
    pooled = {}
    for support, (sizes, seeds) in layouts.items():
        for selection_root, selection_file in summary_targets(support):
            trials = collect_trials(support, sizes, seeds, selection_file, selection_root)
            csv_path = selection_root / "selection_trials.csv"
            summary_path = selection_root / "selection_summary.md"
            write_trials_csv(trials, csv_path)
            write_markdown_summary(trials, sizes, seeds, summary_path)
            pooled.setdefault(selection_root.relative_to(support), []).append(csv_path)
            print(f"Updated summary: {summary_path}", flush=True)
    if upper_study:
        for relative, csv_paths in pooled.items():
            write_overall_score_summary(csv_paths, root / relative / "selection_summary.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--upper-study", action="store_true")
    parser.add_argument("--total-cpus", type=int, default=100)
    parser.add_argument("--cpus-per-trial", type=int, default=5)
    args = parser.parse_args()
    try:
        run_oracle(args.input_dir, upper_study=args.upper_study,
                   total_cpus=args.total_cpus, cpus_per_trial=args.cpus_per_trial)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
