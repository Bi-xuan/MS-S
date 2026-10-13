#!/usr/bin/env python3
"""Add exhaustive upper-support Gaussian BIC fits to saved scaling trials."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.compute_objective_curve import atomic_savez_compressed
from experiments.run_oracle_support_scaling import study_layout, summary_targets, refresh_study_summaries
from experiments.summarize_fixed_support_scaling import collect_trials
from optimizers.bic import select_bic, validate_covariance, validate_settings


def load_bic_inputs(curve_path):
    with np.load(curve_path, allow_pickle=False) as data:
        saved = dict(data)
    required = {"Sigma", "n", "num_samples", "support_scope"}
    missing = required - saved.keys()
    if missing:
        raise ValueError(f"{curve_path} is missing BIC inputs: {', '.join(sorted(missing))}")
    sigma = validate_covariance(saved["Sigma"])
    if float(saved["n"]) != len(sigma) or str(saved["support_scope"]) != "upper":
        raise ValueError(f"{curve_path}: BIC requires matching N and saved support_scope=upper.")
    count = float(saved["num_samples"])
    if not np.isfinite(count) or int(count) != count or count < len(sigma):
        raise ValueError(f"{curve_path}: BIC requires integer num_samples >= N.")
    return saved


def fit_bic_trial(curve_path, options, *, support_workers=1):
    saved = load_bic_inputs(curve_path)
    result = select_bic(saved["Sigma"], int(saved["num_samples"]),
                        n_jobs=support_workers, **options)
    saved.update({f"bic_{name}": value for name, value in result.items()})
    saved["bic_fit_options_json"] = json.dumps(dict(
        options, support_workers=support_workers,
        covariance="X.T @ X / num_samples", mean="known zero",
        omega="joint MLE via analytic profiling", support_scope="upper",
        search="all upper supports including diagonal-only",
        penalty="candidate off-diagonal edge count * log(num_samples)",
        diagonals="free signed, unpenalized, may equal zero",
        tie_rule="fewer edges, then lexicographic edge order", threshold_support=False,
        gradient_coordinates="atanh(diagonals), Lambda_ij / (1 - Lambda_ii * Lambda_jj)",
    ), sort_keys=True)
    atomic_savez_compressed(curve_path, **saved)
    return curve_path


def run_bic(root, *, upper_study=False, total_cpus=100, cpus_per_trial=5,
            support_workers=None, **options):
    root = Path(root)
    if total_cpus < cpus_per_trial or cpus_per_trial < 1:
        raise ValueError("Require TOTAL_CPUS >= CPUS_PER_TRIAL >= 1.")
    support_workers = cpus_per_trial if support_workers is None else support_workers
    if (isinstance(support_workers, bool)
            or not isinstance(support_workers, (int, np.integer))
            or not 1 <= support_workers <= cpus_per_trial):
        raise ValueError("Require integer 1 <= SUPPORT_WORKERS <= CPUS_PER_TRIAL.")
    support_workers = int(support_workers)
    options = dict(dict(diagonal_bound=.999999, max_restarts=10, max_iter=1500, tol=1e-6), **options)
    validate_settings(**options)
    supports = sorted(p for p in root.glob("support_*") if p.is_dir()) if upper_study else [root]
    if not supports:
        raise ValueError(f"No saved support directories under {root}.")
    layouts = {support: study_layout(support) for support in supports}
    curves = [curve for support in supports
              for curve in sorted(support.glob("num_samples_*/seed_*/objective_curve_sigma_hat.npz"))]
    for curve in curves:
        load_bic_inputs(curve)
    for support, (sizes, seeds) in layouts.items():
        for selection_root, selection_file in summary_targets(support):
            collect_trials(support, sizes, seeds, selection_file, selection_root)
    workers = min(total_cpus // cpus_per_trial, len(curves))
    print(f"Fitting {len(curves)} exhaustive BIC trials with {workers} concurrent trials "
          f"and up to {support_workers} single-CPU support workers per trial.", flush=True)
    if workers == 1:
        for curve in curves:
            print(f"Saved BIC: {fit_bic_trial(curve, options, support_workers=support_workers)}",
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fit_bic_trial, curve, options,
                                       support_workers=support_workers) for curve in curves]
            for future in as_completed(futures):
                print(f"Saved BIC: {future.result()}", flush=True)
    refresh_study_summaries(root, layouts, upper_study=upper_study)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--upper-study", action="store_true")
    parser.add_argument("--total-cpus", type=int, default=100)
    parser.add_argument("--cpus-per-trial", type=int, default=5,
                        help="CPU budget per trial; limits concurrent trials (default: 5)")
    parser.add_argument("--support-workers", type=int,
                        help="parallel candidate-support fits per trial "
                             "(default: --cpus-per-trial; must not exceed it)")
    parser.add_argument("--diagonal-bound", type=float, default=.999999)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = vars(parser.parse_args())
    try:
        run_bic(args.pop("input_dir"), **args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
