#!/usr/bin/env python3
"""Run one select_support() trial and export the existing study file formats."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time

if sys.version_info < (3, 10):
    raise SystemExit("This experiment requires Python 3.10+; set PYTHON to a suitable executable.")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import lambda_support_recovery
except ImportError as error:
    raise SystemExit(
        "lambda_support_recovery is required. Use Python 3.10+ with this "
        "project's requirements.txt installed (python -m pip install -r requirements.txt)."
    ) from error

# Reuse only data generation and atomic NPZ writing from the original experiment.
# All support recovery and dimension selection take place in the package API.
from experiments.compute_objective_curve import (
    atomic_savez_compressed,
    covariance_from_lambda_star,
    lambda_star_for_dimension,
    sample_empirical_covariance,
)


def write_json(path, value):
    """Replace a report only after its complete JSON has been written."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json.tmp", delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_trial(*, n, num_samples, seed, n_jobs, output_dir,
              offdiag_abs_min=0.20, offdiag_abs_max=0.60, omega_star=1.0):
    """Use default statistical parameters; the trial seed changes only the data."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lambda_star = lambda_star_for_dimension(
        n, offdiag_abs_min=offdiag_abs_min, offdiag_abs_max=offdiag_abs_max,
    )
    population = covariance_from_lambda_star(lambda_star, omega_star)
    # Preserve the original experiment's observation seed convention.
    sample_seed = seed + 2
    sigma_hat = sample_empirical_covariance(population, num_samples, seed=sample_seed)
    true_edges = {tuple(map(int, edge)) for edge in np.argwhere(np.triu(lambda_star != 0, k=1))}

    print(f"Package: {lambda_support_recovery.__file__}", flush=True)
    print(f"n={n}; num_samples={num_samples}; trial seed={seed}; observation seed={sample_seed}", flush=True)
    print(f"Lambda_star:\n{lambda_star}", flush=True)
    started = time.perf_counter()
    result = lambda_support_recovery.select_support(
        sigma_hat, num_samples=num_samples, support_scope="upper", n_jobs=n_jobs,
        return_result=True, progress=lambda message: print(message, flush=True),
    )
    elapsed = time.perf_counter() - started
    curve = result.curve
    selected_edges = set(result.selected_edges)
    precision = len(selected_edges & true_edges) / len(selected_edges) if selected_edges else 0.0
    exact = selected_edges == true_edges
    curve_path = output_dir / "objective_curve_sigma_hat.npz"
    settings = asdict(curve.fit_settings)
    package_version = getattr(lambda_support_recovery, "__version__", "unknown")
    # Record the installed API's defaults so the statistical settings are auditable.
    defaults = {
        key: parameter.default
        for key, parameter in inspect.signature(lambda_support_recovery.select_support).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    atomic_savez_compressed(
        curve_path,
        curve_type="sigma_hat", n=n, Lambda_star=lambda_star, Sigma=sigma_hat,
        Sigma_population=population, omega_star=omega_star,
        lambda_star_dimension=1 + len(true_edges), lambda_star_support_index=-1,
        lambda_star_support_edges=np.array(sorted(true_edges), dtype=int).reshape(-1, 2),
        lambda_star_min_edge_magnitude=offdiag_abs_min,
        lambda_star_max_edge_magnitude=offdiag_abs_max,
        num_samples=num_samples, random_seed=seed, sample_seed=sample_seed,
        solve_seed=defaults["random_seed"], omega_ref=curve.resolved_omega_ref,
        fit_omega_ref=defaults["fit_omega_ref"], kappa=defaults["kappa"],
        support_scope=curve.support_scope, nested_supports=curve.nested_supports,
        d_m_values=curve.dimensions, objective_values=curve.raw_objectives,
        selected_support_masks=curve.support_masks, selected_support_valid=curve.support_valid,
        fitted_lambdas=curve.fitted_lambdas, fitted_omegas=curve.fitted_omegas,
        fallback_d_m_values=np.array([], dtype=int), fallback_objective_values=np.array([]),
        fit_settings_json=json.dumps(settings, sort_keys=True),
        package_version=package_version, api_defaults_json=json.dumps(defaults, sort_keys=True),
        elapsed_seconds=elapsed,
    )
    report = asdict(result.diagnostics)
    report.update(
        selected_dimension=int(result.selected_dimension), selected_edges=sorted(selected_edges),
        precision=precision, exact_support_recovery=exact, method=result.method,
        input=str(curve_path), objective_floor=result.objective_floor,
        Lm=float(result.lm_values[0]), lm_mode=result.lm_mode,
        package_version=package_version, package_path=str(lambda_support_recovery.__file__),
        api_defaults=defaults, n_jobs=n_jobs, random_seed=seed, sample_seed=sample_seed,
        elapsed_seconds=elapsed,
    )
    write_json(output_dir / "selection_plateau_bootstrap.json", report)
    (output_dir / "result.csv").write_text(
        f"{num_samples},{seed},{result.selected_dimension},{precision:.12g},{str(exact).lower()}\n",
        encoding="utf-8",
    )
    print(f"Selected dimension: {result.selected_dimension}", flush=True)
    print(f"Selected edges: {sorted(selected_edges)}", flush=True)
    print(f"Selected support precision: {precision:.6f}", flush=True)
    print(f"Elapsed: {elapsed:.3f} seconds", flush=True)
    print(f"Saved: {curve_path}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument("--offdiag-abs-min", type=float, default=0.20)
    parser.add_argument("--offdiag-abs-max", type=float, default=0.60)
    parser.add_argument("--omega-star", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Validate setup without fitting or writing files.")
    args = parser.parse_args(argv)
    if args.n < 3 or args.num_samples < args.n or args.seed < 0 or args.n_jobs < 1:
        parser.error("Require n >= 3, num_samples >= n, seed >= 0, and n_jobs >= 1.")
    if not np.isfinite(args.omega_star) or args.omega_star <= 0:
        parser.error("omega-star must be finite and positive.")
    try:
        lambda_star_for_dimension(args.n, offdiag_abs_min=args.offdiag_abs_min, offdiag_abs_max=args.offdiag_abs_max)
    except ValueError as error:
        parser.error(str(error))
    if not args.check_only and args.output_dir is None:
        parser.error("--output-dir is required unless --check-only is used.")
    return args


def main():
    args = parse_args()
    if args.check_only:
        import matplotlib  # Validate the plotting dependency before starting 40 trials.
        print(f"Using lambda_support_recovery {lambda_support_recovery.__version__} from {lambda_support_recovery.__file__}")
        return
    options = vars(args).copy()
    options.pop("check_only")
    run_trial(**options)


if __name__ == "__main__":
    main()
