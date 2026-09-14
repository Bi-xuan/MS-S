"""Select a dimension or penalty scaling parameter from an objective curve."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analyze_penalty import build_penalty_constants, scalar_value
from penalty import pen_n
from plateau_bootstrap import FitSettings, select_plateau_bootstrap
from scaling_selection import (
    DEFAULT_RECOMMENDATION_FACTOR,
    build_dimension_path,
    select_minimal_scale,
)


METHOD_CHOICES = (
    "window",
    "plateau",
    "plateau-bootstrap",
)

DEFAULT_OBJECTIVE_FLOOR = 1e-8


def floor_objective_values(objective_values, objective_floor):
    """Tie finite raw objectives at or below the numerical floor at zero."""

    try:
        objective_floor = float(objective_floor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "objective_floor must be a finite, nonnegative number."
        ) from exc
    if not np.isfinite(objective_floor) or objective_floor < 0.0:
        raise ValueError(
            "objective_floor must be a finite, nonnegative number."
        )

    floored_values = np.asarray(objective_values, dtype=float).copy()
    floored_mask = np.isfinite(floored_values) & (
        floored_values <= objective_floor
    )
    floored_values[floored_mask] = 0.0
    return floored_values, floored_mask


def compute_penalty_values(d_m_values, constants):
    """Compute the unscaled theorem penalty for every candidate dimension."""

    return np.asarray(
        [pen_n(float(d_m), constants) for d_m in d_m_values],
        dtype=float,
    )


def load_selection_inputs(input_path, args):
    """Load an objective curve and derive its unscaled penalty values."""

    with np.load(input_path) as data:
        required_keys = ("d_m_values", "objective_values", "Sigma")
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(
                f"The input NPZ is missing required fields: {missing_keys}."
            )

        d_m_values = data["d_m_values"].copy()
        raw_objective_values = data["objective_values"].copy()
        constants = build_penalty_constants(data, args)
        curve_type = scalar_value(data, "curve_type", "objective curve")

    objective_floor = getattr(
        args,
        "objective_floor",
        DEFAULT_OBJECTIVE_FLOOR,
    )
    objective_values, floored_mask = floor_objective_values(
        raw_objective_values,
        objective_floor,
    )
    penalty_values = compute_penalty_values(d_m_values, constants)
    return {
        "d_m_values": d_m_values,
        "raw_objective_values": raw_objective_values,
        "objective_values": objective_values,
        "objective_floor": float(objective_floor),
        "num_floored_objectives": int(np.count_nonzero(floored_mask)),
        "penalty_values": penalty_values,
        "constants": constants,
        "curve_type": curve_type,
    }


def _report_plateau_selection(plateau):
    """Print the selected plateau and its comparison diagnostics."""

    print("Plateau comparison: succeeded")
    print(f"Chosen plateau dimension: {plateau.dimension}")
    print(
        "Chosen plateau interval: "
        f"[{plateau.left:.12g}, {plateau.right:.12g})"
    )
    print(f"Chosen plateau log-width: {plateau.log_width:.12g}")
    if plateau.runner_up_score is not None:
        print(
            "Runner-up plateau log-width: "
            f"{plateau.runner_up_score:.12g}"
        )
        print(f"Plateau log-width margin: {plateau.score_margin:.12g}")
    print(f"Chosen plateau center: {plateau.center:.12g}")


def report_selection(input_path, selection_data, result):
    """Print the scale-selection inputs and result."""

    constants = selection_data["constants"]
    print(f"Loaded: {input_path}")
    print(f"Curve type: {selection_data['curve_type']}")
    print(f"Matrix dimension: {constants.n}")
    print(f"Number of samples: {constants.num_samples}")
    print(f"Raw-objective floor: {selection_data['objective_floor']:.12g}")
    print(
        "Raw objectives tied at zero: "
        f"{selection_data['num_floored_objectives']}"
    )
    print(f"Method: {result.method}")
    if result.method == "window":
        jump_selection = result.jump_selection
        if jump_selection.succeeded:
            print("Jump selection: succeeded")
            print(f"Largest jump: {result.largest_jump:.12g}")
            print(f"Chosen window size (eta): {result.eta:.12g}")
        else:
            print("Jump selection: failed")
            print(f"Jump failure reason: {jump_selection.failure_reason}")
            print("Largest jump: unavailable (jump selection failed)")
            print(
                "Chosen window size (eta): unavailable "
                "(jump selection failed)"
            )
            print("Rejected jump candidates:")
            for criterion, count in jump_selection.rejection_counts.items():
                print(f"  {criterion}: {count}")
            _report_plateau_selection(result.plateau_selection)
        print(f"Selection source: {result.selection_source}")
    else:
        _report_plateau_selection(result.plateau_selection)
        print(f"Selection source: {result.selection_source}")

    print(f"Minimal scale: {result.minimal_scale:.12g}")
    print(f"Recommendation factor: {result.recommendation_factor:.12g}")
    print(f"Recommended scale: {result.recommended_scale:.12g}")
    print(
        "Selected dimension at recommended scale: "
        f"{result.selected_dimension}"
    )
    if result.recommendation_within_plateau is not None:
        print(
            "Recommended scale within chosen plateau: "
            f"{result.recommendation_within_plateau}"
        )


def run(args):
    """Load one dataset, select its scale, and report the result."""

    input_path = Path(args.input)
    selection_data = load_selection_inputs(input_path, args)
    if getattr(args, "validate_only", False):
        build_dimension_path(
            selection_data["d_m_values"],
            selection_data["objective_values"],
            selection_data["penalty_values"],
        )
        print(f"Scaling-selection input is valid: {input_path}")
        return None

    if args.method == "plateau-bootstrap":
        result = run_bootstrap(input_path, selection_data, args)
        return result

    result = select_minimal_scale(
        selection_data["d_m_values"],
        selection_data["objective_values"],
        selection_data["penalty_values"],
        method=args.method,
        eta=args.eta,
        recommendation_factor=getattr(
            args,
            "recommendation_factor",
            DEFAULT_RECOMMENDATION_FACTOR,
        ),
    )
    report_selection(input_path, selection_data, result)
    return result


def run_bootstrap(input_path, selection_data, args):
    """Load model masks and reproduce the curve's final fitting configuration."""
    with np.load(input_path) as data:
        for key in ("selected_support_masks", "selected_support_valid", "num_samples"):
            if key not in data:
                raise ValueError(f"Bootstrap selection requires saved {key}.")
        # A penalty sample-count override must not change the bootstrap size.
        num_samples = int(data["num_samples"].item())
        if num_samples < 1:
            raise ValueError("Bootstrap selection requires a sampled curve with original num_samples > 0.")
        if "fit_settings_json" in data:
            settings = FitSettings(**json.loads(str(data["fit_settings_json"].item())))
        else:
            if "omega_ref" not in data:
                raise ValueError("Legacy curves require a numeric omega_ref for bootstrap refitting.")
            settings = FitSettings(omega_fixed=float(data["omega_ref"].item()))
            print("Legacy curve: using 800 iterations, tol=1e-7 and 10 Halton restarts; verifying refit objectives.")
        if args.fit_max_restarts is not None:
            settings = FitSettings(**{**asdict(settings), "max_restarts": args.fit_max_restarts})
        result = select_plateau_bootstrap(
            selection_data["d_m_values"], selection_data["objective_values"],
            selection_data["penalty_values"], selection_data["raw_objective_values"],
            data["Sigma"], data["selected_support_masks"], data["selected_support_valid"],
            num_samples, top_plateaus=args.top_plateaus,
            bootstrap_replicates=args.bootstrap_replicates, alpha=args.bootstrap_alpha,
            seed=args.bootstrap_seed, n_jobs=args.n_jobs, fit_settings=settings,
            true_support=(np.abs(data["Lambda_star"]) > 1e-10) if "Lambda_star" in data else None,
            progress=lambda message: print(message, flush=True),
        )
    print(f"Loaded: {input_path}")
    print(f"Method: {result.method}")
    print(f"Top plateaus requested: {result.top_plateaus}")
    print(f"Candidate dimensions (decreasing): {', '.join(map(str, result.candidate_dimensions))}")
    print(f"Bootstrap replicates: {result.bootstrap_replicates}; alpha: {result.alpha}; seed: {result.seed}")
    print(f"Selected dimension: {result.selected_dimension}")
    print(f"Selected edges (zero-based): {result.selected_edges}")
    if result.precision is not None:
        print(f"Selected support precision: {result.precision:.12g}")
    if args.output_json:
        output = Path(args.output_json)
        if output.resolve() == input_path.resolve():
            raise ValueError("The JSON output must not overwrite the input curve.")
        payload = asdict(result)
        payload.update(input=str(input_path), objective_floor=selection_data["objective_floor"],
                       Lm=selection_data["constants"].Lm)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"Saved bootstrap report: {output}")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Select a model from an objective-curve NPZ using penalty-scale "
            "selection or plateau screening plus bootstrap calibration."
        )
    )
    parser.add_argument(
        "input",
        help="Objective-curve NPZ produced by compute_objective_curve.py.",
    )
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="window",
        help="Dimension-selection procedure. Default: window.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate that the curve can form a scaling-selection dimension "
            "path, then exit without selecting a scale."
        ),
    )
    parser.add_argument("--top-plateaus", type=int, default=3,
                        help="Number of longest bounded plateaus to keep for bootstrap selection. Default: 3.")
    parser.add_argument("--bootstrap-replicates", type=int, default=199,
                        help="Bootstrap draws per candidate pair. Default: 199.")
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05,
                        help="Retain the larger candidate only when p < alpha. Default: 0.05.")
    parser.add_argument("--bootstrap-seed", type=int, default=20260913,
                        help="Reproducible bootstrap RNG seed. Default: 20260913.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel bootstrap refit workers. Default: 1.")
    parser.add_argument("--fit-max-restarts", type=int,
                        help="Override saved restart count (legacy curves default to 10).")
    parser.add_argument("--output-json", help="Write the plateau-bootstrap result and all bootstrap gains to JSON.")
    parser.add_argument(
        "--eta",
        type=float,
        help=(
            "Optional minimum adaptive window width for the window method."
        ),
    )
    parser.add_argument(
        "--recommendation-factor",
        type=float,
        default=DEFAULT_RECOMMENDATION_FACTOR,
        help=(
            "Positive multiplier applied to the estimated minimal scale to "
            "obtain the recommended scale. "
            f"Default: {DEFAULT_RECOMMENDATION_FACTOR:g}."
        ),
    )
    parser.add_argument(
        "--objective-floor",
        type=float,
        default=DEFAULT_OBJECTIVE_FLOOR,
        help=(
            "Finite raw objective values at or below this numerical floor "
            "are tied at zero before constructing the penalized dimension "
            f"path. Default: {DEFAULT_OBJECTIVE_FLOOR:g}. Use 0 to floor "
            "only exact zeros."
        ),
    )
    parser.add_argument(
        "--num-samples",
        "--penalty-n",
        dest="num_samples",
        type=int,
        help=(
            "Sample count used by the penalty. Defaults to the positive "
            "num_samples value stored in the NPZ."
        ),
    )
    parser.add_argument(
        "--r",
        type=float,
        default=1.0,
        help="Theorem radius r. Default: 1.",
    )
    parser.add_argument(
        "--Lm",
        "--lm",
        dest="Lm",
        type=float,
        default=1.0,
        help="Model entropy weight Lm. Default: 1.",
    )
    parser.add_argument(
        "--L",
        "--l",
        dest="L",
        type=float,
        default=1.0,
        help="Theorem constant L. Default: 1.",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=10.0,
        help="Theorem tail constant xi. Default: 10.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
