"""Estimate a penalty scaling parameter from an objective-curve NPZ file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analyze_penalty import build_penalty_constants, scalar_value
from penalty import pen_n
from scaling_selection import (
    DEFAULT_RECOMMENDATION_FACTOR,
    select_minimal_scale,
)


METHOD_CHOICES = (
    "maximal-jump",
    "threshold",
    "window",
    "median-jump",
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
    if result.threshold is not None:
        print(f"Threshold: {result.threshold:.12g}")
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

            plateau = result.plateau_selection
            print("Plateau comparison: succeeded")
            print(f"Chosen plateau dimension: {plateau.dimension}")
            print(
                "Chosen plateau interval: "
                f"[{plateau.left:.12g}, {plateau.right:.12g})"
            )
            print(f"Chosen plateau log-width: {plateau.log_width:.12g}")
            print(
                "Chosen plateau persistence score: "
                f"{plateau.persistence_score:.12g}"
            )
            if plateau.runner_up_score is not None:
                print(
                    "Runner-up plateau persistence score: "
                    f"{plateau.runner_up_score:.12g}"
                )
                print(f"Plateau score margin: {plateau.score_margin:.12g}")
            print(f"Chosen plateau center: {plateau.center:.12g}")
        print(f"Selection source: {result.selection_source}")
    elif result.eta is not None:
        print(f"Eta: {result.eta:.12g}")

    if len(result.component_scales) > 1:
        print("Component minimal-scale estimates:")
        for method in ("maximal_jump", "threshold", "window"):
            if method in result.component_scales:
                value = result.component_scales[method]
                print(f"  {method}: {value:.12g}")

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
    constants = selection_data["constants"]
    result = select_minimal_scale(
        selection_data["d_m_values"],
        selection_data["objective_values"],
        selection_data["penalty_values"],
        method=args.method,
        num_samples=constants.num_samples,
        threshold_value=args.threshold_value,
        eta=args.eta,
        recommendation_factor=getattr(
            args,
            "recommendation_factor",
            DEFAULT_RECOMMENDATION_FACTOR,
        ),
    )
    report_selection(input_path, selection_data, result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the minimal penalty scale from an objective-curve NPZ "
            "and report the slope-heuristic recommended scale."
        )
    )
    parser.add_argument(
        "input",
        help="Objective-curve NPZ produced by compute_objective_curve.py.",
    )
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="median-jump",
        help="Scale-selection procedure. Default: median-jump.",
    )
    parser.add_argument(
        "--threshold",
        dest="threshold_value",
        type=float,
        help=(
            "Dimension threshold for threshold or median-jump. By default, "
            "use half the largest candidate dimension with a finite objective."
        ),
    )
    parser.add_argument(
        "--eta",
        type=float,
        help=(
            "Minimum adaptive window width for window, or fixed width for "
            "median-jump. The median-jump default is "
            "sqrt(log(num_samples) / num_samples)."
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
            "Sample count used by the penalty and default eta. Defaults to "
            "the positive num_samples value stored in the NPZ."
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
