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
from scaling_selection import select_minimal_scale


METHOD_CHOICES = (
    "maximal-jump",
    "threshold",
    "window",
    "median-jump",
)


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
        objective_values = data["objective_values"].copy()
        constants = build_penalty_constants(data, args)
        curve_type = scalar_value(data, "curve_type", "objective curve")

    penalty_values = compute_penalty_values(d_m_values, constants)
    return {
        "d_m_values": d_m_values,
        "objective_values": objective_values,
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
    print(f"Method: {result.method}")
    if result.threshold is not None:
        print(f"Threshold: {result.threshold:.12g}")
    if result.eta is not None:
        print(f"Eta: {result.eta:.12g}")

    if len(result.component_scales) > 1:
        print("Component minimal-scale estimates:")
        for method in ("maximal_jump", "threshold", "window"):
            if method in result.component_scales:
                value = result.component_scales[method]
                print(f"  {method}: {value:.12g}")

    print(f"Minimal scale: {result.minimal_scale:.12g}")
    print(f"Recommended scale: {result.recommended_scale:.12g}")
    print(
        "Selected dimension at recommended scale: "
        f"{result.selected_dimension}"
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
            "Window width for window or median-jump. By default, use "
            "sqrt(log(num_samples) / num_samples)."
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
