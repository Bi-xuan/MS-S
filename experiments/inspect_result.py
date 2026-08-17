"""Inspect an objective-curve result and its exact dimension path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.select_scaling_parameter import (
    DEFAULT_OBJECTIVE_FLOOR,
    load_selection_inputs,
)
from scaling_selection import build_dimension_path


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "experiments/output/objective_curve_sigma_hat_from_given_sigma.npz"
)


def report_candidate_results(input_path):
    """Print candidate objectives, recovered supports, and covariance data."""

    with np.load(input_path, allow_pickle=True) as data:
        d_m_values = data["d_m_values"]
        objective_values = data["objective_values"]
        masks = data["selected_support_masks"]
        valid = data["selected_support_valid"]

        for d_m, objective, mask, is_valid in zip(
            d_m_values,
            objective_values,
            masks,
            valid,
        ):
            print(
                f"\nD_m = {d_m}, optimal value = {objective}, "
                f"valid = {is_valid}"
            )
            print(mask.astype(int))

            # List only recovered off-diagonal directed edges.
            edges = [
                (int(i), int(j))
                for i, j in np.argwhere(mask)
                if i != j
            ]
            print("Recovered edges:", edges)

        print("\nCurve type:", data["curve_type"].item())
        if "random_seed" in data:
            print("Random seed:", data["random_seed"].item())
        sigma_hat = data["Sigma"].copy()

    print("Shape:", sigma_hat.shape)
    print("Sigma_hat:")
    print(sigma_hat)


def report_dimension_path(input_path, args):
    """Print the exact scale intervals and their selected dimensions."""

    selection_data = load_selection_inputs(input_path, args)
    path = build_dimension_path(
        selection_data["d_m_values"],
        selection_data["objective_values"],
        selection_data["penalty_values"],
    )

    print("\nExact selected-dimension path:")
    right_endpoints = path.breakpoints[1:] + (float("inf"),)
    for left, right, dimension in zip(
        path.breakpoints,
        right_endpoints,
        path.dimensions,
    ):
        right_text = "infinity" if np.isinf(right) else f"{right:.12g}"
        print(f"[{left:.12g}, {right_text}): dimension {dimension}")


def run(args):
    """Inspect one objective-curve result file."""

    input_path = Path(args.input)
    report_candidate_results(input_path)
    report_dimension_path(input_path, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect candidate solutions and print the exact scale intervals "
            "of the selected-dimension path."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=f"Objective-curve NPZ. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--objective-floor",
        type=float,
        default=DEFAULT_OBJECTIVE_FLOOR,
        help=(
            "Tie finite objective values at or below this value at zero "
            f"before constructing the path. Default: {DEFAULT_OBJECTIVE_FLOOR:g}."
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
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--Lm", "--lm", dest="Lm", type=float, default=1.0)
    parser.add_argument("--L", "--l", dest="L", type=float, default=1.0)
    parser.add_argument("--xi", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
