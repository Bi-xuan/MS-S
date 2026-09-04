#!/usr/bin/env python3
"""Plot covariance-estimation error as a function of omega."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / "experiments" / "output" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.compute_objective_curve import (
    covariance_from_lambda_star,
    lambda_star_for_dimension,
    sample_empirical_covariance,
)


DIMENSION = 4
DEFAULT_NUM_SAMPLES = 100
DEFAULT_SEEDS = (124, 211, 17)
DEFAULT_OMEGA_START = 0.01
DEFAULT_OMEGA_END = 1.0
DEFAULT_OMEGA_STEP = 0.01
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "output" / "sigma_estimation_error_vs_omega.png"
)


def omega_grid(start: float, end: float, step: float) -> np.ndarray:
    """Return an endpoint-inclusive, evenly spaced omega grid."""

    if start < 0.0:
        raise ValueError("omega_start must be nonnegative.")
    if end <= start:
        raise ValueError("omega_end must be greater than omega_start.")
    if step <= 0.0:
        raise ValueError("omega_step must be positive.")

    num_steps = round((end - start) / step)
    reconstructed_end = start + num_steps * step
    return np.linspace(start, reconstructed_end, num_steps + 1)


def compute_frobenius_errors(
    Lambda_star: np.ndarray,
    omega_values: np.ndarray,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> dict[int, np.ndarray]:
    """Compute ||Sigma_hat - Sigma||_F for each omega and random seed."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive.")

    errors = {
        seed: np.empty(len(omega_values), dtype=float)
        for seed in seeds
    }
    for omega_index, omega in enumerate(omega_values):
        Sigma = covariance_from_lambda_star(Lambda_star, float(omega))
        for seed in seeds:
            # Resetting the seed at every omega supplies matched random draws,
            # isolating the effect of changing omega.
            Sigma_hat = sample_empirical_covariance(
                Sigma,
                num_samples=num_samples,
                seed=seed,
            )
            errors[seed][omega_index] = np.linalg.norm(
                Sigma_hat - Sigma,
                ord="fro",
            )

    return errors


def plot_frobenius_errors(
    omega_values: np.ndarray,
    errors: dict[int, np.ndarray],
    output_path: Path,
    num_samples: int,
) -> None:
    """Plot and save the three seed-specific estimation-error curves."""

    figure, axis = plt.subplots(figsize=(9, 6))
    for seed, seed_errors in errors.items():
        axis.plot(
            omega_values,
            seed_errors,
            linewidth=2.0,
            label=f"seed = {seed}",
        )

    axis.set_xlabel(r"$\omega$")
    axis.set_ylabel(r"$\|\hat{\Sigma} - \Sigma\|_F$")
    axis.set_title(
        "Covariance-estimation error vs. omega\n"
        f"n = {DIMENSION}, number of samples = {num_samples}"
    )
    axis.set_xlim(float(omega_values[0]), float(omega_values[-1]))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of Gaussian observations used for each Sigma_hat.",
    )
    parser.add_argument(
        "--omega-start",
        type=float,
        default=DEFAULT_OMEGA_START,
        help=f"First omega value (default: {DEFAULT_OMEGA_START}).",
    )
    parser.add_argument(
        "--omega-end",
        type=float,
        default=DEFAULT_OMEGA_END,
        help=(
            "Requested last omega value. If necessary, the grid uses the nearest "
            "endpoint reachable by an integer number of steps "
            f"(default: {DEFAULT_OMEGA_END})."
        ),
    )
    parser.add_argument(
        "--omega-step",
        type=float,
        default=DEFAULT_OMEGA_STEP,
        help=f"Spacing between omega values (default: {DEFAULT_OMEGA_STEP}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Lambda_star = lambda_star_for_dimension(DIMENSION)
    omega_values = omega_grid(
        args.omega_start,
        args.omega_end,
        args.omega_step,
    )

    print("Lambda_star used in this experiment:")
    print(Lambda_star)

    errors = compute_frobenius_errors(
        Lambda_star,
        omega_values,
        num_samples=args.num_samples,
    )
    plot_frobenius_errors(
        omega_values,
        errors,
        args.output,
        args.num_samples,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
