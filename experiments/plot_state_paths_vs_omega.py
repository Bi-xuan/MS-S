#!/usr/bin/env python3
"""Simulate and plot paths of X_t for several innovation variances."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ms_s_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIMENSION = 4
# The same fixed Lambda_star used by the project's four-dimensional experiments.
LAMBDA = np.array(
    [
        [0.10, 0.00, 0.00, 0.60],
        [0.00, 0.25, 0.00, 0.20],
        [0.00, 0.00, 0.40, -0.45],
        [0.00, 0.00, 0.00, 0.55],
    ]
)
DEFAULT_OMEGAS = (0.01, 0.1, 0.5, 1.0)
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "output" / "state_paths_vs_omega.png"


def simulate_path(
    Lambda: np.ndarray, omega: float, num_steps: int, rng: np.random.Generator
) -> np.ndarray:
    """Return X_0, ..., X_num_steps with X_0 = epsilon_0."""
    if omega < 0 or not np.isfinite(omega):
        raise ValueError("omega must be finite and nonnegative")
    if num_steps < 0:
        raise ValueError("num_steps must be nonnegative")

    dimension = Lambda.shape[0]
    errors = rng.normal(0.0, np.sqrt(omega), size=(num_steps + 1, dimension))
    states = np.empty_like(errors)
    states[0] = errors[0]
    for t in range(1, num_steps + 1):
        states[t] = Lambda.T @ states[t - 1] + errors[t]
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omegas", type=float, nargs="+", default=DEFAULT_OMEGAS)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if any(not np.isfinite(omega) or omega < 0 for omega in args.omegas):
        parser.error("--omegas must contain only finite, nonnegative values")

    rng = np.random.default_rng(args.seed)
    paths = [
        (omega, simulate_path(LAMBDA, omega, args.steps, rng))
        for omega in args.omegas
    ]

    figure, axes = plt.subplots(DIMENSION, 1, figsize=(11, 9), sharex=True)
    times = np.arange(args.steps + 1)
    for coordinate, axis in enumerate(axes):
        for omega, states in paths:
            axis.plot(times, states[:, coordinate], label=rf"$\omega={omega:g}$")
        axis.set_ylabel(rf"$X_{{t,{coordinate + 1}}}$")
        axis.grid(alpha=0.3)
    axes[0].legend(ncol=min(len(paths), 4), fontsize="small")
    axes[-1].set_xlabel("Time $t$")
    figure.suptitle(r"$X_t=\Lambda^T X_{t-1}+\epsilon_t$ with fixed $\Lambda$")
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
