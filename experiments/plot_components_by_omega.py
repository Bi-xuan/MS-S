#!/usr/bin/env python3
"""Plot all components of X_t in a separate panel for each noise variance.

Run from the project root, for example::

    python experiments/plot_components_by_omega.py --omegas 0.01 0.1 0.5 1 --steps 100

Errors are sampled independently across times, components, and omega values.
Each run uses fresh randomness; pass --seed 42 for reproducible draws.
Edit LAMBDA below to use a different fixed transition matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Fixed across every omega, matching plot_state_paths_vs_omega.py.
LAMBDA = np.array(
    [
        [0.10, 0.60, 0.00, 0.00],
        [0.00, 0.25, 0.20, 0.00],
        [0.00, 0.00, 0.40, -0.45],
        [0.00, 0.00, 0.00, 0.55],
    ]
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "components_by_omega.png"


def simulate_path(
    transition: np.ndarray,
    omega: float,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return X_0, ..., X_steps for X_t = transition.T @ X_(t-1) + error_t."""
    transition = np.asarray(transition, dtype=float)
    if (
        transition.ndim != 2
        or transition.shape[0] == 0
        or transition.shape[0] != transition.shape[1]
        or not np.isfinite(transition).all()
    ):
        raise ValueError("transition must be a finite, nonempty square matrix")
    if not np.isfinite(omega) or omega < 0:
        raise ValueError("omega must be finite and nonnegative")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    # omega is the variance, so the normal standard deviation is sqrt(omega).
    errors = rng.normal(0.0, np.sqrt(omega), size=(steps + 1, transition.shape[0]))
    states = np.empty_like(errors)
    states[0] = errors[0]
    for t in range(1, steps + 1):
        states[t] = transition.T @ states[t - 1] + errors[t]
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--omegas", type=float, nargs="+", default=[0.01, 0.1, 0.5, 1.0],
        help="Noise variances to compare (default: 0.01 0.1 0.5 1.0).",
    )
    parser.add_argument("--steps", type=int, default=100, help="Last time index (default: 100).")
    parser.add_argument("--seed", type=int, default=None, help="Optional nonnegative random seed.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path.")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if any(not np.isfinite(omega) or omega < 0 for omega in args.omegas):
        parser.error("--omegas must contain only finite, nonnegative values")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")

    rng = np.random.default_rng(args.seed)
    times = np.arange(args.steps + 1)
    figure, axes = plt.subplots(
        len(args.omegas), 1, figsize=(11, 2.6 * len(args.omegas) + 1),
        sharex=True, sharey=True, squeeze=False,
    )
    for axis, omega in zip(axes[:, 0], args.omegas):
        # A new draw for every omega; no shared or rescaled noise paths.
        states = simulate_path(LAMBDA, omega, args.steps, rng)
        for component in range(LAMBDA.shape[0]):
            axis.plot(
                times, states[:, component], linewidth=1,
                label=rf"$X_{{t,{component + 1}}}$",
            )
        axis.set_title(rf"$\omega = {omega:g}$", loc="left")
        axis.set_ylabel("State value")
        axis.grid(alpha=0.3)
        axis.legend(ncol=min(LAMBDA.shape[0], 4), fontsize="small", loc="upper right")
    axes[-1, 0].set_xlabel("Time $t$")
    figure.suptitle(
        r"$X_0=\epsilon_0,\quad X_t=\Lambda^T X_{t-1}+\epsilon_t$"
        "\nFixed transition matrix; independent errors for each noise variance"
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
