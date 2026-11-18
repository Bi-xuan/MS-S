"""Report numerical convergence of fixed-omega ADMM on the upper-support study."""

import argparse
from math import comb
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from admm import (
    DEFAULT_ADMM_RADIUS,
    admm_solve,
    fixed_omega_admm_beta,
    initialize_admm_state,
)
from experiments.compute_objective_curve import (
    covariance_from_lambda_star,
    indexed_upper_support,
    lambda_star_for_dimension,
    sample_empirical_covariance,
)
from objective import frobenius_objective
from optimizers.support_search import resolve_omega_ref
from supports.common import support_mask_from_edges


def fit_and_report(Sigma_hat, mask, omega_ref, start, args):
    beta = args.beta
    if beta is None:
        beta = fixed_omega_admm_beta(Sigma_hat, omega_ref, args.radius)
    target = Sigma_hat - omega_ref * np.eye(len(Sigma_hat))
    initial, _ = initialize_admm_state(len(Sigma_hat), mask, start, "halton")
    max_norm = np.linalg.norm(initial, ord=2)
    iterations = 0
    primal = dual = stationarity = np.nan
    converged = False

    def check(iteration, L1, L2, previous_L2, _omega, penalty):
        nonlocal iterations, max_norm, primal, dual, stationarity, converged
        iterations = iteration
        max_norm = max(max_norm, np.linalg.norm(L1, ord=2), np.linalg.norm(L2, ord=2))
        primal = np.linalg.norm(L1 - L2, "fro")
        dual = penalty * np.linalg.norm(L2 - previous_L2, "fro")
        SL1 = Sigma_hat @ L1
        gradient = 4.0 * SL1 @ (L1.T @ SL1 - target)
        stationarity = np.linalg.norm(gradient[mask])
        converged = max(primal, dual, stationarity) <= args.tol
        return converged

    Lambda, omega = admm_solve(
        Sigma_hat,
        mask,
        beta=beta,
        max_restarts=1,
        max_iter=args.max_iter,
        omega_fixed=omega_ref,
        init_strategy="halton",
        init_offset=start,
        stop_when=check,
    )
    if not np.all(np.isfinite(Lambda)) or not np.isfinite(omega):
        status = "numerical failure"
        converged = False
    elif converged:
        status = "converged"
    else:
        status = "not converged (iteration limit)"

    print(f"  Halton start {start + 1}: {status}, iterations={iterations}, beta={beta:.6g}")
    print(
        f"  Primal residual={primal:.3e}, dual residual={dual:.3e}, "
        f"support gradient={stationarity:.3e}"
    )
    print(f"  Maximum iterate spectral norm={max_norm:.6f} (radius={args.radius:g})")
    print(f"  Omega={omega:.8f}, objective={frobenius_objective(Sigma_hat, Lambda, omega):.8g}")
    print(f"  Lambda:\n{np.array2string(Lambda, precision=7)}")
    return converged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=124, help="Study seed (default: 124).")
    parser.add_argument("--starts", type=int, default=1, help="Independent Halton starts per support.")
    parser.add_argument("--max-iter", type=int, default=20_000, help="Iteration limit per start.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Absolute tolerance for all residuals.")
    parser.add_argument("--beta", type=float, help="Fixed ADMM penalty (default: computed from radius).")
    parser.add_argument("--radius", type=float, default=DEFAULT_ADMM_RADIUS, help="ADMM radius for beta.")
    args = parser.parse_args()
    if args.seed < 0 or args.starts < 1 or args.max_iter < 1:
        parser.error("seed must be nonnegative and starts and max-iter must be positive")
    if not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("tol must be finite and positive")
    if not np.isfinite(args.radius) or args.radius <= 0:
        parser.error("radius must be finite and positive")
    if args.beta is not None and (not np.isfinite(args.beta) or args.beta <= 0):
        parser.error("beta must be finite and positive")

    n, true_dimension, num_samples = 4, 4, 100
    num_supports = comb(n * (n - 1) // 2, true_dimension - 1)
    print(
        f"n={n}, D_m={true_dimension}, N={num_samples}, study seed={args.seed}, "
        f"starts per support={args.starts}, tol={args.tol:g}"
    )
    num_converged = 0
    for index in range(num_supports):
        edges = indexed_upper_support(n, true_dimension, index)
        Lambda_star = lambda_star_for_dimension(n, edges)
        Sigma = covariance_from_lambda_star(Lambda_star, omega=1.0)
        Sigma_hat = sample_empirical_covariance(Sigma, num_samples, seed=args.seed + 2)
        omega_ref = resolve_omega_ref(Sigma_hat, fit_omega_ref=True)
        mask = support_mask_from_edges(n, edges)
        print(f"\nSupport {index:02d}, off-diagonal entries (zero-based) {edges}")
        for start in range(args.starts):
            num_converged += fit_and_report(Sigma_hat, mask, omega_ref, start, args)

    print(f"\nConverged to numeric tolerances: {num_converged}/{num_supports * args.starts}")


if __name__ == "__main__":
    main()
