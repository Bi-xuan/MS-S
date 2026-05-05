"""Run the active synthetic Lambda/omega optimization experiment. Optimize over one D_m"""

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from admm import covariance_from_lambda_star, lambda_star_spectral_radius
from optimizers.support_search import optimize_lambda, print_optimization_result


def empirical_covariance_from_samples(X):
    num_samples = X.shape[0]
    if num_samples < 2:
        raise ValueError("At least two observations are required to compute Sigma_hat.")

    X_centered = X - X.mean(axis=0, keepdims=True)
    return X_centered.T @ X_centered / (num_samples - 1)


def sample_empirical_covariance(Sigma, num_samples, seed=None):
    rng = np.random.default_rng(seed)
    num_features = Sigma.shape[0]
    X = rng.multivariate_normal(
        mean=np.zeros(num_features),
        cov=Sigma,
        size=num_samples,
    )
    return empirical_covariance_from_samples(X)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the active Lambda optimization experiment."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of worker processes for support-level parallelism.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used for deterministic ADMM support solves.",
    )
    parser.add_argument(
        "--init-strategy",
        choices=["halton", "random"],
        default="halton",
        help="ADMM initialization strategy.",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=5,
        help="Number of ADMM initializations to try per support.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    D_m = 4
    Lambda_star = np.array([
        [0.15, 0.0, 0.0, 0.6],
        [0.0, -0.20, 0.0, -0.45],
        [0.0, 0.0, 0.10, 0.50],
        [0.0, 0.0, 0.0, 0.55],
    ])
    omega_star = 1.0
    omega_ref = omega_star

    lambda_star_radius = lambda_star_spectral_radius(Lambda_star)
    if lambda_star_radius >= 1.0:
        raise ValueError(
            "All eigenvalues of Lambda_star must be smaller than 1 "
            "in absolute value."
        )

    Sigma_given = covariance_from_lambda_star(Lambda_star, omega_star)
    lambda_min_sigma = np.min(np.linalg.eigvalsh(Sigma_given))

    np.random.seed(args.random_seed)
    Lambda, omega, obj = optimize_lambda(
        Sigma_given,
        D_m,
        max_iter=800,
        tol=1e-7,
        max_restarts=args.max_restarts,
        omega_fixed=omega_ref,
        n_jobs=args.n_jobs,
        random_seed=args.random_seed,
        init_strategy=args.init_strategy,
    )
    print(
        "\nTest 7: fixed-omega support selection, then bounded free-omega "
        "optimization"
    )
    print(f"Lambda_star:\n{Lambda_star}")
    print(f"Spectral radius of Lambda_star: {lambda_star_radius:.6f}")
    print(f"omega_star: {omega_star:.6f}")
    print(f"omega_ref for support selection: {omega_ref:.6f}")
    print(f"lambda_min(Sigma): {lambda_min_sigma:.6f}")
    print(f"Sigma:\n{Sigma_given}")
    print_optimization_result(Lambda, omega, obj)
    if omega is not None and np.isfinite(omega):
        print(f"omega <= lambda_min(Sigma): {omega <= lambda_min_sigma + 1e-12}")


if __name__ == "__main__":
    main()
