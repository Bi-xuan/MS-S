"""Inspect select_support on synthetic iid Gaussian data.

Run with Python 3.10+ after installing lambda-support-recovery:
    python test_package.py
"""

import numpy as np
from lambda_support_recovery import select_support

from admm import covariance_from_lambda_star
from experiments.compute_objective_curve import lambda_star_for_dimension


SAMPLE_SIZES = (100, 1_000, 10_000, 100_000)
OMEGA_STAR = 1.0
SEED = 42
N_JOBS = 8

# Smaller fitting/bootstrap budgets keep the parameter comparisons practical.
# The simplest call below uses package defaults apart from the worker count.
SWEEP_SETTINGS = dict(n_jobs=N_JOBS, max_restarts=2, max_iter=120, bootstrap_replicates=39)
SETTINGS = [
    ("Bootstrap, all directed edges", {}),
    ("Bootstrap, upper triangular", dict(support_scope="upper")),
    ("Plateau, all directed edges", dict(method="plateau")),
    ("Plateau, upper triangular", dict(method="plateau", support_scope="upper")),
    ("Plateau, constant Lm", dict(
        method="plateau", support_scope="upper", lm_mode="constant", lm_weight=0.5,
    )),
    ("Plateau, support-count Lm", dict(
        method="plateau", support_scope="upper", lm_mode="support-count", lm_weight=0.3,
    )),
    ("Bootstrap, estimated omega with kappa=0.8", dict(
        support_scope="upper", fit_omega_ref=True, kappa=0.8,
    )),
    ("Bootstrap, known omega=1", dict(
        support_scope="upper", omega_star=OMEGA_STAR,
        omega_ref=OMEGA_STAR, fit_omega_ref=False,
    )),
    ("Plateau, exhaustive upper-triangular search", dict(
        method="plateau", support_scope="upper", nested_supports=False,
    )),
]


def main():
    Lambda_star = lambda_star_for_dimension(4)
    sigma_star = covariance_from_lambda_star(Lambda_star, OMEGA_STAR)
    true_support = Lambda_star != 0
    print("Lambda_star:\n", Lambda_star, sep="", flush=True)
    print("sigma_star (omega_star=1):\n", sigma_star, sep="", flush=True)
    print("True support:\n", true_support.astype(int), sep="", flush=True)

    for num_samples in SAMPLE_SIZES:
        # Resetting the seed gives nested iid samples across sample sizes.
        # Every parameter setting at this N uses the same empirical covariance.
        rng = np.random.default_rng(SEED)
        samples = rng.multivariate_normal(np.zeros(4), sigma_star, size=num_samples)
        sigma_hat = samples.T @ samples / num_samples
        print(f"\n{'=' * 60}\nnum_samples={num_samples}", flush=True)
        print(f"lambda_min(sigma_hat)={np.linalg.eigvalsh(sigma_hat)[0]:.6f}", flush=True)

        print(f"\nSimplest usage (defaults with n_jobs={N_JOBS}):", flush=True)
        support = select_support(sigma_hat, num_samples=num_samples, n_jobs=N_JOBS)
        print(support.astype(int), flush=True)
        print("Exact support recovery:", np.array_equal(support, true_support), flush=True)

        for label, settings in SETTINGS:
            print(f"\n{label}\nParameters: {dict(SWEEP_SETTINGS, **settings)}", flush=True)
            try:
                result = select_support(
                    sigma_hat, num_samples=num_samples, return_result=True,
                    **SWEEP_SETTINGS, **settings,
                )
            except ValueError as error:
                # In particular, omega_ref=1 is inadmissible when the sample
                # covariance's smallest eigenvalue is below 1.
                print(f"Selection failed: {error}", flush=True)
                continue
            print("Support:\n", result.support.astype(int), sep="", flush=True)
            print("Selected dimension:", result.selected_dimension, flush=True)
            print("Selected edges:", result.selected_edges, flush=True)
            print("Fitted omega:", result.fitted_omega, flush=True)
            print("Fitted Lambda:\n", result.fitted_lambda, sep="", flush=True)
            print("Raw objectives:", result.curve.raw_objectives, flush=True)
            print("Exact support recovery:", np.array_equal(result.support, true_support), flush=True)


if __name__ == "__main__":
    main()
