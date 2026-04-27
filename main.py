import numpy as np
from support_utils import get_all_supports
from admm import (
    admm_solve,
    covariance_from_lambda_star,
    lambda_star_spectral_radius,
    sample_lambda_star_from_mask,
)
from objective import frobenius_objective

def threshold_lambda(Lambda, zero_tol):
    Lambda_thr = Lambda.copy()
    Lambda_thr[np.abs(Lambda_thr) < zero_tol] = 0.0
    return Lambda_thr


def effective_offdiag_count(Lambda, zero_tol):
    off_diag_mask = ~np.eye(Lambda.shape[0], dtype=bool)
    return np.count_nonzero(np.abs(Lambda[off_diag_mask]) >= zero_tol)


def has_unused_supported_offdiag(Lambda, mask, zero_tol):
    off_diag_mask = ~np.eye(Lambda.shape[0], dtype=bool)
    supported_offdiag = mask & off_diag_mask
    return np.any(np.abs(Lambda[supported_offdiag]) < zero_tol)


def is_finite_candidate(Lambda, omega, obj):
    return (
        np.all(np.isfinite(Lambda))
        and np.isfinite(omega)
        and np.isfinite(obj)
    )


def satisfies_hard_constraints(
    Lambda,
    mask,
    omega,
    min_supported_offdiag_abs,
    min_omega,
):
    if omega < min_omega:
        return False

    off_diag_mask = ~np.eye(Lambda.shape[0], dtype=bool)
    supported_offdiag = mask & off_diag_mask
    return np.all(
        np.abs(Lambda[supported_offdiag]) >= min_supported_offdiag_abs
    )


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


def print_optimization_result(Lambda, omega, obj):
    if Lambda is None or omega is None or not np.isfinite(obj):
        print("No candidate satisfied the hard constraints.")
        return

    print(f"Best objective: {obj:.6f}")
    print(f"Best omega:     {omega:.6f}")
    print(f"Best Lambda:\n{Lambda}")


def solve_support_with_restarts(
    Sigma,
    mask,
    beta,
    max_iter,
    tol,
    zero_tol,
    max_restarts,
    min_supported_offdiag_abs,
    min_omega,
    omega_fixed=None,
    omega_upper=None,
):
    runs = []

    for _ in range(max_restarts):
        Lambda, omega = admm_solve(
            Sigma,
            mask,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            max_restarts=1,
            omega_fixed=omega_fixed,
            omega_upper=omega_upper,
        )
        Lambda_thr = threshold_lambda(Lambda, zero_tol)
        obj = frobenius_objective(Sigma, Lambda_thr, omega)

        if not is_finite_candidate(Lambda_thr, omega, obj):
            continue

        if not satisfies_hard_constraints(
            Lambda_thr,
            mask,
            omega,
            min_supported_offdiag_abs,
            min_omega,
        ):
            continue

        runs.append((Lambda_thr, omega, obj))

        if not has_unused_supported_offdiag(
            Lambda,
            mask,
            min_supported_offdiag_abs,
        ):
            break

    if not runs:
        return None

    ref_pattern = np.abs(runs[0][0]) >= zero_tol
    consistent_runs = [
        run for run in runs
        if np.array_equal(np.abs(run[0]) >= zero_tol, ref_pattern)
    ]

    if len(consistent_runs) == len(runs):
        return min(consistent_runs, key=lambda run: run[2])

    return None


def optimize_lambda(
    Sigma,
    D_m,
    beta=1.0,
    max_iter=500,
    tol=1e-6,
    zero_tol=1e-5,
    obj_tol=1e-8,
    max_restarts=3,
    min_supported_offdiag_abs=1e-3,
    min_omega=1e-8,
    omega_fixed=None,
):
    """
    Parameters
    ----------
    Sigma  : (n, n) ndarray — fixed covariance matrix
    D_m    : int — D_m = n_edge + 1, so n_edge = D_m - 1
    min_supported_offdiag_abs : float — minimum absolute value for every
        selected off-diagonal Lambda entry
    min_omega : float — minimum allowed omega
    omega_fixed : float or None — if provided, use this value as omega_ref
        to select the support, then re-optimize that support with free omega
        constrained by omega <= lambda_min(Sigma)
    
    Returns
    -------
    best_Lambda : (n, n) ndarray
    best_omega  : float
    best_obj    : float
    """
    n = Sigma.shape[0]
    n_edge = D_m - 1
    supports = get_all_supports(n, n_edge)
    lambda_min_sigma = np.min(np.linalg.eigvalsh(Sigma))

    if lambda_min_sigma <= min_omega:
        return None, None, np.inf

    best_obj = np.inf
    best_Lambda = None
    best_omega = None
    best_mask = None

    for mask in supports:
        result = solve_support_with_restarts(
            Sigma,
            mask,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            zero_tol=zero_tol,
            max_restarts=max_restarts,
            min_supported_offdiag_abs=min_supported_offdiag_abs,
            min_omega=min_omega,
            omega_fixed=omega_fixed,
            omega_upper=lambda_min_sigma if omega_fixed is None else None,
        )

        if result is None:
            continue

        Lambda, omega, obj = result

        if best_Lambda is None:
            best_obj = obj
            best_Lambda = Lambda.copy()
            best_omega = omega
            best_mask = mask.copy()
            continue

        if obj < best_obj - obj_tol:
            best_obj = obj
            best_Lambda = Lambda.copy()
            best_omega = omega
            best_mask = mask.copy()

    if omega_fixed is not None and best_mask is not None:
        final_result = solve_support_with_restarts(
            Sigma,
            best_mask,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            zero_tol=zero_tol,
            max_restarts=max_restarts,
            min_supported_offdiag_abs=min_supported_offdiag_abs,
            min_omega=min_omega,
            omega_fixed=None,
            omega_upper=lambda_min_sigma,
        )

        if final_result is None:
            return None, None, np.inf

        best_Lambda, best_omega, best_obj = final_result

    return best_Lambda, best_omega, best_obj


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Test 1: random Sigma, n = 3
    # -------------------------------------------------------------------------
    # np.random.seed(0)
    # n = 3
    # A = np.random.randn(n, n)
    # Sigma = A @ A.T / n   # random SPD matrix
    #
    # D_m = 4  # 3 edges
    # Lambda, omega, obj = optimize_lambda(Sigma, D_m)
    # print("Test 1: random Sigma, n = 3")
    # print_optimization_result(Lambda, omega, obj)
    #
    # -------------------------------------------------------------------------
    # Test 2: given Sigma, n = 3
    # -------------------------------------------------------------------------
    # Sigma_given = np.array([
    #     [0.94288908, 0.0271487, -0.09748996],
    #     [0.0271487, 2.16931437, 0.14698582],
    #     [-0.09748996, 0.14698582, 1.51557001],
    # ])
    #
    # Lambda, omega, obj = optimize_lambda(Sigma_given, D_m)
    # print("\nTest 2: given Sigma, n = 3")
    # print(f"Sigma:\n{Sigma_given}")
    # print_optimization_result(Lambda, omega, obj)
    #
    # -------------------------------------------------------------------------
    # Test 3: Sigma generated from Lambda_star, n = 3
    # -------------------------------------------------------------------------
    # Lambda_star_mask = np.array([
    #     [1, 0, 1],
    #     [0, 1, 1],
    #     [0, 0, 1],
    # ], dtype=bool)
    # Lambda_star = sample_lambda_star_from_mask(
    #     Lambda_star_mask,
    #     target_spectral_radius=0.9,
    #     seed=0,
    # )
    # omega_star = 1.0
    #
    # lambda_star_radius = lambda_star_spectral_radius(Lambda_star)
    # if lambda_star_radius >= 1.0:
    #     raise ValueError(
    #         "All eigenvalues of Lambda_star must be smaller than 1 "
    #         "in absolute value."
    #     )
    #
    # Sigma_given = covariance_from_lambda_star(Lambda_star, omega_star)
    #
    # Lambda, omega, obj = optimize_lambda(Sigma_given, D_m)
    # print("\nTest 3: Sigma generated from Lambda_star, n = 3")
    # print(f"Lambda_star:\n{Lambda_star}")
    # print(f"Spectral radius of Lambda_star: {lambda_star_radius:.6f}")
    # print(f"Sigma:\n{Sigma_given}")
    # print_optimization_result(Lambda, omega, obj)
    #
    # -------------------------------------------------------------------------
    # Test 4: Sigma_hat from 100 observations, n = 3
    # -------------------------------------------------------------------------
    # Sigma_hat = sample_empirical_covariance(Sigma_given, 100, seed=0)
    #
    # Lambda, omega, obj = optimize_lambda(Sigma_hat, D_m)
    # print("\nTest 4: Sigma_hat from 100 observations, n = 3")
    # print(f"Lambda_star:\n{Lambda_star}")
    # print(f"Spectral radius of Lambda_star: {lambda_star_radius:.6f}")
    # print(f"Sigma_given:\n{Sigma_given}")
    # print(f"Sigma_hat:\n{Sigma_hat}")
    # print_optimization_result(Lambda, omega, obj)

    # -------------------------------------------------------------------------
    # Test 5: random Sigma, n = 4
    # -------------------------------------------------------------------------
    # np.random.seed(0)
    # n = 4
    # A = np.random.randn(n, n)
    # Sigma = A @ A.T / n   # random SPD matrix
    #
    # D_m = 4  # 3 edges
    # Lambda, omega, obj = optimize_lambda(Sigma, D_m)
    # print("Test 5: random Sigma, n = 4")
    # print_optimization_result(Lambda, omega, obj)

    # -------------------------------------------------------------------------
    # Test 6: given Sigma, n = 4
    # -------------------------------------------------------------------------
    # Sigma_given = np.array([
    #     [1.40, 0.12, -0.05, 0.08],
    #     [0.12, 2.10, 0.18, -0.11],
    #     [-0.05, 0.18, 1.65, 0.09],
    #     [0.08, -0.11, 0.09, 1.25],
    # ])
    #
    # Lambda, omega, obj = optimize_lambda(Sigma_given, D_m)
    # print("\nTest 6: given Sigma, n = 4")
    # print(f"Sigma:\n{Sigma_given}")
    # print_optimization_result(Lambda, omega, obj)

    # -------------------------------------------------------------------------
    # Test 7: Sigma generated from Lambda_star, n = 4
    # -------------------------------------------------------------------------
    D_m = 4  # 3 edges
    Lambda_star_mask = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ], dtype=bool)
    Lambda_star = np.array([
        [0.15, 0.0, 0.0, 0.6],
        [0.0, -0.20, 0.0, -0.45],
        [0.0, 0.0, 0.10, 0.50],
        [0.0, 0.0, 0.0, 0.55],
    ])
    omega_star = 1.0
    omega_ref = omega_star
    # omega_ref = 0.5
    
    lambda_star_radius = lambda_star_spectral_radius(Lambda_star)
    if lambda_star_radius >= 1.0:
        raise ValueError(
            "All eigenvalues of Lambda_star must be smaller than 1 "
            "in absolute value."
        )
    
    Sigma_given = covariance_from_lambda_star(Lambda_star, omega_star)
    lambda_min_sigma = np.min(np.linalg.eigvalsh(Sigma_given))
    
    np.random.seed(42)
    Lambda, omega, obj = optimize_lambda(
        Sigma_given,
        D_m,
        max_iter=800,
        tol=1e-7,
        max_restarts=5,
        omega_fixed=omega_ref,
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

    # -------------------------------------------------------------------------
    # Test 7: Sigma generated from Lambda_star, n = 4
    # -------------------------------------------------------------------------
    # D_m = 4  # 3 edges
    # Lambda_star_mask = np.array([
    #     [1, 0, 0, 1],
    #     [0, 1, 0, 1],
    #     [0, 0, 1, 1],
    #     [0, 0, 0, 1],
    # ], dtype=bool)
    # Lambda_star = np.array([
    #     [0.15, 0.0, 0.0, 0.6],
    #     [0.0, -0.20, 0.0, -0.45],
    #     [0.0, 0.0, 0.10, 0.50],
    #     [0.0, 0.0, 0.0, 0.55],
    # ])
    # omega_star = 1.0

    # lambda_star_radius = lambda_star_spectral_radius(Lambda_star)
    # if lambda_star_radius >= 1.0:
    #     raise ValueError(
    #         "All eigenvalues of Lambda_star must be smaller than 1 "
    #         "in absolute value."
    #     )

    # Sigma_given = covariance_from_lambda_star(Lambda_star, omega_star)
    # lambda_min_sigma = np.min(np.linalg.eigvalsh(Sigma_given))

    # np.random.seed(42)
    # Lambda, omega, obj = optimize_lambda(
    #     Sigma_given,
    #     D_m,
    #     max_iter=800,
    #     tol=1e-7,
    #     max_restarts=5,
    #     omega_fixed=None,
    # )
    # print(
    #     "\nTest 7: bounded free-omega support selection"
    # )
    # print(f"Lambda_star:\n{Lambda_star}")
    # print(f"Spectral radius of Lambda_star: {lambda_star_radius:.6f}")
    # print(f"omega_star: {omega_star:.6f}")
    # print(f"lambda_min(Sigma): {lambda_min_sigma:.6f}")
    # print(f"Sigma:\n{Sigma_given}")
    # print_optimization_result(Lambda, omega, obj)
    # if omega is not None and np.isfinite(omega):
    #     print(f"omega <= lambda_min(Sigma): {omega <= lambda_min_sigma + 1e-12}")

    # -------------------------------------------------------------------------
    # Test 8: Sigma_hat from 100 observations, n = 4
    # -------------------------------------------------------------------------
    # Sigma_hat = sample_empirical_covariance(Sigma_given, 100, seed=0)
    #
    # Lambda, omega, obj = optimize_lambda(Sigma_hat, D_m)
    # print("\nTest 8: Sigma_hat from 100 observations, n = 4")
    # print(f"Lambda_star:\n{Lambda_star}")
    # print(f"Spectral radius of Lambda_star: {lambda_star_radius:.6f}")
    # print(f"Sigma_given:\n{Sigma_given}")
    # print(f"Sigma_hat:\n{Sigma_hat}")
    # print_optimization_result(Lambda, omega, obj)
