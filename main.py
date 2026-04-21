import numpy as np
from support_utils import get_all_supports
from admm import admm_solve
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


def sample_lambda_star_from_mask(mask, target_fro_norm=0.5, seed=0):
    if target_fro_norm >= 1.0:
        raise ValueError("target_fro_norm must be smaller than 1.")
    if target_fro_norm <= 0.0:
        raise ValueError("target_fro_norm must be positive.")
    if not np.any(mask):
        raise ValueError("Lambda_star_mask must contain at least one True entry.")

    rng = np.random.default_rng(seed)
    Lambda_star = np.zeros(mask.shape)
    Lambda_star[mask] = rng.normal(size=np.count_nonzero(mask))
    Lambda_star *= target_fro_norm / np.linalg.norm(Lambda_star, "fro")
    return Lambda_star


def covariance_from_lambda_star(Lambda_star, omega):
    n = Lambda_star.shape[0]
    system_matrix = np.eye(n * n) - np.kron(Lambda_star.T, Lambda_star.T)
    rhs = (omega * np.eye(n)).reshape(-1, order="F")
    sigma_vec = np.linalg.solve(system_matrix, rhs)
    Sigma = sigma_vec.reshape((n, n), order="F")
    return 0.5 * (Sigma + Sigma.T)


def solve_support_with_restarts(
    Sigma,
    mask,
    beta,
    max_iter,
    tol,
    zero_tol,
    max_restarts,
):
    runs = []

    for _ in range(max_restarts):
        Lambda, omega = admm_solve(
            Sigma,
            mask,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
        )
        Lambda_thr = threshold_lambda(Lambda, zero_tol)
        obj = frobenius_objective(Sigma, Lambda_thr, omega)

        if not is_finite_candidate(Lambda_thr, omega, obj):
            continue

        runs.append((Lambda_thr, omega, obj))

        if not has_unused_supported_offdiag(Lambda, mask, zero_tol):
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
):
    """
    Parameters
    ----------
    Sigma  : (n, n) ndarray — fixed covariance matrix
    D_m    : int — D_m = n_edge + 1, so n_edge = D_m - 1
    
    Returns
    -------
    best_Lambda : (n, n) ndarray
    best_omega  : float
    best_obj    : float
    """
    n = Sigma.shape[0]
    n_edge = D_m - 1
    supports = get_all_supports(n, n_edge)

    best_obj = np.inf
    best_Lambda = None
    best_omega = None

    for mask in supports:
        result = solve_support_with_restarts(
            Sigma,
            mask,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            zero_tol=zero_tol,
            max_restarts=max_restarts,
        )

        if result is None:
            continue

        Lambda, omega, obj = result

        if best_Lambda is None:
            best_obj = obj
            best_Lambda = Lambda.copy()
            best_omega = omega
            continue

        if obj < best_obj - obj_tol:
            best_obj = obj
            best_Lambda = Lambda.copy()
            best_omega = omega

    return best_Lambda, best_omega, best_obj


if __name__ == "__main__":
    np.random.seed(0)
    n = 3
    A = np.random.randn(n, n)
    Sigma = A @ A.T / n   # random SPD matrix

    D_m = 3  # 2 edges
    Lambda, omega, obj = optimize_lambda(Sigma, D_m)
    print("Test 1: random Sigma")
    print(f"Best objective: {obj:.6f}")
    print(f"Best omega:     {omega:.6f}")
    print(f"Best Lambda:\n{Lambda}")

    Sigma_given = np.array([
        [0.94288908, 0.0271487, -0.09748996],
        [0.0271487, 2.16931437, 0.14698582],
        [-0.09748996, 0.14698582, 1.51557001],
    ])

    Lambda, omega, obj = optimize_lambda(Sigma_given, D_m)
    print("\nTest 2: given Sigma")
    print(f"Sigma:\n{Sigma_given}")
    print(f"Best objective: {obj:.6f}")
    print(f"Best omega:     {omega:.6f}")
    print(f"Best Lambda:\n{Lambda}")

    Lambda_star_mask = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=bool)
    Lambda_star = sample_lambda_star_from_mask(
        Lambda_star_mask,
        target_fro_norm=0.9,
        seed=0,
    )
    omega_star = 1.0

    lambda_star_fro = np.linalg.norm(Lambda_star, "fro")
    if lambda_star_fro >= 1.0:
        raise ValueError("Lambda_star must have Frobenius norm smaller than 1.")

    Sigma_given = covariance_from_lambda_star(Lambda_star, omega_star)

    Lambda, omega, obj = optimize_lambda(Sigma_given, D_m)
    print("\nTest 3: Sigma generated from Lambda_star")
    print(f"Lambda_star:\n{Lambda_star}")
    print(f"Frobenius norm of Lambda_star: {lambda_star_fro:.6f}")
    print(f"Sigma:\n{Sigma_given}")
    print(f"Best objective: {obj:.6f}")
    print(f"Best omega:     {omega:.6f}")
    print(f"Best Lambda:\n{Lambda}")
