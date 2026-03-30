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
        runs.append((Lambda_thr, omega, obj))

        if not has_unused_supported_offdiag(Lambda, mask, zero_tol):
            break

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

    D_m = 4  # 2 edges
    Lambda, omega, obj = optimize_lambda(Sigma, D_m)
    print(f"Best objective: {obj:.6f}")
    print(f"Best omega:     {omega:.6f}")
    print(f"Best Lambda:\n{Lambda}")
