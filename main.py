import numpy as np
from support_utils import get_all_supports
from admm import admm_solve
from objective import frobenius_objective

def optimize_lambda(Sigma, D_m, beta=1.0, max_iter=500, tol=1e-6):
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
        Lambda, omega = admm_solve(Sigma, mask, beta=beta,
                                   max_iter=max_iter, tol=tol)
        obj = frobenius_objective(Sigma, Lambda, omega)
        if obj < best_obj:
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
    print(f"Best objective: {obj:.6f}")
    print(f"Best omega:     {omega:.6f}")
    print(f"Best Lambda:\n{Lambda}")