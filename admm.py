import numpy as np

def impose_support(M, mask):
    M_out = M.copy()
    M_out[~mask] = 0.0
    return M_out

def update_omega(Sigma, Lambda):
    n = Sigma.shape[0]
    residual = Sigma - Lambda.T @ Sigma @ Lambda
    return np.trace(residual) / n

def is_finite_state(*arrays):
    return all(np.all(np.isfinite(arr)) for arr in arrays)


def admm_solve(
    Sigma,
    support_mask,
    beta=1.0,
    max_iter=500,
    tol=1e-6,
    max_restarts=3,
):
    n = Sigma.shape[0]
    for _ in range(max_restarts):
        # Initialize
        L1 = impose_support(np.random.randn(n, n), support_mask)
        L2 = impose_support(np.random.randn(n, n), support_mask)
        alpha = np.zeros((n, n))
        omega = 0.0
        failed = False

        for _ in range(max_iter):
            L1_prev = L1.copy()

            try:
                # Update Lambda_1
                SL2 = Sigma @ L2
                A1 = 2 * SL2 @ SL2.T + beta * np.eye(n)
                B1 = 2 * SL2 @ (Sigma - omega * np.eye(n)) - alpha + beta * L2
                L1 = np.linalg.solve(A1, B1)
                L1 = impose_support(L1, support_mask)

                # Update Lambda_2
                SL1 = Sigma @ L1
                A2 = 2 * SL1 @ SL1.T + beta * np.eye(n)
                B2 = 2 * SL1 @ (Sigma - omega * np.eye(n)) + alpha + beta * L1
                L2 = np.linalg.solve(A2, B2)
                L2 = impose_support(L2, support_mask)
            except np.linalg.LinAlgError:
                failed = True
                break

            # Update omega
            omega = update_omega(Sigma, L1)

            if not is_finite_state(L1, L2, alpha, omega):
                failed = True
                break

            # Update dual variable
            alpha = alpha + beta * (L1 - L2)

            if not is_finite_state(alpha):
                failed = True
                break

            # Check convergence
            if np.linalg.norm(L1 - L1_prev, 'fro') < tol:
                return L1, omega

        if not failed and is_finite_state(L1, L2, alpha, omega):
            return L1, omega

    nan_matrix = np.full((n, n), np.nan)
    return nan_matrix, np.nan

# Example usage and test case
if __name__ == "__main__":

    np.random.seed(42)
    n = 3

    # Build a random SPD matrix as Sigma
    A = np.random.randn(n, n)
    Sigma = A @ A.T / n

    # ── Test 1: full support (all entries free) ──────────────────────────────
    # With no support restriction, ADMM should reduce the objective significantly
    print("=" * 50)
    print("Test 1: full support")
    full_mask = np.ones((n, n), dtype=bool)
    L1, omega = admm_solve(Sigma, full_mask)
    residual = Sigma - L1.T @ Sigma @ L1 - omega * np.eye(n)
    obj = np.linalg.norm(residual, 'fro') ** 2
    print(f"  Objective value : {obj:.8f}")
    print(f"  omega           : {omega:.6f}")
    print(f"  Lambda:\n{L1}")
    print(f"  Residual norm   : {np.linalg.norm(residual, 'fro'):.8f}")

    # ── Test 2: diagonal-only support ────────────────────────────────────────
    # Lambda is forced to be diagonal; checks impose_support is working
    print("=" * 50)
    print("Test 2: diagonal-only support")
    diag_mask = np.eye(n, dtype=bool)
    L1, omega = admm_solve(Sigma, diag_mask)
    # Verify off-diagonal entries are exactly zero
    off_diag_max = np.max(np.abs(L1 - np.diag(np.diag(L1))))
    residual = Sigma - L1.T @ Sigma @ L1 - omega * np.eye(n)
    obj = np.linalg.norm(residual, 'fro') ** 2
    print(f"  Objective value    : {obj:.8f}")
    print(f"  omega              : {omega:.6f}")
    print(f"  Max off-diag in L1 : {off_diag_max:.2e}  (should be 0)")
    print(f"  Lambda:\n{L1}")

    # ── Test 3: zero Sigma edge case ─────────────────────────────────────────
    # If Sigma = 0, the objective is omega^2 * n, minimized at omega = 0
    print("=" * 50)
    print("Test 3: Sigma = 0")
    Sigma_zero = np.zeros((n, n))
    full_mask = np.ones((n, n), dtype=bool)
    L1, omega = admm_solve(Sigma_zero, full_mask)
    print(f"  omega (should be ~0) : {omega:.6f}")
    print(f"  Lambda:\n{L1}")

    # ── Test 4: convergence check ─────────────────────────────────────────────
    # Run with a tight tolerance and confirm L1 ≈ L2 at convergence (primal feasibility)
    print("=" * 50)
    print("Test 4: primal feasibility at convergence")
    A = np.random.randn(n, n)
    Sigma = A @ A.T / n
    full_mask = np.ones((n, n), dtype=bool)

    # Patch admm_solve temporarily to also return L2
    L1 = np.zeros((n, n))
    L2 = np.zeros((n, n))
    alpha = np.zeros((n, n))
    omega = 0.0
    beta = 1.0
    for _ in range(500):
        L1_prev = L1.copy()
        SL2 = Sigma @ L2
        A1 = 2 * SL2 @ SL2.T + beta * np.eye(n)
        B1 = 2 * SL2 @ (Sigma - omega * np.eye(n)) - alpha + beta * L2
        L1 = np.linalg.solve(A1, B1)
        SL1 = Sigma @ L1
        A2 = 2 * SL1 @ SL1.T + beta * np.eye(n)
        B2 = 2 * SL1 @ (Sigma - omega * np.eye(n)) + alpha + beta * L1
        L2 = np.linalg.solve(A2, B2)
        omega = update_omega(Sigma, L1)
        alpha = alpha + beta * (L1 - L2)
        if np.linalg.norm(L1 - L1_prev, 'fro') < 1e-6:
            break
    primal_residual = np.linalg.norm(L1 - L2, 'fro')
    print(f"  ||L1 - L2||_F (should be ~0) : {primal_residual:.2e}")

    # ── Test 5: solve with a given support ───────────────────────────────────
    # Use a fixed support mask and verify the returned Lambda respects it.
    print("=" * 50)
    print("Test 5: check result with given support")
    B = np.array([[1.2, 0.1, 0.3],
                  [0.4, 1.0, 0.2],
                  [0.2, 0.5, 0.9]])
    Sigma = B @ B.T / n
    given_support = np.array([[1, 1, 0],
                              [0, 1, 0],
                              [0, 1, 1]], dtype=bool)
    L1, omega = admm_solve(Sigma, given_support)
    outside_support_max = np.max(np.abs(L1[~given_support]))
    residual = Sigma - L1.T @ Sigma @ L1 - omega * np.eye(n)
    obj = np.linalg.norm(residual, 'fro') ** 2
    print(f"  Given support mask:\n{given_support.astype(int)}")
    print(f"  Objective value         : {obj:.8f}")
    print(f"  omega                   : {omega:.6f}")
    print(f"  Max entry outside mask  : {outside_support_max:.2e}  (should be 0)")
    print(f"  Lambda:\n{L1}")
