"""ADMM solver and simulation helpers for Lambda/omega covariance models."""

import numpy as np


def van_der_corput(index, base):
    value = 0.0
    factor = 1.0 / base

    while index > 0:
        value += factor * (index % base)
        index //= base
        factor /= base

    return value


def first_primes(count):
    primes = []
    candidate = 2

    while len(primes) < count:
        is_prime = True
        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break

        if is_prime:
            primes.append(candidate)

        candidate += 1

    return primes


def halton_point(index, dim):
    primes = first_primes(dim)
    return np.array([
        van_der_corput(index, base)
        for base in primes
    ])


def impose_support(M, mask):
    M_out = M.copy()
    M_out[~mask] = 0.0
    return M_out

def update_omega(Sigma, Lambda, omega_upper=None):
    n = Sigma.shape[0]
    residual = Sigma - Lambda.T @ Sigma @ Lambda
    omega = max(np.trace(residual) / n, 0.0)
    if omega_upper is not None:
        omega = min(omega, omega_upper)
    return omega


def covariance_from_lambda_star(Lambda_star, omega):
    if omega < 0.0:
        raise ValueError("omega must be nonnegative.")
    if lambda_star_spectral_radius(Lambda_star) >= 1.0:
        raise ValueError(
            "All eigenvalues of Lambda_star must be smaller than 1 in absolute value."
        )

    n = Lambda_star.shape[0]
    system_matrix = np.eye(n * n) - np.kron(Lambda_star.T, Lambda_star.T)
    rhs = (omega * np.eye(n)).reshape(-1, order="F")
    sigma_vec = np.linalg.solve(system_matrix, rhs)
    Sigma = sigma_vec.reshape((n, n), order="F")
    return 0.5 * (Sigma + Sigma.T)


def lambda_star_spectral_radius(Lambda_star):
    return np.max(np.abs(np.linalg.eigvals(Lambda_star)))


def sample_lambda_star_from_mask(mask, target_spectral_radius=0.5, seed=0):
    if target_spectral_radius >= 1.0:
        raise ValueError("target_spectral_radius must be smaller than 1.")
    if target_spectral_radius <= 0.0:
        raise ValueError("target_spectral_radius must be positive.")
    if not np.any(mask):
        raise ValueError("Lambda_star_mask must contain at least one True entry.")

    rng = np.random.default_rng(seed)
    Lambda_star = np.zeros(mask.shape)
    Lambda_star[mask] = rng.normal(size=np.count_nonzero(mask))
    spectral_radius = lambda_star_spectral_radius(Lambda_star)
    if spectral_radius == 0.0:
        raise ValueError("Cannot scale Lambda_star with zero spectral radius.")

    Lambda_star *= target_spectral_radius / spectral_radius
    return Lambda_star


def is_finite_state(*arrays):
    return all(np.all(np.isfinite(arr)) for arr in arrays)


def initialize_admm_state(n, support_mask, restart_index, init_strategy):
    if init_strategy == "halton":
        point = halton_point(restart_index + 1, n * n)
        init = 2.0 * (point.reshape(n, n) - 0.5)
        L1 = impose_support(init, support_mask)
        L2 = impose_support(init.copy(), support_mask)
        return L1, L2

    if init_strategy == "random":
        L1 = impose_support(np.random.randn(n, n), support_mask)
        L2 = impose_support(np.random.randn(n, n), support_mask)
        return L1, L2

    raise ValueError(
        "init_strategy must be one of {'halton', 'random'}."
    )


def admm_solve(
    Sigma,
    support_mask,
    beta=1.0,
    max_iter=500,
    tol=1e-6,
    max_restarts=3,
    omega_fixed=None,
    omega_upper=None,
    init_strategy="halton",
    init_offset=0,
):
    if omega_fixed is not None and omega_fixed < 0.0:
        raise ValueError("omega_fixed must be nonnegative.")
    if omega_upper is not None and omega_upper < 0.0:
        raise ValueError("omega_upper must be nonnegative.")

    n = Sigma.shape[0]
    best_Lambda = None
    best_omega = None
    best_obj = np.inf

    for restart_index in range(max_restarts):
        # Initialize
        L1, L2 = initialize_admm_state(
            n,
            support_mask,
            init_offset + restart_index,
            init_strategy,
        )
        alpha = np.zeros((n, n))
        omega = 0.0 if omega_fixed is None else omega_fixed
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

            if omega_fixed is None:
                omega = update_omega(Sigma, L1, omega_upper=omega_upper)

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
                break

        if not failed and is_finite_state(L1, L2, alpha, omega):
            residual = Sigma - L1.T @ Sigma @ L1 - omega * np.eye(n)
            obj = np.linalg.norm(residual, 'fro') ** 2
            if np.isfinite(obj) and obj < best_obj:
                best_Lambda = L1.copy()
                best_omega = omega
                best_obj = obj

    if best_Lambda is not None:
        return best_Lambda, best_omega

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

    # ── Test 6: solve with Sigma generated from Lambda_star ──────────────────
    # Use the Lambda_star-style support: diagonal plus edges into variable 4.
    print("=" * 50)
    print("Test 6: Sigma generated from Lambda_star with given support")
    n = 4
    Lambda_star_mask = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ], dtype=bool)
    Lambda_star = sample_lambda_star_from_mask(
        Lambda_star_mask,
        target_spectral_radius=0.9,
        seed=0,
    )
    omega_star = 1.0

    lambda_star_radius = lambda_star_spectral_radius(Lambda_star)
    if lambda_star_radius >= 1.0:
        raise ValueError(
            "All eigenvalues of Lambda_star must be smaller than 1 "
            "in absolute value."
        )

    Sigma = covariance_from_lambda_star(Lambda_star, omega_star)

    true_residual = Sigma - Lambda_star.T @ Sigma @ Lambda_star - omega_star * np.eye(n)
    true_obj = np.linalg.norm(true_residual, 'fro') ** 2
    identity_residual = Sigma - np.eye(n).T @ Sigma @ np.eye(n)
    identity_obj = np.linalg.norm(identity_residual, 'fro') ** 2

    np.random.seed(42)
    L1, omega = admm_solve(
        Sigma,
        Lambda_star_mask,
        max_iter=2000,
        tol=1e-9,
        max_restarts=10,
    )
    outside_support_max = np.max(np.abs(L1[~Lambda_star_mask]))
    residual = Sigma - L1.T @ Sigma @ L1 - omega * np.eye(n)
    obj = np.linalg.norm(residual, 'fro') ** 2
    print(f"  Lambda_star:\n{Lambda_star}")
    print(f"  omega_star             : {omega_star:.6f}")
    print(f"  True objective value   : {true_obj:.8e}")
    print(f"  Identity objective     : {identity_obj:.8e}")
    print(f"  Generated Sigma:\n{Sigma}")
    print(f"  Given support mask:\n{Lambda_star_mask.astype(int)}")
    print(f"  Objective value         : {obj:.8f}")
    print(f"  omega                   : {omega:.6f}")
    print(f"  Max entry outside mask  : {outside_support_max:.2e}  (should be 0)")
    print(f"  Lambda:\n{L1}")
