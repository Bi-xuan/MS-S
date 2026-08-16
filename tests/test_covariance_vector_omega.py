"""Tests for constructing a covariance with feature-specific omega values."""

import numpy as np
import pytest


def covariance_from_lambda_star(Lambda_star, omega):
    """Solve Sigma - Lambda.T @ Sigma @ Lambda = diag(omega)."""
    Lambda_star = np.asarray(Lambda_star, dtype=float)
    omega = np.asarray(omega, dtype=float)

    n = Lambda_star.shape[0]
    if omega.ndim != 1 or omega.shape[0] != n:
        raise ValueError("omega must be a vector with one entry per feature.")
    if np.any(omega < 0.0):
        raise ValueError("omega entries must be nonnegative.")
    if np.max(np.abs(np.linalg.eigvals(Lambda_star))) >= 1.0:
        raise ValueError(
            "All eigenvalues of Lambda_star must be smaller than 1 in absolute value."
        )

    system_matrix = np.eye(n * n) - np.kron(Lambda_star.T, Lambda_star.T)
    rhs = np.diag(omega).reshape(-1, order="F")
    sigma_vec = np.linalg.solve(system_matrix, rhs)
    Sigma = sigma_vec.reshape((n, n), order="F")
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma, np.linalg.matrix_rank(Sigma)


def test_covariance_from_lambda_star_accepts_vector_omega():
    Lambda_star = np.array([
    [0.20,  0.00,  0.00, 0.04],
    [0.00,  0.30, 0.00, 0.05],
    [0.00,  0.00,  0.25, 0.12],
    [0.00, 0.00,  0.00, 0.15],
])
    omega = np.array([0.4, 0.8, 1.2, 0.7])

    Sigma, rank = covariance_from_lambda_star(Lambda_star, omega)
    print("Calculated Sigma:")
    print(Sigma)
    print(f"Rank of Sigma: {rank}")

    np.testing.assert_allclose(Sigma, Sigma.T)
    np.testing.assert_allclose(
        Sigma - Lambda_star.T @ Sigma @ Lambda_star,
        np.diag(omega),
        atol=1e-12,
    )
    assert rank == 4


@pytest.mark.parametrize("omega", [[0.4, 0.8], [0.4, -0.8, 1.2]])
def test_covariance_from_lambda_star_rejects_invalid_omega(omega):
    with pytest.raises(ValueError):
        covariance_from_lambda_star(np.eye(3) * 0.2, omega)
