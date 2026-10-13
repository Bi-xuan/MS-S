"""Upper-triangular L1 Lyapunov fitting and covariance cross validation.

Only strict-upper entries are penalized. Diagonals have either sign and may
equal zero, but are bounded away from +/-1 for a stationary covariance.
The objective is nonconvex; multiple deterministic starts are used.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize


def gamma_grid(minimum=1e-7, maximum=1.0, count=29, spacing="log"):
    """Return zero plus configurable positive, absolute regularization values."""
    if (not np.isfinite(minimum) or not np.isfinite(maximum)
            or not 0 < minimum <= maximum or count < 1 or int(count) != count):
        raise ValueError("Require finite 0 < gamma_min <= gamma_max and gamma_count >= 1.")
    if spacing not in ("log", "linear"):
        raise ValueError("Gamma spacing must be log or linear.")
    positive = (np.geomspace if spacing == "log" else np.linspace)(minimum, maximum, int(count))
    return np.unique(np.r_[0.0, positive])


def validate_settings(*, kappa=.93, diagonal_bound=.999999, max_restarts=10,
                      max_iter=1500, tol=1e-6, zero_tol=1e-5):
    if not np.isfinite(kappa) or not 0 < kappa <= 1:
        raise ValueError("Lasso kappa must be in (0, 1].")
    if not np.isfinite(diagonal_bound) or not 0 < diagonal_bound < 1:
        raise ValueError("Lasso diagonal_bound must be in (0, 1).")
    for name, value in (("max_restarts", max_restarts), ("max_iter", max_iter)):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"Lasso {name} must be a positive integer.")
    if not np.isfinite(tol) or tol <= 0 or not np.isfinite(zero_tol) or zero_tol < 0:
        raise ValueError("Lasso tol must be positive and zero_tol nonnegative, both finite.")


def _covariance(sigma):
    sigma = np.asarray(sigma, dtype=float)
    if (sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]
            or not np.all(np.isfinite(sigma)) or not np.allclose(sigma, sigma.T)
            or np.linalg.eigvalsh(sigma)[0] <= 0):
        raise ValueError("Lasso training covariance must be finite, symmetric and positive definite.")
    return sigma


def _decode(x, n, rows, cols):
    edges = len(rows)
    matrix = np.diag(x[:n])
    matrix[rows, cols] = x[n:n + edges] - x[n + edges:]
    return matrix


def _encode(matrix, rows, cols):
    edges = matrix[rows, cols]
    return np.r_[np.diag(matrix), np.maximum(edges, 0), np.maximum(-edges, 0)]


def _objective_gradient(x, sigma, omega, gamma, rows, cols, scale):
    n = len(sigma)
    matrix = _decode(x, n, rows, cols)
    residual = sigma - matrix.T @ sigma @ matrix - omega * np.eye(n)
    gradient = -4 * sigma @ matrix @ residual
    # Scaling the *whole* objective improves conditioning without changing gamma.
    value = (np.sum(residual ** 2) + gamma * np.sum(x[n:])) / scale
    derivative = np.r_[np.diag(gradient), gradient[rows, cols] + gamma,
                       -gradient[rows, cols] + gamma] / scale
    return float(value), derivative


def _projected_gradient(x, gradient, n, bound):
    return float(np.max(np.abs(x - np.r_[
        np.clip(x[:n] - gradient[:n], -bound, bound),
        np.maximum(x[n:] - gradient[n:], 0)])))


def _initial_states(sigma, omega, bound, max_restarts):
    n = len(sigma)
    rows, cols = np.triu_indices(n, 1)
    # +/-Lambda have identical objectives, so keep the final sign positive.
    count = min(max_restarts, 2 ** (n - 1))
    if 2 ** (n - 1) <= max_restarts:
        signs = [np.r_[s, 1.] for s in itertools.product((-1., 1.), repeat=n - 1)]
    else:
        rng = np.random.default_rng(0)
        signs = [np.ones(n)]
        while len(signs) < count:
            candidate = np.r_[rng.choice([-1., 1.], n - 1), 1.]
            if not any(np.array_equal(candidate, previous) for previous in signs):
                signs.append(candidate)
    r = np.linalg.cholesky(sigma).T
    try:
        c = np.linalg.cholesky(sigma - omega * np.eye(n)).T
    except np.linalg.LinAlgError:
        # kappa=1 can give a singular factor: optimize from bounded diagonals.
        c = None
    states = []
    for sign in signs:
        matrix = (np.linalg.solve(r, np.diag(sign) @ c) if c is not None
                  else np.diag(.25 * sign))
        np.fill_diagonal(matrix, np.clip(np.diag(matrix), -bound, bound))
        states.append(_encode(matrix, rows, cols))
    return states


def fit_lasso_path(sigma, omega, gammas, *, diagonal_bound=.999999,
                   max_restarts=10, max_iter=1500, tol=1e-6):
    """Fit absolute gammas, using dense and warm starts for each candidate.

    Failed candidates have NaN matrices and infinite objectives. Numerical
    validity requires a finite objective and a small projected gradient.
    """
    validate_settings(diagonal_bound=diagonal_bound, max_restarts=max_restarts,
                      max_iter=max_iter, tol=tol)
    sigma = _covariance(sigma)
    if not np.isfinite(omega) or omega <= 0 or omega > np.linalg.eigvalsh(sigma)[0] * (1 + 1e-12):
        raise ValueError("Lasso omega must be positive and <= lambda_min(S).")
    gammas = np.asarray(gammas, dtype=float)
    if (gammas.ndim != 1 or not len(gammas) or not np.all(np.isfinite(gammas))
            or np.any(gammas < 0) or np.any(np.diff(gammas) <= 0)):
        raise ValueError("Gammas must be finite, nonnegative and strictly increasing.")
    n = len(sigma)
    rows, cols = np.triu_indices(n, 1)
    initial = _initial_states(sigma, omega, diagonal_bound, max_restarts)
    warm = [state.copy() for state in initial]
    bounds = [(-diagonal_bound, diagonal_bound)] * n + [(0., None)] * (2 * len(rows))
    scale = float(np.linalg.norm(sigma, 2) ** 2)
    matrices = np.full((len(gammas), n, n), np.nan)
    details = []
    for index, gamma in enumerate(gammas):
        best = None
        for restart, dense in enumerate(initial):
            branch_best = None
            seeds = [dense] if np.array_equal(warm[restart], dense) else [warm[restart], dense]
            for seed in seeds:
                result = minimize(
                    _objective_gradient, seed, args=(sigma, omega, gamma, rows, cols, scale),
                    jac=True, method="L-BFGS-B", bounds=bounds,
                    options=dict(maxiter=max_iter, ftol=1e-14, gtol=tol * .1, maxls=50),
                )
                value, gradient = _objective_gradient(result.x, sigma, omega, gamma, rows, cols, scale)
                pg = _projected_gradient(result.x, gradient, n, diagonal_bound)
                if np.isfinite(value) and np.isfinite(pg) and pg <= tol:
                    candidate = (value, result.x.copy(), pg, bool(result.success), int(result.nit))
                    if branch_best is None or value < branch_best[0]:
                        branch_best = candidate
            if branch_best is not None:
                warm[restart] = branch_best[1]
                if best is None or branch_best[0] < best[0]:
                    best = branch_best
        if best is None:
            details.append(dict(valid=False, penalized_objective=float("inf"),
                                projected_gradient=float("inf"), optimizer_success=False, iterations=0))
        else:
            value, x, pg, success, iterations = best
            matrices[index] = _decode(x, n, rows, cols)
            details.append(dict(valid=True, penalized_objective=value * scale,
                                projected_gradient=pg, optimizer_success=success, iterations=iterations))
    return matrices, details


def select_lasso(samples, gammas, *, fold_seed=20261013, kappa=.93,
                 zero_tol=1e-5, **fit_options):
    """Five-fold CV with uncentered covariance and unsquared Frobenius error."""
    validate_settings(kappa=kappa, zero_tol=zero_tol, **fit_options)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or len(samples) < 5 or not np.all(np.isfinite(samples)):
        raise ValueError("Lasso requires at least five finite observation rows.")
    gammas = np.asarray(gammas, dtype=float)
    folds = np.array_split(np.random.default_rng(fold_seed).permutation(len(samples)), 5)
    fold_ids = np.empty(len(samples), dtype=np.int8)
    total = samples.T @ samples
    scores = np.full((5, len(gammas)), np.inf)
    paths, diagnostics, omegas, train_covariances, validation_covariances = [], [], [], [], []
    for fold, indices in enumerate(folds):
        fold_ids[indices] = fold
        validation = samples[indices]
        moment = validation.T @ validation
        sigma_train = (total - moment) / (len(samples) - len(indices))
        sigma_validation = moment / len(indices)
        sigma_train = _covariance(sigma_train)
        omega = float(kappa * np.linalg.eigvalsh(sigma_train)[0])
        path, details = fit_lasso_path(sigma_train, omega, gammas, **fit_options)
        for index, (matrix, detail) in enumerate(zip(path, details)):
            if detail["valid"]:
                try:
                    covariance = solve_discrete_lyapunov(matrix.T, omega * np.eye(len(matrix)))
                    if np.all(np.isfinite(covariance)) and np.linalg.eigvalsh(covariance)[0] > 0:
                        scores[fold, index] = np.linalg.norm(sigma_validation - covariance, "fro")
                except np.linalg.LinAlgError:
                    pass
        paths.append(path)
        diagnostics.append(details)
        omegas.append(omega)
        train_covariances.append(sigma_train)
        validation_covariances.append(sigma_validation)
    sums = scores.sum(axis=0)
    if not np.any(np.isfinite(sums)):
        raise ValueError("No lasso gamma produced valid fits in all five folds.")
    selected = int(np.flatnonzero(sums == np.min(sums))[-1])
    sigma = _covariance(total / len(samples))
    omega = float(kappa * np.linalg.eigvalsh(sigma)[0])
    # Include the lower part of the path to retain the warm-start strategy.
    full_path, full_details = fit_lasso_path(sigma, omega, gammas[:selected + 1], **fit_options)
    matrix = full_path[-1]
    if not full_details[-1]["valid"]:
        raise ValueError("Selected lasso gamma failed its full-data refit.")
    mask = np.triu(np.abs(matrix) > zero_tol, 1)
    # Diagonals are always included in model masks; recovery metrics use edges.
    np.fill_diagonal(mask, True)
    covariance = solve_discrete_lyapunov(matrix.T, omega * np.eye(len(matrix)))
    if not np.all(np.isfinite(covariance)) or np.linalg.eigvalsh(covariance)[0] <= 0:
        raise ValueError("Selected lasso fit has no positive stationary covariance.")
    return dict(
        gamma_grid=gammas, selected_index=selected, gamma=float(gammas[selected]),
        fold_ids=fold_ids, cv_errors=scores, cv_error_sums=sums,
        cv_fitted_lambdas=np.array(paths), cv_fitted_omegas=np.array(omegas),
        cv_training_covariances=np.array(train_covariances),
        cv_validation_covariances=np.array(validation_covariances),
        fitted_lambda=matrix, fitted_omega=omega, fitted_covariance=covariance,
        support_mask=mask, support_edges=np.argwhere(np.triu(mask, 1)),
        objective_value=float(np.linalg.norm(sigma - matrix.T @ sigma @ matrix - omega * np.eye(len(matrix)), "fro") ** 2),
        penalized_objective_value=full_details[-1]["penalized_objective"],
        cv_fit_valid=np.array([[d["valid"] for d in fold] for fold in diagnostics]),
        cv_projected_gradients=np.array([[d["projected_gradient"] for d in fold] for fold in diagnostics]),
        full_projected_gradient=full_details[-1]["projected_gradient"],
    )
