"""Exhaustive upper-support Gaussian likelihood selection for small matrices.

Omega is profiled exactly for each Lambda, so both parameters are fitted.
Support masks, rather than thresholded coefficients, determine the penalty.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import itertools
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, cho_factor, cho_solve, solve_discrete_lyapunov
from scipy.optimize import minimize


def validate_settings(*, diagonal_bound=.999999, max_restarts=10,
                      max_iter=1500, tol=1e-6):
    if not np.isfinite(diagonal_bound) or not 0 < diagonal_bound < 1:
        raise ValueError("BIC diagonal_bound must be in (0, 1).")
    for name, value in (("max_restarts", max_restarts), ("max_iter", max_iter)):
        if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f"BIC {name} must be a positive integer.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("BIC tol must be finite and positive.")


def validate_covariance(sigma):
    sigma = np.asarray(sigma, dtype=float)
    if (sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]
            or not 2 <= len(sigma) <= 4 or not np.all(np.isfinite(sigma))
            or not np.allclose(sigma, sigma.T, rtol=1e-12, atol=1e-12)):
        raise ValueError("BIC requires a finite symmetric covariance with 2 <= N <= 4.")
    sigma = (sigma + sigma.T) / 2
    if np.linalg.eigvalsh(sigma)[0] <= 0:
        raise ValueError("BIC requires a positive-definite empirical covariance.")
    return sigma


def upper_support_masks(n):
    """Enumerate by edge count, then lexicographic edge order, including k=0."""
    edges = list(zip(*np.triu_indices(n, 1)))
    for count in range(len(edges) + 1):
        for support in itertools.combinations(edges, count):
            mask = np.eye(n, dtype=bool)
            for i, j in support:
                mask[i, j] = True
            yield mask


def _decode(x, n, rows, cols):
    matrix = np.diag(x[:n])
    matrix[rows, cols] = x[n:]
    return matrix


def _profile(matrix, sigma):
    """Return logdet(C)+tr(C^-1 S), fitted omega, and unit-noise covariance."""
    unit = solve_discrete_lyapunov(matrix.T, np.eye(len(matrix)), method="direct")
    unit = (unit + unit.T) / 2
    factor = cho_factor(unit, lower=True, check_finite=True)
    inverse = cho_solve(factor, np.eye(len(matrix)))
    omega = float(np.sum(inverse * sigma) / len(matrix))
    if not np.isfinite(omega) or omega <= 0:
        raise np.linalg.LinAlgError("No positive profiled omega.")
    logdet = 2 * np.log(np.diag(factor[0])).sum()
    value = float(logdet + len(matrix) * (np.log(omega) + 1))
    return value, omega, unit, inverse


def _objective_gradient(x, sigma, rows, cols):
    matrix = _decode(x, len(sigma), rows, cols)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            value, omega, unit, inverse = _profile(matrix, sigma)
            # Adjoint Lyapunov differentiation; the envelope theorem eliminates
            # the derivative of omega at its analytic likelihood optimum.
            covariance_gradient = inverse - inverse @ sigma @ inverse / omega
            adjoint = solve_discrete_lyapunov(matrix, covariance_gradient, method="direct")
        adjoint = (adjoint + adjoint.T) / 2
        gradient = 2 * unit @ matrix @ adjoint
        derivative = np.r_[np.diag(gradient), gradient[rows, cols]]
        if np.isfinite(value) and np.all(np.isfinite(derivative)):
            return value, derivative
    except (np.linalg.LinAlgError, LinAlgWarning, ValueError, FloatingPointError):
        pass
    return float("inf"), np.zeros_like(x)


def _optimization_decode(y, n, rows, cols):
    diagonal = np.tanh(y[:n])
    return np.r_[diagonal, (1 - diagonal[rows] * diagonal[cols]) * y[n:]]


def _optimization_gradient(y, sigma, rows, cols):
    """An invertible coordinate change improves conditioning near +/-1.

    Lambda_ii=tanh(y_i), Lambda_ij=(1-Lambda_ii*Lambda_jj)*y_ij.
    Edge coordinates remain unbounded and signed; this imposes no extra
    statistical constraints. Convergence is measured in these coordinates.
    """
    n = len(sigma)
    x = _optimization_decode(y, n, rows, cols)
    value, gradient = _objective_gradient(x, sigma, rows, cols)
    diagonal_gradient = gradient[:n].copy()
    np.add.at(diagonal_gradient, rows, -x[cols] * y[n:] * gradient[n:])
    np.add.at(diagonal_gradient, cols, -x[rows] * y[n:] * gradient[n:])
    return value, np.r_[diagonal_gradient * (1 - x[:n] ** 2),
                        gradient[n:] * (1 - x[rows] * x[cols])]


def _initial_states(sigma, mask, bound, max_restarts):
    n = len(sigma)
    rows, cols = np.where(np.triu(mask, 1))
    signs = [np.r_[s, 1.] for s in itertools.product((-1., 1.), repeat=n - 1)]
    r = np.linalg.cholesky(sigma).T
    minimum = np.linalg.eigvalsh(sigma)[0]
    for restart in range(max_restarts):
        # Dense covariance-factor starts cover all diagonal signs modulo the
        # exact global sign symmetry. Additional starts change the noise scale.
        fraction = (.5, .9, .1)[(restart // len(signs)) % 3]
        c = np.linalg.cholesky(sigma - fraction * minimum * np.eye(n)).T
        matrix = np.linalg.solve(r, np.diag(signs[restart % len(signs)]) @ c)
        yield np.r_[np.clip(np.diag(matrix), -bound, bound), matrix[rows, cols]]


def fit_support(sigma, mask, *, diagonal_bound=.999999, max_restarts=10,
                max_iter=1500, tol=1e-6):
    """Fit one support with deterministic restarts and gradient validation."""
    validate_settings(diagonal_bound=diagonal_bound, max_restarts=max_restarts,
                      max_iter=max_iter, tol=tol)
    sigma = validate_covariance(sigma)
    mask = np.asarray(mask, dtype=bool)
    if (mask.shape != sigma.shape or not np.all(np.diag(mask))
            or np.any(np.tril(mask, -1))):
        raise ValueError("BIC support must be upper triangular and include all diagonals.")
    n = len(sigma)
    rows, cols = np.where(np.triu(mask, 1))
    coordinate_bound = np.arctanh(diagonal_bound)
    bounds = [(-coordinate_bound, coordinate_bound)] * n + [(None, None)] * len(rows)
    best = None
    valid_restarts = 0
    for start in _initial_states(sigma, mask, diagonal_bound, max_restarts):
        start = np.r_[np.arctanh(start[:n]),
                      start[n:] / (1 - start[rows] * start[cols])]
        result = minimize(
            _optimization_gradient, start, args=(sigma, rows, cols), jac=True,
            method="L-BFGS-B", bounds=bounds,
            options=dict(maxiter=max_iter, ftol=1e-14, gtol=tol * .1, maxls=50),
        )
        value, gradient = _optimization_gradient(result.x, sigma, rows, cols)
        projected = gradient.copy()
        projected[:n] = result.x[:n] - np.clip(
            result.x[:n] - gradient[:n], -coordinate_bound, coordinate_bound)
        pg = float(np.max(np.abs(projected)))
        if not np.isfinite(value) or not np.isfinite(pg) or pg > tol:
            continue
        matrix = _decode(_optimization_decode(result.x, n, rows, cols), n, rows, cols)
        try:
            value, omega, unit, _ = _profile(matrix, sigma)
            covariance = omega * unit
            np.linalg.cholesky(covariance)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.all(np.isfinite(covariance)):
            continue
        valid_restarts += 1
        if best is None or value < best["objective_value"]:
            best = dict(fitted_lambda=matrix, fitted_omega=omega,
                        fitted_covariance=covariance, objective_value=value,
                        projected_gradient=pg, optimizer_success=bool(result.success),
                        iterations=int(result.nit))
    if best is None:
        raise ValueError("No converged BIC likelihood fit for support "
                         f"{list(zip(rows.tolist(), cols.tolist()))}.")
    best["valid_restarts"] = valid_restarts
    return best


def select_bic(sigma, num_samples, *, n_jobs=1, **fit_options):
    """Fit upper supports in parallel; an unresolved support fails the trial."""
    validate_settings(**fit_options)
    if (isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer))
            or n_jobs < 1):
        raise ValueError("BIC n_jobs must be a positive integer.")
    sigma = validate_covariance(sigma)
    if (isinstance(num_samples, bool) or not np.isfinite(num_samples)
            or int(num_samples) != num_samples or num_samples < len(sigma)):
        raise ValueError("BIC num_samples must be an integer >= N.")
    masks = np.array(list(upper_support_masks(len(sigma))))
    if n_jobs == 1:
        fits = [fit_support(sigma, mask, **fit_options) for mask in masks]
    else:
        with ProcessPoolExecutor(max_workers=min(int(n_jobs), len(masks))) as executor:
            # map preserves mask order regardless of completion order, keeping
            # candidate arrays and the exact BIC tie rule deterministic.
            fits = list(executor.map(partial(fit_support, sigma, **fit_options), masks))
    counts = np.count_nonzero(masks, axis=(1, 2)) - len(sigma)
    losses = np.array([fit["objective_value"] for fit in fits])
    penalties = counts * np.log(num_samples)
    constant = num_samples * len(sigma) * np.log(2 * np.pi)
    reduced_values = num_samples * losses + penalties
    # Masks are ordered by k, then lexicographic edges: argmin gives the
    # agreed deterministic tie rule, without numerical tie tolerances.
    selected = int(np.argmin(reduced_values))
    fit = fits[selected]
    return dict(
        selected_index=selected, dimension=int(counts[selected] + 1),
        edge_count=int(counts[selected]), support_mask=masks[selected],
        support_edges=np.argwhere(np.triu(masks[selected], 1)),
        fitted_lambda=fit["fitted_lambda"], fitted_omega=fit["fitted_omega"],
        fitted_covariance=fit["fitted_covariance"], objective_value=losses[selected],
        log_likelihood=-.5 * (num_samples * losses[selected] + constant),
        penalty_value=penalties[selected], criterion_value=reduced_values[selected] + constant,
        reduced_criterion_value=reduced_values[selected],
        candidate_support_masks=masks, candidate_edge_counts=counts,
        candidate_dimensions=counts + 1, candidate_objective_values=losses,
        candidate_log_likelihoods=-.5 * (num_samples * losses + constant),
        candidate_penalties=penalties, candidate_criterion_values=reduced_values + constant,
        candidate_reduced_criterion_values=reduced_values,
        candidate_fitted_lambdas=np.array([f["fitted_lambda"] for f in fits]),
        candidate_fitted_omegas=np.array([f["fitted_omega"] for f in fits]),
        candidate_fitted_covariances=np.array([f["fitted_covariance"] for f in fits]),
        candidate_projected_gradients=np.array([f["projected_gradient"] for f in fits]),
        candidate_optimizer_success=np.array([f["optimizer_success"] for f in fits]),
        candidate_iterations=np.array([f["iterations"] for f in fits]),
        candidate_valid_restarts=np.array([f["valid_restarts"] for f in fits]),
        candidate_fit_valid=np.ones(len(fits), dtype=bool),
    )
