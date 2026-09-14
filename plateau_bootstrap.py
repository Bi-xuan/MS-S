"""Select among bounded plateaus by sequential, fixed-support bootstrap tests."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import wishart

from optimizers.support_search import solve_support_with_restarts
from scaling_selection import build_dimension_path, ranked_plateaus
from supports.common import off_diagonal_edges, validate_support_mask


@dataclass(frozen=True)
class FitSettings:
    """Settings of the final fit on each selected model mask."""

    omega_fixed: float | None = 1.0
    beta: float = 1.0
    max_iter: int = 800
    tol: float = 1e-7
    zero_tol: float = 1e-5
    max_restarts: int = 10
    min_omega: float = 0.0
    init_strategy: str = "halton"


@dataclass(frozen=True)
class BootstrapComparison:
    smaller_dimension: int
    larger_dimension: int
    observed_gain: float
    exceedances: int
    p_value: float
    retain_larger: bool
    bootstrap_gains: tuple[float, ...]
    null_spectral_radius: float


@dataclass(frozen=True)
class BootstrapSelection:
    selected_dimension: int
    candidate_dimensions: tuple[int, ...]
    plateaus: tuple
    comparisons: tuple[BootstrapComparison, ...]
    selected_edges: tuple[tuple[int, int], ...]
    top_plateaus: int
    bootstrap_replicates: int
    alpha: float
    seed: int
    num_samples: int
    fit_settings: FitSettings
    precision: float | None
    method: str = "plateau-bootstrap"


def _fit(Sigma, mask, settings, *, bootstrap=False):
    options = asdict(settings)
    # Fixed omega is part of the generating model. Sampling may put the
    # empirical minimum eigenvalue below it; do not discard those draws.
    upper = None if bootstrap and settings.omega_fixed is not None else (
        float(np.linalg.eigvalsh(Sigma)[0]) - 1e-6
    )
    result = solve_support_with_restarts(Sigma, mask, omega_upper=upper, **options)
    if result is None:
        raise ValueError("No finite feasible support refit; bootstrap selection aborted.")
    return result


def _bootstrap_gain(task):
    Sigma, smaller, larger, settings = task
    small_fit = _fit(Sigma, smaller, settings, bootstrap=True)
    large_fit = _fit(Sigma, larger, settings, bootstrap=True)
    # The smaller fit is also feasible on the larger mask. This protects
    # against a worse local optimizer solution without changing supports.
    return float(max(0.0, small_fit[2] - large_fit[2]))


def _null_covariance(fit):
    Lambda, omega, _ = fit
    radius = float(np.max(np.abs(np.linalg.eigvals(Lambda))))
    if not np.isfinite(radius) or radius >= 1 or not np.isfinite(omega) or omega <= 0:
        raise ValueError("The smaller fit must have positive omega and spectral radius < 1.")
    Sigma = solve_discrete_lyapunov(Lambda.T, omega * np.eye(len(Lambda)))
    Sigma = (Sigma + Sigma.T) / 2
    if not np.all(np.isfinite(Sigma)):
        raise ValueError("The smaller fit has no finite stationary covariance.")
    np.linalg.cholesky(Sigma)
    return Sigma, radius


def select_plateau_bootstrap(
    dimensions, objectives, penalties, raw_objectives, Sigma, support_masks,
    support_valid, num_samples, *, top_plateaus=3, bootstrap_replicates=199,
    alpha=0.05, seed=20260913, n_jobs=1, fit_settings=None,
    true_support=None, progress=None,
):
    """Screen top log-width plateaus, then test adjacent descending dimensions.

    Screening uses the supplied (possibly floored) objectives. Tests use raw
    objectives, refitting both original masks for every draw. Gaussian samples
    have known zero mean: Wishart(N, Sigma)/N is exactly X.T @ X/N. For N < n,
    draw X explicitly because the Wishart sampler requires N >= n.
    """
    for name, value in (("top_plateaus", top_plateaus),
                        ("bootstrap_replicates", bootstrap_replicates),
                        ("num_samples", num_samples), ("n_jobs", n_jobs)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    settings = fit_settings or FitSettings()
    if settings.init_strategy != "halton":
        raise ValueError("Bootstrap refitting currently requires deterministic Halton starts.")
    if settings.max_restarts < 1 or settings.max_iter < 1 or settings.beta <= 0:
        raise ValueError("Fit iteration/restart counts and beta must be positive.")
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1] or not np.all(np.isfinite(Sigma)):
        raise ValueError("Sigma must be a finite square matrix.")
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Sigma must be symmetric.")
    n = len(Sigma)
    dims = np.asarray(dimensions)
    masks = np.asarray(support_masks, dtype=bool)
    valid = np.asarray(support_valid, dtype=bool)
    raw = np.asarray(raw_objectives, dtype=float)
    if masks.shape != (len(dims), n, n) or valid.shape != dims.shape or raw.shape != dims.shape:
        raise ValueError("Support masks, validity flags and raw objectives must match dimensions.")
    path = build_dimension_path(dims, objectives, penalties)
    plateaus = tuple(ranked_plateaus(path)[:top_plateaus])
    if not plateaus:
        raise ValueError("At least one bounded plateau is required for bootstrap selection.")
    candidates = tuple(sorted((int(p.dimension) for p in plateaus), reverse=True))
    indices = {int(d): i for i, d in enumerate(dims)}
    for d in candidates:
        i = indices[d]
        if not valid[i] or not np.isfinite(raw[i]):
            raise ValueError(f"Missing valid support/objective at dimension {d}.")
        validate_support_mask(masks[i], n, d - 1, off_diagonal_edges(n))
    for larger, smaller in zip(candidates, candidates[1:]):
        if np.any(masks[indices[smaller]] & ~masks[indices[larger]]):
            raise ValueError("Plateau candidates must be nested; recompute with --nested-supports true.")

    fits = {}
    comparisons = []
    selected = candidates[-1]
    # One stream per dimension pair: reproducible independent of worker count,
    # and the same pair uses the same draws when top_plateaus changes.
    for larger, smaller in zip(candidates, candidates[1:]):
        if progress:
            progress(f"Bootstrap comparison D_m={larger} versus {smaller} ({bootstrap_replicates} replicates)")
        for d in (smaller, larger):
            if d not in fits:
                fits[d] = _fit(Sigma, masks[indices[d]], settings)
                if not np.isclose(fits[d][2], raw[indices[d]], rtol=1e-5, atol=1e-10):
                    raise ValueError(
                        f"Refitted objective at D_m={d} differs from the saved curve; "
                        "use matching fitting settings or regenerate the curve."
                    )
        observed = float(max(0.0, raw[indices[smaller]] - raw[indices[larger]]))
        null_sigma, radius = _null_covariance(fits[smaller])
        rng = np.random.default_rng(np.random.SeedSequence([seed, smaller, larger]))
        if num_samples >= n:
            samples = np.asarray(wishart.rvs(
                df=num_samples, scale=null_sigma, size=bootstrap_replicates, random_state=rng,
            )).reshape(bootstrap_replicates, n, n) / num_samples
        else:
            x = rng.multivariate_normal(np.zeros(n), null_sigma, size=(bootstrap_replicates, num_samples))
            samples = x.swapaxes(-1, -2) @ x / num_samples
        tasks = [(s, masks[indices[smaller]], masks[indices[larger]], settings) for s in samples]
        if n_jobs == 1:
            gains = tuple(map(_bootstrap_gain, tasks))
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                gains = tuple(executor.map(_bootstrap_gain, tasks))
        exceedances = int(np.count_nonzero(np.asarray(gains) >= observed))
        p_value = (1 + exceedances) / (bootstrap_replicates + 1)
        retain = p_value < alpha
        comparisons.append(BootstrapComparison(
            smaller, larger, observed, exceedances, p_value, retain, gains, radius,
        ))
        if progress:
            progress(f"p={p_value:.6g}: {'retain larger' if retain else 'prefer smaller'}")
        if retain:
            selected = larger
            break

    selected_mask = masks[indices[selected]].copy()
    np.fill_diagonal(selected_mask, False)
    edges = tuple((int(i), int(j)) for i, j in np.argwhere(selected_mask))
    precision = None
    if true_support is not None:
        truth = np.asarray(true_support, dtype=bool)
        if truth.shape != (n, n):
            raise ValueError("true_support must match Sigma's shape.")
        precision = float(np.count_nonzero(truth & selected_mask) / len(edges)) if edges else None
    return BootstrapSelection(
        selected, candidates, plateaus, tuple(comparisons), edges, top_plateaus,
        bootstrap_replicates, float(alpha), int(seed), int(num_samples), settings, precision,
    )
