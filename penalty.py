"""Penalty from Theorem 13, evaluated at equality in bound (7).

    pen_n(D_m) = sqrt(D_m) * (K_n1 + sqrt(2 * v_n * Lm))

The PDF's sample size n is ``num_samples`` here, and its matrix dimension N
is ``n``. Defaults follow page 2: R_n = log(num_samples) and
delta_n = log(2 * n * num_samples), using natural logarithms.
Only constants needed for the penalty are computed, not the risk remainder.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from math import isfinite, log, log1p, sqrt
from typing import Optional
import warnings

from scipy.integrate import quad


@dataclass(frozen=True)
class PenaltyConstants:
    """Inputs to the penalty; None selects the page 2 truncation defaults.

    ``sigma_trace`` is used only to check R_n**2 > Tr(Sigma). A violation
    emits a warning and the penalty is still evaluated with the given radius.
    """

    num_samples: int
    n: int
    sigma_trace: float
    Lm: float = 1.0
    L: float = 1.0
    r: float = 1.0
    R_n: Optional[float] = None
    delta_n: Optional[float] = None

    def with_overrides(self, **overrides: float) -> "PenaltyConstants":
        """Return a copy with selected inputs replaced."""

        return replace(self, **overrides)


def _constants_from_args(
    constants: Optional[PenaltyConstants],
    **overrides: float,
) -> PenaltyConstants:
    if constants is None:
        return PenaltyConstants(**overrides)
    if overrides:
        return constants.with_overrides(**overrides)
    return constants


def _radius(constants: PenaltyConstants) -> float:
    return log(constants.num_samples) if constants.R_n is None else constants.R_n


def _delta(constants: PenaltyConstants) -> float:
    # Sum logs to avoid an unnecessary product for large sample counts.
    return (log(2 * constants.n) + log(constants.num_samples)
            if constants.delta_n is None else constants.delta_n)


def validate_constants(constants: PenaltyConstants) -> None:
    """Validate inputs and warn if the radius condition is not satisfied."""

    if (not isfinite(constants.num_samples)
            or constants.num_samples != int(constants.num_samples)
            or constants.num_samples < 2):
        raise ValueError("num_samples must be an integer greater than 1.")
    if (not isfinite(constants.n) or constants.n != int(constants.n)
            or constants.n <= 0):
        raise ValueError("n must be a positive integer (matrix dimension).")
    for field_name in ("r", "L", "Lm"):
        value = getattr(constants, field_name)
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{field_name} must be finite and positive.")
    if constants.L >= sqrt(constants.n):
        raise ValueError("The theorem assumes 0 < L < sqrt(n), with n the matrix dimension.")
    if not isfinite(constants.sigma_trace) or constants.sigma_trace < 0:
        raise ValueError("sigma_trace must be finite and nonnegative.")

    R_n = _radius(constants)
    delta_n = _delta(constants)
    if not isfinite(R_n) or R_n <= 0:
        raise ValueError("R_n must be finite and positive.")
    if not isfinite(delta_n) or delta_n <= log(2 * constants.n):
        raise ValueError("delta_n must be finite and greater than log(2 * n).")
    if R_n**2 <= constants.sigma_trace:
        warnings.warn(
            "The theorem requires R_n**2 > Tr(Sigma), but "
            f"R_n**2={R_n**2:.12g} <= sigma_trace={constants.sigma_trace:.12g}. "
            "Computing the penalty with the specified radius; "
            "the theorem's radius condition is not satisfied.",
            RuntimeWarning,
            stacklevel=3,
        )


def compute_R(constants: Optional[PenaltyConstants] = None, **overrides: float) -> float:
    """Return R_n, defaulting to log(num_samples)."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)
    return _radius(constants)


def compute_delta(constants: Optional[PenaltyConstants] = None, **overrides: float) -> float:
    """Return delta_n, defaulting to log(2 * n * num_samples)."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)
    return _delta(constants)


def _c(constants: PenaltyConstants, R_n: float) -> float:
    num_samples = constants.num_samples
    L = constants.L
    return (
        16.0 / num_samples * (1.0 + 1.0 / num_samples) * L**2 * (1.0 + L**2) * R_n**4
        + 8.0 / num_samples * sqrt(constants.n) * (1.0 + 2.0 * L**2) * constants.r * R_n**2
    )


def compute_c(constants: Optional[PenaltyConstants] = None, **overrides: float) -> float:
    """Compute the bounded-differences constant c_n."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)
    return _c(constants, _radius(constants))


def compute_v(constants: Optional[PenaltyConstants] = None, **overrides: float) -> float:
    """Compute v_n = num_samples * c_n**2 / 4."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)
    return constants.num_samples * _c(constants, _radius(constants))**2 / 4.0


def _t(constants: PenaltyConstants, R_n: float, delta_n: float) -> float:
    first_term = 2.0 * delta_n / (3.0 * constants.num_samples)
    return R_n**2 * (
        first_term + sqrt(first_term**2 + 2.0 * delta_n / constants.num_samples)
    )


def compute_t(constants: Optional[PenaltyConstants] = None, **overrides: float) -> float:
    """Compute the covariance concentration constant t_bar(delta_n)."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)
    return _t(constants, _radius(constants), _delta(constants))


def _L_prime(constants: PenaltyConstants, R_n: float, t_bar: float) -> float:
    L = constants.L
    return (
        4.0 * t_bar * L * (4.0 * R_n**2 * (1.0 + L**2) + constants.r)
        + 2.0 * constants.n * t_bar * (1.0 + L**2)
    )


def compute_L_prime(
    constants: Optional[PenaltyConstants] = None,
    **overrides: float,
) -> float:
    """Compute the chaining Lipschitz constant L'_n."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)
    R_n = _radius(constants)
    return _L_prime(constants, R_n, _t(constants, R_n, _delta(constants)))


def _entropy_integrand(u: float) -> float:
    if u == 0.0:
        return float("inf")
    return sqrt(log1p(1.0 / u))


@lru_cache(maxsize=1)
def _entropy_integral() -> float:
    # This integral is independent of all model and sample parameters.
    integral, _ = quad(_entropy_integrand, 0.0, 0.25, points=[0.0])
    return integral


def compute_K(constants: Optional[PenaltyConstants] = None, **overrides: float) -> float:
    """Compute K_{n,1} = 48 * L'_n * (L + r) * integral_0^{1/4}."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)
    R_n = _radius(constants)
    L_prime = _L_prime(constants, R_n, _t(constants, R_n, _delta(constants)))
    return 48.0 * L_prime * (constants.L + constants.r) * _entropy_integral()


def pen_n(
    D_m: float,
    constants: Optional[PenaltyConstants] = None,
    **overrides: float,
) -> float:
    """Evaluate equality in penalty bound (7) for model dimension D_m."""

    if not isfinite(D_m) or D_m < 0:
        raise ValueError("D_m must be finite and nonnegative.")

    constants = _constants_from_args(constants, **overrides)
    derived = theorem_constants(constants)
    return sqrt(D_m) * (derived["K"] + sqrt(2.0 * derived["v"] * constants.Lm))


def theorem_constants(
    constants: Optional[PenaltyConstants] = None,
    **overrides: float,
) -> dict[str, float]:
    """Return penalty constants R_n, delta_n, c_n, v_n, t_bar, L'_n, K_{n,1}.

    The short dictionary keys R, delta, c, v, t, L_prime, K correspond to
    those quantities, respectively. No risk remainder is computed.
    """

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)
    R_n = _radius(constants)
    delta_n = _delta(constants)
    c_n = _c(constants, R_n)
    t_bar = _t(constants, R_n, delta_n)
    L_prime = _L_prime(constants, R_n, t_bar)
    return {
        "R": R_n,
        "delta": delta_n,
        "c": c_n,
        "v": constants.num_samples * c_n**2 / 4.0,
        "t": t_bar,
        "L_prime": L_prime,
        "K": 48.0 * L_prime * (constants.L + constants.r) * _entropy_integral(),
    }
