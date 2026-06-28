"""Penalty utilities for theorem-based model selection.

The implemented lower-bound penalty is

    pen_n(D_m) = sqrt(D_m) * (K + sqrt(2 * v * L_m)).

Whenever the theorem gives an inequality, this module uses the lower bound.
In particular, R is set to the lower admissible value and the penalty is set
to the lower admissible value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import log, log1p, sqrt
from typing import Optional

from scipy.integrate import quad


@dataclass(frozen=True)
class PenaltyConstants:
    """Constants entering the theorem penalty.

    Defaults follow the analysis convention:
        Lm = L = 1, xi = 10.

    The remaining constants must come from the problem instance or from the
    theorem quantities being analyzed.
    """

    num_samples: int
    n: int
    r: float
    lambda_sum: float
    lambda_inf_norm: float
    lambda_2_norm: float
    sigma_fro_norm: float
    sigma_op_norm: float
    sigma_trace: float
    Lm: float = 1.0
    L: float = 1.0
    xi: float = 10.0

    def with_overrides(self, **overrides: float) -> "PenaltyConstants":
        """Return a copy with selected constants replaced."""

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


def validate_constants(constants: PenaltyConstants) -> None:
    """Validate theorem constants before evaluating the penalty."""

    if constants.num_samples <= 1:
        raise ValueError("num_samples must be greater than 1.")
    if constants.n <= 0:
        raise ValueError("n must be positive.")
    if constants.r <= 0:
        raise ValueError("r must be positive.")
    if constants.L <= 0:
        raise ValueError("L must be positive.")
    if constants.L >= sqrt(constants.n):
        raise ValueError("The theorem assumes 0 < L < sqrt(n).")
    if constants.Lm <= 0:
        raise ValueError("Lm must be positive.")
    if constants.xi <= 0:
        raise ValueError("xi must be positive.")

    nonnegative_fields = (
        "lambda_sum",
        "lambda_inf_norm",
        "lambda_2_norm",
        "sigma_fro_norm",
        "sigma_op_norm",
        "sigma_trace",
    )
    for field_name in nonnegative_fields:
        if getattr(constants, field_name) < 0:
            raise ValueError(f"{field_name} must be nonnegative.")


def compute_R(constants: Optional[PenaltyConstants] = None, **overrides: float) -> float:
    """Compute R using the theorem lower bound."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)

    log_term = log(constants.num_samples) + constants.xi
    return sqrt(
        constants.lambda_sum + 2.0 * constants.lambda_inf_norm * log_term
     + 2.0 * constants.lambda_2_norm * sqrt(log_term))


def compute_c(constants: Optional[PenaltyConstants] = None, **overrides: float) -> float:
    """Compute the constant c from the theorem."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)

    num_samples = constants.num_samples
    n = constants.n
    L = constants.L
    r = constants.r
    R = compute_R(constants)

    first_factor = 8.0 / (num_samples - 1.0) + 4.0 / num_samples
    second_factor = (
        8.0 * num_samples / (num_samples - 1.0)
        + 8.0 / (num_samples - 1.0)
        + 4.0 / num_samples
    )

    return (
        first_factor
        * second_factor
        * 4.0
        * L**2
        * (1.0 + L**2)
        * R**4
        + 2.0
        * sqrt(n)
        * first_factor
        * (2.0 + 4.0 * L**2)
        * r
        * R**2
    )


def compute_v(constants: Optional[PenaltyConstants] = None, **overrides: float) -> float:
    """Compute v = num_samples * c^2 / 2."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)
    c = compute_c(constants)
    return constants.num_samples * c**2 / 2.0


def compute_L_prime(
    constants: Optional[PenaltyConstants] = None,
    **overrides: float,
) -> float:
    """Compute L' from the theorem."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)

    num_samples = constants.num_samples
    L = constants.L
    r = constants.r
    R = compute_R(constants)
    sigma_fro_sq = constants.sigma_fro_norm**2
    sigma_op = constants.sigma_op_norm
    sigma_op_sq = sigma_op**2

    return (
        32.0
        * (
            num_samples**2 / (num_samples - 1.0) ** 2 * R**4
            + num_samples / (num_samples - 1.0) * sigma_op_sq
        )
        * L**3
        + 8.0
        * (num_samples / (num_samples - 1.0) * R**2 + sigma_op)
        * L**2
        + 8.0
        * (
            num_samples**2 / (num_samples - 1.0) ** 2 * R**4
            + 2.0 * r * num_samples / (num_samples - 1.0) * R**2
            + sigma_op_sq
            + 2.0 * r * sigma_op
            + (sigma_fro_sq + sigma_op_sq) / (num_samples - 1.0)
        )
        * L
        + 2.0
        * (
            num_samples / (num_samples - 1.0) * R**2
            + abs(constants.sigma_trace)
        )
    )


def _entropy_integrand(u: float) -> float:
    if u == 0.0:
        return float("inf")
    return sqrt(log1p(1.0 / u))


def compute_K(constants: Optional[PenaltyConstants] = None, **overrides: float) -> float:
    """Compute K with scipy.integrate.quad for the entropy integral."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)

    L_prime = compute_L_prime(constants)
    upper_bound = L_prime / 4.0
    integral, _ = quad(_entropy_integrand, 0.0, upper_bound, points=[0.0])
    return 96.0 * L_prime * (constants.L + constants.r) * integral


def compute_B(constants: Optional[PenaltyConstants] = None, **overrides: float) -> float:
    """Compute B from the theorem."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)

    L = constants.L
    return (
        constants.sigma_fro_norm**2 * (1.0 + L**4)
        + constants.sigma_op_norm**2 * L**4
        + constants.sigma_trace**2
    )


def pen_n(
    D_m: float,
    constants: Optional[PenaltyConstants] = None,
    **overrides: float,
) -> float:
    """Evaluate the lower-bound theorem penalty for model dimension D_m."""

    if D_m < 0:
        raise ValueError("D_m must be nonnegative.")

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)

    K = compute_K(constants)
    v = compute_v(constants)
    return sqrt(D_m) * (K + sqrt(2.0 * v * constants.Lm))


def theorem_constants(
    constants: Optional[PenaltyConstants] = None,
    **overrides: float,
) -> dict[str, float]:
    """Return all derived constants used by the penalty."""

    constants = _constants_from_args(constants, **overrides)
    validate_constants(constants)

    R = compute_R(constants)
    c = compute_c(constants)
    v = constants.num_samples * c**2 / 2.0
    L_prime = compute_L_prime(constants)
    K = compute_K(constants)
    B = compute_B(constants)

    return {
        "R": R,
        "c": c,
        "v": v,
        "L_prime": L_prime,
        "K": K,
        "B": B,
    }
