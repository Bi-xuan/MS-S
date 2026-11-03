"""Regression tests for the penalty in Theorem 13, bound (7)."""

from dataclasses import asdict
from math import isfinite, log, sqrt
from pathlib import Path
import sys
import warnings

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from penalty import (
    PenaltyConstants,
    compute_K,
    compute_L_prime,
    compute_R,
    compute_c,
    compute_delta,
    compute_t,
    compute_v,
    pen_n,
    theorem_constants,
)


# Integral_0^{1/4} sqrt(log(1 + 1/u)) du, independently evaluated after
# substituting u = exp(-s) and integrating from log(4) to infinity.
ENTROPY_INTEGRAL = 0.3898133333427541


@pytest.fixture
def constants():
    return PenaltyConstants(
        num_samples=100, n=5, sigma_trace=10.0, Lm=1.5, L=0.75,
    )


def reference_constants(constants):
    """Translate the PDF directly, separately from the production helpers."""
    sample_count = constants.num_samples
    dimension = constants.n
    R = log(sample_count) if constants.R_n is None else constants.R_n
    delta = log(2 * dimension * sample_count) if constants.delta_n is None else constants.delta_n
    L, r = constants.L, constants.r
    c = (
        16 / sample_count * (1 + 1 / sample_count) * L**2 * (1 + L**2) * R**4
        + 8 / sample_count * sqrt(dimension) * (1 + 2 * L**2) * r * R**2
    )
    v = sample_count * c**2 / 4
    t = R**2 * (
        2 * delta / (3 * sample_count)
        + sqrt((2 * delta / (3 * sample_count))**2 + 2 * delta / sample_count)
    )
    L_prime = 4 * t * L * (4 * R**2 * (1 + L**2) + r) + 2 * dimension * t * (1 + L**2)
    K = 48 * L_prime * (L + r) * ENTROPY_INTEGRAL
    return dict(R=R, delta=delta, c=c, v=v, t=t, L_prime=L_prime, K=K)


@pytest.mark.parametrize("overrides", [
    {}, {"R_n": 7.0}, {"delta_n": 9.0},
    {"R_n": 7.0, "delta_n": 9.0, "r": 0.4, "L": 0.5},
])
def test_theorem_penalty_formulas(constants, overrides):
    constants = constants.with_overrides(**overrides)
    expected = reference_constants(constants)
    assert constants.r == overrides.get("r", 1.0)
    assert compute_R(constants) == pytest.approx(expected["R"])
    assert compute_delta(constants) == pytest.approx(expected["delta"])
    assert compute_c(constants) == pytest.approx(expected["c"])
    assert compute_v(constants) == pytest.approx(expected["v"])
    assert compute_t(constants) == pytest.approx(expected["t"])
    assert compute_L_prime(constants) == pytest.approx(expected["L_prime"])
    assert compute_K(constants) == pytest.approx(expected["K"], rel=1e-10)
    assert theorem_constants(constants) == pytest.approx(expected, rel=1e-10)
    assert pen_n(3, constants) == pytest.approx(
        sqrt(3) * (expected["K"] + sqrt(2 * expected["v"] * constants.Lm)),
        rel=1e-10,
    )


def test_keyword_inputs_and_per_call_overrides(constants):
    assert pen_n(3, **asdict(constants)) == pytest.approx(pen_n(3, constants))
    overrides = dict(R_n=8.0, delta_n=12.0, Lm=3.0)
    assert pen_n(3, constants, **overrides) == pytest.approx(
        pen_n(3, constants.with_overrides(**overrides))
    )
    assert constants.R_n is None
    assert constants.delta_n is None
    assert constants.Lm == 1.5


def test_default_truncation_parameters_follow_sample_count(constants):
    changed = constants.with_overrides(num_samples=200)
    assert compute_R(changed) == pytest.approx(log(200))
    assert compute_delta(changed) == pytest.approx(log(2 * constants.n * 200))
    # A covariance-scale change cannot alter the page 2 defaults.
    assert pen_n(3, constants, sigma_trace=1) == pen_n(3, constants)


@pytest.mark.parametrize("overrides", [
    {"sigma_trace": 100.0},
    {"R_n": 2.0},
    {"R_n": 2.0, "sigma_trace": 4.0},
])
def test_invalid_radius_condition_warns_once_and_still_computes(constants, overrides):
    constants = constants.with_overrides(**overrides)
    expected = reference_constants(constants)
    with pytest.warns(RuntimeWarning, match=r"R_n\*\*2 > Tr\(Sigma\)") as recorded:
        result = pen_n(3, constants)
    assert len(recorded) == 1
    assert isfinite(result)
    assert result == pytest.approx(
        sqrt(3) * (expected["K"] + sqrt(2 * expected["v"] * constants.Lm))
    )


def test_valid_radius_override_does_not_warn(constants):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isfinite(pen_n(3, constants, R_n=11.0, sigma_trace=100.0))


def test_minimum_sample_count_still_computes_with_radius_warning(constants):
    with pytest.warns(RuntimeWarning):
        assert isfinite(pen_n(3, constants, num_samples=2))


@pytest.mark.parametrize("overrides, message", [
    ({"num_samples": 1}, "num_samples"),
    ({"num_samples": 2.5}, "num_samples"),
    ({"num_samples": float("nan")}, "num_samples"),
    ({"n": 0}, "n must"),
    ({"n": 2.5}, "n must"),
    ({"R_n": 0.0}, "R_n"),
    ({"R_n": float("inf")}, "R_n"),
    ({"R_n": float("nan")}, "R_n"),
    ({"delta_n": log(10)}, "delta_n"),
    ({"delta_n": 0.0}, "delta_n"),
    ({"delta_n": float("inf")}, "delta_n"),
    ({"delta_n": float("nan")}, "delta_n"),
    ({"sigma_trace": -1.0}, "sigma_trace"),
    ({"sigma_trace": float("nan")}, "sigma_trace"),
    ({"Lm": 0.0}, "Lm"),
    ({"Lm": float("nan")}, "Lm"),
    ({"r": 0.0}, "r must"),
    ({"L": 0.0}, "L must"),
    ({"L": sqrt(5)}, "0 < L < sqrt"),
])
def test_invalid_constants_are_rejected(constants, overrides, message):
    with pytest.raises(ValueError, match=message):
        pen_n(3, constants, **overrides)


@pytest.mark.parametrize("dimension", [-1.0, float("nan"), float("inf")])
def test_invalid_dimension_is_rejected(constants, dimension):
    with pytest.raises(ValueError, match="D_m"):
        pen_n(dimension, constants)


def test_dimension_and_weight_dependence(constants):
    assert pen_n(0, constants) == 0.0
    assert pen_n(9, constants) == pytest.approx(3 * pen_n(1, constants))
    base = theorem_constants(constants)
    assert pen_n(3, constants, Lm=4 * constants.Lm) == pytest.approx(
        sqrt(3) * (base["K"] + 2 * sqrt(2 * base["v"] * constants.Lm))
    )


def test_penalty_and_variance_decay_at_large_sample_sizes(constants):
    sample_sizes = [10**8, 10**10, 10**12]
    derived = [theorem_constants(constants, num_samples=n) for n in sample_sizes]
    penalties = [pen_n(3, constants, num_samples=n) for n in sample_sizes]
    assert penalties[0] > penalties[1] > penalties[2] > 0
    assert derived[0]["v"] > derived[1]["v"] > derived[2]["v"] > 0


def test_only_penalty_constants_are_reported(constants):
    assert set(theorem_constants(constants)) == {
        "R", "delta", "c", "v", "t", "L_prime", "K",
    }
