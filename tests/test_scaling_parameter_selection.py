"""Regression tests for preprocessing objective curves before scale selection."""

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.select_scaling_parameter import (
    floor_objective_values,
    load_selection_inputs,
)
from scaling_selection import (
    DimensionPath,
    adaptive_window,
    build_dimension_path,
    select_minimal_scale,
    window,
)


def test_floor_applies_only_to_finite_raw_objectives():
    raw_objectives = np.array([1e-12, 1e-8, 2e-8, np.inf])

    floored, mask = floor_objective_values(raw_objectives, 1e-8)

    np.testing.assert_array_equal(mask, [True, True, False, False])
    np.testing.assert_array_equal(floored[:3], [0.0, 0.0, 2e-8])
    assert np.isinf(floored[3])
    np.testing.assert_array_equal(
        raw_objectives,
        [1e-12, 1e-8, 2e-8, np.inf],
    )


def test_floor_makes_dimension_path_prefer_smallest_plateau_model():
    dimensions = np.array([5, 6, 7, 8])
    raw_objectives = np.array([1e-4, 8e-9, 1e-13, 2e-14])
    penalties = np.sqrt(dimensions)
    floored, _ = floor_objective_values(raw_objectives, 1e-8)

    path = build_dimension_path(dimensions, floored, penalties)

    assert path.dimension_at(0.0) == 6


def test_loading_floors_raw_objectives_but_not_penalties(tmp_path):
    input_path = tmp_path / "curve.npz"
    np.savez(
        input_path,
        d_m_values=np.array([1, 2, 3]),
        objective_values=np.array([1e-4, 8e-9, 1e-13]),
        Sigma=np.eye(4),
        n=4,
        num_samples=100,
        curve_type="test_curve",
    )
    args = Namespace(
        objective_floor=1e-8,
        num_samples=None,
        r=1.0,
        Lm=1.0,
        L=1.0,
        xi=10.0,
    )

    selection_data = load_selection_inputs(input_path, args)

    np.testing.assert_array_equal(
        selection_data["raw_objective_values"],
        [1e-4, 8e-9, 1e-13],
    )
    np.testing.assert_array_equal(
        selection_data["objective_values"],
        [1e-4, 0.0, 0.0],
    )
    assert selection_data["num_floored_objectives"] == 2
    assert np.all(selection_data["penalty_values"] > 0.0)
    assert np.all(np.diff(selection_data["penalty_values"]) > 0.0)


def test_recommendation_factor_controls_recommended_scale_and_dimension():
    dimensions = np.array([1, 2])
    objectives = np.array([1.0, 0.0])
    penalties = np.array([1.0, 2.0])

    result = select_minimal_scale(
        dimensions,
        objectives,
        penalties,
        method="threshold",
        threshold_value=1,
        recommendation_factor=0.5,
    )

    assert result.minimal_scale == pytest.approx(1.0)
    assert result.recommendation_factor == pytest.approx(0.5)
    assert result.recommended_scale == pytest.approx(0.5)
    assert result.selected_dimension == 2


def test_recommendation_factor_defaults_to_two():
    result = select_minimal_scale(
        [1, 2],
        [1.0, 0.0],
        [1.0, 2.0],
        method="threshold",
        threshold_value=1,
    )

    assert result.recommendation_factor == pytest.approx(2.0)
    assert result.recommended_scale == pytest.approx(2.0)
    assert result.selected_dimension == 1


@pytest.mark.parametrize(
    "recommendation_factor",
    [0.0, -1.0, np.inf, np.nan, "bad"],
)
def test_recommendation_factor_must_be_positive(recommendation_factor):
    with pytest.raises(ValueError, match="recommendation_factor must be"):
        select_minimal_scale(
            [1, 2],
            [1.0, 0.0],
            [1.0, 2.0],
            method="threshold",
            threshold_value=1,
            recommendation_factor=recommendation_factor,
        )


@pytest.mark.parametrize("objective_floor", [-1.0, np.inf, np.nan, "bad"])
def test_floor_rejects_invalid_values(objective_floor):
    with pytest.raises(ValueError, match="finite, nonnegative"):
        floor_objective_values([1.0], objective_floor)


def fragmented_transition_path():
    return DimensionPath(
        breakpoints=(0.0, 0.98, 1.0, 1.03, 4.0, 8.0),
        dimensions=(12, 11, 10, 9, 8, 7),
        max_candidate_dimension=12.0,
    )


def test_adaptive_window_selects_smallest_stable_aggregated_jump():
    result = adaptive_window(fragmented_transition_path())

    entry_width = 0.5 * np.log(1.03 / 0.98)
    expected_eta = np.expm1(entry_width / 0.8 * (1.0 + 1e-12))
    assert result.eta == pytest.approx(expected_eta)
    assert result.center == pytest.approx(np.sqrt(0.98 * 1.03))
    assert result.largest_jump == pytest.approx(3.0)
    assert result.transition_scales == pytest.approx((0.98, 1.0, 1.03))
    assert window(fragmented_transition_path(), result.eta) == pytest.approx(
        result.center
    )


def test_adaptive_window_treats_eta_as_a_minimum_bandwidth():
    result = adaptive_window(fragmented_transition_path(), minimum_eta=0.1)

    expected_eta = np.expm1(np.log1p(0.1) / 0.8 * (1.0 + 1e-12))
    assert result.eta == pytest.approx(expected_eta)
    assert result.center == pytest.approx(np.sqrt(0.98 * 1.03))
    assert result.largest_jump == pytest.approx(3.0)


def test_adaptive_window_falls_back_without_merging_unseparated_transitions():
    path = DimensionPath(
        breakpoints=(0.0, 1.0, 2.0, 4.0),
        dimensions=(5, 4, 3, 2),
        max_candidate_dimension=5.0,
    )

    result = adaptive_window(path)

    assert result.center == pytest.approx(4.0)
    assert result.largest_jump == pytest.approx(1.0)
    assert result.transition_scales == (4.0,)


def test_adaptive_window_fallback_matches_a_user_supplied_minimum_window():
    result = adaptive_window(fragmented_transition_path(), minimum_eta=10.0)

    assert result.eta == pytest.approx(10.0)
    assert result.center == pytest.approx(np.sqrt(0.98 * 8.0))
    assert result.largest_jump == pytest.approx(5.0)
    assert window(fragmented_transition_path(), result.eta) == pytest.approx(
        result.center
    )


def test_window_method_uses_adaptive_center_as_minimal_scale():
    # Adjacent affine criteria cross at the requested transition scales.
    result = select_minimal_scale(
        [7, 8, 9, 10, 11, 12],
        [15.01, 7.01, 3.01, 1.98, 0.98, 0.0],
        [7, 8, 9, 10, 11, 12],
        method="window",
    )

    assert result.minimal_scale == pytest.approx(np.sqrt(0.98 * 1.03))
    assert result.largest_jump == pytest.approx(3.0)
    assert result.eta == pytest.approx(
        np.expm1(0.5 * np.log(1.03 / 0.98) / 0.8 * (1.0 + 1e-12))
    )
