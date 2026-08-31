"""Checks for strict upper-triangular exhaustive support search."""

from math import comb

import numpy as np
import pytest

from experiments.compute_objective_curve import (
    compute_objective_curve,
    indexed_upper_support,
    lambda_star_for_dimension,
)
from optimizers.support_search import optimize_lambda
from supports.exact import get_upper_triangular_supports


def off_diagonal_edges_in_mask(mask):
    n = mask.shape[0]
    return {
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j and mask[i, j]
    }


def test_upper_triangular_supports_enumerate_only_strict_upper_entries():
    n = 4
    n_edge = 2

    supports = list(get_upper_triangular_supports(n, n_edge))

    assert len(supports) == comb(n * (n - 1) // 2, n_edge)
    for mask in supports:
        assert np.all(np.diag(mask))
        edges = off_diagonal_edges_in_mask(mask)
        assert len(edges) == n_edge
        assert all(i < j for i, j in edges)


def test_upper_triangular_supports_reject_too_many_edges():
    with pytest.raises(ValueError, match="n_edge must be between 0 and 3"):
        list(get_upper_triangular_supports(3, 4))


def test_indexed_true_supports_cover_all_twenty_dimension_four_cases():
    supports = [indexed_upper_support(4, 4, index) for index in range(20)]

    assert len(set(supports)) == comb(6, 3) == 20
    assert supports[0] == ((0, 1), (0, 2), (0, 3))
    assert supports[-1] == ((1, 2), (1, 3), (2, 3))


def test_indexed_true_support_uses_existing_lambda_value_convention():
    support = indexed_upper_support(4, 4, 7)

    Lambda_star = lambda_star_for_dimension(4, support)

    np.testing.assert_allclose(np.diag(Lambda_star), [0.10, 0.25, 0.40, 0.55])
    np.testing.assert_allclose(
        [Lambda_star[edge] for edge in support],
        [0.60, 0.20, -0.45],
    )
    assert np.all(
        np.abs([Lambda_star[edge] for edge in support]) >= 0.20
    )
    recovered_support = {
        (int(i), int(j))
        for i, j in np.argwhere(Lambda_star)
        if i != j
    }
    assert recovered_support == set(support)


def test_indexed_true_support_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="support_index must be between 0 and 19"):
        indexed_upper_support(4, 4, 20)


def test_optimize_lambda_upper_scope_returns_upper_triangular_support():
    Sigma = np.eye(3)

    Lambda, omega, obj, metadata = optimize_lambda(
        Sigma,
        D_m=2,
        max_iter=2,
        max_restarts=1,
        n_jobs=1,
        random_seed=7,
        refine_after_fixed_omega=False,
        return_metadata=True,
        support_scope="upper",
    )

    assert Lambda is not None
    assert np.isfinite(omega)
    assert np.isfinite(obj)
    assert metadata["support_scope"] == "upper"
    assert len(metadata["selected_support_edges"]) == 1
    assert all(i < j for i, j in metadata["selected_support_edges"])


def test_optimize_lambda_upper_scope_rejects_preselection():
    with pytest.raises(ValueError, match="cannot be combined"):
        optimize_lambda(
            np.eye(3),
            D_m=2,
            preselect_k=2,
            support_scope="upper",
        )


def test_upper_scope_objective_curve_uses_reduced_dimension_range():
    (
        d_m_values,
        _,
        _,
        _,
        selected_support_masks,
        selected_support_valid,
    ) = compute_objective_curve(
        np.eye(2),
        max_iter=2,
        max_restarts=1,
        n_jobs=1,
        random_seed=7,
        refine_after_fixed_omega=False,
        support_scope="upper",
    )

    assert d_m_values.tolist() == [1, 2]
    assert np.all(selected_support_valid)
    for mask in selected_support_masks:
        assert not mask[1, 0]
