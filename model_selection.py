"""Utilities for penalized model selection."""

from __future__ import annotations

import numpy as np


def select_dimension(
    d_m_values,
    objective_values,
    penalty_values,
    C,
):
    """Select the dimension minimizing objective(D_m) + C * p(D_m).

    Non-finite objective values are excluded from the candidate set. If
    several candidates attain the same minimum, the smallest dimension is
    selected.

    Parameters
    ----------
    d_m_values
        One-dimensional candidate model dimensions.
    objective_values
        Raw objective value associated with each candidate dimension.
    penalty_values
        Unscaled penalty p(D_m) associated with each candidate dimension.
    C
        Finite, nonnegative penalty scaling parameter.

    Returns
    -------
    int or float
        The selected candidate dimension.
    """

    d_m_values = np.asarray(d_m_values)
    objective_values = np.asarray(objective_values, dtype=float)
    penalty_values = np.asarray(penalty_values, dtype=float)

    if d_m_values.ndim != 1:
        raise ValueError("d_m_values must be one-dimensional.")
    if objective_values.ndim != 1:
        raise ValueError("objective_values must be one-dimensional.")
    if penalty_values.ndim != 1:
        raise ValueError("penalty_values must be one-dimensional.")
    if len(d_m_values) == 0:
        raise ValueError("At least one candidate dimension is required.")
    if not (
        len(d_m_values) == len(objective_values) == len(penalty_values)
    ):
        raise ValueError(
            "d_m_values, objective_values, and penalty_values must have "
            "the same length."
        )

    try:
        C = float(C)
    except (TypeError, ValueError) as exc:
        raise ValueError("C must be a finite, nonnegative number.") from exc
    if not np.isfinite(C) or C < 0.0:
        raise ValueError("C must be a finite, nonnegative number.")

    try:
        finite_dimensions = np.isfinite(d_m_values)
        nonnegative_dimensions = d_m_values >= 0
    except TypeError as exc:
        raise ValueError("d_m_values must contain numeric values.") from exc
    if not np.all(finite_dimensions & nonnegative_dimensions):
        raise ValueError("d_m_values must contain finite, nonnegative values.")
    if len(np.unique(d_m_values)) != len(d_m_values):
        raise ValueError("d_m_values must contain unique candidate dimensions.")
    if not np.all(np.isfinite(penalty_values)):
        raise ValueError("penalty_values must contain only finite values.")
    if np.any(penalty_values < 0.0):
        raise ValueError("penalty_values must be nonnegative.")

    finite_objectives = np.isfinite(objective_values)
    if not np.any(finite_objectives):
        raise ValueError("At least one objective value must be finite.")

    with np.errstate(over="ignore", invalid="ignore"):
        criterion_values = objective_values + C * penalty_values
    eligible = finite_objectives & np.isfinite(criterion_values)
    if not np.any(eligible):
        raise ValueError("No finite penalized criterion value is available.")

    minimum = np.min(criterion_values[eligible])
    tied_indices = np.flatnonzero(eligible & (criterion_values == minimum))
    selected_index = tied_indices[np.argmin(d_m_values[tied_indices])]
    return d_m_values[selected_index].item()


def select_dimension_path(
    d_m_values,
    objective_values,
    penalty_values,
    C_values,
):
    """Evaluate the selected model dimension over a grid of penalty scales."""

    C_values = np.asarray(C_values, dtype=float)
    if C_values.ndim != 1:
        raise ValueError("C_values must be one-dimensional.")
    if len(C_values) == 0:
        raise ValueError("At least one penalty scaling parameter is required.")
    if not np.all(np.isfinite(C_values)) or np.any(C_values < 0.0):
        raise ValueError("C_values must contain finite, nonnegative values.")
    if np.any(np.diff(C_values) < 0.0):
        raise ValueError("C_values must be sorted in nondecreasing order.")

    return np.asarray(
        [
            select_dimension(
                d_m_values,
                objective_values,
                penalty_values,
                C,
            )
            for C in C_values
        ]
    )
