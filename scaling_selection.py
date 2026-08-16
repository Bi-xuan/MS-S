"""Data-driven selection of the multiplier in a model-selection penalty.

The procedures implemented here follow the jump-based definitions in
Appendix D.2 of Arlot (2019).  They estimate the minimal-penalty scale from
the exact, piecewise-constant path of selected model dimensions.  The slope
heuristic then recommends using twice that scale for final model selection.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import expm1, log, log1p, sqrt
from typing import Mapping

import numpy as np


DEFAULT_RECOMMENDATION_FACTOR = 2.0
DEFAULT_WINDOW_PERTURBATION = 0.2
DEFAULT_WINDOW_AGGREGATION_RATIO = 1.5
DEFAULT_WINDOW_SEPARATION_RATIO = 2.5


@dataclass(frozen=True)
class DimensionPath:
    """Exact selected-dimension path over nonnegative penalty scales.

    ``dimensions[i]`` is selected on the interval beginning at
    ``breakpoints[i]``.  The first breakpoint is always zero, and the final
    interval extends to infinity.
    """

    breakpoints: tuple[float, ...]
    dimensions: tuple[int | float, ...]
    max_candidate_dimension: float

    def __post_init__(self) -> None:
        if not self.breakpoints or self.breakpoints[0] != 0.0:
            raise ValueError("The first dimension-path breakpoint must be zero.")
        if len(self.breakpoints) != len(self.dimensions):
            raise ValueError("breakpoints and dimensions must have the same length.")
        if any(not np.isfinite(value) for value in self.breakpoints):
            raise ValueError("Dimension-path breakpoints must be finite.")
        if any(value < 0.0 for value in self.breakpoints):
            raise ValueError("Dimension-path breakpoints must be nonnegative.")
        if any(
            right <= left
            for left, right in zip(self.breakpoints, self.breakpoints[1:])
        ):
            raise ValueError("Dimension-path breakpoints must be strictly increasing.")

    @property
    def transition_scales(self) -> tuple[float, ...]:
        """Return the positive scales at which the selected dimension changes."""

        return self.breakpoints[1:]

    def dimension_at(self, scale: float) -> int | float:
        """Return the dimension selected at a finite, nonnegative scale."""

        scale = _finite_nonnegative_float(scale, "scale")
        index = bisect_right(self.breakpoints, scale) - 1
        return self.dimensions[index]


@dataclass(frozen=True)
class ScalingSelection:
    """Result of a minimal-penalty scale-selection procedure."""

    method: str
    minimal_scale: float
    recommended_scale: float
    selected_dimension: int | float
    component_scales: Mapping[str, float]
    threshold: float | None = None
    eta: float | None = None
    largest_jump: float | None = None
    recommendation_factor: float = DEFAULT_RECOMMENDATION_FACTOR


@dataclass(frozen=True)
class AdaptiveWindowSelection:
    """Adaptive window bandwidth and its dominant jump cluster."""

    center: float
    eta: float
    largest_jump: float
    transition_scales: tuple[float, ...]


@dataclass(frozen=True)
class _DominantCluster:
    first: int
    last: int
    strength: float
    center: float
    num_tied: int


@dataclass(frozen=True)
class _Line:
    slope: float
    intercept: float
    dimension: int | float


def _finite_nonnegative_float(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite, nonnegative number.") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite, nonnegative number.")
    return value


def _positive_float(value, name: str) -> float:
    value = _finite_nonnegative_float(value, name)
    if value == 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_inputs(d_m_values, objective_values, penalty_values):
    dimensions_original = np.asarray(d_m_values)
    try:
        dimensions = np.asarray(d_m_values, dtype=float)
        objectives = np.asarray(objective_values, dtype=float)
        penalties = np.asarray(penalty_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "d_m_values, objective_values, and penalty_values must be numeric."
        ) from exc

    arrays = {
        "d_m_values": dimensions,
        "objective_values": objectives,
        "penalty_values": penalties,
    }
    for name, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")

    if len(dimensions) == 0:
        raise ValueError("At least one candidate dimension is required.")
    if not (len(dimensions) == len(objectives) == len(penalties)):
        raise ValueError(
            "d_m_values, objective_values, and penalty_values must have "
            "the same length."
        )
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions < 0.0):
        raise ValueError("d_m_values must contain finite, nonnegative values.")
    if len(np.unique(dimensions)) != len(dimensions):
        raise ValueError("d_m_values must contain unique candidate dimensions.")
    if not np.all(np.isfinite(penalties)) or np.any(penalties < 0.0):
        raise ValueError("penalty_values must contain finite, nonnegative values.")

    finite_objectives = np.isfinite(objectives)
    if not np.any(finite_objectives):
        raise ValueError("At least one objective value must be finite.")

    dimension_order = np.argsort(dimensions)
    if np.any(np.diff(penalties[dimension_order]) < 0.0):
        raise ValueError(
            "penalty_values must be nondecreasing with model dimension."
        )

    return (
        dimensions_original,
        dimensions,
        objectives,
        penalties,
        finite_objectives,
    )


def _intersection(left: _Line, right: _Line) -> float:
    """Return where a lower-slope right line overtakes the left line."""

    return (right.intercept - left.intercept) / (left.slope - right.slope)


def build_dimension_path(
    d_m_values,
    objective_values,
    penalty_values,
) -> DimensionPath:
    """Build the exact lower-envelope path of selected dimensions.

    Each candidate model defines the affine criterion

    ``objective_values[m] + C * penalty_values[m]``.

    The returned path is exact up to floating-point arithmetic and therefore
    does not depend on a user-chosen grid of ``C`` values.  Non-finite
    objective values are excluded.  Penalties must be nondecreasing with
    model dimension so that the selected-complexity path has the interpretation
    required by the jump procedures.
    """

    (
        dimensions_original,
        dimensions,
        objectives,
        penalties,
        finite_objectives,
    ) = _validate_inputs(d_m_values, objective_values, penalty_values)

    eligible_indices = np.flatnonzero(finite_objectives)
    max_candidate_dimension = float(np.max(dimensions[eligible_indices]))

    # For identical penalty slopes, only the smallest-intercept line can be
    # selected.  If the criteria are identical, retain the smaller dimension,
    # matching the convention used by model_selection.select_dimension.
    lines_by_slope: dict[float, _Line] = {}
    for index in eligible_indices:
        dimension_value = dimensions_original[index]
        if getattr(dimension_value, "shape", None) == ():
            dimension_value = dimension_value.item()
        line = _Line(
            slope=float(penalties[index]),
            intercept=float(objectives[index]),
            dimension=dimension_value,
        )
        previous = lines_by_slope.get(line.slope)
        if previous is None or line.intercept < previous.intercept:
            lines_by_slope[line.slope] = line
        elif (
            line.intercept == previous.intercept
            and float(line.dimension) < float(previous.dimension)
        ):
            lines_by_slope[line.slope] = line

    # With slopes in decreasing order, each accepted line becomes optimal to
    # the right of its start value.  Popping non-increasing starts constructs
    # the lower envelope in linear time after sorting.
    ordered_lines = sorted(
        lines_by_slope.values(),
        key=lambda line: line.slope,
        reverse=True,
    )
    envelope: list[_Line] = []
    starts: list[float] = []
    for line in ordered_lines:
        start = float("-inf")
        while envelope:
            start = _intersection(envelope[-1], line)
            if start > starts[-1]:
                break
            envelope.pop()
            starts.pop()
        if not envelope:
            start = float("-inf")
        envelope.append(line)
        starts.append(start)

    active_index = bisect_right(starts, 0.0) - 1
    breakpoints = [0.0]
    selected_dimensions = [envelope[active_index].dimension]
    for index in range(active_index + 1, len(envelope)):
        start = float(starts[index])
        if not np.isfinite(start):
            continue
        if start <= 0.0:
            selected_dimensions[0] = envelope[index].dimension
            continue
        if envelope[index].dimension == selected_dimensions[-1]:
            continue
        breakpoints.append(start)
        selected_dimensions.append(envelope[index].dimension)

    return DimensionPath(
        breakpoints=tuple(breakpoints),
        dimensions=tuple(selected_dimensions),
        max_candidate_dimension=max_candidate_dimension,
    )


def maximal_jump(path: DimensionPath) -> float:
    """Return the last location of the largest downward dimension jump."""

    if len(path.dimensions) < 2:
        raise ValueError(
            "Maximal jump is undefined because the path has no transition."
        )

    jumps = np.asarray(path.dimensions[:-1], dtype=float) - np.asarray(
        path.dimensions[1:], dtype=float
    )
    largest_jump = float(np.max(jumps))
    if largest_jump <= 0.0:
        raise ValueError("Maximal jump requires at least one downward dimension jump.")

    tied_indices = np.flatnonzero(
        np.isclose(jumps, largest_jump, rtol=1e-12, atol=1e-12)
    )
    transition_index = int(tied_indices[-1])
    return float(path.transition_scales[transition_index])


def threshold(
    path: DimensionPath,
    threshold_value: float | None = None,
) -> float:
    """Return the first scale selecting a dimension at most the threshold.

    By default, the threshold is half the largest candidate dimension having
    a finite objective value.
    """

    if threshold_value is None:
        threshold_value = path.max_candidate_dimension / 2.0
    threshold_value = _finite_nonnegative_float(
        threshold_value,
        "threshold_value",
    )

    for breakpoint, dimension in zip(path.breakpoints, path.dimensions):
        if float(dimension) <= threshold_value:
            return float(breakpoint)
    raise ValueError(
        "The selected dimension never reaches the requested threshold."
    )


def window(path: DimensionPath, eta: float) -> float:
    """Return the geometric center of the last maximal window interval."""

    eta = _positive_float(eta, "eta")
    if len(path.dimensions) < 2:
        raise ValueError("Window is undefined because the path has no transition.")

    factor = 1.0 + eta
    boundaries = {0.0}
    for transition in path.transition_scales:
        boundaries.add(float(transition) / factor)
        boundaries.add(float(transition) * factor)
    ordered_boundaries = sorted(boundaries)

    intervals: list[tuple[float, float, float]] = []
    for left, right in zip(ordered_boundaries, ordered_boundaries[1:]):
        if left == right:
            continue
        probe = sqrt(left * right) if left > 0.0 else right / 2.0
        jump = float(path.dimension_at(probe / factor)) - float(
            path.dimension_at(probe * factor)
        )
        intervals.append((left, right, jump))

    if not intervals:
        raise ValueError("Window is undefined because no finite interval is available.")
    largest_jump = max(interval[2] for interval in intervals)
    if largest_jump <= 0.0:
        raise ValueError("Window requires at least one downward dimension jump.")

    maximal_intervals = [
        interval
        for interval in intervals
        if np.isclose(interval[2], largest_jump, rtol=1e-12, atol=1e-12)
    ]

    # Merge adjacent maximizing intervals, then keep the final connected
    # component as prescribed in Appendix D.2.
    components: list[list[float]] = []
    for left, right, _ in maximal_intervals:
        if components and left == components[-1][1]:
            components[-1][1] = right
        else:
            components.append([left, right])
    left, right = components[-1]
    return sqrt(left * right)


def _dominant_cluster(
    log_transitions: np.ndarray,
    jumps: np.ndarray,
    log_half_width: float,
) -> _DominantCluster:
    """Return the last strongest transition cluster fitting in a log window."""

    cumulative_jumps = np.concatenate(([0.0], np.cumsum(jumps)))
    best_strength = float("-inf")
    tied_clusters: list[tuple[int, int, float]] = []
    last = -1
    width = 2.0 * log_half_width

    for first in range(len(log_transitions)):
        last = max(last, first)
        while (
            last + 1 < len(log_transitions)
            and log_transitions[last + 1] - log_transitions[first]
            <= width + 1e-14
        ):
            last += 1

        strength = float(cumulative_jumps[last + 1] - cumulative_jumps[first])
        center = float((log_transitions[first] + log_transitions[last]) / 2.0)
        if strength > best_strength and not np.isclose(
            strength,
            best_strength,
            rtol=1e-12,
            atol=1e-12,
        ):
            best_strength = strength
            tied_clusters = [(first, last, center)]
        elif np.isclose(strength, best_strength, rtol=1e-12, atol=1e-12):
            tied_clusters.append((first, last, center))

    # This matches window(): retain the final maximizing component on the
    # scale axis when several windows have the same strength.
    first, last, center = max(tied_clusters, key=lambda item: item[2])
    return _DominantCluster(
        first=first,
        last=last,
        strength=best_strength,
        center=center,
        num_tied=len(tied_clusters),
    )


def _cluster_is_separated(
    cluster: _DominantCluster,
    log_transitions: np.ndarray,
    separation_ratio: float,
) -> bool:
    """Return whether a multi-jump cluster is isolated from its neighbors."""

    internal_gaps = np.diff(
        log_transitions[cluster.first : cluster.last + 1]
    )
    if len(internal_gaps) == 0:
        return False

    external_gaps = []
    if cluster.first > 0:
        external_gaps.append(
            log_transitions[cluster.first]
            - log_transitions[cluster.first - 1]
        )
    if cluster.last + 1 < len(log_transitions):
        external_gaps.append(
            log_transitions[cluster.last + 1]
            - log_transitions[cluster.last]
        )
    if not external_gaps:
        return False

    return min(external_gaps) >= separation_ratio * float(np.max(internal_gaps))


def adaptive_window(
    path: DimensionPath,
    minimum_eta: float | None = None,
    *,
    perturbation: float = DEFAULT_WINDOW_PERTURBATION,
    aggregation_ratio: float = DEFAULT_WINDOW_AGGREGATION_RATIO,
    separation_ratio: float = DEFAULT_WINDOW_SEPARATION_RATIO,
) -> AdaptiveWindowSelection:
    """Choose the smallest stable bandwidth around a dominant jump cluster.

    The search is performed on the exact transition scales in ``log(C)``.
    It tests only bandwidths at which a new consecutive transition cluster
    can fit inside a window.  An aggregate is accepted when it remains the
    unique dominant cluster under a relative ``perturbation`` of the log
    half-width, contains meaningful mass beyond its largest member, and is
    separated from the nearest transition outside the cluster.

    If no multi-jump cluster meets those conditions, the method conservatively
    returns the dominant window at the minimum bandwidth.  With the default
    numerical minimum, this is normally the last largest individual jump.
    """

    if len(path.dimensions) < 2:
        raise ValueError("Window is undefined because the path has no transition.")

    perturbation = _positive_float(perturbation, "perturbation")
    if perturbation >= 1.0:
        raise ValueError("perturbation must be less than one.")
    aggregation_ratio = _positive_float(
        aggregation_ratio,
        "aggregation_ratio",
    )
    separation_ratio = _positive_float(separation_ratio, "separation_ratio")

    if minimum_eta is None:
        # The exact path has no sampling-grid resolution.  This is the
        # smallest practical multiplicative window distinguishable from one.
        minimum_eta = 8.0 * np.finfo(float).eps
    else:
        minimum_eta = _positive_float(minimum_eta, "minimum_eta")
    minimum_log_width = log1p(minimum_eta)

    transition_scales = np.asarray(path.transition_scales, dtype=float)
    log_transitions = np.log(transition_scales)
    jumps = np.asarray(path.dimensions[:-1], dtype=float) - np.asarray(
        path.dimensions[1:],
        dtype=float,
    )
    if np.any(jumps <= 0.0):
        raise ValueError(
            "Window requires every dimension-path transition to be downward."
        )

    critical_widths = {0.0}
    for first in range(len(log_transitions) - 1):
        for last in range(first + 1, len(log_transitions)):
            critical_widths.add(
                float((log_transitions[last] - log_transitions[first]) / 2.0)
            )

    # At a cluster's entry event, move far enough into its plateau that a
    # negative perturbation remains beyond the event.  Testing all pairwise
    # events also detects plateaus whose start is caused by a competing cluster.
    candidate_widths = sorted(
        {
            max(width, minimum_log_width)
            / (1.0 - perturbation)
            * (1.0 + 1e-12)
            for width in critical_widths
        }
    )

    for log_half_width in candidate_widths:
        probes = (
            (1.0 - perturbation) * log_half_width,
            log_half_width,
            (1.0 + perturbation) * log_half_width,
        )
        clusters = [
            _dominant_cluster(log_transitions, jumps, probe)
            for probe in probes
        ]
        cluster = clusters[1]
        signature = (cluster.first, cluster.last)
        if any(
            (candidate.first, candidate.last) != signature
            or candidate.num_tied != 1
            for candidate in clusters
        ):
            continue
        if cluster.last == cluster.first:
            continue

        member_jumps = jumps[cluster.first : cluster.last + 1]
        if cluster.strength < aggregation_ratio * float(np.max(member_jumps)):
            continue
        if not _cluster_is_separated(
            cluster,
            log_transitions,
            separation_ratio,
        ):
            continue

        eta = expm1(log_half_width)
        return AdaptiveWindowSelection(
            center=float(np.exp(cluster.center)),
            eta=float(eta),
            largest_jump=float(cluster.strength),
            transition_scales=tuple(
                float(value)
                for value in transition_scales[cluster.first : cluster.last + 1]
            ),
        )

    fallback = _dominant_cluster(
        log_transitions,
        jumps,
        minimum_log_width,
    )
    return AdaptiveWindowSelection(
        center=float(np.exp(fallback.center)),
        eta=float(minimum_eta),
        largest_jump=float(fallback.strength),
        transition_scales=tuple(
            float(value)
            for value in transition_scales[fallback.first : fallback.last + 1]
        ),
    )


def median_jump(
    path: DimensionPath,
    *,
    threshold_value: float | None = None,
    eta: float,
) -> tuple[float, dict[str, float]]:
    """Return the median of maximal-jump, threshold, and window scales."""

    component_scales = {
        "maximal_jump": maximal_jump(path),
        "threshold": threshold(path, threshold_value),
        "window": window(path, eta),
    }
    value = float(np.median(list(component_scales.values())))
    return value, component_scales


def _default_eta(num_samples) -> float:
    try:
        numeric_samples = float(num_samples)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "A positive integer num_samples is required for the default eta."
        ) from exc
    if (
        not np.isfinite(numeric_samples)
        or numeric_samples <= 1.0
        or not numeric_samples.is_integer()
    ):
        raise ValueError(
            "A positive integer num_samples greater than one is required "
            "for the default eta."
        )
    return sqrt(log(numeric_samples) / numeric_samples)


def _normalize_method(method: str) -> str:
    if not isinstance(method, str):
        raise ValueError("method must be a string.")
    normalized = method.strip().lower().replace("-", "_")
    aliases = {
        "maximal_jump": "maximal_jump",
        "max_jump": "maximal_jump",
        "threshold": "threshold",
        "window": "window",
        "median_jump": "median_jump",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "method must be one of: maximal_jump, threshold, window, "
            "or median_jump."
        ) from exc


def select_minimal_scale(
    d_m_values,
    objective_values,
    penalty_values,
    *,
    method: str = "median_jump",
    num_samples=None,
    threshold_value: float | None = None,
    eta: float | None = None,
    recommendation_factor: float = DEFAULT_RECOMMENDATION_FACTOR,
) -> ScalingSelection:
    """Estimate a minimal-penalty scale and report the recommended scale.

    Parameters
    ----------
    d_m_values, objective_values, penalty_values
        Candidate dimensions and the intercepts and slopes of their penalized
        criteria.
    method
        One of ``"maximal_jump"``, ``"threshold"``, ``"window"``, or
        ``"median_jump"``.  Hyphens may be used in place of underscores.
    num_samples
        Number of observations.  It is used only to obtain the default
        ``eta = sqrt(log(num_samples) / num_samples)`` for ``median_jump``.
    threshold_value
        Dimension threshold.  The default is half the largest candidate
        dimension having a finite objective.
    eta
        For ``window``, an optional positive lower bound for the adaptively
        selected window width.  For ``median_jump``, a fixed positive window
        width; if omitted, it is computed from ``num_samples``.
    recommendation_factor
        Positive multiplier applied to the estimated minimal scale. The
        default is 2, as prescribed by the slope heuristic.

    Returns
    -------
    ScalingSelection
        ``minimal_scale`` is the estimated minimal-penalty constant.
        ``recommended_scale`` is ``recommendation_factor * minimal_scale``,
        and ``selected_dimension`` is the dimension selected at that scale.
    """

    normalized_method = _normalize_method(method)
    recommendation_factor = _positive_float(
        recommendation_factor,
        "recommendation_factor",
    )
    path = build_dimension_path(
        d_m_values,
        objective_values,
        penalty_values,
    )

    resolved_threshold = None
    resolved_eta = None
    if normalized_method in {"threshold", "median_jump"}:
        resolved_threshold = (
            path.max_candidate_dimension / 2.0
            if threshold_value is None
            else _finite_nonnegative_float(threshold_value, "threshold_value")
        )
    if normalized_method == "median_jump":
        resolved_eta = _default_eta(num_samples) if eta is None else _positive_float(
            eta,
            "eta",
        )

    if normalized_method == "maximal_jump":
        minimal_scale = maximal_jump(path)
        component_scales = {"maximal_jump": minimal_scale}
    elif normalized_method == "threshold":
        minimal_scale = threshold(path, resolved_threshold)
        component_scales = {"threshold": minimal_scale}
    elif normalized_method == "window":
        window_result = adaptive_window(path, minimum_eta=eta)
        minimal_scale = window_result.center
        resolved_eta = window_result.eta
        largest_jump = window_result.largest_jump
        component_scales = {"window": minimal_scale}
    else:
        minimal_scale, component_scales = median_jump(
            path,
            threshold_value=resolved_threshold,
            eta=resolved_eta,
        )

    recommended_scale = recommendation_factor * minimal_scale
    if not np.isfinite(recommended_scale):
        raise ValueError("The recommended scale is not finite.")

    return ScalingSelection(
        method=normalized_method,
        minimal_scale=float(minimal_scale),
        recommended_scale=float(recommended_scale),
        selected_dimension=path.dimension_at(recommended_scale),
        component_scales=dict(component_scales),
        threshold=resolved_threshold,
        eta=resolved_eta,
        largest_jump=(largest_jump if normalized_method == "window" else None),
        recommendation_factor=recommendation_factor,
    )
