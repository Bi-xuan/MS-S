"""Score directed supports using Jaccard distance and maximal-class recovery.

Supports contain zero-based directed edges; self-loops are excluded from all
distances. Maximal-class families include isolated vertices and their members
range over all directed supports, including edges below the diagonal.
"""

from __future__ import annotations

from functools import lru_cache
from math import exp, isfinite

from maximal_class_graph import graph_to_maximal_class, list_graphs_in_maximal_class


def validate_score_a(a: float) -> float:
    a = float(a)
    if not isfinite(a) or not 0 < a < 1:
        raise ValueError("score a must be finite and satisfy 0 < a < 1.")
    return a


def _edges(support) -> frozenset[tuple[int, int]]:
    return frozenset((i, j) for i, j in support if i != j)


def jaccard_distance(left, right) -> float:
    """Return (FP + FN) / (TP + FP + FN), with d(empty, empty) = 0."""

    left, right = _edges(left), _edges(right)
    union = left | right
    return len(left ^ right) / len(union) if union else 0.0


@lru_cache(maxsize=512)
def _maximal_class(edges: frozenset, n: int) -> frozenset:
    return frozenset(graph_to_maximal_class(edges, nodes=tuple(range(n))))


@lru_cache(maxsize=128)
def _class_members(family: frozenset) -> tuple[frozenset, ...]:
    graphs = list_graphs_in_maximal_class(family)
    # The package encodes non-loop directed edges in row-major order.
    edge_order = tuple((i, j) for i in graphs.nodes for j in graphs.nodes if i != j)
    return tuple(
        frozenset(edge for bit, edge in enumerate(edge_order) if mask & (1 << bit))
        for mask in graphs.masks
    )


def same_maximal_class(left, right, n: int) -> bool:
    """Test membership in the same full maximal-class family."""

    return _maximal_class(_edges(left), n) == _maximal_class(_edges(right), n)


def distance_to_maximal_class(support, family) -> float:
    """Distance to the nearest directed support in a maximal-class family."""

    support = _edges(support)
    family = frozenset(frozenset(group) for group in family)
    return min(jaccard_distance(support, member) for member in _class_members(family))


def maximal_class_distance(left_family, right_family) -> float:
    """Distance between the two nearest members of two maximal-class families."""

    left = frozenset(frozenset(group) for group in left_family)
    right = frozenset(frozenset(group) for group in right_family)
    if left == right:
        return 0.0
    return min(
        jaccard_distance(first, second)
        for first in _class_members(left)
        for second in _class_members(right)
    )


def score_support(selected, truth, n: int, a: float = 0.5) -> float:
    """Return Q(selected, truth) with configurable 0 < a < 1."""

    a = validate_score_a(a)
    selected, truth = _edges(selected), _edges(truth)
    true_class = _maximal_class(truth, n)
    if _maximal_class(selected, n) == true_class:
        return a + (1 - a) * exp(-jaccard_distance(selected, truth))
    return a * exp(-distance_to_maximal_class(selected, true_class))
