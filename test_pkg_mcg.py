"""Inspect maximal-class families and their graphs for three N=4 supports.

Install and run with Python 3.9+:
    python -m pip install "maximal-class-graph[matrix] @ git+https://github.com/Bi-xuan/maximal_class_graph.git" pytest
    python test_pkg_mcg.py

Alternatively, run as tests and show the printed graphs:
    python -m pytest -q -s test_pkg_mcg.py

Vertices are labeled 0, 1, 2, 3. Entry [i, j] denotes i -> j, and
every support includes the diagonal (self-loops).
"""

import numpy as np
import pytest
from maximal_class_graph import (
    convert_bitmask_graphs,
    graph_to_maximal_class,
    list_graphs_in_maximal_class,
)


N = 4
NODES = tuple(range(N))
SUPPORTS = (
    (
        "Disconnected pairs",
        np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ], dtype=bool),
        (frozenset({0, 1}), frozenset({2, 3})),
    ),
    (
        "Overlapping maximal classes",
        np.array([
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ], dtype=bool),
        (frozenset({0, 2, 3}), frozenset({1, 2, 3})),
    ),
    (
        "Directed cycle with an isolated vertex",
        np.array([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=bool),
        (frozenset({0, 1, 2}), frozenset({3})),
    ),
        (
        "Identifiable support",
        np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ], dtype=bool),
        (frozenset({0, 3}), frozenset({1,3}), frozenset({2, 3})),
    ),
)


@pytest.mark.parametrize(
    "name,support,expected_family",
    SUPPORTS,
    ids=[case[0] for case in SUPPORTS],
)
def test_maximal_classes_and_graphs(name, support, expected_family):
    assert support.shape == (N, N)
    family = graph_to_maximal_class(support, input_format="matrix", nodes=NODES)
    assert frozenset(family) == frozenset(expected_family)

    print(f"\n{'=' * 60}\n{name} (N={N})")
    print("Support:")
    print(support.astype(int))
    print("Maximal classes:", [sorted(cls) for cls in family])

    # Enumeration takes the entire family, including overlapping classes and
    # singleton classes for isolated vertices.
    results = list_graphs_in_maximal_class(family)
    graphs = convert_bitmask_graphs(results, output_format="matrix")
    assert results.nodes == NODES
    assert graphs
    assert len(results.masks) == len(set(results.masks))
    assert any(np.array_equal(graph, support) for graph in graphs)

    print(f"Corresponding graphs for this maximal-class family ({len(graphs)} total):")
    for index, graph in enumerate(graphs, start=1):
        assert graph.shape == (N, N)
        assert np.all(np.diag(graph))
        recovered = graph_to_maximal_class(
            graph, input_format="matrix", nodes=results.nodes,
        )
        assert frozenset(recovered) == frozenset(family)
        print(f"  Graph {index}: {graph.astype(int).tolist()}")


def main():
    for name, support, expected_family in SUPPORTS:
        test_maximal_classes_and_graphs(name, support, expected_family)


if __name__ == "__main__":
    main()
