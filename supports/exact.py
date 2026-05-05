"""Exact exhaustive support enumeration for small node counts."""

from itertools import combinations
from math import comb

try:
    from .common import off_diagonal_edges, support_mask_from_edges, validate_n_edge
except ImportError:
    from common import off_diagonal_edges, support_mask_from_edges, validate_n_edge


def get_all_supports(n, n_edge):
    """
    Yields (n x n) binary masks where:
    - all diagonal entries are always True (unconstrained)
    - exactly n_edge off-diagonal entries are True (enumerated over all combinations)
    """
    validate_n_edge(n, n_edge)
    off_diag = off_diagonal_edges(n)
    for chosen in combinations(off_diag, n_edge):
        yield support_mask_from_edges(n, chosen)

# Example usage and test case
if __name__ == "__main__":
    n, n_edge = 3, 2
    expected_count = comb(n * (n - 1), n_edge)

    support_count = 0
    first_mask = None

    for mask in get_all_supports(n, n_edge):
        if first_mask is None:
            first_mask = mask.copy()
        total = mask.sum()
        diag_count = sum(mask[i, i] for i in range(n))
        offdiag_count = total - diag_count
        assert diag_count == n,      f"Expected {n} diagonal entries, got {diag_count}"
        assert offdiag_count == n_edge, f"Expected {n_edge} off-diag entries, got {offdiag_count}"
        support_count += 1

    assert support_count == expected_count, (
        f"Expected {expected_count} supports, got {support_count}"
    )

    print(f"Number of supports: {support_count}")  # C(6, 2) = 15

    # Inspect the first support
    print("\nFirst mask:")
    print(first_mask.astype(int))

    print("\nAll masks passed verification.")
