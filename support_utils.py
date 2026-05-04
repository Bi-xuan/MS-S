from math import comb
from itertools import combinations
import numpy as np

def get_all_supports(n, n_edge):
    """
    Yields (n x n) binary masks where:
    - all diagonal entries are always True (unconstrained)
    - exactly n_edge off-diagonal entries are True (enumerated over all combinations)
    """
    off_diag = [(i, j) for i in range(n) for j in range(n) if i != j]
    for chosen in combinations(off_diag, n_edge):
        mask = np.zeros((n, n), dtype=bool)
        # Always include all diagonal entries
        for i in range(n):
            mask[i, i] = True
        # Add the chosen off-diagonal entries
        for (i, j) in chosen:
            mask[i, j] = True
        yield mask

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
