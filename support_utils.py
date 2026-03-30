# Enumerate all supports
from itertools import combinations
import numpy as np

def get_all_supports(n, n_edge):
    """
    Returns a list of (n x n) binary masks where:
    - all diagonal entries are always True (unconstrained)
    - exactly n_edge off-diagonal entries are True (enumerated over all combinations)
    """
    off_diag = [(i, j) for i in range(n) for j in range(n) if i != j]
    supports = []
    for chosen in combinations(off_diag, n_edge):
        mask = np.zeros((n, n), dtype=bool)
        # Always include all diagonal entries
        for i in range(n):
            mask[i, i] = True
        # Add the chosen off-diagonal entries
        for (i, j) in chosen:
            mask[i, j] = True
        supports.append(mask)
    return supports

# Example usage and test case
if __name__ == "__main__":
    from itertools import combinations
    import numpy as np

    n, n_edge = 3, 2
    supports = get_all_supports(n, n_edge)

    print(f"Number of supports: {len(supports)}")  # C(6, 2) = 15

    # Inspect the first support
    print("\nFirst mask:")
    print(supports[0].astype(int))

    # Verify every mask has exactly n + n_edge True entries
    for mask in supports:
        total = mask.sum()
        diag_count = sum(mask[i, i] for i in range(n))
        offdiag_count = total - diag_count
        assert diag_count == n,      f"Expected {n} diagonal entries, got {diag_count}"
        assert offdiag_count == n_edge, f"Expected {n_edge} off-diag entries, got {offdiag_count}"

    print("\nAll masks passed verification.")
