"""Check ADMM initialization strategies."""

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import admm
from admm import admm_solve, halton_point, initialize_admm_state


def make_problem():
    Sigma = np.array([
        [1.4, 0.2, 0.1],
        [0.2, 1.2, 0.0],
        [0.1, 0.0, 1.1],
    ])
    support_mask = np.eye(3, dtype=bool)
    return Sigma, support_mask


def test_halton_initialization_is_seed_independent():
    _, support_mask = make_problem()

    np.random.seed(1)
    L1_a, L2_a = initialize_admm_state(3, support_mask, 0, "halton")

    np.random.seed(999)
    L1_b, L2_b = initialize_admm_state(3, support_mask, 0, "halton")

    assert np.array_equal(L1_a, L1_b)
    assert np.array_equal(L2_a, L2_b)
    assert np.array_equal(L1_a, L2_a)
    assert not np.all(L1_a[support_mask] == -1.0)


def test_random_initialization_still_available():
    _, support_mask = make_problem()

    np.random.seed(1)
    L1_a, L2_a = initialize_admm_state(3, support_mask, 0, "random")

    np.random.seed(999)
    L1_b, L2_b = initialize_admm_state(3, support_mask, 0, "random")

    assert not np.array_equal(L1_a, L1_b)
    assert not np.array_equal(L2_a, L2_b)


def test_invalid_initialization_strategy():
    _, support_mask = make_problem()

    try:
        initialize_admm_state(3, support_mask, 0, "bad")
    except ValueError:
        return

    raise AssertionError("Expected ValueError for invalid init_strategy.")


def test_admm_solve_accepts_both_initialization_strategies():
    Sigma, support_mask = make_problem()

    for init_strategy in ("halton", "random"):
        np.random.seed(42)
        Lambda, omega = admm_solve(
            Sigma,
            support_mask,
            max_iter=20,
            max_restarts=2,
            init_strategy=init_strategy,
        )

        assert np.all(np.isfinite(Lambda))
        assert np.isfinite(omega)


def test_fixed_omega_admm_uses_radius_and_solves_restricted_blocks(monkeypatch):
    Sigma, support_mask = make_problem()
    support_mask[0, 1] = True
    omega = 0.9
    radius = 2.0
    target = Sigma - omega * np.eye(len(Sigma))
    sigma_norm = np.linalg.norm(Sigma, 2)
    L = 6 * sigma_norm**2 * radius**2 + 2 * sigma_norm * np.linalg.norm(target, 2)
    beta = 1.1 * max(1, 3 * L)
    _, other = initialize_admm_state(len(Sigma), support_mask, 0, "halton")
    solve_on_support = admm.solve_on_support
    updates = []

    def check_update(system, rhs, column_rows):
        other_block = updates[-1] if updates else other
        product = Sigma @ other_block
        np.testing.assert_allclose(system, 2 * product @ product.T + beta * np.eye(len(Sigma)))
        result = solve_on_support(system, rhs, column_rows)
        np.testing.assert_allclose((system @ result - rhs)[support_mask], 0, atol=1e-12)
        np.testing.assert_array_equal(result[~support_mask], 0)
        updates.append(result)
        return result

    monkeypatch.setattr(admm, "solve_on_support", check_update)
    fitted, _ = admm_solve(
        Sigma, support_mask, omega_fixed=omega, radius=radius, max_iter=1, max_restarts=1,
    )
    assert len(updates) == 2
    np.testing.assert_array_equal(fitted, updates[0])


def print_halton_sequence(n, num_points):
    support_mask = np.ones((n, n), dtype=bool)

    for restart_index in range(num_points):
        raw_point = halton_point(restart_index + 1, n * n)
        scaled_matrix, _ = initialize_admm_state(
            n,
            support_mask,
            restart_index,
            "halton",
        )

        print(f"\nrestart_index = {restart_index}")
        print("raw Halton point:")
        print(raw_point.reshape(n, n))
        print("scaled ADMM initializer:")
        print(scaled_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-halton",
        action="store_true",
        help="Print the raw Halton points and scaled ADMM initializers.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Matrix dimension to display with --show-halton.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=5,
        help="Number of Halton restart points to display.",
    )
    args = parser.parse_args()

    if args.show_halton:
        print_halton_sequence(args.n, args.num_points)
    else:
        test_halton_initialization_is_seed_independent()
        test_random_initialization_still_available()
        test_invalid_initialization_strategy()
        test_admm_solve_accepts_both_initialization_strategies()
        print("ADMM initialization tests passed.")
