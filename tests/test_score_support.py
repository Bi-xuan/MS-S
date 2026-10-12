"""Verify the score formula, EC distances, and study summary integration."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.summarize_fixed_support_scaling import (
    _load_trial,
    write_overall_score_summary,
)
from score_support import (
    distance_to_maximal_class,
    jaccard_distance,
    maximal_class_distance,
    same_maximal_class,
    score_support,
)


ROOT = Path(__file__).resolve().parents[1]
TRUTH = frozenset({(0, 1), (1, 2)})


def test_score_exact_same_ec_and_nearest_member():
    assert score_support(TRUTH, TRUTH, n=3) == 1.0
    transitive = TRUTH | {(0, 2)}
    assert same_maximal_class(transitive, TRUTH, n=3)
    assert jaccard_distance(transitive, TRUTH) == pytest.approx(1 / 3)
    assert score_support(transitive, TRUTH, n=3) == 1.0

    # This edge is absent from truth but can be retained in a two-edge member
    # of truth's EC, so distance to the EC is 1/2 rather than distance 1 to truth.
    selected = {(0, 2)}
    assert not same_maximal_class(selected, TRUTH, n=3)
    assert jaccard_distance(selected, TRUTH) == 1.0
    assert distance_to_maximal_class(selected, ({0, 1, 2},)) == 0.5
    assert score_support(selected, TRUTH, n=3) == 0.5


def test_distances_include_lower_edges_and_exclude_self_loops():
    assert jaccard_distance([], []) == 0.0
    assert jaccard_distance({(0, 0), (0, 1)}, {(0, 1)}) == 0.0
    assert same_maximal_class({(1, 0)}, {(0, 1)}, n=2)
    assert distance_to_maximal_class({(1, 0)}, ({0, 1},)) == 0.0
    assert score_support({(1, 0), (1, 1)}, {(0, 1)}, n=2) == 1.0
    assert maximal_class_distance(({0, 1},), ({0}, {1})) == 1.0
    # A two-edge cycle on {0, 1} needs only one added edge to reach vertex 2.
    assert maximal_class_distance(({0, 1}, {2}), ({0, 1, 2},)) == pytest.approx(1 / 3)


@pytest.mark.parametrize("selected, truth, expected", [
    (set(), set(), 1.0),
    (set(), TRUTH, 0.0),
    ({(0, 1)}, set(), 0.0),
    ({(0, 1)}, TRUTH, 0.5),
    ({(0, 1), (1, 2), (2, 3)}, TRUTH, 2 / 3),
])
def test_score_boundaries_and_isolated_vertices(selected, truth, expected):
    assert score_support(selected, truth, n=4) == pytest.approx(expected)


def save_trial(directory, samples, seed, selected_dimension, *, edge_order=None):
    directory.mkdir(parents=True, exist_ok=True)
    edge_order = edge_order or [(0, 1), (1, 2), (0, 2)]
    masks = np.repeat(np.eye(3, dtype=bool)[None], 4, axis=0)
    for index, (i, j) in enumerate(edge_order, 1):
        masks[index:, i, j] = True
    curve = directory / "objective_curve_sigma_hat.npz"
    np.savez(
        curve, n=3, num_samples=samples, random_seed=seed,
        support_scope="upper", nested_supports=True,
        lambda_star_support_edges=np.array(sorted(TRUTH)),
        d_m_values=np.arange(1, 5), selected_support_masks=masks,
        selected_support_valid=np.ones(4, dtype=bool),
    )
    selection = directory / "selection_plateau_bootstrap.json"
    selection.write_text(json.dumps({
        "selected_dimension": selected_dimension,
        "selected_edges": edge_order[:selected_dimension - 1],
    }))
    return curve, selection


def test_best_on_path_still_uses_minimum_fp_plus_fn(tmp_path):
    curve, selection = save_trial(
        tmp_path, 100, 1, 2, edge_order=[(0, 1), (0, 2), (1, 2)])
    summary = _load_trial(curve, selection, 100, 1)
    assert summary.best_on_path_selection  # Ties the full support at FP + FN = 1.
    assert not summary.true_support_on_path
    assert not summary.exact_support_recovery
    assert not summary.ec_recovery
    assert summary.score == 0.5
    assert summary.score < score_support(TRUTH | {(0, 2)}, TRUTH, n=3)


def test_overall_average_weights_individual_trials(tmp_path):
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    paths[0].write_text("num_samples,score\n100,1\n100,1\n1000,0.8\n")
    paths[1].write_text("num_samples,score\n100,0.25\n1000,0.2\n")
    output = tmp_path / "selection_summary.md"
    write_overall_score_summary(paths, output)
    assert output.read_text().splitlines() == [
        "| Number of samples | Average EC score |", "|---:|---:|",
        "| 100 | 0.750000 |", "| 1000 | 0.500000 |",
    ]


def test_upper_study_aggregate_writes_ec_recovery_and_scores(tmp_path):
    seeds_by_size = {
        100: [124, 139, 141, 147, 153, 156, 173, 192, 237, 243],
        1000: [6, 15, 23, 46, 51, 53, 59, 61, 74, 85],
        10000: [3, 4, 6, 8, 12, 13, 18, 21, 22, 27],
        1000000: [2, 3, 124, 139, 147, 153, 156, 173, 192, 237],
    }
    for support, dimension in [(0, 3), (1, 4)]:
        for samples, seeds in seeds_by_size.items():
            for seed in seeds:
                save_trial(tmp_path / f"support_{support:02d}/num_samples_{samples}/seed_{seed}",
                           samples, seed, dimension)
    env = dict(os.environ, OUTPUT_ROOT=str(tmp_path), NUM_SUPPORTS="2",
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(
        ["bash", str(ROOT / "experiments/run_upper_support_scaling_study.sh"), "aggregate"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    per_support_score = 1.0
    expected_row = f"| 100 | 0.666667 | 0/10 | 10/10 | {per_support_score:.6f} | 0/10 | 10/10 |"
    support_summary = (tmp_path / "support_01/selection_summary.md").read_text()
    assert expected_row in support_summary
    assert "EC recovery (x/10) | Average EC score over 10 seeds" in support_summary
    assert f"| 100 | {(1 + per_support_score) / 2:.6f} |" in (
        tmp_path / "selection_summary.md").read_text()
    with (tmp_path / "support_01/selection_trials.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert all(row["ec_recovery"] == "true" for row in rows)
    assert all(float(row["score"]) == pytest.approx(per_support_score) for row in rows)
