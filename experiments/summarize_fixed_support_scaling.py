#!/usr/bin/env python3
"""Summarize final and best-available supports in a fixed-support study."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "experiments" / "output" / "fixed_support_scaling_n4"
DEFAULT_SAMPLE_SIZES = (100, 1_000, 10_000, 1_000_000)
SEED_DIR_RE = re.compile(r"seed_(\d+)$")


@dataclass(frozen=True)
class TrialSummary:
    num_samples: int
    random_seed: int
    selected_dimension: int
    precision: float
    exact_support_recovery: bool
    best_on_path_selection: bool
    true_support_on_path: bool


def _mask_edges(mask: np.ndarray) -> frozenset[tuple[int, int]]:
    """Return zero-based upper-triangular edges from a support mask."""

    return frozenset(
        (int(i), int(j))
        for i, j in np.argwhere(np.triu(np.asarray(mask, dtype=bool), k=1))
    )


def _load_trial(
    curve_path: Path,
    selection_path: Path,
    expected_num_samples: int,
    expected_seed: int,
) -> TrialSummary:
    """Load one trial and compare its final support with its full path."""

    with np.load(curve_path, allow_pickle=False) as data:
        required = {
            "n",
            "num_samples",
            "random_seed",
            "support_scope",
            "nested_supports",
            "lambda_star_support_edges",
            "d_m_values",
            "selected_support_masks",
            "selected_support_valid",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"{curve_path} is missing required fields: "
                f"{', '.join(sorted(missing))}"
            )

        n = int(np.asarray(data["n"]))
        num_samples = int(np.asarray(data["num_samples"]))
        random_seed = int(np.asarray(data["random_seed"]))
        support_scope = str(np.asarray(data["support_scope"]))
        nested_supports = bool(np.asarray(data["nested_supports"]))
        dimensions = np.asarray(data["d_m_values"], dtype=int)
        masks = np.asarray(data["selected_support_masks"], dtype=bool)
        valid = np.asarray(data["selected_support_valid"], dtype=bool)
        true_edges = frozenset(
            tuple(map(int, edge))
            for edge in np.asarray(data["lambda_star_support_edges"])
        )

    if num_samples != expected_num_samples:
        raise ValueError(
            f"{curve_path} records num_samples={num_samples}, expected "
            f"{expected_num_samples}."
        )
    if random_seed != expected_seed:
        raise ValueError(
            f"{curve_path} records random_seed={random_seed}, expected "
            f"{expected_seed}."
        )
    if support_scope != "upper":
        raise ValueError(f"{curve_path} must use support_scope='upper'.")
    if not nested_supports:
        raise ValueError(f"{curve_path} must contain nested supports.")
    if masks.shape != (len(dimensions), n, n):
        raise ValueError(
            f"{curve_path} has support-mask shape {masks.shape}, expected "
            f"{(len(dimensions), n, n)}."
        )
    if valid.shape != dimensions.shape or not np.all(valid):
        raise ValueError(f"{curve_path} contains an invalid support on its path.")
    if len(set(map(int, dimensions))) != len(dimensions):
        raise ValueError(f"{curve_path} contains duplicate model dimensions.")

    candidate_edges = frozenset(_mask_edges(np.ones((n, n), dtype=bool)))
    if not true_edges or not true_edges < candidate_edges:
        raise ValueError(
            f"{curve_path} must have at least one true and one false upper edge."
        )

    path_edges = tuple(_mask_edges(mask) for mask in masks)
    # Symmetric-difference size is exactly FP + FN for edge supports.
    path_errors = np.asarray(
        [len(edges.symmetric_difference(true_edges)) for edges in path_edges],
        dtype=int,
    )

    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing final selection: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    required_selection = {"selected_dimension", "selected_edges"}
    missing_selection = required_selection.difference(selection)
    if missing_selection:
        raise ValueError(
            f"{selection_path} is missing required fields: "
            f"{', '.join(sorted(missing_selection))}"
        )

    selected_dimension = int(selection["selected_dimension"])
    selected_indices = np.flatnonzero(dimensions == selected_dimension)
    if len(selected_indices) != 1:
        raise ValueError(
            f"{selection_path} selects dimension {selected_dimension}, which is "
            "not present exactly once on its path."
        )
    selected_index = int(selected_indices[0])
    selected_edge_list = [tuple(map(int, edge)) for edge in selection["selected_edges"]]
    selected_edges = frozenset(selected_edge_list)
    if len(selected_edges) != len(selected_edge_list):
        raise ValueError(f"{selection_path} contains duplicate selected edges.")
    if not selected_edges <= candidate_edges:
        raise ValueError(f"{selection_path} contains edges outside the upper triangle.")
    if selected_edges != path_edges[selected_index]:
        raise ValueError(
            f"{selection_path} final support does not match the saved support "
            f"at dimension {selected_dimension}."
        )

    true_positive_count = len(selected_edges & true_edges)
    precision = (
        true_positive_count / len(selected_edges)
        if selected_edges
        else 0.0
    )
    saved_precision = selection.get("precision")
    if saved_precision is not None and not np.isclose(
        float(saved_precision), precision, rtol=1e-12, atol=1e-12
    ):
        raise ValueError(
            f"{selection_path} records precision={saved_precision}, but the "
            f"supports imply precision={precision}."
        )

    selected_error = int(path_errors[selected_index])
    best_error = int(np.min(path_errors))
    return TrialSummary(
        num_samples=num_samples,
        random_seed=random_seed,
        selected_dimension=selected_dimension,
        precision=float(precision),
        exact_support_recovery=selected_edges == true_edges,
        best_on_path_selection=selected_error == best_error,
        true_support_on_path=best_error == 0,
    )


def collect_trials(
    input_dir: Path,
    sample_sizes: tuple[int, ...],
    expected_seeds: int,
    selection_file: str = "selection_plateau_bootstrap.json",
) -> list[TrialSummary]:
    """Collect validated trials in sample-size and numeric-seed order."""

    trials = []
    for num_samples in sample_sizes:
        sample_dir = input_dir / f"num_samples_{num_samples}"
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Missing sample directory: {sample_dir}")

        seed_directories = []
        for path in sample_dir.iterdir():
            match = SEED_DIR_RE.fullmatch(path.name)
            if path.is_dir() and match:
                seed_directories.append((int(match.group(1)), path))
        seed_directories.sort()
        if len(seed_directories) != expected_seeds:
            raise ValueError(
                f"Expected {expected_seeds} seed directories under {sample_dir}, "
                f"found {len(seed_directories)}."
            )

        for random_seed, seed_dir in seed_directories:
            trials.append(
                _load_trial(
                    seed_dir / "objective_curve_sigma_hat.npz",
                    seed_dir / selection_file,
                    num_samples,
                    random_seed,
                )
            )
    return trials


def write_trials_csv(trials: list[TrialSummary], output_path: Path) -> None:
    """Write auditable per-seed metrics."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(
            (
                "num_samples",
                "random_seed",
                "selected_dimension",
                "precision",
                "exact_support_recovery",
                "best_on_path_selection",
                "true_support_on_path",
            )
        )
        for trial in trials:
            writer.writerow(
                (
                    trial.num_samples,
                    trial.random_seed,
                    trial.selected_dimension,
                    f"{trial.precision:.12g}",
                    str(trial.exact_support_recovery).lower(),
                    str(trial.best_on_path_selection).lower(),
                    str(trial.true_support_on_path).lower(),
                )
            )


def write_markdown_summary(
    trials: list[TrialSummary],
    sample_sizes: tuple[int, ...],
    expected_seeds: int,
    output_path: Path,
) -> None:
    """Write the aggregate fixed-support Markdown table."""

    lines = [
        f"| Number of samples | Average precision over {expected_seeds} seeds | "
        f"Exact support recovery (x/{expected_seeds}) | "
        f"Best on path selection (x/{expected_seeds}) | "
        f"True support on path (x/{expected_seeds}) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for num_samples in sample_sizes:
        sample_trials = [
            trial for trial in trials if trial.num_samples == num_samples
        ]
        if len(sample_trials) != expected_seeds:
            raise ValueError(
                f"Expected {expected_seeds} collected trials for num_samples="
                f"{num_samples}, found {len(sample_trials)}."
            )
        mean_precision = float(np.mean([trial.precision for trial in sample_trials]))
        exact_count = sum(trial.exact_support_recovery for trial in sample_trials)
        best_count = sum(trial.best_on_path_selection for trial in sample_trials)
        truth_count = sum(trial.true_support_on_path for trial in sample_trials)
        lines.append(
            f"| {num_samples} | {mean_precision:.6f} | "
            f"{exact_count}/{expected_seeds} | {best_count}/{expected_seeds} | "
            f"{truth_count}/{expected_seeds} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Fixed-support experiment output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SAMPLE_SIZES,
        help="Sample-size directories to summarize (default: %(default)s)",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=10,
        help="Required number of seeds per sample size (default: %(default)s)",
    )
    parser.add_argument(
        "--selection-file",
        default="selection_plateau_bootstrap.json",
        help="Selection JSON filename within each seed directory.",
    )
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=None,
        help="Per-seed CSV path (default: <input-dir>/selection_trials.csv)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Markdown summary path (default: <input-dir>/selection_summary.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_seeds < 1:
        raise ValueError("expected-seeds must be positive.")
    sample_sizes = tuple(args.sample_sizes)
    if not sample_sizes or any(value < 1 for value in sample_sizes):
        raise ValueError("sample-sizes must contain positive integers.")
    if len(set(sample_sizes)) != len(sample_sizes):
        raise ValueError("sample-sizes must not contain duplicates.")

    trials_output = args.trials_output or args.input_dir / "selection_trials.csv"
    summary_output = args.summary_output or args.input_dir / "selection_summary.md"
    trials = collect_trials(args.input_dir, sample_sizes, args.expected_seeds,
                            args.selection_file)
    write_trials_csv(trials, trials_output)
    write_markdown_summary(
        trials,
        sample_sizes,
        args.expected_seeds,
        summary_output,
    )
    print(f"Saved trial results to {trials_output}")
    print(f"Saved summary table to {summary_output}")


if __name__ == "__main__":
    main()
