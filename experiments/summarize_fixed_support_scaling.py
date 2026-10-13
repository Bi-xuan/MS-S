#!/usr/bin/env python3
"""Summarize final and best-available supports in a fixed-support study."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from math import fsum, isfinite
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from score_support import same_maximal_class, score_support
from supports.common import validate_support_mask

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
    ec_recovery: bool
    score: float
    best_on_path_selection: bool
    true_support_on_path: bool
    oracle_exact_support_recovery: bool | None = None
    oracle_ec_recovery: bool | None = None
    oracle_score: float | None = None
    lasso_exact_support_recovery: bool | None = None
    lasso_ec_recovery: bool | None = None
    lasso_score: float | None = None


def _mask_edges(mask: np.ndarray, support_scope: str = "all") -> frozenset[tuple[int, int]]:
    """Return directed edges within the requested scope, excluding the diagonal."""

    mask = np.asarray(mask, dtype=bool).copy()
    np.fill_diagonal(mask, False)
    if support_scope == "upper":
        mask = np.triu(mask, k=1)
    return frozenset(
        (int(i), int(j))
        for i, j in np.argwhere(mask)
    )


def _load_trial(
    curve_path: Path,
    selection_path: Path,
    expected_num_samples: int,
    expected_seed: int,
    support_scope: str | None = None,
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
        saved_scope = str(np.asarray(data["support_scope"]))
        dimensions = np.asarray(data["d_m_values"], dtype=int)
        masks = np.asarray(data["selected_support_masks"], dtype=bool)
        valid = np.asarray(data["selected_support_valid"], dtype=bool)
        true_edges = frozenset(
            tuple(map(int, edge))
            for edge in np.asarray(data["lambda_star_support_edges"])
        )
        oracle_edges = None
        if "oracle_support_mask" in data:
            oracle_dimension = int(np.asarray(data["oracle_dimension"]))
            if oracle_dimension != int(np.asarray(data["lambda_star_dimension"])):
                raise ValueError(f"{curve_path} has an incorrect oracle dimension.")
            allowed_edges = _mask_edges(np.ones((n, n), dtype=bool), saved_scope)
            oracle_mask = validate_support_mask(
                data["oracle_support_mask"], n, oracle_dimension - 1, allowed_edges,
            )
            oracle_edges = _mask_edges(oracle_mask)
        lasso_edges = None
        if "lasso_support_mask" in data:
            lasso_mask = np.asarray(data["lasso_support_mask"], dtype=bool)
            if (lasso_mask.shape != (n, n) or not np.all(np.diag(lasso_mask))
                    or np.any(np.tril(lasso_mask, -1))):
                raise ValueError(f"{curve_path} has an invalid upper-triangular lasso support.")
            lasso_edges = _mask_edges(lasso_mask)

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
    if saved_scope not in ("all", "upper"):
        raise ValueError(f"{curve_path} has unsupported support_scope={saved_scope!r}.")
    if support_scope is not None and saved_scope != support_scope:
        raise ValueError(f"{curve_path} records support_scope={saved_scope!r}, expected {support_scope!r}.")
    if masks.shape != (len(dimensions), n, n):
        raise ValueError(
            f"{curve_path} has support-mask shape {masks.shape}, expected "
            f"{(len(dimensions), n, n)}."
        )
    if valid.shape != dimensions.shape or not np.all(valid):
        raise ValueError(f"{curve_path} contains an invalid support on its path.")
    if len(set(map(int, dimensions))) != len(dimensions):
        raise ValueError(f"{curve_path} contains duplicate model dimensions.")

    candidate_edges = _mask_edges(np.ones((n, n), dtype=bool), saved_scope)
    if not true_edges or not true_edges < candidate_edges:
        raise ValueError(
            f"{curve_path} must have at least one true and one false edge within scope={saved_scope!r}."
        )

    path_edges = tuple(_mask_edges(mask) for mask in masks)
    if any(not edges <= candidate_edges for edges in path_edges):
        raise ValueError(f"{curve_path} contains edges outside support_scope={saved_scope!r}.")
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
        raise ValueError(f"{selection_path} contains edges outside support_scope={saved_scope!r}.")
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
        ec_recovery=same_maximal_class(selected_edges, true_edges, n),
        score=score_support(selected_edges, true_edges, n),
        best_on_path_selection=selected_error == best_error,
        true_support_on_path=best_error == 0,
        oracle_exact_support_recovery=(oracle_edges == true_edges if oracle_edges is not None else None),
        oracle_ec_recovery=(same_maximal_class(oracle_edges, true_edges, n) if oracle_edges is not None else None),
        oracle_score=(score_support(oracle_edges, true_edges, n) if oracle_edges is not None else None),
        lasso_exact_support_recovery=(lasso_edges == true_edges if lasso_edges is not None else None),
        lasso_ec_recovery=(same_maximal_class(lasso_edges, true_edges, n) if lasso_edges is not None else None),
        lasso_score=(score_support(lasso_edges, true_edges, n) if lasso_edges is not None else None),
    )


def collect_trials(
    input_dir: Path,
    sample_sizes: tuple[int, ...],
    expected_seeds: int,
    selection_file: str = "selection_plateau_bootstrap.json",
    selection_root: Path | None = None,
    support_scope: str | None = None,
) -> list[TrialSummary]:
    """Collect validated trials in sample-size and numeric-seed order."""

    selection_root = selection_root or input_dir
    trials = []
    resolved_scope = support_scope
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
            curve_path = seed_dir / "objective_curve_sigma_hat.npz"
            if resolved_scope is None:
                with np.load(curve_path, allow_pickle=False) as data:
                    resolved_scope = str(np.asarray(data["support_scope"]))
            trials.append(
                _load_trial(
                    curve_path,
                    selection_root / sample_dir.name / seed_dir.name / selection_file,
                    num_samples,
                    random_seed,
                    resolved_scope,
                )
            )
    return trials


def _has_oracle(trials: list[TrialSummary]) -> bool:
    present = [trial.oracle_score is not None for trial in trials]
    if any(present) and not all(present):
        raise ValueError("Oracle results are missing for some trials; run oracle on the complete study.")
    return bool(present) and all(present)


def _has_lasso(trials: list[TrialSummary]) -> bool:
    present = [trial.lasso_score is not None for trial in trials]
    if any(present) and not all(present):
        raise ValueError("Lasso results are missing for some trials; run lasso on the complete study.")
    return bool(present) and all(present)


def write_trials_csv(trials: list[TrialSummary], output_path: Path) -> None:
    """Write auditable per-seed metrics."""

    has_oracle = _has_oracle(trials)
    has_lasso = _has_lasso(trials)
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
                "ec_recovery",
                "score",
                "best_on_path_selection",
                "true_support_on_path",
            ) + (("oracle_exact_support_recovery", "oracle_ec_recovery", "oracle_score")
                 if has_oracle else ())
            + (("lasso_exact_support_recovery", "lasso_ec_recovery", "lasso_score")
               if has_lasso else ())
        )
        for trial in trials:
            writer.writerow(
                (
                    trial.num_samples,
                    trial.random_seed,
                    trial.selected_dimension,
                    f"{trial.precision:.12g}",
                    str(trial.exact_support_recovery).lower(),
                    str(trial.ec_recovery).lower(),
                    f"{trial.score:.17g}",
                    str(trial.best_on_path_selection).lower(),
                    str(trial.true_support_on_path).lower(),
                ) + ((str(trial.oracle_exact_support_recovery).lower(),
                      str(trial.oracle_ec_recovery).lower(), f"{trial.oracle_score:.17g}")
                     if has_oracle else ())
                + ((str(trial.lasso_exact_support_recovery).lower(),
                    str(trial.lasso_ec_recovery).lower(), f"{trial.lasso_score:.17g}")
                   if has_lasso else ())
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
        f"EC recovery (x/{expected_seeds}) | "
        f"Average EC score over {expected_seeds} seeds | "
        f"Best on path selection (x/{expected_seeds}) | "
        f"True support on path (x/{expected_seeds}) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    has_oracle = _has_oracle(trials)
    has_lasso = _has_lasso(trials)
    if has_oracle:
        lines[0] += (f" Oracle exact support recovery (x/{expected_seeds}) |"
                     f" Oracle EC recovery (x/{expected_seeds}) |"
                     f" Average oracle EC score over {expected_seeds} seeds |")
        lines[1] += "---:|---:|---:|"
    if has_lasso:
        lines[0] += (f" Lasso exact support recovery (x/{expected_seeds}) |"
                     f" Lasso EC recovery (x/{expected_seeds}) |"
                     f" Average lasso EC score over {expected_seeds} seeds |")
        lines[1] += "---:|---:|---:|"
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
        ec_count = sum(trial.ec_recovery for trial in sample_trials)
        mean_score = fsum(trial.score for trial in sample_trials) / len(sample_trials)
        best_count = sum(trial.best_on_path_selection for trial in sample_trials)
        truth_count = sum(trial.true_support_on_path for trial in sample_trials)
        lines.append(
            f"| {num_samples} | {mean_precision:.6f} | "
            f"{exact_count}/{expected_seeds} | {ec_count}/{expected_seeds} | "
            f"{mean_score:.6f} | {best_count}/{expected_seeds} | "
            f"{truth_count}/{expected_seeds} |"
        )
        if has_oracle:
            oracle_exact = sum(trial.oracle_exact_support_recovery for trial in sample_trials)
            oracle_ec = sum(trial.oracle_ec_recovery for trial in sample_trials)
            oracle_mean = fsum(trial.oracle_score for trial in sample_trials) / len(sample_trials)
            lines[-1] += f" {oracle_exact}/{expected_seeds} | {oracle_ec}/{expected_seeds} | {oracle_mean:.6f} |"
        if has_lasso:
            lasso_exact = sum(trial.lasso_exact_support_recovery for trial in sample_trials)
            lasso_ec = sum(trial.lasso_ec_recovery for trial in sample_trials)
            lasso_mean = fsum(trial.lasso_score for trial in sample_trials) / len(sample_trials)
            lines[-1] += f" {lasso_exact}/{expected_seeds} | {lasso_ec}/{expected_seeds} | {lasso_mean:.6f} |"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overall_score_summary(trial_paths: list[Path], output_path: Path) -> None:
    """Average individual trial scores by sample size across all supports."""

    scores = defaultdict(list)
    oracle_scores = defaultdict(list)
    oracle_presence = []
    lasso_scores = defaultdict(list)
    lasso_presence = []
    for trial_path in trial_paths:
        with trial_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"num_samples", "score"} <= set(reader.fieldnames or ()):
                raise ValueError(f"{trial_path} must contain num_samples and score columns.")
            has_oracle = "oracle_score" in (reader.fieldnames or ())
            oracle_presence.append(has_oracle)
            has_lasso = "lasso_score" in (reader.fieldnames or ())
            lasso_presence.append(has_lasso)
            row_count = 0
            for row in reader:
                num_samples, score = int(row["num_samples"]), float(row["score"])
                if num_samples < 1 or not isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"{trial_path} contains an invalid sample size or score.")
                scores[num_samples].append(score)
                if has_oracle:
                    oracle_score = float(row["oracle_score"])
                    if not isfinite(oracle_score) or not 0 <= oracle_score <= 1:
                        raise ValueError(f"{trial_path} contains an invalid oracle score.")
                    oracle_scores[num_samples].append(oracle_score)
                if has_lasso:
                    lasso_score = float(row["lasso_score"])
                    if not isfinite(lasso_score) or not 0 <= lasso_score <= 1:
                        raise ValueError(f"{trial_path} contains an invalid lasso score.")
                    lasso_scores[num_samples].append(lasso_score)
                row_count += 1
            if not row_count:
                raise ValueError(f"{trial_path} contains no trials.")
    if not scores:
        raise ValueError("No trial scores to summarize.")
    if any(oracle_presence) and not all(oracle_presence):
        raise ValueError("Oracle scores are missing from some per-support trial CSVs.")
    if any(lasso_presence) and not all(lasso_presence):
        raise ValueError("Lasso scores are missing from some per-support trial CSVs.")
    lines = ["| Number of samples | Average EC score |", "|---:|---:|"]
    if oracle_scores:
        lines[0] += " Average oracle EC score |"
        lines[1] += "---:|"
    if lasso_scores:
        lines[0] += " Average lasso EC score |"
        lines[1] += "---:|"
    for num_samples, values in sorted(scores.items()):
        lines.append(f"| {num_samples} | {fsum(values) / len(values):.6f} |")
        if oracle_scores:
            values = oracle_scores[num_samples]
            lines[-1] += f" {fsum(values) / len(values):.6f} |"
        if lasso_scores:
            values = lasso_scores[num_samples]
            lines[-1] += f" {fsum(values) / len(values):.6f} |"
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
        "--support-scope",
        choices=("all", "upper"),
        default=None,
        help="Require this scope; otherwise infer it from the saved curves.",
    )
    parser.add_argument(
        "--selection-file",
        default="selection_plateau_bootstrap.json",
        help="Selection JSON filename within each seed directory.",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=None,
        help="Root containing per-trial selection JSONs (default: input directory).",
    )
    parser.add_argument(
        "--overall-trials",
        nargs="+",
        type=Path,
        help="Write only an overall score table from these per-support trial CSVs.",
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
    if args.overall_trials:
        summary_output = args.summary_output or args.input_dir / "selection_summary.md"
        write_overall_score_summary(args.overall_trials, summary_output)
        print(f"Saved overall score table to {summary_output}")
        return
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
                            args.selection_file, args.selection_root,
                            args.support_scope)
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
