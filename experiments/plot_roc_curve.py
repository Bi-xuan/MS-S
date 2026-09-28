#!/usr/bin/env python3
"""Plot mean edge-recovery ROC curves for the fixed-support experiment.

Each seed contributes one ROC path indexed by model dimension. One plot shows
the pointwise mean ROC paths over the 10 seeds for each requested sample size.
A separate plot shows the final support selected in each seed and its mean
operating point.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / "experiments" / "output" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SAMPLE_SIZES = (100, 1_000, 10_000, 1_000_000)
EXPECTED_SEED_COUNT = 10
SEED_DIR_RE = re.compile(r"seed_(\d+)$")


def _seed_curve_files(sample_dir: Path) -> list[tuple[int, Path]]:
    """Return numerically sorted seed/curve pairs under a sample directory."""

    files = []
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing sample directory: {sample_dir}")
    for path in sample_dir.iterdir():
        match = SEED_DIR_RE.fullmatch(path.name)
        curve_path = path / "objective_curve_sigma_hat.npz"
        if path.is_dir() and match and curve_path.is_file():
            files.append((int(match.group(1)), curve_path))
    files.sort()
    if len(files) != EXPECTED_SEED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SEED_COUNT} seed curves under {sample_dir}, "
            f"found {len(files)}."
        )
    return files


def _upper_edges(n: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))


def _load_seed_path(
    curve_path: Path,
    expected_num_samples: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    frozenset[tuple[int, int]],
    tuple[frozenset[tuple[int, int]], ...],
]:
    """Load and validate one nested upper-support recovery path."""

    with np.load(curve_path, allow_pickle=False) as data:
        required = {
            "n",
            "num_samples",
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
    if support_scope != "upper":
        raise ValueError(f"{curve_path} must use support_scope='upper'.")
    if not nested_supports:
        raise ValueError(f"{curve_path} must contain nested supports.")

    candidate_edges = _upper_edges(n)
    expected_dimensions = np.arange(1, len(candidate_edges) + 2)
    if not np.array_equal(dimensions, expected_dimensions):
        raise ValueError(
            f"{curve_path} has dimensions {dimensions.tolist()}, expected "
            f"{expected_dimensions.tolist()}."
        )
    if masks.shape != (len(dimensions), n, n):
        raise ValueError(
            f"{curve_path} has support-mask shape {masks.shape}, expected "
            f"{(len(dimensions), n, n)}."
        )
    if valid.shape != dimensions.shape or not np.all(valid):
        raise ValueError(f"{curve_path} contains an invalid support on its ROC path.")

    candidate_set = frozenset(candidate_edges)
    if not true_edges or not true_edges < candidate_set:
        raise ValueError(
            f"{curve_path} must have at least one true and one false upper edge."
        )

    off_diagonal_masks = masks.copy()
    diagonal = np.arange(n)
    off_diagonal_masks[:, diagonal, diagonal] = False
    upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(off_diagonal_masks & ~upper_triangle):
        raise ValueError(f"{curve_path} contains selected edges outside the upper triangle.")
    if np.any(off_diagonal_masks[:-1] & ~off_diagonal_masks[1:]):
        raise ValueError(f"{curve_path} support masks are not nested.")
    selected_counts = off_diagonal_masks.sum(axis=(1, 2))
    if not np.array_equal(selected_counts, dimensions - 1):
        raise ValueError(
            f"{curve_path} support sizes do not match their model dimensions."
        )

    true_mask = np.zeros((n, n), dtype=bool)
    for i, j in true_edges:
        true_mask[i, j] = True
    false_mask = upper_triangle & ~true_mask
    true_positive_rates = np.count_nonzero(
        off_diagonal_masks & true_mask,
        axis=(1, 2),
    ) / np.count_nonzero(true_mask)
    false_positive_rates = np.count_nonzero(
        off_diagonal_masks & false_mask,
        axis=(1, 2),
    ) / np.count_nonzero(false_mask)
    path_edges = tuple(
        frozenset((int(i), int(j)) for i, j in np.argwhere(mask))
        for mask in off_diagonal_masks
    )
    return (
        dimensions,
        false_positive_rates,
        true_positive_rates,
        true_edges,
        path_edges,
    )


def _load_selected_index(
    selection_path: Path,
    dimensions: np.ndarray,
    path_edges: tuple[frozenset[tuple[int, int]], ...],
) -> int:
    """Locate and validate the final selected support on its saved ROC path."""

    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing final selection: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    required = {"selected_dimension", "selected_edges"}
    missing = required.difference(selection)
    if missing:
        raise ValueError(
            f"{selection_path} is missing required fields: "
            f"{', '.join(sorted(missing))}"
        )

    selected_dimension = int(selection["selected_dimension"])
    matches = np.flatnonzero(dimensions == selected_dimension)
    if len(matches) != 1:
        raise ValueError(
            f"{selection_path} selects dimension {selected_dimension}, which is "
            "not present exactly once on its ROC path."
        )
    selected_index = int(matches[0])
    selected_edge_list = [tuple(map(int, edge)) for edge in selection["selected_edges"]]
    selected_edges = frozenset(selected_edge_list)
    if len(selected_edges) != len(selected_edge_list):
        raise ValueError(f"{selection_path} contains duplicate selected edges.")
    if selected_edges != path_edges[selected_index]:
        raise ValueError(
            f"{selection_path} final support does not match the saved support "
            f"at dimension {selected_dimension}."
        )
    return selected_index


def collect_mean_roc(
    input_dir: Path,
) -> dict[
    int,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Collect mean ROC paths and per-seed final operating points."""

    curves = {}
    reference_dimensions = None
    reference_truth = None
    for sample_size in SAMPLE_SIZES:
        sample_dir = input_dir / f"num_samples_{sample_size}"
        false_positive_rates = []
        true_positive_rates = []
        selected_false_positive_rates = []
        selected_true_positive_rates = []
        for _, curve_path in _seed_curve_files(sample_dir):
            dimensions, fpr, tpr, true_edges, path_edges = _load_seed_path(
                curve_path, sample_size
            )
            if reference_dimensions is None:
                reference_dimensions = dimensions
                reference_truth = true_edges
            elif not np.array_equal(dimensions, reference_dimensions):
                raise ValueError(f"{curve_path} has an inconsistent dimension grid.")
            elif true_edges != reference_truth:
                raise ValueError(f"{curve_path} has an inconsistent ground-truth support.")
            false_positive_rates.append(fpr)
            true_positive_rates.append(tpr)
            selected_index = _load_selected_index(
                curve_path.with_name("selection_plateau_bootstrap.json"),
                dimensions,
                path_edges,
            )
            selected_false_positive_rates.append(fpr[selected_index])
            selected_true_positive_rates.append(tpr[selected_index])

        curves[sample_size] = (
            dimensions,
            np.mean(false_positive_rates, axis=0),
            np.mean(true_positive_rates, axis=0),
            np.asarray(selected_false_positive_rates),
            np.asarray(selected_true_positive_rates),
        )
    return curves


def _jitter_overlaps(
    x_values: np.ndarray,
    y_values: np.ndarray,
    phase: float,
    radius: float = 0.012,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply deterministic display-only jitter to coincident points."""

    x_display = np.asarray(x_values, dtype=float).copy()
    y_display = np.asarray(y_values, dtype=float).copy()
    groups: dict[tuple[float, float], list[int]] = {}
    for index, point in enumerate(zip(x_values, y_values)):
        groups.setdefault(tuple(map(float, point)), []).append(index)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        angles = phase + np.linspace(0.0, 2.0 * np.pi, len(indices), endpoint=False)
        x_display[indices] += radius * np.cos(angles)
        y_display[indices] += radius * np.sin(angles)
    return x_display, y_display


def plot_mean_roc(input_dir: Path, output_path: Path) -> None:
    """Draw only the sample-size mean ROC paths on one set of axes."""

    curves = collect_mean_roc(input_dir)
    figure, axis = plt.subplots(figsize=(7.2, 6.4))
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    for color, sample_size in zip(colors, SAMPLE_SIZES):
        _, mean_fpr, mean_tpr, _, _ = curves[sample_size]
        axis.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5.5,
            label=f"num_samples={sample_size:,}",
        )

    axis.plot(
        [0, 1],
        [0, 1],
        color="0.45",
        linewidth=1.2,
        linestyle="--",
        label="Random baseline",
        zorder=0,
    )
    axis.set_xlim(-0.025, 1.025)
    axis.set_ylim(-0.025, 1.025)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Mean edge-recovery ROC curves across 10 seeds (n=4)")
    axis.grid(True, color="0.88", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=True)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_final_selections(input_dir: Path, output_path: Path) -> None:
    """Draw seed-level and mean final-support operating points separately."""

    curves = collect_mean_roc(input_dir)
    figure, axis = plt.subplots(figsize=(9.2, 6.4))
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    sample_handles = []
    for curve_index, (color, sample_size) in enumerate(zip(colors, SAMPLE_SIZES)):
        _, _, _, selected_fpr, selected_tpr = curves[sample_size]
        display_fpr, display_tpr = _jitter_overlaps(
            selected_fpr,
            selected_tpr,
            phase=curve_index * np.pi / 7.0,
        )
        axis.scatter(
            display_fpr,
            display_tpr,
            color=color,
            marker="D",
            s=42,
            alpha=0.32,
            edgecolors="none",
            zorder=3,
        )
        axis.scatter(
            np.mean(selected_fpr),
            np.mean(selected_tpr),
            color=color,
            marker="*",
            s=230,
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
        )
        sample_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=7,
                label=f"num_samples={sample_size:,}",
            )
        )

    axis.plot(
        [0, 1],
        [0, 1],
        color="0.45",
        linewidth=1.2,
        linestyle="--",
        label="Random baseline",
        zorder=0,
    )
    axis.set_xlim(-0.025, 1.025)
    axis.set_ylim(-0.025, 1.025)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Final support operating points across 10 seeds (n=4)")
    axis.grid(True, color="0.88", linewidth=0.8)
    axis.set_axisbelow(True)
    sample_legend = axis.legend(
        handles=sample_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        title="Sample size",
    )
    axis.add_artist(sample_legend)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor="0.45",
                markeredgecolor="none",
                alpha=0.4,
                markersize=6,
                label="Seed final selection",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markerfacecolor="0.6",
                markeredgecolor="black",
                markeredgewidth=0.85,
                markersize=13,
                label="Mean final selection",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 0.48),
        frameon=True,
        title="Final supports",
    )
    figure.text(
        0.5,
        0.015,
        "Coincident seed points use display-only jitter; stars use unjittered means.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "output"
            / "fixed_support_scaling_n4"
        ),
        help="Fixed-support experiment directory (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <input-dir>/roc_curve.png)",
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=None,
        help=(
            "Final-selection PNG path "
            "(default: <input-dir>/final_support_operating_points.png)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input_dir / "roc_curve.png"
    selection_output_path = (
        args.selection_output
        or args.input_dir / "final_support_operating_points.png"
    )
    plot_mean_roc(args.input_dir, output_path)
    plot_final_selections(args.input_dir, selection_output_path)
    print(f"Saved ROC plot to {output_path}")
    print(f"Saved final-support plot to {selection_output_path}")


if __name__ == "__main__":
    main()
