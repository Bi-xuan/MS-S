#!/usr/bin/env python3
"""Plot per-support coverage outcomes from the upper-support scaling study."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUTCOMES = (
    "Both succeed",
    "Window only",
    "Plateau only",
    "Neither succeeds",
)
COLORS = ("#1B9E77", "#66C2A5", "#A6D854", "#D73027")
INVALID_OUTCOME = "Invalid input"
INVALID_COLOR = "#6C3483"


def classify(row: dict[str, str]) -> str:
    if row["scaling_input_status"].strip().lower() != "valid":
        return INVALID_OUTCOME
    window_succeeds = row["window_correct"].strip().lower() == "true"
    plateau_succeeds = row["plateau_correct"].strip().lower() == "true"
    if window_succeeds and plateau_succeeds:
        return "Both succeed"
    if window_succeeds:
        return "Window only"
    if plateau_succeeds:
        return "Plateau only"
    return "Neither succeeds"


def load_counts(summary_path: Path) -> tuple[list[int], dict[int, Counter[str]]]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    with summary_path.open(newline="", encoding="utf-8") as summary_file:
        reader = csv.DictReader(summary_file)
        required = {
            "support_index",
            "random_seed",
            "scaling_input_status",
            "window_correct",
            "plateau_correct",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")
        for row in reader:
            counts[int(row["support_index"])][classify(row)] += 1

    if not counts:
        raise ValueError("The selection summary contains no trials.")
    return sorted(counts), counts


def plot_coverage(summary_path: Path, output_path: Path, pdf_path: Path | None) -> None:
    support_indices, support_counts = load_counts(summary_path)
    overall = Counter()
    for counter in support_counts.values():
        overall.update(counter)

    row_counts = [overall, *(support_counts[index] for index in support_indices)]
    row_labels = ["Overall", *(f"Support {index + 1}" for index in support_indices)]
    valid_totals = [sum(counter[outcome] for outcome in OUTCOMES) for counter in row_counts]
    invalid_totals = [counter[INVALID_OUTCOME] for counter in row_counts]

    figure_height = max(6.5, 0.42 * len(row_labels) + 1.8)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    left = [0.0] * len(row_labels)
    for outcome, color in zip(OUTCOMES, COLORS):
        widths = [
            100.0 * counter[outcome] / total if total else 0.0
            for counter, total in zip(row_counts, valid_totals)
        ]
        axis.barh(
            row_labels,
            widths,
            left=left,
            height=0.72,
            label=outcome,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        for row_number, (start, width, counter) in enumerate(zip(left, widths, row_counts)):
            if width >= 7.0:
                axis.text(
                    start + width / 2,
                    row_number,
                    str(counter[outcome]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if outcome in {"Both succeed", "Neither succeeds"} else "#183028",
                )
        left = [start + width for start, width in zip(left, widths)]

    for row_number, (counter, valid_total, invalid_total) in enumerate(
        zip(row_counts, valid_totals, invalid_totals)
    ):
        covered = valid_total - counter["Neither succeeds"]
        coverage_label = (
            f"{100 * covered / valid_total:.0f}% ({covered}/{valid_total})"
            if valid_total
            else "N/A (0/0)"
        )
        if not valid_total:
            axis.barh(
                row_number,
                100.0,
                height=0.72,
                color="#F2F2F2",
                edgecolor="#BDBDBD",
                hatch="///",
                zorder=0,
            )
        axis.text(
            101.2,
            row_number,
            coverage_label,
            ha="left",
            va="center",
            fontsize=9,
            clip_on=False,
        )
        if invalid_total:
            axis.scatter(
                117.5,
                row_number,
                marker="X",
                s=34,
                color=INVALID_COLOR,
                clip_on=False,
                zorder=4,
            )
            axis.text(
                119.0,
                row_number,
                f"{invalid_total} invalid",
                ha="left",
                va="center",
                fontsize=8.5,
                color=INVALID_COLOR,
                clip_on=False,
            )

    axis.invert_yaxis()
    axis.set_xlim(0, 132)
    axis.set_xlabel("Share of trials (%)")
    axis.set_title("Dimension-selection coverage by true support", pad=4)
    axis.set_xticks(range(0, 101, 20))
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="none",
            markersize=7,
            markerfacecolor=INVALID_COLOR,
            markeredgecolor=INVALID_COLOR,
        )
    )
    labels.append("Invalid input (excluded)")
    axis.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.065),
        ncol=5,
        frameon=False,
    )
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Path to selection_summary.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--pdf-output", type=Path, help="Optional output PDF path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_coverage(args.summary, args.output, args.pdf_output)
    print(f"Saved coverage plot to {args.output}")
    if args.pdf_output is not None:
        print(f"Saved coverage plot to {args.pdf_output}")


if __name__ == "__main__":
    main()
