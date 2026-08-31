"""Plot an exact dimension path and its selected scaling feature."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / "experiments" / "output" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments import select_scaling_parameter as scaling_parameter_selection
from scaling_selection import build_dimension_path


def _selection_feature_bounds(result):
    """Return the finite bounds of the feature selected by the method."""

    if result.method == "window":
        factor = 1.0 + result.eta
        return (
            result.minimal_scale / factor,
            result.minimal_scale * factor,
        )

    plateau = result.plateau_selection
    return plateau.left, plateau.right


def resolve_plot_bounds(path, result, C_min, C_max):
    """Choose finite positive bounds for displaying the exact path."""

    anchors = list(path.transition_scales)
    anchors.extend(_selection_feature_bounds(result))
    anchors.append(result.minimal_scale)
    positive_anchors = np.asarray(
        [value for value in anchors if np.isfinite(value) and value > 0.0],
        dtype=float,
    )
    if len(positive_anchors) == 0:
        raise ValueError("The dimension path has no positive scale to plot.")

    if C_min is None:
        C_min = float(np.min(positive_anchors) / 2.0)
    if C_max is None:
        C_max = float(np.max(positive_anchors) * 2.0)

    if not np.isfinite(C_min) or C_min <= 0.0:
        raise ValueError("C_min must be a finite, positive number.")
    if not np.isfinite(C_max) or C_max <= C_min:
        raise ValueError("C_max must be finite and greater than C_min.")
    return float(C_min), float(C_max)


def _plot_window_selection(ax, result, C_min, C_max):
    """Highlight the adaptive window and its dominant transition cluster."""

    jump_selection = result.jump_selection
    factor = 1.0 + result.eta
    window_left = result.minimal_scale / factor
    window_right = result.minimal_scale * factor
    visible_left = max(window_left, C_min)
    visible_right = min(window_right, C_max)
    if visible_left < visible_right:
        ax.axvspan(
            visible_left,
            visible_right,
            color="tab:orange",
            alpha=0.18,
            label="Selected window",
        )
    if C_min <= result.minimal_scale <= C_max:
        ax.axvline(
            result.minimal_scale,
            color="tab:orange",
            linewidth=1.8,
            label=r"Selected minimal scale $C_\star$",
        )

    first_transition = True
    for transition in jump_selection.transition_scales:
        if C_min <= transition <= C_max:
            ax.axvline(
                transition,
                color="tab:red",
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label=(
                    "Dominant-cluster transitions"
                    if first_transition
                    else None
                ),
            )
            first_transition = False


def _plot_plateau_selection(ax, result, C_min, C_max):
    """Highlight the plateau selected by the persistent-plateau method."""

    plateau = result.plateau_selection
    left = max(plateau.left, C_min)
    right = min(plateau.right, C_max)
    if left < right:
        ax.axvspan(
            left,
            right,
            color="tab:orange",
            alpha=0.2,
            label="Selected plateau",
        )
        ax.hlines(
            plateau.dimension,
            left,
            right,
            color="tab:orange",
            linewidth=4.0,
        )


def plot_dimension_path(path, result, output_path, title, C_min, C_max):
    """Draw and save the exact piecewise-constant dimension path."""

    internal_breakpoints = [
        value
        for value in path.transition_scales
        if C_min < value < C_max
    ]
    plot_scales = np.asarray(
        [C_min, *internal_breakpoints, C_max],
        dtype=float,
    )
    plot_dimensions = np.asarray(
        [path.dimension_at(scale) for scale in plot_scales],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(
        plot_scales,
        plot_dimensions,
        where="post",
        linewidth=1.7,
        color="tab:blue",
        label="Dimension path",
    )

    if result.method == "window":
        _plot_window_selection(ax, result, C_min, C_max)
    else:
        _plot_plateau_selection(ax, result, C_min, C_max)

    ax.set_xscale("log")
    ax.set_xlim(C_min, C_max)
    ax.set_xlabel("Penalty scale C")
    ax.set_ylabel(r"Selected dimension $\widehat{D}(C)$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    unique_dimensions = np.unique(path.dimensions)
    if len(unique_dimensions) <= 15:
        ax.set_yticks(unique_dimensions)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def report_transitions(path):
    """Print the exact dimension changes in the path."""

    if not path.transition_scales:
        print("Exact dimension transitions: none")
        return

    print("Exact dimension transitions:")
    for scale, before, after in zip(
        path.transition_scales,
        path.dimensions[:-1],
        path.dimensions[1:],
    ):
        print(f"  C = {scale:.12g}: {before} -> {after}")


def run(args):
    """Load an NPZ, select its scale, and plot its exact dimension path."""

    input_path = Path(args.input)
    if input_path.suffix.lower() != ".npz":
        raise ValueError("The input must be an .npz file.")
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_dimension_path.png")
    )

    selection_data = scaling_parameter_selection.load_selection_inputs(
        input_path,
        args,
    )
    result = scaling_parameter_selection.select_minimal_scale(
        selection_data["d_m_values"],
        selection_data["objective_values"],
        selection_data["penalty_values"],
        method=args.method,
        eta=args.eta,
        recommendation_factor=args.recommendation_factor,
    )
    path = build_dimension_path(
        selection_data["d_m_values"],
        selection_data["objective_values"],
        selection_data["penalty_values"],
    )
    C_min, C_max = resolve_plot_bounds(
        path,
        result,
        args.C_min,
        args.C_max,
    )

    constants = selection_data["constants"]
    curve_label = str(selection_data["curve_type"]).replace("_", " ")
    title = f"Dimension Path ({curve_label}, n={constants.n})"
    plot_dimension_path(
        path,
        result,
        output_path,
        title,
        C_min,
        C_max,
    )

    scaling_parameter_selection.report_selection(
        input_path,
        selection_data,
        result,
    )
    print(f"Saved plot to: {output_path}")
    print(f"Plot range: [{C_min:.12g}, {C_max:.12g}]")
    report_transitions(path)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select a penalty scale from an objective-curve NPZ and plot "
            "the exact resulting dimension path."
        )
    )
    parser.add_argument(
        "input",
        help="Objective-curve NPZ produced by compute_objective_curve.py.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output PNG path. Defaults to INPUT_STEM_dimension_path.png "
            "next to the input."
        ),
    )
    parser.add_argument(
        "--method",
        choices=scaling_parameter_selection.METHOD_CHOICES,
        default="window",
        help="Scale-selection procedure. Default: window.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        help="Optional minimum adaptive window width for the window method.",
    )
    parser.add_argument(
        "--recommendation-factor",
        type=float,
        default=scaling_parameter_selection.DEFAULT_RECOMMENDATION_FACTOR,
        help=(
            "Positive multiplier applied to the minimal scale. "
            "Default: 2."
        ),
    )
    parser.add_argument(
        "--objective-floor",
        type=float,
        default=scaling_parameter_selection.DEFAULT_OBJECTIVE_FLOOR,
        help=(
            "Finite raw objectives at or below this value are tied at zero. "
            "Default: 1e-8."
        ),
    )
    parser.add_argument(
        "--c-min",
        "--C-min",
        dest="C_min",
        type=float,
        help="Smallest positive C displayed on the logarithmic axis.",
    )
    parser.add_argument(
        "--c-max",
        "--C-max",
        dest="C_max",
        type=float,
        help="Largest C displayed on the logarithmic axis.",
    )
    parser.add_argument(
        "--num-samples",
        "--penalty-n",
        dest="num_samples",
        type=int,
        help=(
            "Sample count used by the penalty. Defaults to the positive "
            "num_samples value stored in the NPZ."
        ),
    )
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--Lm", "--lm", dest="Lm", type=float, default=1.0)
    parser.add_argument("--L", "--l", dest="L", type=float, default=1.0)
    parser.add_argument("--xi", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
