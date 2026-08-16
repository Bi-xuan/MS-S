"""Plot the selected model dimension over a grid of penalty scales."""

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

from analyze_penalty import build_penalty_constants, scalar_value
from model_selection import select_dimension_path
from penalty import pen_n


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "experiments"
    / "output"
    / "objective_curve_sigma_hat_from_given_sigma.npz"
)

METADATA_KEYS = (
    "curve_type",
    "n",
    "omega_star",
    "omega_ref",
    "random_seed",
    "solve_seed",
    "fallback_seed",
    "stop_obj_threshold",
    "support_scope",
    "preselect_k",
    "preselect_direction_policy",
)


def load_curve_data(input_path):
    """Load the objective curve and relevant experiment metadata."""

    with np.load(input_path) as data:
        required_keys = (
            "d_m_values",
            "objective_values",
            "Sigma",
            "num_samples",
        )
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(
                f"The input NPZ is missing required fields: {missing_keys}."
            )

        d_m_values = data["d_m_values"].copy()
        objective_values = data["objective_values"].copy()
        sigma = data["Sigma"].copy()
        num_samples = int(scalar_value(data, "num_samples"))
        metadata = {
            key: scalar_value(data, key)
            for key in METADATA_KEYS
            if key in data
        }

    return {
        "d_m_values": d_m_values,
        "objective_values": objective_values,
        "Sigma": sigma,
        "num_samples": num_samples,
        "metadata": metadata,
    }


def compute_penalty_values(d_m_values, constants):
    """Compute the unscaled penalty vector once for all candidate dimensions."""

    return np.asarray(
        [pen_n(float(d_m), constants) for d_m in d_m_values],
        dtype=float,
    )


def reference_scale(objective_values, penalty_values):
    """Estimate a data-dependent scale from the two curve spans."""

    finite_objectives = objective_values[np.isfinite(objective_values)]
    if len(finite_objectives) == 0:
        raise ValueError("At least one objective value must be finite.")

    objective_span = float(np.ptp(finite_objectives))
    penalty_span = float(np.ptp(penalty_values))
    if objective_span > 0.0 and penalty_span > 0.0:
        return objective_span / penalty_span
    return 1.0


def construct_C_grid(
    objective_values,
    penalty_values,
    grid_type,
    C_min,
    C_max,
    num_C,
):
    """Construct a user-specified or data-adaptive grid of penalty scales."""

    if num_C < 2:
        raise ValueError("num_C must be at least 2.")

    scale_reference = reference_scale(objective_values, penalty_values)
    if C_min is None:
        C_min = scale_reference * (1e-4 if grid_type == "log" else 0.0)
    if C_max is None:
        C_max = scale_reference * (1e4 if grid_type == "log" else 10.0)

    if not np.isfinite(C_min) or not np.isfinite(C_max):
        raise ValueError("C grid bounds must be finite.")
    if C_min < 0.0:
        raise ValueError("C_min must be nonnegative.")
    if C_max <= C_min:
        raise ValueError("C_max must be greater than C_min.")

    if grid_type == "log":
        if C_min <= 0.0:
            raise ValueError("C_min must be positive for a logarithmic grid.")
        C_values = np.geomspace(C_min, C_max, num=num_C)
    else:
        C_values = np.linspace(C_min, C_max, num=num_C)

    return C_values, scale_reference


def plot_dimension_path(
    C_values,
    selected_dimensions,
    output_path,
    title,
):
    """Draw and save the dimension path on a linear x-axis."""

    plt.figure(figsize=(8, 5))
    plt.step(
        C_values,
        selected_dimensions,
        where="post",
        linewidth=1.7,
        color="tab:blue",
    )
    plt.xscale("linear")
    plt.xlabel("Penalty scale C")
    plt.ylabel(r"Selected dimension $\widehat{D}(C)$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    unique_dimensions = np.unique(selected_dimensions)
    if len(unique_dimensions) <= 15:
        plt.yticks(unique_dimensions)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def report_transitions(C_values, selected_dimensions):
    """Print the observed dimension changes along the sampled path."""

    transition_indices = np.flatnonzero(
        selected_dimensions[1:] != selected_dimensions[:-1]
    )
    if len(transition_indices) == 0:
        print("Observed dimension transitions: none")
        return

    print("Observed dimension transitions:")
    for index in transition_indices:
        print(
            f"  C in [{C_values[index]:.12g}, "
            f"{C_values[index + 1]:.12g}]: "
            f"{selected_dimensions[index]} -> "
            f"{selected_dimensions[index + 1]}"
        )


def run(args):
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_dimension_path.png")
    )

    curve_data = load_curve_data(input_path)
    constants_args = argparse.Namespace(
        num_samples=args.num_samples,
        r=args.r,
        Lm=args.Lm,
        L=args.L,
        xi=args.xi,
    )
    penalty_data = {
        "Sigma": curve_data["Sigma"],
        "num_samples": curve_data["num_samples"],
    }
    if "n" in curve_data["metadata"]:
        penalty_data["n"] = curve_data["metadata"]["n"]
    constants = build_penalty_constants(penalty_data, constants_args)

    d_m_values = curve_data["d_m_values"]
    objective_values = curve_data["objective_values"]
    penalty_values = compute_penalty_values(d_m_values, constants)
    C_values, scale_reference = construct_C_grid(
        objective_values,
        penalty_values,
        args.grid,
        args.C_min,
        args.C_max,
        args.num_C,
    )
    selected_dimensions = select_dimension_path(
        d_m_values,
        objective_values,
        penalty_values,
        C_values,
    )

    metadata = curve_data["metadata"]
    curve_type = metadata.get("curve_type", "objective curve")
    n = int(metadata.get("n", curve_data["Sigma"].shape[0]))
    curve_label = str(curve_type).replace("_", " ")
    title = f"Dimension Path ({curve_label}, n={n})"
    plot_dimension_path(
        C_values,
        selected_dimensions,
        output_path,
        title,
    )

    print(f"Loaded {input_path}")
    print(f"Saved plot to {output_path}")
    print(f"Curve type: {curve_type}")
    print(f"Matrix dimension n: {n}")
    print(f"Penalty num_samples: {constants.num_samples}")
    print(f"Grid type: {args.grid}")
    print(f"Grid points: {len(C_values)}")
    print(f"Data-adaptive reference scale: {scale_reference:.12g}")
    print(f"C range: [{C_values[0]:.12g}, {C_values[-1]:.12g}]")
    print(
        f"Selected dimension range: "
        f"{selected_dimensions[0]} -> {selected_dimensions[-1]}"
    )
    report_transitions(C_values, selected_dimensions)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the penalized model selector over a C-grid and plot "
            "the resulting dimension path."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=(
            "Objective-curve NPZ produced by compute_objective_curve.py. "
            f"Default: {DEFAULT_INPUT}"
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Output PNG path. Defaults to INPUT_STEM_dimension_path.png "
            "next to the input."
        ),
    )
    parser.add_argument(
        "--grid",
        choices=("log", "linear"),
        default="log",
        help="C-grid spacing. Default: log.",
    )
    parser.add_argument(
        "--c-min",
        "--C-min",
        dest="C_min",
        type=float,
        help=(
            "Smallest C. By default this is 1e-4 times the data-adaptive "
            "reference scale."
        ),
    )
    parser.add_argument(
        "--c-max",
        "--C-max",
        dest="C_max",
        type=float,
        help=(
            "Largest C. By default this is 1e4 times the data-adaptive "
            "reference scale for a log grid, or 10 times it for a linear grid."
        ),
    )
    parser.add_argument(
        "--num-c",
        "--num-C",
        dest="num_C",
        type=int,
        default=1000,
        help="Number of C-grid points. Default: 1000.",
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
    parser.add_argument(
        "--r",
        type=float,
        default=1.0,
        help="Theorem radius r. Default: 1.",
    )
    parser.add_argument(
        "--Lm",
        type=float,
        default=1.0,
        help="Model entropy weight Lm. Default: 1.",
    )
    parser.add_argument(
        "--L",
        type=float,
        default=1.0,
        help="Theorem constant L. Default: 1.",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=10.0,
        help="Theorem tail constant xi. Default: 10.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
