"""Plot objective-curve NPZ outputs produced by compute_objective_curve.py."""

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / "experiments" / "output" / ".matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np


def penalty_values_for_d_m(d_m_values):
    return np.sqrt(d_m_values)


def plot_curve(
    x_values,
    y_values,
    output_path,
    title,
    xlabel,
    ylabel,
    use_data_xticks=True,
    fallback_x_values=None,
    fallback_y_values=None,
):
    plt.figure(figsize=(8, 5))
    if len(x_values) > 0:
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.5,
            color="tab:blue",
            label="ADMM optimized",
        )
    if fallback_x_values is not None and len(fallback_x_values) > 0:
        plt.scatter(
            fallback_x_values,
            fallback_y_values,
            marker="^",
            color="tab:orange",
            label="Best random feasible fallback",
            zorder=3,
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if use_data_xticks:
        tick_values = x_values
        if fallback_x_values is not None:
            tick_values = np.union1d(tick_values, fallback_x_values)
        if len(tick_values) <= 15:
            plt.xticks(tick_values)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_curve_data(input_path):
    with np.load(input_path) as data:
        return {
            "curve_type": data["curve_type"].item(),
            "n": int(data["n"]),
            "d_m_values": data["d_m_values"],
            "objective_values": data["objective_values"],
            "fallback_d_m_values": data["fallback_d_m_values"],
            "fallback_objective_values": data["fallback_objective_values"],
        }


def plot_d_m_curve(input_path, output_path, title_template):
    data = load_curve_data(input_path)
    title = title_template.format(n=data["n"])
    plot_curve(
        data["d_m_values"],
        data["objective_values"],
        output_path,
        title=title,
        xlabel="D_m",
        ylabel="Nested optimal objective value",
        fallback_x_values=data["fallback_d_m_values"],
        fallback_y_values=data["fallback_objective_values"],
    )
    print(f"Saved plot to {output_path}")


def plot_penalty_curve(input_path, output_path, title_template):
    data = load_curve_data(input_path)
    title = title_template.format(n=data["n"])
    plot_curve(
        penalty_values_for_d_m(data["d_m_values"]),
        data["objective_values"],
        output_path,
        title=title,
        xlabel="pen_n(m)",
        ylabel="Nested optimal objective value",
        use_data_xticks=False,
        fallback_x_values=penalty_values_for_d_m(data["fallback_d_m_values"]),
        fallback_y_values=data["fallback_objective_values"],
    )
    print(f"Saved plot to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot objective curves from NPZ data produced by compute_objective_curve.py."
    )
    parser.add_argument(
        "--given-input",
        default="experiments/output/objective_curve_given_sigma.npz",
        help="NPZ curve data for the given Sigma.",
    )
    parser.add_argument(
        "--sigma-hat-input",
        default="experiments/output/objective_curve_sigma_hat_from_given_sigma.npz",
        help="NPZ curve data for Sigma_hat built from the given Sigma.",
    )
    parser.add_argument(
        "--given-output",
        default="experiments/output/objective_vs_dm_given_sigma.png",
        help="Path to save the given-Sigma D_m plot.",
    )
    parser.add_argument(
        "--sigma-hat-output",
        default="experiments/output/objective_vs_dm_sigma_hat_from_given_sigma.png",
        help="Path to save the Sigma_hat D_m plot.",
    )
    parser.add_argument(
        "--sigma-hat-penalty-output",
        default="experiments/output/objective_vs_pen_n_m_sigma_hat_from_given_sigma.png",
        help="Path to save the Sigma_hat pen_n(m) plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    plot_d_m_curve(
        args.given_input,
        args.given_output,
        "Nested Optimal Objective Value vs D_m (Given Sigma, n={n})",
    )
    plot_d_m_curve(
        args.sigma_hat_input,
        args.sigma_hat_output,
        "Nested Optimal Objective Value vs D_m (Sigma_hat from Given Sigma, n={n})",
    )
    plot_penalty_curve(
        args.sigma_hat_input,
        args.sigma_hat_penalty_output,
        "Nested Optimal Objective Value vs pen_n(m) (Sigma_hat from Given Sigma, n={n})",
    )
