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


def true_dimension_from_lambda_star(Lambda_star, zero_tol=1e-12):
    off_diag_mask = ~np.eye(Lambda_star.shape[0], dtype=bool)
    return np.count_nonzero(np.abs(Lambda_star[off_diag_mask]) > zero_tol) + 1


def maximal_dimension_from_n(n):
    return n * (n - 1) // 2


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
    highlight_x=None,
    highlight_y=None,
    max_dimension_x=None,
    max_dimension_y=None,
):
    plt.figure(figsize=(8, 5))
    if len(x_values) > 0:
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.5,
            color="tab:blue",
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
    if highlight_x is not None and highlight_y is not None:
        plt.scatter(
            [highlight_x],
            [highlight_y],
            s=160,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=2.0,
            label="True dimension",
            zorder=4,
        )
    if max_dimension_x is not None and max_dimension_y is not None:
        plt.scatter(
            [max_dimension_x],
            [max_dimension_y],
            s=120,
            marker="s",
            facecolors="none",
            edgecolors="tab:green",
            linewidths=2.0,
            label="Max D_m",
            zorder=4,
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
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_labeled_curves(
    curve_data_list,
    output_path,
    title,
    xlabel,
    ylabel,
    x_transform=lambda values: values,
    use_data_xticks=True,
):
    if len(curve_data_list) == 0:
        raise ValueError("At least one curve is required for plotting.")

    reference_data = curve_data_list[0]
    true_dimension = true_dimension_from_lambda_star(reference_data["Lambda_star"])
    max_dimension = maximal_dimension_from_n(reference_data["n"])
    true_x = x_transform(np.array([true_dimension]))[0]
    max_x = x_transform(np.array([max_dimension]))[0]
    all_x_values = []

    plt.figure(figsize=(8, 5))
    for data in curve_data_list:
        x_values = x_transform(data["d_m_values"])
        all_x_values.append(x_values)
        label = sample_count_label(data["num_samples"])
        line = plt.plot(
            x_values,
            data["objective_values"],
            marker="o",
            linewidth=1.5,
            label=label,
        )[0]

        fallback_x_values = x_transform(data["fallback_d_m_values"])
        if len(fallback_x_values) > 0:
            plt.scatter(
                fallback_x_values,
                data["fallback_objective_values"],
                marker="^",
                color=line.get_color(),
                alpha=0.8,
                zorder=3,
            )
            all_x_values.append(fallback_x_values)

    plt.axvline(
        true_x,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="True dimension",
    )
    plt.axvline(
        max_x,
        color="tab:green",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label="Max D_m",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if use_data_xticks and len(all_x_values) > 0:
        tick_values = np.unique(np.concatenate(all_x_values))
        if len(tick_values) <= 15:
            plt.xticks(tick_values)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def sample_count_label(num_samples):
    if num_samples is None or num_samples < 0:
        return "num_samples unknown"
    return f"num_samples = {num_samples}"


def load_curve_data(input_path):
    with np.load(input_path) as data:
        return {
            "curve_type": data["curve_type"].item(),
            "n": int(data["n"]),
            "Lambda_star": data["Lambda_star"],
            "num_samples": int(data["num_samples"]) if "num_samples" in data else None,
            "d_m_values": data["d_m_values"],
            "objective_values": data["objective_values"],
            "fallback_d_m_values": data["fallback_d_m_values"],
            "fallback_objective_values": data["fallback_objective_values"],
        }


def plot_d_m_curve(input_path, output_path, title_template, ylabel):
    data = load_curve_data(input_path)
    title = title_template.format(n=data["n"])
    true_dimension = true_dimension_from_lambda_star(data["Lambda_star"])
    true_index = np.where(data["d_m_values"] == true_dimension)[0]
    highlight_y = None
    if len(true_index) > 0:
        highlight_y = data["objective_values"][true_index[0]]
    max_dimension = maximal_dimension_from_n(data["n"])
    max_dimension_index = np.where(data["d_m_values"] == max_dimension)[0]
    max_dimension_y = None
    if len(max_dimension_index) > 0:
        max_dimension_y = data["objective_values"][max_dimension_index[0]]
    plot_curve(
        data["d_m_values"],
        data["objective_values"],
        output_path,
        title=title,
        xlabel="D_m",
        ylabel=ylabel,
        fallback_x_values=data["fallback_d_m_values"],
        fallback_y_values=data["fallback_objective_values"],
        highlight_x=true_dimension if len(true_index) > 0 else None,
        highlight_y=highlight_y,
        max_dimension_x=max_dimension if len(max_dimension_index) > 0 else None,
        max_dimension_y=max_dimension_y,
    )
    print(f"Saved plot to {output_path}")


def load_sigma_hat_curve_data(input_paths):
    curve_data_list = [load_curve_data(input_path) for input_path in input_paths]
    for input_path, data in zip(input_paths, curve_data_list):
        if data["curve_type"] != "sigma_hat_from_given_sigma":
            raise ValueError(
                f"Expected Sigma_hat curve data in {input_path}, "
                f"but found curve_type={data['curve_type']!r}."
            )

    n_values = {data["n"] for data in curve_data_list}
    if len(n_values) != 1:
        raise ValueError(
            "All Sigma_hat inputs must have the same n to be plotted together."
        )
    return sorted(
        curve_data_list,
        key=lambda data: float("inf") if data["num_samples"] is None else data["num_samples"],
    )


def plot_sigma_hat_d_m_curves(input_paths, output_path, title_template, ylabel):
    curve_data_list = load_sigma_hat_curve_data(input_paths)
    title = title_template.format(n=curve_data_list[0]["n"])
    plot_labeled_curves(
        curve_data_list,
        output_path,
        title=title,
        xlabel="D_m",
        ylabel=ylabel,
    )
    print(f"Saved plot to {output_path}")


def plot_sigma_hat_penalty_curves(input_paths, output_path, title_template, ylabel):
    curve_data_list = load_sigma_hat_curve_data(input_paths)
    title = title_template.format(n=curve_data_list[0]["n"])
    plot_labeled_curves(
        curve_data_list,
        output_path,
        title=title,
        xlabel="pen_n(m)",
        ylabel=ylabel,
        x_transform=penalty_values_for_d_m,
        use_data_xticks=False,
    )
    print(f"Saved plot to {output_path}")


def plot_penalty_curve(input_path, output_path, title_template, ylabel):
    data = load_curve_data(input_path)
    title = title_template.format(n=data["n"])
    true_dimension = true_dimension_from_lambda_star(data["Lambda_star"])
    true_index = np.where(data["d_m_values"] == true_dimension)[0]
    highlight_y = None
    if len(true_index) > 0:
        highlight_y = data["objective_values"][true_index[0]]
    max_dimension = maximal_dimension_from_n(data["n"])
    max_dimension_index = np.where(data["d_m_values"] == max_dimension)[0]
    max_dimension_y = None
    if len(max_dimension_index) > 0:
        max_dimension_y = data["objective_values"][max_dimension_index[0]]
    plot_curve(
        penalty_values_for_d_m(data["d_m_values"]),
        data["objective_values"],
        output_path,
        title=title,
        xlabel="pen_n(m)",
        ylabel=ylabel,
        use_data_xticks=False,
        fallback_x_values=penalty_values_for_d_m(data["fallback_d_m_values"]),
        fallback_y_values=data["fallback_objective_values"],
        highlight_x=np.sqrt(true_dimension) if len(true_index) > 0 else None,
        highlight_y=highlight_y,
        max_dimension_x=np.sqrt(max_dimension) if len(max_dimension_index) > 0 else None,
        max_dimension_y=max_dimension_y,
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
        nargs="+",
        default=["experiments/output/objective_curve_sigma_hat_from_given_sigma.npz"],
        help=(
            "One or more NPZ curve data files for Sigma_hat built from the "
            "given Sigma. Multiple files are overlaid and colored by num_samples."
        ),
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
        "Optimal Objective Value vs D_m (Given Sigma, n={n})",
        ylabel="Optimal objective value",
    )
    plot_sigma_hat_d_m_curves(
        args.sigma_hat_input,
        args.sigma_hat_output,
        "Optimal Objective Value vs D_m (Sigma_hat from Given Sigma, n={n})",
        ylabel="Optimal objective value",
    )
    plot_sigma_hat_penalty_curves(
        args.sigma_hat_input,
        args.sigma_hat_penalty_output,
        "Optimal Objective Value vs pen_n(m) (Sigma_hat from Given Sigma, n={n})",
        ylabel="Optimal objective value",
    )
