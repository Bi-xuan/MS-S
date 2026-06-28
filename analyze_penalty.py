"""Plot penalized objective values from an objective-curve NPZ output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / "experiments" / "output" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from penalty import PenaltyConstants, pen_n


def scalar_value(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value


def true_dimension_from_lambda_star(lambda_star, zero_tol):
    off_diag_mask = ~np.eye(lambda_star.shape[0], dtype=bool)
    return np.count_nonzero(np.abs(lambda_star[off_diag_mask]) > zero_tol) + 1


def eigenvalue_constants_from_sigma(sigma):
    eigenvalues = np.linalg.eigvalsh(sigma)
    return {
        "lambda_sum": float(np.sum(eigenvalues)),
        "lambda_inf_norm": float(np.max(np.abs(eigenvalues))),
        "lambda_2_norm": float(np.linalg.norm(eigenvalues)),
    }


def sigma_constants(sigma):
    return {
        "sigma_fro_norm": float(np.linalg.norm(sigma, ord="fro")),
        "sigma_op_norm": float(np.linalg.norm(sigma, ord=2)),
        "sigma_trace": float(np.trace(sigma)),
    }


def infer_sample_count(data, num_samples_override):
    if num_samples_override is not None:
        return num_samples_override

    num_samples = scalar_value(data, "num_samples")
    if num_samples is not None and int(num_samples) > 0:
        return int(num_samples)

    raise ValueError(
        "num_samples is required because the NPZ does not contain a positive "
        "num_samples. Pass it with --num-samples."
    )


def infer_r(data, r_override):
    if r_override is not None:
        return r_override

    omega_ref = scalar_value(data, "omega_ref")
    if omega_ref is not None and np.isfinite(omega_ref) and float(omega_ref) > 0:
        return float(omega_ref)

    omega_star = scalar_value(data, "omega_star")
    if omega_star is not None and np.isfinite(omega_star) and float(omega_star) > 0:
        return float(omega_star)

    raise ValueError(
        "r is required because neither omega_ref nor omega_star gives a positive "
        "finite default. Pass it with --r."
    )


def build_penalty_constants(data, args):
    if "Sigma" not in data:
        raise ValueError("The input NPZ must contain Sigma to derive penalty constants.")

    sigma = data["Sigma"]
    matrix_dimension = int(scalar_value(data, "n", sigma.shape[0]))
    return PenaltyConstants(
        num_samples=infer_sample_count(data, args.num_samples),
        n=matrix_dimension,
        r=infer_r(data, args.r),
        Lm=args.Lm,
        L=args.L,
        xi=args.xi,
        **eigenvalue_constants_from_sigma(sigma),
        **sigma_constants(sigma),
    )


def penalized_values(d_m_values, objective_values, constants, penalty_scale):
    penalties = np.array([pen_n(float(d_m), constants) for d_m in d_m_values])
    scaled_penalties = penalty_scale * penalties
    return objective_values + scaled_penalties, penalties, scaled_penalties


def finite_argmin(values):
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return None
    finite_indices = np.flatnonzero(finite_mask)
    return finite_indices[np.argmin(values[finite_mask])]


def plot_penalized_curve(
    d_m_values,
    penalized_objective_values,
    true_dimension,
    selected_dimension,
    output_path,
    title,
    fallback_d_m_values=None,
    fallback_penalized_objective_values=None,
):
    plt.figure(figsize=(8, 5))
    plt.plot(
        d_m_values,
        penalized_objective_values,
        marker="o",
        linewidth=1.5,
        color="tab:blue",
        label="Exact support search",
    )

    if fallback_d_m_values is not None and len(fallback_d_m_values) > 0:
        plt.scatter(
            fallback_d_m_values,
            fallback_penalized_objective_values,
            marker="^",
            color="tab:orange",
            label="Best random feasible fallback",
            zorder=3,
        )

    true_index = np.where(d_m_values == true_dimension)[0]
    if len(true_index) > 0:
        plt.scatter(
            [true_dimension],
            [penalized_objective_values[true_index[0]]],
            s=170,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=2.0,
            label="True dimension",
            zorder=4,
        )
    else:
        plt.axvline(
            true_dimension,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="True dimension",
        )

    if selected_dimension is not None:
        selected_index = np.where(d_m_values == selected_dimension)[0]
        if len(selected_index) > 0:
            plt.scatter(
                [selected_dimension],
                [penalized_objective_values[selected_index[0]]],
                s=250,
                facecolors="none",
                edgecolors="tab:green",
                linewidths=2.0,
                label="Selected dimension",
                zorder=5,
            )

    all_d_m_values = d_m_values
    if fallback_d_m_values is not None and len(fallback_d_m_values) > 0:
        all_d_m_values = np.union1d(all_d_m_values, fallback_d_m_values)

    plt.xlabel("D_m")
    plt.ylabel("Optim objective + pen_n(D_m)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(all_d_m_values) <= 15:
        plt.xticks(all_d_m_values)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def analyze(args):
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_penalized_objective_vs_dm.png"
    )

    with np.load(input_path) as data:
        d_m_values = data["d_m_values"]
        objective_values = data["objective_values"]
        lambda_star = data["Lambda_star"]
        constants = build_penalty_constants(data, args)
        true_dimension = true_dimension_from_lambda_star(lambda_star, args.zero_tol)

        penalized_objective_values, penalties, scaled_penalties = penalized_values(
            d_m_values,
            objective_values,
            constants,
            args.penalty_scale,
        )

        fallback_d_m_values = data["fallback_d_m_values"]
        fallback_penalized_objective_values = None
        if len(fallback_d_m_values) > 0:
            fallback_penalized_objective_values, _, _ = penalized_values(
                fallback_d_m_values,
                data["fallback_objective_values"],
                constants,
                args.penalty_scale,
            )

        curve_type = scalar_value(data, "curve_type", "objective curve")
        n = int(scalar_value(data, "n", lambda_star.shape[0]))

    best_index = finite_argmin(penalized_objective_values)
    selected_dimension = None
    if best_index is not None:
        selected_dimension = d_m_values[best_index]

    title = f"Penalized Objective vs D_m ({curve_type}, n={n})"
    plot_penalized_curve(
        d_m_values,
        penalized_objective_values,
        true_dimension,
        selected_dimension,
        output_path,
        title,
        fallback_d_m_values=fallback_d_m_values,
        fallback_penalized_objective_values=fallback_penalized_objective_values,
    )

    print(f"Loaded {input_path}")
    print(f"Saved plot to {output_path}")
    print(f"Penalty num_samples: {constants.num_samples}")
    print(f"Penalty n (matrix dimension): {constants.n}")
    print(f"Penalty scale: {args.penalty_scale:.12g}")
    print(f"True dimension: {true_dimension}")
    if best_index is not None:
        print(f"Selected D_m: {d_m_values[best_index]}")
        print(f"Objective: {objective_values[best_index]:.12g}")
        print(f"Raw penalty: {penalties[best_index]:.12g}")
        print(f"Scaled penalty: {scaled_penalties[best_index]:.12g}")
        print(f"Penalized objective: {penalized_objective_values[best_index]:.12g}")
    else:
        print("Selected D_m: unavailable because all penalized values are non-finite.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute optim obj + pen_n(D_m) from an objective-curve NPZ and "
            "plot the penalized objective against D_m."
        )
    )
    parser.add_argument("input", help="Input NPZ file produced by compute_objective_curve.py.")
    parser.add_argument(
        "--output",
        help=(
            "Path for the output PNG. Defaults to INPUT_STEM_penalized_objective_vs_dm.png "
            "next to the input."
        ),
    )
    parser.add_argument(
        "--num-samples",
        "--penalty-n",
        dest="num_samples",
        type=int,
        help=(
            "Sample count used as PenaltyConstants.num_samples. Defaults to positive num_samples "
            "in the NPZ."
        ),
    )
    parser.add_argument(
        "--penalty-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to pen_n(D_m) before adding it to the objective. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--r",
        type=float,
        help="Theorem radius r. Defaults to positive omega_ref, then omega_star, from the NPZ.",
    )
    parser.add_argument(
        "--Lm",
        type=float,
        default=1.0,
        help="Model entropy weight Lm used by pen_n. Default: 1.",
    )
    parser.add_argument(
        "--L",
        type=float,
        default=1.0,
        help="Theorem constant L used by pen_n. Default: 1.",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=10.0,
        help="Theorem tail constant xi used by pen_n. Default: 10.",
    )
    parser.add_argument(
        "--zero-tol",
        type=float,
        default=1e-12,
        help="Tolerance for counting nonzero off-diagonal Lambda_star entries.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
