import argparse
import numpy as np
import matplotlib.pyplot as plt

from main import optimize_lambda


def compute_objective_curve(
    Sigma,
    beta=1.0,
    max_iter=500,
    tol=1e-6,
    zero_tol=1e-5,
    obj_tol=1e-8,
    max_restarts=3,
):
    n = Sigma.shape[0]
    max_dm = n * (n - 1) + 1

    d_m_values = []
    objective_values = []

    for d_m in range(1, max_dm + 1):
        print(f"Solving for D_m = {d_m}...")
        Lambda, omega, obj = optimize_lambda(
            Sigma,
            d_m,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            zero_tol=zero_tol,
            obj_tol=obj_tol,
            max_restarts=max_restarts,
        )

        if (
            Lambda is None
            or omega is None
            or not np.all(np.isfinite(Lambda))
            or not np.isfinite(omega)
            or not np.isfinite(obj)
        ):
            print(f"  Skipping D_m = {d_m} because no finite solution was found.")
            continue

        d_m_values.append(d_m)
        objective_values.append(obj)
        print(f"  Objective = {obj:.6f}")

    return np.array(d_m_values), np.array(objective_values)


def plot_objective_curve(d_m_values, objective_values, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(d_m_values, objective_values, marker="o", linewidth=1.5)
    plt.xlabel("D_m")
    plt.ylabel("Optimal objective value")
    plt.title("Optimal Objective Value vs D_m")
    plt.grid(True, alpha=0.3)
    plt.xticks(d_m_values)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot optimal objective value as a function of D_m for a fixed Sigma."
    )
    parser.add_argument(
        "--output",
        default="objective_vs_dm.png",
        help="Path to save the plot image.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    Sigma = np.array([
        [2.0, 0.0, 0.3],
        [0.0, 1.5, 0.4],
        [0.3, 0.4, 1.2],
    ])

    d_m_values, objective_values = compute_objective_curve(Sigma)

    if len(d_m_values) == 0:
        print("No finite solutions were found for any D_m.")
    else:
        plot_objective_curve(d_m_values, objective_values, args.output)
        print(f"Saved plot to {args.output}")
