import argparse
import numpy as np
import matplotlib.pyplot as plt

from main import optimize_lambda


def numerical_integral_sqrt_log_term(upper, num_grid=5000):
    if upper <= 0:
        return 0.0

    eps = min(1e-8, upper * 1e-4)
    grid = np.linspace(eps, upper, num_grid)
    integrand = np.sqrt(np.log1p(1.0 / grid))
    return np.trapezoid(integrand, grid)


def empirical_covariance_from_samples(X):
    num_samples = X.shape[0]
    if num_samples < 2:
        raise ValueError("At least two observations are required to compute Sigma_hat.")

    X_centered = X - X.mean(axis=0, keepdims=True)
    return X_centered.T @ X_centered / (num_samples - 1)


def sample_empirical_covariance(Sigma, num_samples, seed=None):
    rng = np.random.default_rng(seed)
    num_features = Sigma.shape[0]
    X = rng.multivariate_normal(
        mean=np.zeros(num_features),
        cov=Sigma,
        size=num_samples,
    )
    return empirical_covariance_from_samples(X)


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


def compute_penalty_curve(Sigma, sample_size, ell, r_value):
    matrix_dim = Sigma.shape[0]
    d_m_values = np.arange(1, matrix_dim * (matrix_dim - 1) + 2)

    eigenvalues = np.linalg.eigvalsh(Sigma)
    lambda_inf = np.max(np.abs(eigenvalues))
    lambda_two = np.linalg.norm(eigenvalues, 2)

    m_value = np.log(sample_size)
    r_bound = np.sqrt(
        np.sum(eigenvalues)
        + 2.0 * lambda_inf * m_value
        + 2.0 * lambda_two * np.sqrt(m_value)
    )

    v_value = sample_size / 2.0
    v_prime = 2.0 * v_value

    sigma_op = np.linalg.norm(Sigma, 2)
    sigma_fro = np.linalg.norm(Sigma, "fro")
    trace_sigma = np.trace(Sigma)

    l_prime = (
        32.0
        * (
            (sample_size**2 / (sample_size - 1) ** 2) * r_bound**4
            + ((sample_size + 1) / (sample_size - 1)) * sigma_op**2
        )
        * ell**3
        + 8.0
        * ((sample_size / (sample_size - 1)) * r_bound**2 + sigma_op)
        * ell**2
        + 8.0
        * (
            (sample_size**2 / (sample_size - 1) ** 2) * r_bound**4
            + 2.0 * r_value * (sample_size / (sample_size - 1)) * r_bound**2
            + sigma_op**2
            + 2.0 * r_value * sigma_op
            + (1.0 / (sample_size - 1)) * sigma_op * (sigma_fro + sigma_op)
        )
        * ell
        + 2.0 * ((sample_size / (sample_size - 1)) * r_bound**2 + abs(trace_sigma))
    )

    integral_value = numerical_integral_sqrt_log_term(l_prime / 4.0)
    k_value = 48.0 * (ell + r_value) * integral_value

    c_value = (
        ((8.0 / (sample_size - 1)) + (4.0 / sample_size))
        * ((8.0 * sample_size / (sample_size - 1)) + (8.0 / (sample_size - 1)) + (4.0 / sample_size))
        * (1.0 + 3.0 * ell**2)
        * r_bound**4
        + 2.0
        * np.sqrt(matrix_dim)
        * ((8.0 / (sample_size - 1)) + (4.0 / sample_size))
        * (2.0 + 4.0 * ell**2)
        * r_value
        * r_bound**2
    )
    v_value = sample_size * c_value**2 / 2.0
    v_prime = 2.0 * v_value

    penalty_values = np.sqrt(d_m_values) * (k_value + np.sqrt(2.0 * v_prime) * ell)

    b_value = sigma_fro**2 * ell**4 * (1.0 + 1.0 / ell**2) + sigma_fro**2 + trace_sigma**2

    constants = {
        "R": r_bound,
        "v": v_value,
        "v_prime": v_prime,
        "K": k_value,
        "L_prime": l_prime,
        "B": b_value,
    }
    return d_m_values, penalty_values, constants


def penalty_values_for_d_m(d_m_values, penalty_d_m_values, penalty_values):
    penalty_map = {
        int(d_m): penalty
        for d_m, penalty in zip(penalty_d_m_values, penalty_values)
    }
    return np.array([penalty_map[int(d_m)] for d_m in d_m_values])


def resample_curve_evenly_in_x(x_values, y_values, num_points=None):
    if len(x_values) <= 1:
        return x_values, y_values

    order = np.argsort(x_values)
    x_sorted = x_values[order]
    y_sorted = y_values[order]

    if num_points is None:
        num_points = len(x_sorted)

    x_even = np.linspace(x_sorted[0], x_sorted[-1], num_points)
    y_even = np.interp(x_even, x_sorted, y_sorted)
    return x_even, y_even


def plot_curve(x_values, y_values, output_path, title, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(x_values) <= 15:
        plt.xticks(x_values)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot optimal objective value as a function of D_m for a fixed Sigma."
    )
    parser.add_argument(
        "--given-output",
        default="objective_vs_dm_given.png",
        help="Path to save the plot image for the given Sigma.",
    )
    parser.add_argument(
        "--random-output",
        default="objective_vs_dm_random.png",
        help="Path to save the plot image for the random Sigma.",
    )
    parser.add_argument(
        "--sigma-hat-output",
        default="objective_vs_dm_sigma_hat.png",
        help="Path to save the plot image for optimal value vs pen_n(m) using Sigma_hat built from the given Sigma.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of i.i.d. Gaussian observations used to build Sigma_hat.",
    )
    parser.add_argument(
        "--ell",
        type=float,
        default=0.5,
        help="Positive constant L_m = L in (0, 1).",
    )
    parser.add_argument(
        "--r-value",
        type=float,
        default=0.5,
        help="Positive constant r in (0, 1).",
    )
    parser.add_argument(
        "--sigma-hat-plot-points",
        type=int,
        default=None,
        help="Number of evenly spaced points to use on the pen_n(m) axis for the Sigma_hat plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    Sigma_given = np.array([
        [2.0, 0.0, 0.3],
        [0.0, 1.5, 0.4],
        [0.3, 0.4, 1.2],
    ])

    print("Given Sigma:")
    print(Sigma_given)
    d_m_values, objective_values = compute_objective_curve(Sigma_given)

    if len(d_m_values) == 0:
        print("No finite solutions were found for the given Sigma.")
    else:
        plot_curve(
            d_m_values,
            objective_values,
            args.given_output,
            title="Optimal Objective Value vs D_m (Given Sigma)",
            xlabel="D_m",
            ylabel="Optimal objective value",
        )
        print(f"Saved given-Sigma plot to {args.given_output}")

    np.random.seed(1)
    n = Sigma_given.shape[0]
    A = np.random.randn(n, n)
    Sigma_random = A @ A.T / n

    print("\nRandom Sigma:")
    print(Sigma_random)
    d_m_values, objective_values = compute_objective_curve(Sigma_random)

    if len(d_m_values) == 0:
        print("No finite solutions were found for the random Sigma.")
    else:
        plot_curve(
            d_m_values,
            objective_values,
            args.random_output,
            title="Optimal Objective Value vs D_m (Random Sigma)",
            xlabel="D_m",
            ylabel="Optimal objective value",
        )
        print(f"Saved random-Sigma plot to {args.random_output}")

    Sigma_hat_given = sample_empirical_covariance(
        Sigma_given,
        num_samples=args.num_samples,
        seed=0,
    )

    print("\nEmpirical covariance Sigma_hat from given Sigma:")
    print(Sigma_hat_given)

    d_m_values, objective_values = compute_objective_curve(Sigma_hat_given)

    if len(d_m_values) == 0:
        print("No finite solutions were found for Sigma_hat built from the given Sigma.")
    else:
        plot_curve(
            d_m_values,
            objective_values,
            args.sigma_hat_output,
            title="Optimal Objective Value vs D_m (Sigma_hat from Given Sigma)",
            xlabel = "D_m",
            ylabel="Optimal objective value",
        )
        print(f"Saved given-Sigma_hat plot to {args.sigma_hat_output}")

    penalty_d_m_values, penalty_values, penalty_constants = compute_penalty_curve(
        Sigma_given,
        sample_size=args.num_samples,
        ell=args.ell,
        r_value=args.r_value,
    )
    print("\nPenalty constants based on Sigma_given:")
    for key, value in penalty_constants.items():
        print(f"{key} = {value}")

    sigma_hat_penalties = penalty_values_for_d_m(
        d_m_values,
        penalty_d_m_values,
        penalty_values,
    )
    sigma_hat_penalties_even, objective_values_even = resample_curve_evenly_in_x(
        sigma_hat_penalties,
        objective_values,
        num_points=args.sigma_hat_plot_points,
    )

    plot_curve(
        sigma_hat_penalties_even,
        objective_values_even,
        args.sigma_hat_output,
        title="Optimal Objective Value vs pen_n(m) (Sigma_hat from Given Sigma)",
        xlabel="pen_n(m)",
        ylabel="Optimal objective value",
    )
    print(f"Saved Sigma_hat objective-vs-penalty plot to {args.sigma_hat_output}")
