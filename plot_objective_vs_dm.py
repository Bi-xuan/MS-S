import argparse
import numpy as np
import matplotlib.pyplot as plt

from main import optimize_lambda
from objective import frobenius_objective
from support_utils import get_all_supports


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


def covariance_from_lambda_star(Lambda_star, omega):
    n = Lambda_star.shape[0]
    system_matrix = np.eye(n * n) - np.kron(Lambda_star.T, Lambda_star.T)
    rhs = (omega * np.eye(n)).reshape(-1, order="F")
    sigma_vec = np.linalg.solve(system_matrix, rhs)
    Sigma = sigma_vec.reshape((n, n), order="F")
    return 0.5 * (Sigma + Sigma.T)


def sample_lambda_star_from_mask(mask, target_fro_norm=0.5, seed=0):
    if target_fro_norm >= 1.0:
        raise ValueError("target_fro_norm must be smaller than 1.")
    if target_fro_norm <= 0.0:
        raise ValueError("target_fro_norm must be positive.")
    if not np.any(mask):
        raise ValueError("Lambda_star_mask must contain at least one True entry.")

    rng = np.random.default_rng(seed)
    Lambda_star = np.zeros(mask.shape)
    Lambda_star[mask] = rng.normal(size=np.count_nonzero(mask))
    Lambda_star *= target_fro_norm / np.linalg.norm(Lambda_star, "fro")
    return Lambda_star


def optimal_omega_for_lambda(Sigma, Lambda):
    n = Sigma.shape[0]
    residual_without_omega = Sigma - Lambda.T @ Sigma @ Lambda
    return np.trace(residual_without_omega) / n


def random_feasible_candidate(Sigma, mask, rng, max_fro_norm=0.95):
    Lambda = np.zeros(mask.shape)
    Lambda[mask] = rng.normal(size=np.count_nonzero(mask))
    lambda_norm = np.linalg.norm(Lambda, "fro")
    if lambda_norm == 0.0:
        return None

    target_norm = rng.uniform(0.0, max_fro_norm)
    Lambda *= target_norm / lambda_norm
    omega = optimal_omega_for_lambda(Sigma, Lambda)
    obj = frobenius_objective(Sigma, Lambda, omega)

    if (
        not np.all(np.isfinite(Lambda))
        or not np.isfinite(omega)
        or not np.isfinite(obj)
    ):
        return None

    return Lambda, omega, obj


def best_random_feasible_objective(
    Sigma,
    d_m,
    rng,
    num_candidates=500,
    max_fro_norm=0.95,
):
    n = Sigma.shape[0]
    supports = get_all_supports(n, d_m - 1)
    if not supports:
        return None

    best = None
    for _ in range(num_candidates):
        mask = supports[rng.integers(len(supports))]
        candidate = random_feasible_candidate(
            Sigma,
            mask,
            rng,
            max_fro_norm=max_fro_norm,
        )
        if candidate is None:
            continue

        if best is None or candidate[2] < best[2]:
            best = candidate

    return best


def compute_objective_curve(
    Sigma,
    beta=1.0,
    max_iter=500,
    tol=1e-6,
    zero_tol=1e-5,
    obj_tol=1e-8,
    max_restarts=3,
    fallback_candidates=500,
    fallback_max_fro_norm=0.95,
    fallback_seed=0,
):
    n = Sigma.shape[0]
    max_dm = n * (n - 1) + 1

    d_m_values = []
    objective_values = []
    fallback_d_m_values = []
    fallback_objective_values = []
    fallback_rng = np.random.default_rng(fallback_seed)

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
            fallback = best_random_feasible_objective(
                Sigma,
                d_m,
                fallback_rng,
                num_candidates=fallback_candidates,
                max_fro_norm=fallback_max_fro_norm,
            )
            if fallback is not None:
                fallback_d_m_values.append(d_m)
                fallback_objective_values.append(fallback[2])
                print(f"  Random feasible fallback objective = {fallback[2]:.6f}")
            continue

        d_m_values.append(d_m)
        objective_values.append(obj)
        print(f"  Objective = {obj:.6f}")

    return (
        np.array(d_m_values),
        np.array(objective_values),
        np.array(fallback_d_m_values),
        np.array(fallback_objective_values),
    )


def penalty_values_for_d_m(d_m_values):
    return np.sqrt(d_m_values)


def output_path_for_dimension(output_path, n, add_suffix):
    if not add_suffix:
        return output_path

    stem, dot, extension = output_path.rpartition(".")
    if not dot:
        return f"{output_path}_n{n}"
    return f"{stem}_n{n}.{extension}"


def lambda_star_mask_for_dimension(n):
    if n == 3:
        return np.array([
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ], dtype=bool)

    if n == 4:
        return np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ], dtype=bool)

    raise ValueError(f"No Lambda_star mask is defined for dimension {n}.")


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot optimal objective value as a function of D_m for a fixed Sigma."
    )
    parser.add_argument(
        "--given-output",
        default="objective_vs_dm_given_sigma.png",
        help="Path to save the plot image for the given Sigma.",
    )
    parser.add_argument(
        "--sigma-hat-output",
        default="objective_vs_dm_sigma_hat_from_given_sigma.png",
        help="Path to save the plot image for optimal value vs D_m using Sigma_hat built from the given Sigma.",
    )
    parser.add_argument(
        "--sigma-hat-penalty-output",
        default="objective_vs_pen_n_m_sigma_hat_from_given_sigma.png",
        help="Path to save the plot image for optimal value vs pen_n(m) using Sigma_hat built from the given Sigma.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of i.i.d. Gaussian observations used to build Sigma_hat.",
    )
    parser.add_argument(
        "--fallback-candidates",
        type=int,
        default=500,
        help="Number of random feasible candidates to try when ADMM fails for a D_m.",
    )
    parser.add_argument(
        "--lambda-star-dims",
        type=int,
        nargs="+",
        default=[3, 4],
        choices=[3, 4],
        help="Dimensions of Lambda_star to test.",
    )
    return parser.parse_args()


def run_experiment(args, n, add_output_suffix):
    Lambda_star_mask = lambda_star_mask_for_dimension(n)
    Lambda_star = sample_lambda_star_from_mask(
        Lambda_star_mask,
        target_fro_norm=0.9,
        seed=0,
    )
    omega_star = 1.0

    lambda_star_fro = np.linalg.norm(Lambda_star, "fro")
    if lambda_star_fro >= 1.0:
        raise ValueError("Lambda_star must have Frobenius norm smaller than 1.")

    Sigma_given = covariance_from_lambda_star(Lambda_star, omega_star)

    given_output = output_path_for_dimension(
        args.given_output,
        n,
        add_output_suffix,
    )
    sigma_hat_output = output_path_for_dimension(
        args.sigma_hat_output,
        n,
        add_output_suffix,
    )
    sigma_hat_penalty_output = output_path_for_dimension(
        args.sigma_hat_penalty_output,
        n,
        add_output_suffix,
    )

    print(f"\n=== Lambda_star dimension n = {n} ===")
    print("Lambda_star:")
    print(Lambda_star)
    print(f"Frobenius norm of Lambda_star: {lambda_star_fro:.6f}")

    print("Given Sigma:")
    print(Sigma_given)
    (
        d_m_values,
        objective_values,
        fallback_d_m_values,
        fallback_objective_values,
    ) = compute_objective_curve(
        Sigma_given,
        fallback_candidates=args.fallback_candidates,
        fallback_seed=0,
    )

    if len(d_m_values) == 0 and len(fallback_d_m_values) == 0:
        print("No finite solutions were found for the given Sigma.")
    else:
        plot_curve(
            d_m_values,
            objective_values,
            given_output,
            title=f"Optimal Objective Value vs D_m (Given Sigma, n={n})",
            xlabel="D_m",
            ylabel="Optimal objective value",
            fallback_x_values=fallback_d_m_values,
            fallback_y_values=fallback_objective_values,
        )
        print(f"Saved given-Sigma plot to {given_output}")

    Sigma_hat_given = sample_empirical_covariance(
        Sigma_given,
        num_samples=args.num_samples,
        seed=0,
    )

    print("\nEmpirical covariance Sigma_hat from given Sigma:")
    print(Sigma_hat_given)

    (
        d_m_values,
        objective_values,
        fallback_d_m_values,
        fallback_objective_values,
    ) = compute_objective_curve(
        Sigma_hat_given,
        fallback_candidates=args.fallback_candidates,
        fallback_seed=1,
    )

    if len(d_m_values) == 0 and len(fallback_d_m_values) == 0:
        print("No finite solutions were found for Sigma_hat built from the given Sigma.")
    else:
        plot_curve(
            d_m_values,
            objective_values,
            sigma_hat_output,
            title=f"Optimal Objective Value vs D_m (Sigma_hat from Given Sigma, n={n})",
            xlabel="D_m",
            ylabel="Optimal objective value",
            fallback_x_values=fallback_d_m_values,
            fallback_y_values=fallback_objective_values,
        )
        print(f"Saved given-Sigma_hat plot to {sigma_hat_output}")

    sigma_hat_penalties = penalty_values_for_d_m(d_m_values)
    sigma_hat_fallback_penalties = penalty_values_for_d_m(fallback_d_m_values)

    plot_curve(
        sigma_hat_penalties,
        objective_values,
        sigma_hat_penalty_output,
        title=f"Optimal Objective Value vs pen_n(m) (Sigma_hat from Given Sigma, n={n})",
        xlabel="pen_n(m)",
        ylabel="Optimal objective value",
        use_data_xticks=False,
        fallback_x_values=sigma_hat_fallback_penalties,
        fallback_y_values=fallback_objective_values,
    )
    print(f"Saved Sigma_hat objective-vs-pen_n(m) plot to {sigma_hat_penalty_output}")


if __name__ == "__main__":
    args = parse_args()

    for index, n in enumerate(args.lambda_star_dims):
        run_experiment(args, n, add_output_suffix=index > 0)
