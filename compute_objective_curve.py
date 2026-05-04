import argparse

import numpy as np

from main import optimize_lambda


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
    if omega < 0.0:
        raise ValueError("omega must be nonnegative.")
    if lambda_star_spectral_radius(Lambda_star) >= 1.0:
        raise ValueError(
            "All eigenvalues of Lambda_star must be smaller than 1 in absolute value."
        )

    n = Lambda_star.shape[0]
    system_matrix = np.eye(n * n) - np.kron(Lambda_star.T, Lambda_star.T)
    rhs = (omega * np.eye(n)).reshape(-1, order="F")
    sigma_vec = np.linalg.solve(system_matrix, rhs)
    Sigma = sigma_vec.reshape((n, n), order="F")
    return 0.5 * (Sigma + Sigma.T)


def lambda_star_spectral_radius(Lambda_star):
    return np.max(np.abs(np.linalg.eigvals(Lambda_star)))


def compute_objective_curve(
    Sigma,
    beta=1.0,
    max_iter=500,
    tol=1e-6,
    zero_tol=1e-5,
    obj_tol=1e-8,
    max_restarts=5,
    fallback_candidates=500,
    fallback_max_fro_norm=0.95,
    random_seed=42,
    fallback_seed=None,
    min_supported_offdiag_abs=1e-3,
    min_omega=0.0,
    omega_ref=None,
    stop_obj_threshold=1e-6,
    n_jobs=1,
):
    n = Sigma.shape[0]
    max_dm = n * (n - 1) + 1

    exact_d_m_values = []
    exact_objective_values = []
    fallback_d_m_values = []
    fallback_objective_values = []

    for d_m in range(1, max_dm + 1):
        print(f"Solving for D_m = {d_m}...")
        np.random.seed(random_seed)
        Lambda, omega, obj = optimize_lambda(
            Sigma,
            d_m,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            zero_tol=zero_tol,
            obj_tol=obj_tol,
            max_restarts=max_restarts,
            min_supported_offdiag_abs=min_supported_offdiag_abs,
            min_omega=min_omega,
            omega_fixed=omega_ref,
            n_jobs=n_jobs,
            random_seed=random_seed,
        )

        if (
            Lambda is None
            or omega is None
            or not np.all(np.isfinite(Lambda))
            or not np.isfinite(omega)
            or not np.isfinite(obj)
        ):
            print(
                f"  No finite solution was found for D_m = {d_m}; "
                "recording objective = Inf."
            )
            exact_d_m_values.append(d_m)
            exact_objective_values.append(np.inf)
            continue

        print(f"  Objective = {obj:.6f}")
        if obj < stop_obj_threshold:
            print(
                f"  Objective is below {stop_obj_threshold:.1e}; "
                "stopping remaining D_m values without saving this point."
            )
            break
        exact_d_m_values.append(d_m)
        exact_objective_values.append(obj)

    d_m_values = np.array(exact_d_m_values)
    objective_values = np.minimum.accumulate(np.array(exact_objective_values))

    return (
        d_m_values,
        objective_values,
        np.array([
            d_m
            for d_m, obj in zip(fallback_d_m_values, fallback_objective_values)
            if obj >= stop_obj_threshold
        ]),
        np.array([
            obj
            for obj in fallback_objective_values
            if obj >= stop_obj_threshold
        ]),
    )


def lambda_star_for_dimension(n):
    if n < 2:
        raise ValueError("n must be at least 2.")

    Lambda_star = np.zeros((n, n))
    diag_values = np.linspace(0.10, 0.55, n)
    last_col_values = np.linspace(0.60, -0.45, n - 1)

    np.fill_diagonal(Lambda_star, diag_values)
    Lambda_star[:n - 1, n - 1] = last_col_values
    return Lambda_star


def output_path_for_dimension(output_path, n, add_suffix):
    if not add_suffix:
        return output_path

    stem, dot, extension = output_path.rpartition(".")
    if not dot:
        return f"{output_path}_n{n}"
    return f"{stem}_n{n}.{extension}"


def save_curve_result(
    output_path,
    curve_type,
    n,
    Lambda_star,
    Sigma,
    omega_star,
    omega_ref,
    random_seed,
    solve_seed,
    fallback_seed,
    num_samples,
    stop_obj_threshold,
    curve_result,
):
    (
        d_m_values,
        objective_values,
        fallback_d_m_values,
        fallback_objective_values,
    ) = curve_result

    np.savez_compressed(
        output_path,
        curve_type=curve_type,
        n=n,
        Lambda_star=Lambda_star,
        Sigma=Sigma,
        omega_star=omega_star,
        omega_ref=omega_ref,
        random_seed=random_seed,
        solve_seed=solve_seed,
        fallback_seed=fallback_seed,
        num_samples=num_samples,
        stop_obj_threshold=stop_obj_threshold,
        d_m_values=d_m_values,
        objective_values=objective_values,
        fallback_d_m_values=fallback_d_m_values,
        fallback_objective_values=fallback_objective_values,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute objective curves and save NPZ data for later plotting."
    )
    parser.add_argument(
        "--given-output",
        default="objective_curve_given_sigma.npz",
        help="Path to save computed curve data for the given Sigma.",
    )
    parser.add_argument(
        "--sigma-hat-output",
        default="objective_curve_sigma_hat_from_given_sigma.npz",
        help="Path to save computed curve data for Sigma_hat built from the given Sigma.",
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
        help="Deprecated; failed D_m solves are now recorded as Inf.",
    )
    parser.add_argument(
        "--stop-obj-threshold",
        type=float,
        default=1e-6,
        help="Stop solving larger D_m values once a finite objective is below this threshold.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of worker processes for support-level parallelism.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Base seed used to derive all random seeds in the experiment.",
    )
    parser.add_argument(
        "--lambda-star-dims",
        type=int,
        nargs="+",
        default=[4],
        help="Dimensions of Lambda_star to compute.",
    )
    return parser.parse_args()


def run_experiment(args, n, add_output_suffix):
    given_solve_seed = args.random_seed
    given_fallback_seed = args.random_seed + 1
    sigma_hat_sample_seed = args.random_seed + 2
    sigma_hat_solve_seed = args.random_seed + 3
    sigma_hat_fallback_seed = args.random_seed + 4

    Lambda_star = lambda_star_for_dimension(n)
    omega_star = 1.0
    omega_ref = omega_star

    lambda_star_radius = lambda_star_spectral_radius(Lambda_star)
    if lambda_star_radius >= 1.0:
        raise ValueError(
            "All eigenvalues of Lambda_star must be smaller than 1 "
            "in absolute value."
        )

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

    print(f"\n=== Lambda_star dimension n = {n} ===")
    print("Lambda_star:")
    print(Lambda_star)
    print(f"Spectral radius of Lambda_star: {lambda_star_radius:.6f}")
    print(f"omega_star: {omega_star:.6f}")
    print(f"omega_ref for support selection: {omega_ref:.6f}")

    print("Given Sigma:")
    print(Sigma_given)
    given_curve = compute_objective_curve(
        Sigma_given,
        max_iter=800,
        tol=1e-7,
        fallback_candidates=args.fallback_candidates,
        random_seed=given_solve_seed,
        fallback_seed=given_fallback_seed,
        omega_ref=omega_ref,
        stop_obj_threshold=args.stop_obj_threshold,
        n_jobs=args.n_jobs,
    )
    save_curve_result(
        given_output,
        "given_sigma",
        n,
        Lambda_star,
        Sigma_given,
        omega_star,
        omega_ref,
        args.random_seed,
        given_solve_seed,
        given_fallback_seed,
        -1,
        args.stop_obj_threshold,
        given_curve,
    )
    print(f"Saved given-Sigma curve data to {given_output}")

    Sigma_hat_given = sample_empirical_covariance(
        Sigma_given,
        num_samples=args.num_samples,
        seed=sigma_hat_sample_seed,
    )

    print("\nEmpirical covariance Sigma_hat from given Sigma:")
    print(Sigma_hat_given)
    sigma_hat_curve = compute_objective_curve(
        Sigma_hat_given,
        max_iter=800,
        tol=1e-7,
        fallback_candidates=args.fallback_candidates,
        random_seed=sigma_hat_solve_seed,
        fallback_seed=sigma_hat_fallback_seed,
        omega_ref=omega_ref,
        stop_obj_threshold=args.stop_obj_threshold,
        n_jobs=args.n_jobs,
    )
    save_curve_result(
        sigma_hat_output,
        "sigma_hat_from_given_sigma",
        n,
        Lambda_star,
        Sigma_hat_given,
        omega_star,
        omega_ref,
        args.random_seed,
        sigma_hat_solve_seed,
        sigma_hat_fallback_seed,
        args.num_samples,
        args.stop_obj_threshold,
        sigma_hat_curve,
    )
    print(f"Saved Sigma_hat curve data to {sigma_hat_output}")


if __name__ == "__main__":
    args = parse_args()

    for index, n in enumerate(args.lambda_star_dims):
        run_experiment(args, n, add_output_suffix=index > 0)
