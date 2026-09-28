"""Fitted reference omega and the fixed-omega eigenvalue bound."""

import json
import numpy as np
import pytest

from experiments import compute_objective_curve as curve
from optimizers.support_search import optimize_lambda, resolve_omega_ref
from scaling_selection import FitSettings


def test_fitted_reference_uses_smallest_eigenvalue_and_stays_fixed():
    sigma_hat = np.diag([1.0, 2.0])
    Lambda, omega, objective = optimize_lambda(
        sigma_hat, D_m=1, fit_omega_ref=True, kappa=0.93,
        max_iter=20, max_restarts=1,
    )

    assert Lambda is not None
    assert omega == pytest.approx(0.93)
    assert np.isfinite(objective)


def test_fixed_reference_can_equal_smallest_eigenvalue_with_preselection():
    Sigma = np.eye(2)
    Lambda, omega, objective = optimize_lambda(
        Sigma, D_m=2, omega_ref=1.0, omega_upper_gap=0.1,
        preselect_k=1, max_iter=20, max_restarts=1,
    )

    assert Lambda is not None
    assert omega == 1.0
    assert np.isfinite(objective)


@pytest.mark.parametrize("omega_ref, fit_omega_ref, kappa", [
    (0.5, True, 0.93),
    (None, True, 0.0),
    (None, True, 1.1),
])
def test_invalid_fitted_reference_configuration(omega_ref, fit_omega_ref, kappa):
    with pytest.raises(ValueError):
        resolve_omega_ref(np.eye(2), omega_ref, fit_omega_ref, kappa)


def test_objective_curves_share_sigma_hat_reference_without_changing_generation(tmp_path):
    args = curve.parse_args([
        "--given-output", str(tmp_path / "given.npz"),
        "--sigma-hat-output", str(tmp_path / "sample.npz"),
        "--lambda-star-dims", "2", "--support-scope", "upper",
        "--omega-star", "0.5", "--omega-ref", "none",
        "--fit-omega-ref", "true", "--kappa", "0.5",
        "--num-samples", "1000", "--max-restarts", "1",
        "--refine-after-fixed-omega", "false",
    ])
    curve.run_experiment(args, 2, False)

    Lambda_star = curve.lambda_star_for_dimension(2)
    expected_given = curve.covariance_from_lambda_star(Lambda_star, 0.5)
    expected_hat = curve.sample_empirical_covariance(
        expected_given, 1000, seed=args.random_seed + 2,
    )
    expected_ref = 0.5 * np.linalg.eigvalsh(expected_hat)[0]

    with np.load(args.given_output) as given, np.load(args.sigma_hat_output) as sample:
        np.testing.assert_allclose(given["Sigma"], expected_given)
        np.testing.assert_allclose(sample["Sigma"], expected_hat)
        assert given["omega_ref"].item() == pytest.approx(expected_ref)
        assert sample["omega_ref"].item() == pytest.approx(expected_ref)
        assert given["fit_omega_ref"].item()
        assert sample["fit_omega_ref"].item()
        FitSettings(**json.loads(sample["fit_settings_json"].item()))

    curve.run_experiment(args, 2, False)
    args.kappa = 0.6
    with pytest.raises(ValueError, match="omega_ref"):
        curve.run_experiment(args, 2, False)
