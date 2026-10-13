"""Check the lasso estimator, fold scoring, archive preservation and launchers."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import check_grad

from experiments import run_lasso_support_scaling as runner
from experiments.compute_objective_curve import covariance_from_lambda_star
from experiments.summarize_fixed_support_scaling import (
    _load_trial, collect_trials, write_markdown_summary, write_overall_score_summary,
)
from optimizers import lasso

ROOT = Path(__file__).resolve().parents[1]


def test_absolute_objective_gradient_and_unpenalized_diagonals():
    rng = np.random.default_rng(3)
    b = rng.normal(size=(3, 3))
    sigma = b.T @ b + np.eye(3)
    rows, cols = np.triu_indices(3, 1)
    x = rng.uniform(.1, .4, 9)
    x[:3] *= [-1, 1, -1]
    args = (sigma, .8, .31, rows, cols, 7.)
    value, gradient = lasso._objective_gradient(x, *args)
    matrix = lasso._decode(x, 3, rows, cols)
    expected = (np.sum((sigma - matrix.T @ sigma @ matrix - .8 * np.eye(3)) ** 2)
                + .31 * x[3:].sum()) / 7
    assert value == pytest.approx(expected)
    assert check_grad(lambda y: lasso._objective_gradient(y, *args)[0],
                      lambda y: lasso._objective_gradient(y, *args)[1], x) < 2e-6
    without_penalty = lasso._objective_gradient(x, sigma, .8, 0, rows, cols, 7.)[1]
    np.testing.assert_array_equal(gradient[:3], without_penalty[:3])
    np.testing.assert_allclose(gradient[3:] - without_penalty[3:], .31 / 7)


def test_gamma_grid_configuration():
    default = lasso.gamma_grid()
    assert len(default) == 30 and default[0] == 0
    assert default[1] == pytest.approx(1e-7) and default[-1] == 1
    np.testing.assert_allclose(lasso.gamma_grid(.1, .4, 4, "linear"), [0, .1, .2, .3, .4])
    for args in ((0, 1, 3, "log"), (1, .1, 3, "linear"), (.1, 1, 0, "log")):
        with pytest.raises(ValueError):
            lasso.gamma_grid(*args)


def test_real_fits_are_triangular_stable_and_fit_zero_gamma():
    matrix = np.array([[-.3, .25, -.2], [0, .4, .1], [0, 0, -.55]])
    sigma = covariance_from_lambda_star(matrix, .8)
    omega = .93 * np.linalg.eigvalsh(sigma)[0]
    path, details = lasso.fit_lasso_path(sigma, omega, [0, .01, 1.], max_restarts=4)
    assert all(item["valid"] for item in details)
    assert np.all(np.tril(path, -1) == 0)
    assert np.all(np.abs(np.diagonal(path, axis1=1, axis2=2)) < 1)
    zero_covariance = solve_discrete_lyapunov(path[0].T, omega * np.eye(3))
    np.testing.assert_allclose(zero_covariance, sigma, rtol=1e-8, atol=1e-8)
    for fitted, detail, gamma in zip(path, details, [0, .01, 1.]):
        loss = np.sum((sigma - fitted.T @ sigma @ fitted - omega * np.eye(3)) ** 2)
        assert detail["penalized_objective"] == pytest.approx(loss + gamma * np.abs(np.triu(fitted, 1)).sum(), abs=1e-8)


def test_cv_uses_training_reference_absolute_gamma_and_unsquared_fold_errors(monkeypatch):
    samples = np.random.default_rng(6).normal(size=(103, 3))
    gammas = np.array([0, .2, .7])
    calls = []

    def fit(sigma, omega, grid, **options):
        calls.append((sigma.copy(), omega, grid.copy()))
        path = np.array([np.diag([.2 + .4 * gamma, -.3, .4]) for gamma in grid])
        path[:, 0, 1] = 1e-6
        details = [dict(valid=True, penalized_objective=1., projected_gradient=0.) for _ in grid]
        return path, details

    monkeypatch.setattr(lasso, "fit_lasso_path", fit)
    result = lasso.select_lasso(samples, gammas, fold_seed=42, kappa=.8)
    expected_errors = []
    for fold in range(5):
        train = samples[result["fold_ids"] != fold]
        heldout = samples[result["fold_ids"] == fold]
        sigma_train = train.T @ train / len(train)
        np.testing.assert_allclose(calls[fold][0], sigma_train)
        assert calls[fold][1] == pytest.approx(.8 * np.linalg.eigvalsh(sigma_train)[0])
        np.testing.assert_array_equal(calls[fold][2], gammas)
        np.testing.assert_allclose(result["cv_validation_covariances"][fold], heldout.T @ heldout / len(heldout))
        errors = []
        for matrix in result["cv_fitted_lambdas"][fold]:
            covariance = solve_discrete_lyapunov(matrix.T, calls[fold][1] * np.eye(3))
            errors.append(np.linalg.norm(heldout.T @ heldout / len(heldout) - covariance, "fro"))
        expected_errors.append(errors)
    np.testing.assert_allclose(result["cv_errors"], expected_errors)
    np.testing.assert_allclose(result["cv_error_sums"], np.sum(expected_errors, axis=0))
    assert result["selected_index"] == int(np.argmin(np.sum(expected_errors, axis=0)))
    np.testing.assert_allclose(calls[-1][0], samples.T @ samples / len(samples))
    assert calls[-1][1] == pytest.approx(.8 * np.linalg.eigvalsh(calls[-1][0])[0])
    assert not result["support_mask"][0, 1]
    assert result["fitted_lambda"][0, 1] == 1e-6  # threshold affects support, not refit
    np.testing.assert_array_equal(result["fold_ids"], lasso.select_lasso(samples, gammas, fold_seed=42, kappa=.8)["fold_ids"])
    assert sorted(np.bincount(result["fold_ids"])) == [20, 20, 21, 21, 21]


def test_cv_ties_choose_larger_gamma_and_failed_refit_raises(monkeypatch):
    samples = np.random.default_rng(2).normal(size=(30, 2))
    count = 0

    def fit(sigma, omega, grid, **options):
        nonlocal count
        count += 1
        return np.zeros((len(grid), 2, 2)), [dict(
            valid=(count != 12), penalized_objective=0., projected_gradient=0.) for _ in grid]

    monkeypatch.setattr(lasso, "fit_lasso_path", fit)
    result = lasso.select_lasso(samples, [0, .1, 1.])
    assert result["gamma"] == 1.
    np.testing.assert_array_equal(np.diag(result["fitted_lambda"]), [0, 0])
    with pytest.raises(ValueError, match="full-data refit"):
        lasso.select_lasso(samples, [0, .1, 1.])


def save_trial(root, seed=1, package=False):
    directory = root / "num_samples_100" / f"seed_{seed}"
    directory.mkdir(parents=True)
    matrix = np.diag([.2, -.4, .5])
    matrix[0, 2] = .25
    population = covariance_from_lambda_star(matrix, 1.)
    samples = np.random.default_rng(seed + 2).multivariate_normal(np.zeros(3), population, size=100)
    mask = np.eye(3, dtype=bool)
    mask[0, 2] = True
    curve = directory / "objective_curve_sigma_hat.npz"
    saved = dict(n=3, Sigma=samples.T @ samples / 100, num_samples=100,
                 random_seed=seed, Lambda_star=matrix, omega_star=1.,
                 lambda_star_support_edges=[[0, 2]], support_scope="upper",
                 nested_supports=True, d_m_values=[2], selected_support_masks=mask[None],
                 selected_support_valid=[True], lambda_star_dimension=2,
                 oracle_dimension=2, oracle_support_mask=mask)
    if package:
        saved.update(sample_seed=seed + 2, Sigma_population=population)
    np.savez_compressed(curve, **saved)
    (directory / "selection_plateau_bootstrap.json").write_text(json.dumps(
        dict(selected_dimension=2, selected_edges=[[0, 2]])))
    return curve


@pytest.mark.parametrize("package", [False, True])
def test_real_saved_trial_preserves_archive_and_reports_lasso(tmp_path, package):
    curve = save_trial(tmp_path, package=package)
    with np.load(curve) as saved:
        before = dict(saved)
    runner.run_lasso(tmp_path, gammas=[0, .01, 1.], total_cpus=1, cpus_per_trial=1, max_restarts=4)
    with np.load(curve, allow_pickle=False) as saved:
        for key, value in before.items():
            np.testing.assert_array_equal(saved[key], value)
        assert saved["lasso_cv_errors"].shape == (5, 3)
        assert saved["lasso_cv_fitted_lambdas"].shape == (5, 3, 3, 3)
        np.testing.assert_allclose(saved["lasso_cv_error_sums"], saved["lasso_cv_errors"].sum(axis=0))
        assert saved["lasso_gamma"] in [0, .01, 1.]
        assert json.loads(str(saved["lasso_fit_options_json"]))["gamma_units"] == "absolute"
    trial = _load_trial(curve, curve.parent / "selection_plateau_bootstrap.json", 100, 1)
    assert trial.lasso_score is not None and trial.oracle_score == 1
    summary = (tmp_path / "selection_summary.md").read_text()
    assert "Lasso exact support recovery (x/1)" in summary
    assert "Lasso EC recovery (x/1)" in summary
    assert "Average lasso EC score over 1 seeds" in summary
    assert "Oracle exact support recovery" in summary


def test_upper_study_refreshes_reselect_and_pooled_tables(tmp_path, monkeypatch):
    def selected(samples, grid, **options):
        mask = np.eye(3, dtype=bool)
        mask[0, 2] = True
        return dict(support_mask=mask)

    monkeypatch.setattr(runner, "select_lasso", selected)
    for support in ("support_00", "support_01"):
        root = tmp_path / support
        for seed in (1, 2):
            curve = save_trial(root, seed)
            destination = root / "reselect_lm_support_count_plateau" / "num_samples_100" / f"seed_{seed}"
            destination.mkdir(parents=True)
            (destination / "selection_plateau.json").write_text(
                (curve.parent / "selection_plateau_bootstrap.json").read_text())
        (destination.parents[1] / "selection_summary.md").write_text("old summary")
    runner.run_lasso(tmp_path, upper_study=True, total_cpus=1, cpus_per_trial=1)
    for support in ("support_00", "support_01"):
        for relative in (".", "reselect_lm_support_count_plateau"):
            summary = (tmp_path / support / relative / "selection_summary.md").read_text()
            assert summary.splitlines()[-1].endswith(" 2/2 | 2/2 | 1.000000 |")
            assert "Average lasso EC score over 2 seeds" in summary
    for relative in (".", "reselect_lm_support_count_plateau"):
        assert "Average lasso EC score" in (tmp_path / relative / "selection_summary.md").read_text()


def test_mismatched_data_and_failed_fit_preserve_archive(tmp_path, monkeypatch):
    curve = save_trial(tmp_path)
    with np.load(curve) as saved:
        arrays = dict(saved)
    arrays["Sigma"] *= 2
    np.savez_compressed(curve, **arrays)
    before = curve.read_bytes()
    with pytest.raises(ValueError, match="do not match"):
        runner.fit_lasso_trial(curve, [0, 1.], {})
    assert curve.read_bytes() == before
    arrays["Sigma"] /= 2
    np.savez_compressed(curve, **arrays)
    before = curve.read_bytes()
    summary = tmp_path / "selection_summary.md"
    summary.write_text("old summary")

    def failed(*args, **kwargs):
        raise ValueError("No valid lasso fit")

    monkeypatch.setattr(runner, "select_lasso", failed)
    with pytest.raises(ValueError, match="No valid"):
        runner.run_lasso(tmp_path, total_cpus=1, cpus_per_trial=1)
    assert curve.read_bytes() == before and summary.read_text() == "old summary"


def test_summaries_reject_partial_lasso_results(tmp_path):
    one = save_trial(tmp_path, 1)
    save_trial(tmp_path, 2)
    with np.load(one) as saved:
        arrays = dict(saved)
    arrays["lasso_support_mask"] = arrays["oracle_support_mask"]
    np.savez_compressed(one, **arrays)
    trials = collect_trials(tmp_path, (100,), 2)
    with pytest.raises(ValueError, match="missing"):
        write_markdown_summary(trials, (100,), 2, tmp_path / "summary.md")
    paths = [tmp_path / "one.csv", tmp_path / "two.csv"]
    paths[0].write_text("num_samples,score,lasso_score\n100,.5,1\n")
    paths[1].write_text("num_samples,score\n100,.5\n")
    with pytest.raises(ValueError, match="missing"):
        write_overall_score_summary(paths, tmp_path / "summary.md")


@pytest.mark.parametrize("script,upper", [
    ("run_fixed_support_scaling.sh", False),
    ("run_fixed_support_scaling_package.sh", False),
    ("run_upper_support_scaling_study.sh", True),
])
def test_launchers_route_lasso_with_configurable_grid(tmp_path, script, upper):
    stub = tmp_path / "python-stub"
    calls = tmp_path / "calls"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CALLS"\n')
    stub.chmod(0o755)
    env = dict(os.environ, PYTHON=str(stub), CALLS=str(calls), TOTAL_CPUS="6",
               CPUS_PER_TRIAL="2", LASSO_GAMMA_MIN=".01", LASSO_GAMMA_MAX="3",
               LASSO_GAMMA_COUNT="17", LASSO_GAMMA_SPACING="linear", LASSO_CV_SEED="42")
    completed = subprocess.run(["bash", str(ROOT / "experiments" / script), "lasso", str(tmp_path)],
                               env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    arguments = calls.read_text().splitlines()
    assert arguments[:2] == ["experiments/run_lasso_support_scaling.py", str(tmp_path)]
    assert ("--upper-study" in arguments) is upper
    for flag, value in (("--total-cpus", "6"), ("--cpus-per-trial", "2"),
                        ("--gamma-min", ".01"), ("--gamma-max", "3"),
                        ("--gamma-count", "17"), ("--gamma-spacing", "linear"), ("--fold-seed", "42")):
        assert arguments[arguments.index(flag) + 1] == value
