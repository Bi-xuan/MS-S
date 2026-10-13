"""Likelihood mathematics, exhaustive selection, persistence, and reporting."""

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import check_grad

from experiments import run_bic_support_scaling as runner
from experiments.summarize_fixed_support_scaling import (
    collect_trials, write_markdown_summary, write_overall_score_summary,
)
from optimizers import bic

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_omega_likelihood_and_gradients():
    sigma = np.array([[1.3, .2, -.1], [.2, 1.5, .3], [-.1, .3, 2.]])
    rows, cols = np.triu_indices(3, 1)
    x = np.array([-.3, .4, -.5, .2, -.1, .3])
    matrix = bic._decode(x, 3, rows, cols)
    omega = .8
    value, unit, _ = bic._likelihood(matrix, sigma, omega)
    covariance = omega * unit
    expected = np.linalg.slogdet(covariance)[1] + np.trace(np.linalg.solve(covariance, sigma))
    assert value == pytest.approx(expected)
    np.testing.assert_allclose(covariance, matrix.T @ covariance @ matrix + omega * np.eye(3))
    assert check_grad(lambda z: bic._objective_gradient(z, sigma, omega, rows, cols)[0],
                      lambda z: bic._objective_gradient(z, sigma, omega, rows, cols)[1], x) < 2e-6
    y = np.r_[np.arctanh(x[:3]), x[3:] / (1 - x[rows] * x[cols])]
    assert check_grad(lambda z: bic._optimization_gradient(z, sigma, omega, rows, cols)[0],
                      lambda z: bic._optimization_gradient(z, sigma, omega, rows, cols)[1], y) < 2e-6


def test_all_64_real_fits_and_bic_formula():
    sigma = np.array([[1.3, .2, -.1, .05], [.2, 1.5, .3, -.1],
                      [-.1, .3, 2., .2], [.05, -.1, .2, 1.1]])
    count = 1000
    result = bic.select_bic(sigma, count, omega_ref=.7)
    masks = result["candidate_support_masks"]
    assert masks.shape == (64, 4, 4)
    assert len({mask.tobytes() for mask in masks}) == 64
    assert np.all(result["candidate_fit_valid"])
    assert np.all(result["candidate_projected_gradients"] <= 1e-6)
    assert np.bincount(result["candidate_edge_counts"]).tolist() == [1, 6, 15, 20, 15, 6, 1]
    assert not np.any(np.tril(masks, -1))
    for index, (matrix, covariance, omega) in enumerate(zip(
            result["candidate_fitted_lambdas"], result["candidate_fitted_covariances"],
            result["candidate_fitted_omegas"])):
        assert np.all(matrix[~masks[index]] == 0)
        assert np.max(np.abs(np.diag(matrix))) <= .999999
        assert omega == .7
        np.linalg.cholesky(covariance)
        np.testing.assert_allclose(covariance, matrix.T @ covariance @ matrix + omega * np.eye(4),
                                   rtol=1e-8, atol=1e-8)
        loss = np.linalg.slogdet(covariance)[1] + np.trace(np.linalg.solve(covariance, sigma))
        assert result["candidate_objective_values"][index] == pytest.approx(loss, abs=1e-9)
        log_likelihood = -.5 * count * (4 * np.log(2 * np.pi) + loss)
        assert result["candidate_log_likelihoods"][index] == pytest.approx(log_likelihood)
        assert result["candidate_criterion_values"][index] == pytest.approx(
            -2 * log_likelihood + result["candidate_edge_counts"][index] * np.log(count))
    # The unrestricted upper model must attain the saturated Gaussian optimum.
    assert result["candidate_objective_values"][-1] == pytest.approx(np.linalg.slogdet(sigma)[1] + 4)
    np.testing.assert_allclose(result["candidate_fitted_covariances"][-1], sigma, atol=1e-8)
    assert result["selected_index"] == np.argmin(result["candidate_reduced_criterion_values"])
    np.testing.assert_array_equal(result["support_mask"], masks[result["selected_index"]])


def test_diagonal_covariance_selects_empty_edges_at_fixed_noise():
    sigma = np.diag([1., 2., 3., 4.])
    result = bic.select_bic(sigma, 100, omega_ref=.7)
    assert result["dimension"] == 1 and result["edge_count"] == 0
    assert result["support_edges"].shape == (0, 2)
    np.testing.assert_allclose(result["fitted_covariance"], sigma, atol=1e-8)
    assert result["objective_value"] == pytest.approx(np.log(np.diag(sigma)).sum() + 4)


def test_fixed_noise_above_smallest_variance_has_correct_constrained_fit():
    sigma = np.diag([.5, 2.])
    fit = bic.fit_support(sigma, np.eye(2, dtype=bool), omega_ref=1.)
    assert fit["fitted_omega"] == 1.
    np.testing.assert_allclose(fit["fitted_covariance"], np.diag([1., 2.]), atol=1e-7)
    assert fit["objective_value"] == pytest.approx(np.log(2.) + 1.5)


def test_fixed_saved_noise_converges_for_reported_boundary_failure():
    # support_02 / num_samples_100 / seed_124 from the stopped BIC study.
    sigma = np.array([
        [1.1271493136308992, .0855597283218267, .050415228253321315, .14339681651911157],
        [.0855597283218267, 1.7776701834685296, -.06895051826535627, -.295899296013806],
        [.050415228253321315, -.06895051826535627, 1.4859198657956256, .029789720922752217],
        [.14339681651911157, -.295899296013806, .029789720922752217, 2.410578803532628],
    ])
    mask = np.eye(4, dtype=bool)
    mask[0, 1] = mask[0, 2] = True
    omega_ref = 1.0054331746771485
    fit = bic.fit_support(sigma, mask, omega_ref=omega_ref)
    assert fit["fitted_omega"] == omega_ref
    assert fit["projected_gradient"] <= 1e-6
    assert np.max(np.abs(np.diag(fit["fitted_lambda"]))) < .8
    covariance = fit["fitted_covariance"]
    expected = np.linalg.slogdet(covariance)[1] + np.trace(np.linalg.solve(covariance, sigma))
    assert fit["objective_value"] == pytest.approx(expected, abs=1e-10)


def test_parallel_support_fits_match_serial_results_and_candidate_order():
    sigma = np.array([[1.3, .2, -.1], [.2, 1.5, .3], [-.1, .3, 2.]])
    serial = bic.select_bic(sigma, 100, omega_ref=.7)
    parallel = bic.select_bic(sigma, 100, omega_ref=.7, n_jobs=2)
    assert serial.keys() == parallel.keys()
    for name in serial:
        np.testing.assert_allclose(parallel[name], serial[name], rtol=1e-11, atol=1e-11,
                                   err_msg=name)


def test_optimizer_success_alone_does_not_accept_an_unconverged_fit(monkeypatch):
    def stopped(objective, start, **kwargs):
        return SimpleNamespace(x=start, success=True, nit=0)

    monkeypatch.setattr(bic, "minimize", stopped)
    with pytest.raises(ValueError, match="No converged BIC"):
        bic.fit_support(np.array([[1.2, .3], [.3, 2.1]]), np.eye(2, dtype=bool), omega_ref=.7)


def test_exact_ties_use_edge_count_then_lexicographic_order_and_count_mask(monkeypatch):
    def fit(sigma, mask, **options):
        # Coefficients happen to vanish: candidate positions still count.
        loss = 3. if np.count_nonzero(mask) == 4 else 100.
        return dict(fitted_lambda=np.zeros((3, 3)), fitted_covariance=np.eye(3), fitted_omega=1.,
                    objective_value=loss, projected_gradient=0., optimizer_success=True,
                    iterations=0, valid_restarts=1)

    monkeypatch.setattr(bic, "fit_support", fit)
    result = bic.select_bic(np.eye(3), 100, omega_ref=1.)
    assert result["edge_count"] == 1
    np.testing.assert_array_equal(result["support_edges"], [[0, 1]])
    assert result["penalty_value"] == pytest.approx(np.log(100))
    # At n=1 there would be no edge penalty, but n>=N is required. Instead
    # compensate the likelihood to make every penalized score exactly zero.
    def tied(sigma, mask, **options):
        answer = fit(sigma, mask, **options)
        answer["objective_value"] = -(np.count_nonzero(mask) - 3) * np.log(100) / 100
        return answer

    monkeypatch.setattr(bic, "fit_support", tied)
    result = bic.select_bic(np.eye(3), 100, omega_ref=1.)
    assert result["edge_count"] == 0


def save_trial(root, seed=1):
    directory = root / "num_samples_100" / f"seed_{seed}"
    directory.mkdir(parents=True)
    mask = np.eye(4, dtype=bool)
    mask[0, 2] = True
    curve = directory / "objective_curve_sigma_hat.npz"
    # No observations, generating Lambda, omega_star, or saved fit settings:
    # BIC needs the empirical covariance, count, and saved fixed omega_ref.
    np.savez_compressed(curve, n=4, Sigma=np.array([
        [1.3, .03, .2, .01], [.03, 1.5, .01, .02], [.2, .01, 2., .03], [.01, .02, .03, 1.1]]),
        num_samples=100, omega_ref=.7 + .01 * seed, random_seed=seed, lambda_star_support_edges=[[0, 2]],
        lambda_star_dimension=2, support_scope="upper", nested_supports=True,
        d_m_values=[2], selected_support_masks=mask[None], selected_support_valid=[True],
        oracle_dimension=2, oracle_support_mask=mask, lasso_support_mask=mask)
    (directory / "selection_plateau_bootstrap.json").write_text(json.dumps(
        dict(selected_dimension=2, selected_edges=[[0, 2]])))
    return curve


@pytest.mark.parametrize("num_trials,cpus_per_trial", [(1, 1), (1, 2), (2, 2)])
def test_real_runner_preserves_archive_and_refreshes_comparisons(
        tmp_path, num_trials, cpus_per_trial):
    curve = save_trial(tmp_path)
    for seed in range(2, num_trials + 1):
        save_trial(tmp_path, seed)
    with np.load(curve) as saved:
        before = dict(saved)
    # A rerun must replace results and metadata from the previous profiled fit.
    np.savez_compressed(curve, **before, bic_fitted_omega=999.,
                        bic_candidate_fitted_omegas=np.full(64, 999.),
                        bic_fit_options_json="joint MLE via analytic profiling")
    runner.run_bic(tmp_path, total_cpus=num_trials * cpus_per_trial,
                   cpus_per_trial=cpus_per_trial)
    with np.load(curve, allow_pickle=False) as saved:
        for name, value in before.items():
            np.testing.assert_array_equal(saved[name], value)
        assert saved["bic_candidate_support_masks"].shape == (64, 4, 4)
        settings = json.loads(str(saved["bic_fit_options_json"]))
        assert settings["omega"] == "fixed saved omega_ref"
        assert settings["omega_ref"] == float(saved["omega_ref"])
        assert not settings["threshold_support"]
        assert settings["support_workers"] == cpus_per_trial
    for trial in tmp_path.glob("num_samples_*/seed_*/objective_curve_sigma_hat.npz"):
        with np.load(trial) as saved:
            assert saved["bic_candidate_support_masks"].shape == (64, 4, 4)
            assert saved["bic_fitted_omega"] == saved["omega_ref"]
            np.testing.assert_array_equal(saved["bic_candidate_fitted_omegas"],
                                          np.full(64, saved["omega_ref"]))
    summary = (tmp_path / "selection_summary.md").read_text()
    assert f"BIC exact support recovery (x/{num_trials})" in summary
    assert f"BIC EC recovery (x/{num_trials})" in summary
    assert f"Average BIC EC score over {num_trials} seeds" in summary
    assert "Lasso exact support recovery" in summary and "Oracle exact support recovery" in summary
    assert "bic_score" in (tmp_path / "selection_trials.csv").read_text()
    trials = collect_trials(tmp_path, (100,), num_trials)
    write_markdown_summary(trials, (100,), num_trials, tmp_path / "regenerated.md")
    assert (tmp_path / "regenerated.md").read_text() == summary


def test_upper_study_reselect_and_pooled_summaries(tmp_path, monkeypatch):
    def selected(sigma, count, **options):
        mask = np.eye(4, dtype=bool)
        mask[0, 2] = True
        return dict(support_mask=mask, edge_count=1)

    monkeypatch.setattr(runner, "select_bic", selected)
    for support in ("support_00", "support_01"):
        root = tmp_path / support
        curve = save_trial(root)
        destination = root / "reselect_lm_support_count_plateau" / "num_samples_100" / "seed_1"
        destination.mkdir(parents=True)
        (destination / "selection_plateau.json").write_text(
            (curve.parent / "selection_plateau_bootstrap.json").read_text())
        (destination.parents[1] / "selection_summary.md").write_text("old summary")
    runner.run_bic(tmp_path, upper_study=True, total_cpus=1, cpus_per_trial=1)
    for relative in (".", "reselect_lm_support_count_plateau"):
        for support in ("support_00", "support_01"):
            summary = (tmp_path / support / relative / "selection_summary.md").read_text()
            assert "Average BIC EC score over 1 seeds" in summary
            assert summary.splitlines()[-1].endswith(" 1/1 | 1/1 | 1.000000 |")
        assert "Average BIC EC score" in (tmp_path / relative / "selection_summary.md").read_text()


def test_unresolved_candidate_preserves_archive_and_summary(tmp_path, monkeypatch):
    curve = save_trial(tmp_path)
    before = curve.read_bytes()
    summary = tmp_path / "selection_summary.md"
    summary.write_text("old summary")
    calls = []

    def failed(sigma, mask, **options):
        calls.append(mask)
        raise ValueError("No converged BIC likelihood fit")

    monkeypatch.setattr(bic, "fit_support", failed)
    with pytest.raises(ValueError, match="No converged"):
        runner.run_bic(tmp_path, total_cpus=1, cpus_per_trial=1)
    assert len(calls) == 1
    assert curve.read_bytes() == before and summary.read_text() == "old summary"


def test_parallel_unresolved_candidate_preserves_archive_and_summary(tmp_path):
    curve = save_trial(tmp_path)
    before = curve.read_bytes()
    summary = tmp_path / "selection_summary.md"
    summary.write_text("old summary")
    with pytest.raises(ValueError, match="No converged"):
        runner.run_bic(tmp_path, total_cpus=2, cpus_per_trial=2,
                       max_restarts=1, max_iter=1, tol=1e-30)
    assert curve.read_bytes() == before and summary.read_text() == "old summary"


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_support_worker_counts_are_rejected(tmp_path, workers):
    with pytest.raises(ValueError, match="n_jobs"):
        bic.select_bic(np.eye(3), 100, omega_ref=1., n_jobs=workers)
    with pytest.raises(ValueError, match="SUPPORT_WORKERS"):
        runner.run_bic(tmp_path, total_cpus=2, cpus_per_trial=2, support_workers=workers)


def test_support_worker_override_and_cpu_budget(tmp_path, monkeypatch):
    curve = save_trial(tmp_path)
    before = curve.read_bytes()
    with pytest.raises(ValueError, match="SUPPORT_WORKERS"):
        runner.run_bic(tmp_path, total_cpus=2, cpus_per_trial=2, support_workers=3)
    assert curve.read_bytes() == before
    original = runner.select_bic
    calls = []

    def selected(sigma, count, **options):
        calls.append(options["n_jobs"])
        return original(sigma, count, **options)

    monkeypatch.setattr(runner, "select_bic", selected)
    runner.run_bic(tmp_path, total_cpus=2, cpus_per_trial=2, support_workers=1)
    assert calls == [1]
    with np.load(curve) as saved:
        assert json.loads(str(saved["bic_fit_options_json"]))["support_workers"] == 1


def test_partial_comparisons_are_rejected(tmp_path):
    curve = save_trial(tmp_path, 1)
    save_trial(tmp_path, 2)
    with np.load(curve) as saved:
        arrays = dict(saved)
    arrays.update(bic_support_mask=arrays["lasso_support_mask"], bic_edge_count=1)
    np.savez_compressed(curve, **arrays)
    with pytest.raises(ValueError, match="BIC results are missing"):
        write_markdown_summary(collect_trials(tmp_path, (100,), 2), (100,), 2, tmp_path / "summary.md")
    one, two = tmp_path / "one.csv", tmp_path / "two.csv"
    one.write_text("num_samples,score,bic_score\n100,.5,1\n")
    two.write_text("num_samples,score\n100,.5\n")
    with pytest.raises(ValueError, match="BIC scores are missing"):
        write_overall_score_summary([one, two], tmp_path / "summary.md")


def test_input_validation_and_upper_scope_preflight(tmp_path):
    for sigma in (np.eye(10), np.ones((4, 4)), np.eye(4) * np.nan):
        with pytest.raises(ValueError):
            bic.select_bic(sigma, 100, omega_ref=.7)
    for count in (0, 3, 10.5, float("inf")):
        with pytest.raises(ValueError, match="num_samples"):
            bic.select_bic(np.eye(4), count, omega_ref=.7)
    curve = save_trial(tmp_path)
    with np.load(curve) as saved:
        arrays = dict(saved)
    arrays["support_scope"] = "all"
    np.savez_compressed(curve, **arrays)
    before = curve.read_bytes()
    with pytest.raises(ValueError, match="support_scope=upper"):
        runner.run_bic(tmp_path, total_cpus=1, cpus_per_trial=1)
    assert curve.read_bytes() == before


@pytest.mark.parametrize("omega_ref", [0., -1., np.nan, np.inf, True, "None", [.7, .8]])
def test_invalid_fixed_reference_is_rejected_before_writes(tmp_path, omega_ref):
    with pytest.raises(ValueError, match="omega_ref"):
        bic.select_bic(np.eye(3), 100, omega_ref=omega_ref)
    with pytest.raises(ValueError, match="omega_ref"):
        bic.fit_support(np.eye(3), np.eye(3, dtype=bool), omega_ref=omega_ref)
    first = save_trial(tmp_path, 1)
    invalid = save_trial(tmp_path, 2)
    with np.load(invalid) as saved:
        arrays = dict(saved)
    arrays["omega_ref"] = omega_ref
    np.savez_compressed(invalid, **arrays)
    before = [curve.read_bytes() for curve in (first, invalid)]
    summary = tmp_path / "selection_summary.md"
    summary.write_text("old summary")
    with pytest.raises(ValueError, match="omega_ref"):
        runner.run_bic(tmp_path, total_cpus=1, cpus_per_trial=1)
    assert [curve.read_bytes() for curve in (first, invalid)] == before
    assert summary.read_text() == "old summary"


def test_missing_saved_reference_is_rejected_before_writes(tmp_path):
    curve = save_trial(tmp_path)
    with np.load(curve) as saved:
        arrays = dict(saved)
    del arrays["omega_ref"]
    np.savez_compressed(curve, **arrays)
    before = curve.read_bytes()
    with pytest.raises(ValueError, match="missing BIC inputs: omega_ref"):
        runner.run_bic(tmp_path, total_cpus=1, cpus_per_trial=1)
    assert curve.read_bytes() == before


@pytest.mark.parametrize("command", ["BIC", "bic"])
@pytest.mark.parametrize("support_workers", [None, "1"])
@pytest.mark.parametrize("script,upper", [
    ("run_fixed_support_scaling.sh", False),
    ("run_fixed_support_scaling_package.sh", False),
    ("run_upper_support_scaling_study.sh", True),
])
def test_launchers_route_bic(tmp_path, script, upper, command, support_workers):
    stub = tmp_path / "python-stub"
    calls = tmp_path / "calls"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CALLS"\n')
    stub.chmod(0o755)
    env = dict(os.environ, PYTHON=str(stub), CALLS=str(calls), TOTAL_CPUS="6",
               CPUS_PER_TRIAL="2", BIC_DIAGONAL_BOUND=".99", BIC_MAX_RESTARTS="12",
               BIC_MAX_ITER="2000", BIC_TOL="1e-5")
    env.pop("BIC_SUPPORT_WORKERS", None)
    if support_workers is not None:
        env["BIC_SUPPORT_WORKERS"] = support_workers
    completed = subprocess.run(["bash", str(ROOT / "experiments" / script), command, str(tmp_path)],
                               env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    arguments = calls.read_text().splitlines()
    assert arguments[:2] == ["experiments/run_bic_support_scaling.py", str(tmp_path)]
    assert ("--upper-study" in arguments) is upper
    for flag, value in (("--total-cpus", "6"), ("--cpus-per-trial", "2"),
                        ("--support-workers", support_workers or "2"),
                        ("--diagonal-bound", ".99"), ("--max-restarts", "12"),
                        ("--max-iter", "2000"), ("--tol", "1e-5")):
        assert arguments[arguments.index(flag) + 1] == value
