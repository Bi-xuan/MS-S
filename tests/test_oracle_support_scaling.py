"""Verify fresh dimension-constrained fitting and saved-study oracle reports."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from experiments import run_oracle_support_scaling as oracle
from experiments.summarize_fixed_support_scaling import _load_trial, write_overall_score_summary

ROOT = Path(__file__).resolve().parents[1]


def save_trial(root, seed=1, *, package=False, scope="upper"):
    directory = root / "num_samples_100" / f"seed_{seed}"
    directory.mkdir(parents=True)
    path_mask = np.eye(3, dtype=bool)
    path_mask[0, 1] = True
    settings = dict(omega_fixed=.5, beta=1., max_iter=12, tol=1e-7,
                    zero_tol=1e-5, max_restarts=2, init_strategy="halton")
    if not package:
        settings["min_omega"] = 0.
    curve = directory / "objective_curve_sigma_hat.npz"
    np.savez_compressed(curve, n=3, Sigma=np.eye(3), num_samples=100,
                        random_seed=seed, solve_seed=seed + 3, omega_ref=.5,
                        fit_settings_json=json.dumps(settings),
                        lambda_star_dimension=2, lambda_star_support_edges=[[0, 2]],
                        support_scope=scope, nested_supports=True,
                        d_m_values=[2], objective_values=[1.],
                        selected_support_masks=path_mask[None], selected_support_valid=[True])
    (directory / "selection_plateau_bootstrap.json").write_text(json.dumps(
        dict(selected_dimension=2, selected_edges=[[0, 1]])))
    return curve


def exact_fit(sigma, dimension, **options):
    assert dimension == 2
    assert options["return_metadata"]
    assert "previous_support_mask" not in options
    mask = np.eye(3, dtype=bool)
    mask[0, 2] = True
    return np.zeros((3, 3)), .5, .125, dict(
        selected_support_mask=mask, selected_support_edges=[(0, 2)])


@pytest.mark.parametrize("package", [False, True])
@pytest.mark.parametrize("scope", ["upper", "all"])
def test_fresh_fit_uses_saved_settings_and_preserves_existing_fields(tmp_path, monkeypatch, package, scope):
    curve = save_trial(tmp_path, package=package, scope=scope)
    with np.load(curve) as saved:
        original = dict(saved)
    calls = []

    def fit(sigma, dimension, **options):
        calls.append(options)
        return exact_fit(sigma, dimension, **options)

    monkeypatch.setattr(oracle, "optimize_lambda", fit)
    oracle.fit_oracle_trial(curve, 3)
    assert len(calls) == 1
    assert calls[0]["support_scope"] == scope
    assert calls[0]["omega_ref"] == .5
    assert calls[0]["random_seed"] == 4
    assert calls[0]["max_restarts"] == 2
    assert calls[0]["max_iter"] == 12
    assert calls[0]["min_omega"] == 0.
    assert calls[0]["n_jobs"] == 3
    with np.load(curve, allow_pickle=False) as saved:
        for key, value in original.items():
            np.testing.assert_array_equal(saved[key], value)
        assert saved["oracle_dimension"] == 2
        assert saved["oracle_support_mask"][0, 2]
        assert not saved["oracle_support_mask"][0, 1]
        np.testing.assert_array_equal(saved["oracle_support_edges"], [[0, 2]])
        assert saved["oracle_objective_value"] == .125


def test_upper_study_updates_per_support_reselect_and_pooled_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr(oracle, "optimize_lambda", exact_fit)
    for support in ("support_00", "support_01"):
        root = tmp_path / support
        for seed in (1, 2):
            curve = save_trial(root, seed)
            reselect = root / "reselect_lm_support_count_plateau"
            destination = reselect / "num_samples_100" / f"seed_{seed}"
            destination.mkdir(parents=True)
            (destination / "selection_plateau.json").write_text(
                (curve.parent / "selection_plateau_bootstrap.json").read_text())
        (reselect / "selection_summary.md").write_text("old summary")
    oracle.run_oracle(tmp_path, upper_study=True, total_cpus=1, cpus_per_trial=1)
    for support in ("support_00", "support_01"):
        root = tmp_path / support
        curve = root / "num_samples_100/seed_1/objective_curve_sigma_hat.npz"
        trial = _load_trial(curve, curve.parent / "selection_plateau_bootstrap.json", 100, 1)
        assert not trial.exact_support_recovery
        assert trial.oracle_exact_support_recovery
        assert trial.oracle_ec_recovery
        assert trial.oracle_score == 1.
        for relative in (".", "reselect_lm_support_count_plateau"):
            table = (root / relative / "selection_summary.md").read_text()
            assert "Oracle exact support recovery (x/2)" in table
            assert "Oracle EC recovery (x/2)" in table
            assert table.splitlines()[-1].endswith(" 2/2 | 2/2 | 1.000000 |")
            csv = (root / relative / "selection_trials.csv").read_text()
            assert "oracle_exact_support_recovery,oracle_ec_recovery,oracle_score" in csv
    for relative in (".", "reselect_lm_support_count_plateau"):
        table = (tmp_path / relative / "selection_summary.md").read_text()
        assert table.splitlines()[0] == "| Number of samples | Average EC score | Average oracle EC score |"
        assert table.splitlines()[-1].endswith(" 1.000000 |")


def test_failed_fit_preserves_npz_and_summary(tmp_path, monkeypatch):
    curve = save_trial(tmp_path)
    original = curve.read_bytes()
    summary = tmp_path / "selection_summary.md"
    summary.write_text("old summary")
    monkeypatch.setattr(oracle, "optimize_lambda", lambda *args, **kwargs: (None, None, np.inf, {}))
    with pytest.raises(ValueError, match="no finite oracle fit"):
        oracle.run_oracle(tmp_path, total_cpus=1, cpus_per_trial=1)
    assert curve.read_bytes() == original
    assert summary.read_text() == "old summary"


def test_oracle_fits_when_true_dimension_is_absent_from_saved_path(tmp_path, monkeypatch):
    curve = save_trial(tmp_path)
    with np.load(curve) as saved:
        arrays = dict(saved)
    arrays["d_m_values"] = np.array([3])
    arrays["selected_support_masks"][0, 1, 2] = True
    np.savez_compressed(curve, **arrays)
    (curve.parent / "selection_plateau_bootstrap.json").write_text(json.dumps(
        dict(selected_dimension=3, selected_edges=[[0, 1], [1, 2]])))
    monkeypatch.setattr(oracle, "optimize_lambda", exact_fit)
    oracle.run_oracle(tmp_path, total_cpus=1, cpus_per_trial=1)
    summary = (tmp_path / "selection_summary.md").read_text()
    assert summary.splitlines()[-1].endswith(" 1/1 | 1/1 | 1.000000 |")


@pytest.mark.parametrize("script,upper", [
    ("run_fixed_support_scaling.sh", False),
    ("run_fixed_support_scaling_package.sh", False),
    ("run_upper_support_scaling_study.sh", True),
])
def test_all_launchers_route_oracle_without_fitting_curves_or_plotting(tmp_path, script, upper):
    stub = tmp_path / "python-stub"
    calls = tmp_path / "calls.json"
    stub.write_text("#!/bin/sh\n" + 'printf "%s\\n" "$@" > "$CALLS"\n')
    stub.chmod(0o755)
    env = dict(os.environ, PYTHON=str(stub), CALLS=str(calls),
               TOTAL_CPUS="6", CPUS_PER_TRIAL="2")
    completed = subprocess.run(["bash", str(ROOT / "experiments" / script),
                                "oracle", str(tmp_path / "saved-output")],
                               env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    arguments = calls.read_text().splitlines()
    assert arguments[0] == "experiments/run_oracle_support_scaling.py"
    assert arguments[1] == str(tmp_path / "saved-output")
    assert ("--upper-study" in arguments) is upper
    assert arguments[-4:] == ["--total-cpus", "6", "--cpus-per-trial", "2"]


def test_pooled_summary_rejects_partial_oracle_metrics(tmp_path):
    paths = [tmp_path / "one.csv", tmp_path / "two.csv"]
    paths[0].write_text("num_samples,score,oracle_score\n100,.5,1\n")
    paths[1].write_text("num_samples,score\n100,.5\n")
    with pytest.raises(ValueError, match="missing"):
        write_overall_score_summary(paths, tmp_path / "summary.md")
