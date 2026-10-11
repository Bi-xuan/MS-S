"""Verify the single select_support call, study exports, and local reselection."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import upper_support_package as study
from experiments.summarize_fixed_support_scaling import _load_trial


ROOT = Path(__file__).resolve().parents[1]


def trial_args(output, *options):
    return study.parse_args([
        "--lambda-star-support-index", "0", "--num-samples", "100",
        "--random-seed", "124", "--n-jobs", "1",
        "--sigma-hat-output", str(output / "curve.npz"),
        "--given-output", str(output / "sigma_hat.npz"),
        "--output-json", str(output / "selection.json"), *options,
    ])


@pytest.fixture(scope="module")
def package_trial(tmp_path_factory):
    output = tmp_path_factory.mktemp("package-trial")
    calls = []
    package = study.recovery

    def select(sigma_hat, **options):
        calls.append((sigma_hat.copy(), options))
        return package.select_support(sigma_hat, **options)

    # Expose only select_support, plus metadata. Any other package function or
    # class access in the adapter would fail, while exercising the real estimator.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "recovery", SimpleNamespace(
            select_support=select, __version__=package.__version__, __file__=package.__file__,
        ))
        args = trial_args(output, "--bootstrap-replicates", "3", "--top-plateaus", "2")
        result = study.run_trial(args)
    return args, result, calls


def test_trial_calls_only_select_support_once_and_exports_result(package_trial):
    args, result, calls = package_trial
    assert len(calls) == 1
    _, options = calls[0]
    assert options["return_result"] is True
    assert options["method"] == "plateau-bootstrap"
    assert options["bootstrap_replicates"] == 3
    assert options["top_plateaus"] == 2
    report = json.loads(args.output_json.read_text())
    assert report["selected_dimension"] == result.selected_dimension
    assert report["selected_edges"] == [list(edge) for edge in result.selected_edges]
    np.testing.assert_allclose(report["Lm"], result.lm_values)
    assert result.diagnostics.comparisons  # Exercise actual bootstrap refits.
    summary = _load_trial(args.output, args.output_json, 100, 124)
    assert summary.selected_dimension == result.selected_dimension
    assert report["package_path"] == str(study.recovery.__file__)


def test_curve_preserves_support_and_seeds(package_trial):
    args, result, calls = package_trial
    covariance, options = calls[0]
    assert options["random_seed"] == 127
    assert options["omega_ref"] == 1.0
    assert options["fit_omega_ref"] is False
    with np.load(args.output, allow_pickle=False) as data:
        assert int(data["lambda_star_support_index"]) == 0
        assert int(data["lambda_star_dimension"]) == 4
        assert int(data["sample_seed"]) == 126
        assert int(data["solve_seed"]) == 127
        np.testing.assert_array_equal(data["lambda_star_support_edges"], [(0, 1), (0, 2), (0, 3)])
        np.testing.assert_array_equal(data["Sigma"], covariance)
        np.testing.assert_array_equal(data["fitted_lambdas"], result.curve.fitted_lambdas)
        np.testing.assert_array_equal(covariance, study.sample_empirical_covariance(
            data["Sigma_population"], 100, seed=126,
        ))


def test_legacy_reselection_uses_local_selector_without_changing_curve(package_trial, tmp_path):
    args, _, _ = package_trial
    with np.load(args.output, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files
                if key not in ("fitted_lambdas", "fitted_omegas", "package_version", "package_path")}
    settings = json.loads(str(data["fit_settings_json"]))
    settings["min_omega"] = 0.0
    data["fit_settings_json"] = json.dumps(settings)
    trial = tmp_path / "support_00/num_samples_100/seed_124"
    trial.mkdir(parents=True)
    legacy = trial / "objective_curve_sigma_hat.npz"
    np.savez_compressed(legacy, **data)
    original = legacy.read_bytes()
    env = dict(os.environ, PYTHON=sys.executable, N_JOBS="1", MPLCONFIGDIR=str(tmp_path / "mpl"))
    completed = subprocess.run(
        ["bash", str(ROOT / "experiments/run_upper_support_scaling_study.sh"), "reselect", str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert legacy.read_bytes() == original
    selection = tmp_path / "support_00/reselect_lm_support_count_plateau/num_samples_100/seed_124/selection_plateau.json"
    _load_trial(legacy, selection, 100, 124)
    assert "package_path" not in json.loads(selection.read_text())
    assert (tmp_path / "reselect_lm_support_count_plateau/selection_summary.md").exists()


@pytest.mark.parametrize("options", [
    ["--refine-after-fixed-omega", "true"],
    ["--omega-ref", "none", "--fit-omega-ref", "false"],
    ["--nested-supports", "false"],
])
def test_unsupported_settings_rejected_before_fitting(tmp_path, options):
    with pytest.raises(SystemExit):
        trial_args(tmp_path, *options)


def test_estimated_reference_and_study_settings_forwarded_to_select_support(tmp_path, monkeypatch):
    calls = []

    def capture(sigma_hat, **options):
        calls.append(options)
        raise ValueError("Stop after capturing options")

    monkeypatch.setattr(study.recovery, "select_support", capture)
    args = trial_args(
        tmp_path, "--fit-omega-ref", "true", "--kappa", "0.8", "--max-restarts", "7",
        "--objective-floor", "1e-7", "--top-plateaus", "2", "--bootstrap-alpha", "0.1",
        "--bootstrap-seed", "456", "--bootstrap-replicates", "39", "--n-jobs", "2",
    )
    with pytest.raises(ValueError, match="capturing"):
        study.run_trial(args)
    assert len(calls) == 1
    assert {key: calls[0][key] for key in (
        "omega_ref", "fit_omega_ref", "kappa", "max_restarts", "objective_floor",
        "top_plateaus", "bootstrap_alpha", "bootstrap_seed", "bootstrap_replicates", "n_jobs",
    )} == dict(
        omega_ref=None, fit_omega_ref=True, kappa=0.8, max_restarts=7, objective_floor=1e-7,
        top_plateaus=2, bootstrap_alpha=0.1, bootstrap_seed=456, bootstrap_replicates=39, n_jobs=2,
    )
    assert not args.output.exists()  # Package failure does not export a partial result.
