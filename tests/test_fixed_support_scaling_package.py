"""Check package API defaults, saved-study compatibility, and launcher failures."""

from dataclasses import dataclass
from functools import wraps
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run_fixed_support_scaling_package_trial as trial
from experiments.summarize_fixed_support_scaling import _load_trial
from experiments.plot_roc_curve import _load_seed_path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/run_fixed_support_scaling_package.sh"


@dataclass
class Diagnostics:
    selected_dimension: int = 3
    selected_edges: tuple = ((0, 2), (1, 2))


def test_api_keeps_statistical_defaults_and_exports_compatible_files(tmp_path, monkeypatch):
    calls = []
    masks = np.repeat(np.eye(3, dtype=bool)[None], 4, axis=0)
    for index, edge in enumerate(((0, 2), (1, 2), (0, 1)), 1):
        masks[index:] = masks[index - 1]
        masks[index:, edge[0], edge[1]] = True

    @wraps(trial.lambda_support_recovery.select_support)
    def select(sigma_hat, **kwargs):
        calls.append((sigma_hat.copy(), kwargs))
        settings = trial.lambda_support_recovery.FitSettings(omega_fixed=.5)
        curve = SimpleNamespace(
            fit_settings=settings, resolved_omega_ref=.5,
            support_scope="upper", nested_supports=True,
            dimensions=np.arange(1, 5), raw_objectives=np.array([3., 2., 1., 0.]),
            support_masks=masks, support_valid=np.ones(4, dtype=bool),
            fitted_lambdas=np.zeros((4, 3, 3)), fitted_omegas=np.full(4, .5),
        )
        return SimpleNamespace(
            curve=curve, selected_edges=((0, 2), (1, 2)), selected_dimension=3,
            diagnostics=Diagnostics(), method="plateau-bootstrap", objective_floor=1e-8,
            lm_values=np.ones(4),
            lm_mode="constant",
        )

    monkeypatch.setattr(trial.lambda_support_recovery, "select_support", select)
    trial.run_trial(n=3, num_samples=100, seed=124, n_jobs=5, output_dir=tmp_path)
    covariance, options = calls[0]
    assert set(options) == {"num_samples", "support_scope", "n_jobs", "return_result", "progress"}
    assert options["num_samples"] == 100
    assert options["support_scope"] == "upper"
    assert options["n_jobs"] == 5
    assert options["return_result"] is True
    expected = trial.sample_empirical_covariance(
        trial.covariance_from_lambda_star(trial.lambda_star_for_dimension(3), 1.), 100, seed=126,
    )
    np.testing.assert_array_equal(covariance, expected)

    curve_path = tmp_path / "objective_curve_sigma_hat.npz"
    selection_path = tmp_path / "selection_plateau_bootstrap.json"
    summary = _load_trial(curve_path, selection_path, 100, 124)
    assert summary.exact_support_recovery
    assert summary.best_on_path_selection
    assert summary.true_support_on_path
    assert summary.precision == 1.
    dimensions, fpr, tpr, _, _ = _load_seed_path(curve_path, 100)
    np.testing.assert_array_equal(dimensions, np.arange(1, 5))
    assert fpr[-1] == tpr[-1] == 1.
    report = json.loads(selection_path.read_text())
    assert report["input"] == str(curve_path)
    assert report["sample_seed"] == 126
    assert report["api_defaults"]["random_seed"] == 42
    assert report["api_defaults"]["bootstrap_replicates"] == 199
    with np.load(curve_path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["Sigma"], covariance)
        assert saved["solve_seed"] == 42
        assert json.loads(str(saved["fit_settings_json"]))["max_restarts"] == 10


@pytest.mark.parametrize("arguments", [
    ["--n", "2"], ["--num-samples", "5"], ["--seed", "-1"],
    ["--n-jobs", "0"], ["--omega-star", "0"], ["--offdiag-abs-min", "0.7"],
])
def test_invalid_trial_settings_fail_before_fitting(arguments):
    with pytest.raises(SystemExit):
        trial.parse_args(["--num-samples", "100", "--seed", "124", "--check-only", *arguments])


@pytest.mark.parametrize("settings", [
    {"TOTAL_CPUS": "4", "CPUS_PER_TRIAL": "5"}, {"TOTAL_CPUS": "0"},
    {"CPUS_PER_TRIAL": "bad"}, {"N": "2"}, {"SUPPORT_SCOPE": "all"},
])
def test_launcher_rejects_invalid_settings_before_starting_workers(settings, tmp_path):
    env = dict(os.environ, TOTAL_CPUS="100", CPUS_PER_TRIAL="5", N="10",
               SUPPORT_SCOPE="upper", OUTPUT_ROOT=str(tmp_path / "output"), PYTHON="/nonexistent")
    env.update(settings)
    completed = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 2
    assert "Error:" in completed.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("fail_seed", [None, "124"])
def test_launcher_schedules_all_trials_and_skips_summary_after_failure(tmp_path, fail_seed):
    # Exercise the real Bash orchestration; record Python calls without running
    # numerical fitting or plotting in the scheduling test.
    stub = tmp_path / "python-stub"
    stub.write_text('''#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ["CALLS"]).open("a") as handle:
    handle.write(json.dumps(args) + "\\n")
if "--seed" in args and "--check-only" not in args:
    identity = [args[args.index("--num-samples") + 1], args[args.index("--seed") + 1]]
    with Path(os.environ["EVENTS"]).open("a") as handle:
        handle.write(json.dumps(["start", identity]) + "\\n")
    time.sleep(.03)
    with Path(os.environ["EVENTS"]).open("a") as handle:
        handle.write(json.dumps(["end", identity]) + "\\n")
    if args[args.index("--seed") + 1] == os.environ.get("FAIL_SEED"):
        sys.exit(1)
''')
    stub.chmod(0o755)
    calls_path = tmp_path / "calls.jsonl"
    events_path = tmp_path / "events.jsonl"
    env = dict(os.environ, TOTAL_CPUS="10", CPUS_PER_TRIAL="5", N="10",
               SUPPORT_SCOPE="upper", OUTPUT_ROOT=str(tmp_path / "output"),
               PYTHON=str(stub), CALLS=str(calls_path), EVENTS=str(events_path))
    if fail_seed:
        env["FAIL_SEED"] = fail_seed
    else:
        env.pop("FAIL_SEED", None)
    completed = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    fits = [args for args in calls if "--seed" in args and "--check-only" not in args]
    summaries = [args for args in calls if any("summarize_fixed_support" in value for value in args)]
    if fail_seed:
        assert completed.returncode != 0
        assert "summary and plots were skipped" in completed.stderr
        assert not summaries
    else:
        assert completed.returncode == 0, completed.stderr
        assert "2 concurrent trials x 5 workers" in completed.stdout
        assert len(fits) == 40
        pairs = {(args[args.index("--num-samples")+1], args[args.index("--seed")+1]) for args in fits}
        assert len(pairs) == 40
        assert {size: sum(a == size for a, _ in pairs) for size in ("100", "1000", "10000", "1000000")} == {
            "100": 10, "1000": 10, "10000": 10, "1000000": 10,
        }
        assert len(summaries) == 1
        assert sum(any("plot_" in value for value in args) for args in calls) == 2
        active = set()
        peak = 0
        for line in events_path.read_text().splitlines():
            event, identity = json.loads(line)
            identity = tuple(identity)
            if event == "start":
                active.add(identity)
                peak = max(peak, len(active))
            else:
                active.remove(identity)
        assert not active
        assert peak == 2
