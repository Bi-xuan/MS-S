"""Exercise SEARCH_OPTION through the fixed-support Bash runner."""

import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/run_fixed_support_scaling.sh"


@pytest.fixture
def runner(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "-c":
    os.execv(sys.executable, [sys.executable, *args])
with Path(os.environ["CALLS"]).open("a") as f:
    f.write(json.dumps(args) + "\\n")
if "compute_objective_curve.py" in args[0]:
    Path(args[args.index("--sigma-hat-output") + 1]).touch()
if "select_scaling_parameter.py" in args[0]:
    Path(args[args.index("--output-json") + 1]).write_text(json.dumps({"precision": 1.0}))
    print("Selected dimension: 4")
''')
    stub.chmod(0o755)
    source = SCRIPT.read_text().replace(
        'PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"',
        "PROJECT_ROOT=" + shlex.quote(str(tmp_path)),
    ).replace("NUM_SAMPLE_VALUES=(100 1000 10000 1000000)", "NUM_SAMPLE_VALUES=(100)")
    source = source.replace("seeds=(124 139 141 147 153 156 173 192 237 243)", "seeds=(124)")
    harness = tmp_path / "runner.sh"
    harness.write_text(source)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               CALLS=str(tmp_path / "calls.jsonl"), TOTAL_CPUS="1", CPUS_PER_TRIAL="1", N_JOBS="1")
    for key in ("SEARCH_OPTION", "OUTPUT_ROOT", "NESTED_SUPPORTS", "N"):
        env.pop(key, None)
    return harness, env, tmp_path


@pytest.mark.parametrize("mode", [None, "forward", "forward-backward-nested"])
def test_mode_reaches_curve_and_keeps_nested_selection(runner, mode):
    harness, env, root = runner
    if mode:
        env["SEARCH_OPTION"] = mode
    result = subprocess.run(["bash", str(harness)], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    compute = next(call for call in calls if "compute_objective_curve.py" in call[0])
    assert compute[compute.index("--search-option") + 1] == (mode or "forward")
    assert compute[compute.index("--nested-supports") + 1] == "true"
    suffix = "_forward_backward_nested" if mode == "forward-backward-nested" else ""
    output = root / f"experiments/output/fixed_support_scaling_n4{suffix}"
    assert compute[compute.index("--sigma-hat-output") + 1] == str(output.relative_to(root) / "num_samples_100/seed_124/objective_curve_sigma_hat.npz")
    assert (output / "num_samples_100/seed_124/result.csv").read_text().strip() == "100,124,4,1.0,true"
    selection = next(call for call in calls if "select_scaling_parameter.py" in call[0])
    assert selection[selection.index("--method") + 1] == "plateau-bootstrap"


def test_explicit_output_root_is_preserved(runner):
    harness, env, root = runner
    env.update(SEARCH_OPTION="forward-backward-nested", OUTPUT_ROOT=str(root / "custom"))
    result = subprocess.run(["bash", str(harness)], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert (root / "custom/num_samples_100/seed_124/result.csv").exists()


@pytest.mark.parametrize("settings", [{"SEARCH_OPTION": "invalid"}, {"NESTED_SUPPORTS": "false"}])
def test_invalid_mode_fails_before_starting_workers(runner, settings):
    harness, env, root = runner
    env.update(settings)
    result = subprocess.run(["bash", str(harness)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "Error:" in result.stderr
    assert not Path(env["CALLS"]).exists()
    assert not (root / "experiments").exists()


def test_reselect_does_not_recompute_or_apply_environment_search_mode(runner):
    harness, env, root = runner
    output = root / "saved"
    trial = output / "num_samples_100/seed_124"
    trial.mkdir(parents=True)
    (trial / "objective_curve_sigma_hat.npz").touch()
    env.update(SEARCH_OPTION="ignored-for-reselect", OUTPUT_ROOT=str(output))
    result = subprocess.run(["bash", str(harness), "reselect"], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert not any("compute_objective_curve.py" in call[0] for call in calls)
    assert (output / "reselect_lm_support_count_plateau/num_samples_100/seed_124/result.csv").exists()
