"""Exercise the Bash reselect worker pool without running numerical solvers."""

import os
import csv
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/run_upper_support_scaling_study.sh"


@pytest.mark.parametrize("top_plateaus", [None, "2"])
def test_study_runs_all_methods_and_forwards_bootstrap_options(tmp_path, top_plateaus):
    output = tmp_path / "output"
    trial = output / "support_18/seed_2"
    trial.mkdir(parents=True)
    (trial / "objective_curve_sigma_hat.npz").touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"
    python_stub.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with (Path(os.environ["OUTPUT_ROOT"]) / "calls.jsonl").open("a") as f:
    f.write(json.dumps(args) + "\\n")
if "--method" in args:
    method = args[args.index("--method") + 1]
    if method == "plateau-bootstrap":
        print("Selected dimension: 4")
        print("Selected support precision: 0.75")
    else:
        print("Selected dimension at recommended scale: 4")
''')
    python_stub.chmod(0o755)
    env = dict(os.environ, OUTPUT_ROOT=str(output), N_JOBS="1",
               PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    env.pop("TOP_PLATEAUS", None)
    if top_plateaus is not None:
        env["TOP_PLATEAUS"] = top_plateaus
    result = subprocess.run(["bash", str(SCRIPT), "reselect"], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "unbound variable" not in result.stderr
    with (trial / "result.csv").open() as f:
        row = next(csv.DictReader(f))
    for method in ("window", "plateau", "bootstrap"):
        assert row[f"{method}_status"] == "ok"
        assert row[f"{method}_dimension"] == "4"
    assert row["bootstrap_precision"] == "0.75"
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    boot = next(call for call in calls if "plateau-bootstrap" in call)
    assert boot[boot.index("--top-plateaus") + 1] == (top_plateaus or "3")
    assert boot[boot.index("--n-jobs") + 1] == "1"


@pytest.fixture
def reselect_harness(tmp_path):
    # Keep discovery and worker scheduling intact; replace the expensive trial
    # and plot operations with events so concurrency and failures are observable.
    source = SCRIPT.read_text()
    overrides = r'''
run_trial() {
    printf 'start %s %s\n' "$2" "${N_JOBS}" >> "${OUTPUT_ROOT}/events"
    sleep 0.1
    if [[ "$2" == "${FAIL_SEED:-none}" ]]; then
        printf 'failed %s\n' "$2" >> "${OUTPUT_ROOT}/events"
        return 1
    fi
    printf 'end %s\n' "$2" >> "${OUTPUT_ROOT}/events"
}
aggregate_results() {
    printf 'aggregate %s\n' "$#" >> "${OUTPUT_ROOT}/events"
}
'''
    script = tmp_path / "experiments/study.sh"
    script.parent.mkdir()
    script.write_text(source.replace('command="${1:-auto}"', overrides + '\ncommand="${1:-auto}"'))
    output = tmp_path / "output"
    for seed in range(1, 7):
        trial = output / f"support_18/seed_{seed}"
        trial.mkdir(parents=True)
        (trial / "objective_curve_sigma_hat.npz").write_bytes(b"saved curve")
    env = dict(os.environ, OUTPUT_ROOT=str(output))
    for key in ("N_JOBS", "SLURM_CPUS_PER_TASK", "NSLOTS", "FAIL_SEED"):
        env.pop(key, None)
    return script, output, env


@pytest.mark.parametrize(
    "settings, expected_workers",
    [({"N_JOBS": "1"}, 1), ({"N_JOBS": "2"}, 2),
     ({"N_JOBS": "100"}, 6), ({"NSLOTS": "2"}, 2),
     ({"N_JOBS": "1", "NSLOTS": "2"}, 1)],
)
def test_reselect_bounds_concurrency_and_aggregates_after_workers(
    reselect_harness, settings, expected_workers,
):
    script, output, env = reselect_harness
    result = subprocess.run(
        ["bash", str(script), "reselect"], env=dict(env, **settings),
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    active = set()
    started = []
    peak = 0
    events = (output / "events").read_text().splitlines()
    for event in events:
        parts = event.split()
        if parts[0] == "start":
            assert parts[2] == "1"  # No nested N_JOBS-way parallelism.
            active.add(parts[1])
            started.append(parts[1])
            peak = max(peak, len(active))
        elif parts[0] == "end":
            active.remove(parts[1])
        else:
            assert parts == ["aggregate", "6"]
            assert not active
    assert sorted(started) == [str(seed) for seed in range(1, 7)]
    assert peak == expected_workers
    assert events[-1] == "aggregate 6"


def test_reselect_worker_failure_prevents_aggregation(reselect_harness):
    script, output, env = reselect_harness
    result = subprocess.run(
        ["bash", str(script), "reselect"],
        env=dict(env, N_JOBS="2", FAIL_SEED="1"),
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode != 0
    assert "aggregation was skipped" in result.stderr
    events = (output / "events").read_text().splitlines()
    assert "failed 1" in events
    assert "end 6" in events  # The other worker was awaited before returning.
    assert not any(event.startswith("aggregate") for event in events)


@pytest.mark.parametrize("n_jobs", ["0", "-1", "bad", "1.5"])
def test_reselect_rejects_invalid_worker_count(reselect_harness, n_jobs):
    script, output, env = reselect_harness
    result = subprocess.run(
        ["bash", str(script), "reselect"], env=dict(env, N_JOBS=n_jobs),
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 2
    assert "N_JOBS to be a positive integer" in result.stderr
    assert not (output / "events").exists()
