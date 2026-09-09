"""Exercise the Bash reselect worker pool without running numerical solvers."""

import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/run_upper_support_scaling_study.sh"


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
