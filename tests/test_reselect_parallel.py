"""Exercise the Bash reselect worker pool without running numerical solvers."""

import os
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/run_upper_support_scaling_study.sh"


@pytest.fixture
def study_env(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"
    python_stub.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "-c":
    os.execv(sys.executable, [sys.executable, *args])
with (Path(os.environ["OUTPUT_ROOT"]) / "calls.jsonl").open("a") as f:
    f.write(json.dumps(args) + "\\n")
if "compute_objective_curve.py" in args[0]:
    Path(args[args.index("--sigma-hat-output") + 1]).touch()
if "summarize_fixed_support_scaling.py" in args[0]:
    for flag in ("--trials-output", "--summary-output"):
        if flag in args:
            path = Path(args[args.index(flag) + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("summary")
if "--method" in args:
    Path(args[args.index("--output-json") + 1]).write_text(
        json.dumps({"precision": 0.75}))
    print("Selected dimension at recommended scale: 4")
''')
    python_stub.chmod(0o755)
    env = dict(os.environ, OUTPUT_ROOT=str(output), N_JOBS="1",
               PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    for key in ("SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK", "NSLOTS"):
        env.pop(key, None)
    return output, env


@pytest.mark.parametrize("lm_weight", [None, "0.2"])
def test_reselect_uses_support_count_plateau_and_separate_outputs(study_env, lm_weight):
    output, env = study_env
    trial = output / "support_18/num_samples_100/seed_2"
    trial.mkdir(parents=True)
    (trial / "objective_curve_sigma_hat.npz").touch()
    env.pop("LM_WEIGHT", None)
    if lm_weight is not None:
        env["LM_WEIGHT"] = lm_weight
    result = subprocess.run(["bash", str(SCRIPT), "reselect"], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "unbound variable" not in result.stderr
    selection_dir = output / "support_18/reselect_lm_support_count_plateau/num_samples_100/seed_2"
    assert (selection_dir / "result.csv").read_text().strip() == "100,2,4,0.75,false"
    assert not (trial / "result.csv").exists()
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    selection = next(call for call in calls if "--method" in call)
    assert selection[selection.index("--method") + 1] == "plateau"
    assert selection[selection.index("--lm-mode") + 1] == "support-count"
    assert selection[selection.index("--Lm") + 1] == (lm_weight or "0.1")
    assert selection[selection.index("--n-jobs") + 1] == "1"
    assert "--bootstrap-replicates" not in selection
    summary = next(call for call in calls if "summarize_fixed_support_scaling.py" in call[0])
    assert summary[summary.index("--sample-sizes") + 1] == "100"
    assert summary[summary.index("--expected-seeds") + 1] == "1"
    assert summary[summary.index("--selection-file") + 1] == "selection_plateau.json"
    assert summary[summary.index("--score-a") + 1] == "0.5"
    overall = next(call for call in calls if "--overall-trials" in call)
    assert overall[overall.index("--overall-trials") + 1] == str(
        output / "support_18/reselect_lm_support_count_plateau/selection_trials.csv")
    assert (output / "reselect_lm_support_count_plateau/selection_summary.md").exists()


@pytest.fixture
def reselect_harness(tmp_path):
    # Keep discovery and worker scheduling intact; replace the expensive trial
    # and summary operations with events so concurrency and failures are observable.
    source = SCRIPT.read_text()
    overrides = r'''
run_trial() {
    printf 'start %s %s\n' "$3" "$4" >> "${OUTPUT_ROOT}/events"
    sleep 0.1
    if [[ "$3" == "${FAIL_SEED:-none}" ]]; then
        printf 'failed %s\n' "$3" >> "${OUTPUT_ROOT}/events"
        return 1
    fi
    printf 'end %s\n' "$3" >> "${OUTPUT_ROOT}/events"
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
        trial = output / f"support_18/num_samples_100/seed_{seed}"
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
            assert parts[2] == "true"  # Use the curve-only selection path.
            active.add(parts[1])
            started.append(parts[1])
            peak = max(peak, len(active))
        elif parts[0] == "end":
            active.remove(parts[1])
        else:
            assert parts == ["aggregate", "2"]
            assert not active
    assert sorted(started) == [str(seed) for seed in range(1, 7)]
    assert peak == expected_workers
    assert events[-1] == "aggregate 2"


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


@pytest.mark.parametrize("task_id,support,samples,seed", [
    (0, 0, 100, 124), (10, 0, 1000, 6), (20, 0, 10000, 3),
    (30, 0, 1000000, 2), (40, 1, 100, 124), (799, 19, 1000000, 237),
])
def test_trial_maps_support_sample_and_seed_and_forwards_configuration(
    study_env, task_id, support, samples, seed,
):
    output, env = study_env
    env.update(CPUS_PER_TRIAL="2", TOTAL_CPUS="6", TOP_PLATEAUS="2",
               BOOTSTRAP_REPLICATES="39", BOOTSTRAP_ALPHA="0.1",
               BOOTSTRAP_SEED="456", LAMBDA_STAR_OFFDIAG_ABS_MIN="0.3",
               LAMBDA_STAR_OFFDIAG_ABS_MAX="0.5", MAX_RESTARTS="7",
               OMEGA_STAR="2", OMEGA_REF="1.5", KAPPA="0.8",
               FIT_OMEGA_REF="true", REFINE_AFTER_FIXED_OMEGA="true",
               SUPPORT_SCOPE="upper", NESTED_SUPPORTS="true", OBJECTIVE_FLOOR="1e-7")
    result = subprocess.run(["bash", str(SCRIPT), "trial", str(task_id)], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    trial = output / f"support_{support:02d}/num_samples_{samples}/seed_{seed}"
    assert (trial / "objective_curve_sigma_hat.npz").is_file()
    assert (trial / "compute.log").is_file()
    assert (trial / "selection_plateau_bootstrap.json").is_file()
    assert (trial / "selection_plateau_bootstrap.log").is_file()
    assert (trial / "result.csv").read_text().strip() == f"{samples},{seed},4,0.75,false"
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    compute, selection = calls
    for flag, value in {
        "--lambda-star-support-index": support, "--num-samples": samples,
        "--random-seed": seed, "--lambda-star-dims": 4, "--lambda-star-dimension": 4,
        "--n-jobs": 2, "--lambda-star-offdiag-abs-min": "0.3",
        "--lambda-star-offdiag-abs-max": "0.5", "--max-restarts": 7,
        "--omega-star": 2, "--omega-ref": "1.5", "--kappa": "0.8",
        "--fit-omega-ref": "true", "--refine-after-fixed-omega": "true",
        "--support-scope": "upper", "--nested-supports": "true",
    }.items():
        assert compute[compute.index(flag) + 1] == str(value)
    assert compute[compute.index("--given-output") + 1] == str(trial / "sigma_hat.npz")
    for flag, value in {
        "--method": "plateau-bootstrap", "--n-jobs": 2, "--top-plateaus": 2,
        "--bootstrap-replicates": 39, "--bootstrap-alpha": "0.1",
        "--bootstrap-seed": 456, "--objective-floor": "1e-7",
    }.items():
        assert selection[selection.index(flag) + 1] == str(value)


def test_aggregate_writes_fixed_support_outputs_for_each_support(study_env):
    output, env = study_env
    # A stub summarizer records the invocation and writes its requested outputs.
    for support in range(20):
        (output / f"support_{support:02d}").mkdir()
    result = subprocess.run(["bash", str(SCRIPT), "aggregate"], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 21
    for support, call in enumerate(calls[:-1]):
        support_dir = output / f"support_{support:02d}"
        assert call[1] == str(support_dir)
        assert call[call.index("--sample-sizes") + 1:call.index("--expected-seeds")] == [
            "100", "1000", "10000", "1000000"]
        assert call[call.index("--expected-seeds") + 1] == "10"
        assert (support_dir / "selection_trials.csv").is_file()
        assert (support_dir / "selection_summary.md").is_file()
    assert calls[-1][1] == "--overall-trials"
    assert calls[-1][2:calls[-1].index("--summary-output")] == [
        str(output / f"support_{support:02d}/selection_trials.csv")
        for support in range(20)
    ]
    assert (output / "selection_summary.md").is_file()


def test_run_all_schedules_all_800_trials_before_aggregation(tmp_path):
    source = SCRIPT.read_text()
    overrides = r'''
run_trial() {
    printf '%s,%s,%s\n' "$1" "$2" "$3" >> "${OUTPUT_ROOT}/events"
}
aggregate_results() {
    printf 'aggregate\n' >> "${OUTPUT_ROOT}/events"
}
'''
    script = tmp_path / "study.sh"
    script.write_text(source.replace('command="${1:-auto}"', overrides + '\ncommand="${1:-auto}"'))
    env = dict(os.environ, OUTPUT_ROOT=str(tmp_path), TOTAL_CPUS="6", CPUS_PER_TRIAL="2")
    env.pop("SLURM_ARRAY_TASK_ID", None)
    result = subprocess.run(["bash", str(script), "run-all"], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "3 concurrent trials x 2 CPUs" in result.stdout
    events = (tmp_path / "events").read_text().splitlines()
    assert events[-1] == "aggregate"
    assert len(events) == 801
    assert len(set(events[:-1])) == 800
    for support in range(20):
        for samples in (100, 1000, 10000, 1000000):
            assert sum(row.startswith(f"{support},{samples},") for row in events[:-1]) == 10


@pytest.mark.parametrize("settings,message", [
    ({"NUM_SUPPORTS": "21"}, "between 1 and 20"),
    ({"N": "3"}, "only 3 supports exist"),
    ({"TOTAL_CPUS": "2", "CPUS_PER_TRIAL": "5"}, "TOTAL_CPUS >= CPUS_PER_TRIAL"),
    ({"LAMBDA_STAR_OFFDIAG_ABS_MIN": "0.7"}, "0 < MIN <= MAX"),
])
def test_study_rejects_invalid_configuration(study_env, settings, message):
    output, env = study_env
    result = subprocess.run(["bash", str(SCRIPT), "trial", "0"], env=dict(env, **settings),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert message in result.stderr
    assert not list(output.glob("support_*"))
