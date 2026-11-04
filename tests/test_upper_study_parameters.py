"""Check saved methodology stays accurate when commands reuse older curves."""

from concurrent.futures import ThreadPoolExecutor
import json
import shutil

import numpy as np
import pytest

from experiments.write_upper_study_parameters import write_parameters


def saved_trial(root, seed=124, method="plateau"):
    trial = root / f"support_00/num_samples_100/seed_{seed}"
    trial.mkdir(parents=True)
    np.savez(trial / "objective_curve_sigma_hat.npz", n=4, lambda_star_dimension=4,
             search_option="greedy", support_scope="upper", nested_supports=False,
             fit_omega_ref=True, kappa=.93, omega_ref=1.2,
             lambda_star_min_edge_magnitude=.2, lambda_star_max_edge_magnitude=.6,
             fit_settings_json=json.dumps({"omega_fixed": 1.2, "max_restarts": 10, "tol": 1e-7}),
             bic_fit_options_json=json.dumps({"penalty": "edge count * log(n)"}))
    report = dict(method=method, objective_floor=1e-8, Lm_mode="support-count", Lm=[.1, .6, 1.5])
    if method == "plateau-bootstrap":
        report.update(top_plateaus=3, bootstrap_replicates=199, alpha=.05, seed=20260913, nested=False)
    selection = trial / "selection_plateau_bootstrap.json"
    selection.write_text(json.dumps(report))
    return selection


def test_reselection_records_saved_search_instead_of_current_shell_defaults(tmp_path):
    saved_trial(tmp_path, method="plateau-bootstrap")
    path = write_parameters(tmp_path, "reselect", environ={
        "SEARCH_OPTION": "forward", "SUPPORT_SCOPE": "all", "NESTED_SUPPORTS": "true",
        "RESELECT_METHOD": "plateau-bootstrap", "LM_WEIGHT": "0.1", "N_JOBS": "8",
        "STUDY_SAMPLE_SIZES": "100 1000 10000 1000000", "TOTAL_TRIALS": "800",
    })
    data = json.loads(path.read_text())
    assert data["configured_parameters"]["SEARCH_OPTION"] == "forward"
    assert data["configured_parameters"]["NESTED_SUPPORTS"] is True
    saved = data["saved_results"]
    assert saved["search"] == {"search_option": "greedy", "support_scope": "upper", "nested_supports": False}
    assert saved["fitting"]["max_restarts"] == 10
    assert saved["selection"]["method"] == "plateau-bootstrap"
    assert saved["selection"]["bootstrap_replicates"] == 199
    assert saved["data_generation"]["fit_omega_ref"] is True
    assert saved["random_seeds_by_sample_size"] == {"100": [124]}
    assert saved["bic"]["penalty"] == "edge count * log(n)"
    assert saved["metadata_read_errors"] == []


def test_aggregate_preserves_actual_selection_and_reports_mixed_settings(tmp_path):
    saved_trial(tmp_path, seed=124, method="plateau")
    saved_trial(tmp_path, seed=139, method="plateau-bootstrap")
    data = json.loads(write_parameters(tmp_path, "aggregate", environ={"RESELECT_METHOD": "plateau"}).read_text())
    assert data["saved_results"]["selection"]["method"] == {
        "values": ["plateau", "plateau-bootstrap"], "distinct_count": 2,
    }
    assert data["saved_results"]["curve_count"] == 2
    assert data["saved_results"]["random_seeds_by_sample_size"] == {"100": [124, 139]}


def test_plateau_metadata_excludes_bootstrap_defaults_and_inactive_report_fields(tmp_path):
    selection = saved_trial(tmp_path, method="plateau")
    report = json.loads(selection.read_text())
    report.update(top_plateaus=3, bootstrap_replicates=199, alpha=.05, seed=20260913, nested=False)
    selection.write_text(json.dumps(report))
    selection.with_suffix(".log").write_text("Recommendation factor: 2\n")
    data = json.loads(write_parameters(tmp_path, "aggregate", environ={
        "RESELECT_METHOD": "plateau-bootstrap", "LM_WEIGHT": "9", "TOP_PLATEAUS": "10",
        "BOOTSTRAP_REPLICATES": "999", "BOOTSTRAP_ALPHA": ".01", "BOOTSTRAP_SEED": "42",
        "OBJECTIVE_FLOOR": "1e-6",
    }).read_text())
    assert data["saved_results"]["selection"] == {
        "method": "plateau", "objective_floor": 1e-8, "Lm_mode": "support-count",
        "Lm": [.1, .6, 1.5], "recommendation_factor": 2.0,
    }
    assert data["configured_parameters"] == {}
    assert "current_selector_defaults" not in data
    assert "plateau-bootstrap" not in json.dumps(data)


def test_bootstrap_metadata_uses_saved_settings_and_omits_plateau_options(tmp_path):
    selection = saved_trial(tmp_path, method="plateau-bootstrap")
    report = json.loads(selection.read_text())
    report["recommendation_factor"] = 2.0
    selection.write_text(json.dumps(report))
    data = json.loads(write_parameters(tmp_path, "aggregate", environ={
        "RESELECT_METHOD": "plateau", "BOOTSTRAP_REPLICATES": "999",
    }).read_text())
    saved = data["saved_results"]["selection"]
    assert saved["method"] == "plateau-bootstrap"
    assert saved["bootstrap_replicates"] == 199
    assert saved["alpha"] == .05 and saved["seed"] == 20260913
    assert "recommendation_factor" not in saved
    assert "RESELECT_METHOD" not in data["configured_parameters"]


def test_incomplete_concurrent_report_is_explicit_then_refreshed(tmp_path):
    selection = saved_trial(tmp_path)
    complete = selection.read_text()
    selection.write_text('{"method":')
    data = json.loads(write_parameters(tmp_path, "trial", task_id=0, environ={}).read_text())
    assert len(data["saved_results"]["metadata_read_errors"]) == 1
    assert "selection" not in data["saved_results"]
    selection.write_text(complete)
    data = json.loads(write_parameters(tmp_path, "aggregate", environ={}).read_text())
    assert not data["saved_results"]["metadata_read_errors"]
    assert data["saved_results"]["selection"]["method"] == "plateau"


def test_concurrent_array_snapshots_leave_complete_json_and_no_temporary_files(tmp_path):
    saved_trial(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda task: write_parameters(tmp_path, "trial", task_id=task, environ={}), range(12)))
    data = json.loads(paths[-1].read_text())
    assert data["last_command"] == "trial"
    assert data["task_id"] in range(12)
    assert data["saved_results"]["curve_count"] == 1
    assert not list(tmp_path.glob(".methodological_parameters_*"))


@pytest.mark.parametrize("method,mode,lm", [("plateau", "constant", 1.0),
                                          ("plateau-bootstrap", "support-count", [.1, .6, 1.5])])
def test_comparison_metadata_reads_only_its_own_reports_at_both_levels(tmp_path, method, mode, lm):
    canonical = saved_trial(tmp_path, method="plateau")
    shutil.copytree(tmp_path / "support_00", tmp_path / "support_01")
    original = canonical.read_bytes()
    subdir = "reselect_unique/experiment"
    for support_name in ("support_00", "support_01"):
        destination = tmp_path / support_name / subdir / "num_samples_100/seed_124"
        destination.mkdir(parents=True)
        report = dict(method=method, Lm_mode=mode, Lm=lm, objective_floor=1e-8)
        if method == "plateau-bootstrap":
            report.update(top_plateaus=2, bootstrap_replicates=3, alpha=.05, seed=42, nested=False)
        (destination / "selection.json").write_text(json.dumps(report))
    for source, output, count in ((tmp_path, tmp_path / subdir, 2),
                                  (tmp_path / "support_00", tmp_path / "support_00" / subdir, 1)):
        data = json.loads(write_parameters(output, "reselect", source_root=source,
            selection_subdir=subdir, selection_file="selection.json", environ={}).read_text())
        saved = data["saved_results"]
        assert saved["curve_count"] == count and saved["support_count"] == count
        assert saved["selection"]["method"] == method
        assert saved["selection"]["Lm_mode"] == mode and saved["selection"]["Lm"] == lm
        assert saved["search"]["search_option"] == "greedy"
        assert not saved["metadata_read_errors"] and not saved["missing_selection_reports"]
        assert data["source_root"] == str(source)
    assert canonical.read_bytes() == original
