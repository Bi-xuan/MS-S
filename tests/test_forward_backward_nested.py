"""Endpoint corrections, complete prefix replacement, caching and resume."""

import json

import numpy as np
import pytest

from experiments import compute_objective_curve as curve
from optimizers import support_search as search
from optimizers.nested_search import NestedCurveResult


A, B, C = (0, 1), (0, 2), (1, 2)
MODE = "forward-backward-nested"


@pytest.fixture
def landscape(monkeypatch):
    scores = {(): 10., (A,): 1., (B,): 2., (C,): 3.,
              (A, B): .6, (A, C): .3, (B, C): .1, (A, B, C): .01}
    calls = []

    def worker(task):
        index, sigma, mask = task[:3]
        edges = tuple(search.support_edges_from_mask(mask))
        calls.append((edges, task[-2]))
        score = scores[edges]
        return index, mask, (None if not np.isfinite(score) else
                             (np.zeros_like(sigma), .1, score))

    monkeypatch.setattr(search, "solve_support_worker", worker)
    return scores, calls


def compute(**kwargs):
    return curve.compute_objective_curve(
        np.eye(3), support_scope="upper", search_option=MODE, **kwargs,
    )


def edges(result):
    return [tuple(search.support_edges_from_mask(mask)) for mask in result[4]]


def assert_nested(result):
    np.testing.assert_array_equal(result[0], np.arange(1, len(result[0]) + 1))
    for index, mask in enumerate(result[4]):
        assert np.all(np.diag(mask))
        assert np.count_nonzero(mask) == 3 + index
        if index:
            assert np.all(~result[4][index - 1] | mask)


def test_correction_replaces_entire_prefix_and_caches_fits(landscape):
    scores, calls = landscape
    checkpoints = []
    result = compute(save_callback=checkpoints.append)
    assert edges(result) == [(), (B,), (B, C), (A, B, C)]
    assert result[1].tolist() == [10., 2., .1, .01]
    assert result.search_state["corrections"] == 1
    assert result.search_state["complete"]
    assert result.search_state["best_seen"][1] == 1.  # Worse prefix is intentional.
    assert len(calls) == len({support for support, seed in calls}) == len(scores)
    assert edges(checkpoints[2])[-1] == (A, C)
    assert edges(checkpoints[3]) == [(), (B,), (B, C)]
    for checkpoint in checkpoints:
        assert_nested(checkpoint)


def test_forward_keeps_existing_path(landscape):
    default = curve.compute_objective_curve(np.eye(3), support_scope="upper")
    explicit = curve.compute_objective_curve(
        np.eye(3), support_scope="upper", search_option="forward",
    )
    assert edges(default) == [(), (A,), (A, C), (A, B, C)]
    for actual, expected in zip(explicit, default):
        np.testing.assert_array_equal(actual, expected)


def test_parallel_completion_order_and_preselection_do_not_change_path(landscape, monkeypatch):
    baseline = compute()
    monkeypatch.setattr(search, "iter_parallel_support_results",
                        lambda tasks, workers: reversed([search.solve_support_worker(t) for t in tasks]))
    parallel = curve.compute_objective_curve(
        np.eye(3), support_scope="all", preselect_edges=[C, A, B], n_jobs=3,
        search_option=MODE,
    )
    for actual, expected in zip(parallel, baseline):
        np.testing.assert_array_equal(actual, expected)


def save_checkpoint(path, result):
    curve.save_curve_result(
        path, "given_sigma", 3, np.zeros((3, 3)), np.eye(3), 1., .1,
        42, 42, 43, -1, 1e-6, None, "directed", None, None, result,
        support_scope="upper", search_option=MODE,
    )


@pytest.mark.parametrize("after_correction", [False, True])
def test_disk_resume_matches_uninterrupted_search(landscape, tmp_path, after_correction):
    full = compute()
    path = tmp_path / "checkpoint.npz"

    def interrupt(result):
        save_checkpoint(path, result)
        if (result.search_state["corrections"] > 0 if after_correction else len(result[0]) == 2):
            raise InterruptedError

    with pytest.raises(InterruptedError):
        compute(save_callback=interrupt)
    loaded = curve.load_existing_curve_result(
        path, "given_sigma", 3, {"search_option": MODE, "nested_supports": True},
    )
    assert isinstance(loaded, NestedCurveResult)
    resumed = compute(initial_curve_result=loaded)
    for actual, expected in zip(resumed, full):
        np.testing.assert_array_equal(actual, expected)
    assert resumed.search_state == full.search_state


def test_completed_checkpoint_skips_fits(landscape, tmp_path):
    _, calls = landscape
    full = compute()
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, full)
    loaded = curve.load_existing_curve_result(path, "given_sigma", 3)
    calls.clear()
    assert compute(initial_curve_result=loaded) is loaded
    assert calls == []


def test_missing_history_restarts_instead_of_trusting_rewritten_prefix(landscape, capsys):
    full = compute()
    prefix = tuple(a[:3] if i in (0, 1, 4, 5) else a for i, a in enumerate(full))
    restarted = compute(initial_curve_result=prefix)
    for actual, expected in zip(restarted, full):
        np.testing.assert_array_equal(actual, expected)
    assert "restarting the search" in capsys.readouterr().out


def test_failed_extension_blocks_later_dimensions_and_resumes(landscape):
    scores, calls = landscape
    for support in list(scores):
        if support:
            scores[support] = np.inf
    result = compute()
    assert result[5].tolist() == [True, False, False, False]
    assert np.all(np.isposinf(result[1][1:]))
    calls.clear()
    assert compute(initial_curve_result=result) is result
    assert calls == []


def test_unbuildable_correction_preserves_previous_path(landscape):
    scores, _ = landscape
    scores[(B,)] = scores[(C,)] = np.inf
    result = compute()
    assert edges(result) == [(), (A,), (A, C), (A, B, C)]
    assert result.search_state["corrections"] == 0


def test_correction_tolerance_and_ties(landscape):
    scores, _ = landscape
    scores[(B, C)] = scores[(A, C)] - 1e-10
    assert edges(compute())[-2] == (A, C)
    scores[(B, C)] = scores[(A, C)]
    assert edges(compute(obj_tol=0))[-2] == (A, C)


def test_cli_metadata_and_mode_mismatch(tmp_path):
    args = curve.parse_args([
        "--given-output", str(tmp_path / "given.npz"),
        "--sigma-hat-output", str(tmp_path / "sample.npz"),
        "--search-option", MODE, "--support-scope", "upper",
        "--omega-ref", ".1", "--max-restarts", "1",
    ])
    curve.run_experiment(args, 2, False)
    for path in (args.given_output, args.sigma_hat_output):
        with np.load(path, allow_pickle=False) as saved:
            assert saved["search_option"].item() == MODE
            assert saved["nested_supports"].item()
            assert json.loads(saved["search_state_json"].item())["complete"]
    curve.run_experiment(args, 2, False)
    args.search_option = "forward"
    with pytest.raises(ValueError, match="search_option"):
        curve.run_experiment(args, 2, False)


def test_legacy_forward_mode_and_conflicting_settings(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, curve_type="given_sigma", n=3, nested_supports=True,
             d_m_values=[1], objective_values=[1.], fallback_d_m_values=[],
             fallback_objective_values=[])
    assert curve.load_existing_curve_result(path, "given_sigma", 3, {"search_option": "forward"})
    with pytest.raises(ValueError, match="search_option"):
        curve.load_existing_curve_result(path, "given_sigma", 3, {"search_option": MODE})
    with pytest.raises(ValueError, match="nested_supports=True"):
        compute(nested_supports=False)
    with pytest.raises(ValueError, match="obj_tol"):
        compute(obj_tol=-1)
    with pytest.raises(SystemExit):
        curve.parse_args(["--search-option", "invalid"])


def test_real_random_starts_resume_reproducibly(tmp_path):
    sigma = curve.covariance_from_lambda_star(curve.lambda_star_for_dimension(3), 1.)
    settings = dict(support_scope="upper", search_option=MODE, omega_ref=.93,
                    init_strategy="random", max_restarts=2, max_iter=100)
    checkpoints = []
    full = curve.compute_objective_curve(sigma, save_callback=checkpoints.append, **settings)
    resumed = curve.compute_objective_curve(sigma, initial_curve_result=checkpoints[1], **settings)
    for actual, expected in zip(resumed, full):
        np.testing.assert_array_equal(actual, expected)
    assert_nested(resumed)
