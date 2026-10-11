"""Verify scope-specific metrics, including selected lower-triangular edges."""

import json

import numpy as np
import pytest

from experiments import plot_roc_curve as roc
from experiments.plot_edge_recovery_heatmap import collect_frequencies, plot_heatmap
from experiments.summarize_fixed_support_scaling import _load_trial, collect_trials


def save_trial(root, scope, samples=100, seed=1, nested_supports=True):
    directory = root / f"num_samples_{samples}" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    truth = [(0, 2), (1, 2)]
    edges = ([(2, 0), *truth, (0, 1), (1, 0), (2, 1)]
             if scope == "all" else [*truth, (0, 1)])
    selected_dimension = 4 if scope == "all" else 3
    masks = np.repeat(np.eye(3, dtype=bool)[None], len(edges) + 1, axis=0)
    for index, (i, j) in enumerate(edges, 1):
        masks[index:, i, j] = True
    if not nested_supports:
        # The one-edge support loses its edge at the next dimension.
        masks[1] = np.eye(3, dtype=bool)
        masks[1, 0, 1] = True
    curve = directory / "objective_curve_sigma_hat.npz"
    np.savez(curve, n=3, num_samples=samples, random_seed=seed,
             support_scope=scope, nested_supports=nested_supports,
             lambda_star_support_edges=np.array(truth),
             d_m_values=np.arange(1, len(edges) + 2), selected_support_masks=masks,
             selected_support_valid=np.ones(len(edges) + 1, dtype=bool))
    selection = directory / "selection_plateau_bootstrap.json"
    selection.write_text(json.dumps({
        "input": str(curve), "selected_dimension": selected_dimension,
        "selected_edges": edges[:selected_dimension - 1],
    }))
    return curve, selection


@pytest.mark.parametrize("scope", ["all", "upper"])
def test_scope_controls_summary_roc_and_heatmap(tmp_path, scope):
    curve, selection = save_trial(tmp_path, scope)
    summary = _load_trial(curve, selection, 100, 1, support_scope=scope)
    assert summary.precision == pytest.approx(2 / 3 if scope == "all" else 1)
    assert summary.exact_support_recovery == (scope == "upper")
    assert summary.true_support_on_path == (scope == "upper")
    assert summary.best_on_path_selection
    assert collect_trials(tmp_path, (100,), 1, support_scope=scope) == [summary]

    dimensions, fpr, tpr, _, path_edges = roc._load_seed_path(curve, 100, scope)
    if scope == "all":
        np.testing.assert_allclose(fpr, [0, .25, .25, .25, .5, .75, 1])
        np.testing.assert_allclose(tpr, [0, 0, .5, 1, 1, 1, 1])
        assert (2, 0) in path_edges[3]
    else:
        np.testing.assert_allclose(fpr, [0, 0, 0, 1])
        np.testing.assert_allclose(tpr, [0, .5, 1, 1])
    assert roc._load_selected_index(selection, dimensions, path_edges) == summary.selected_dimension - 1

    edges, _, _, frequencies, counts, denominators = collect_frequencies(tmp_path, scope)
    assert len(edges) == (6 if scope == "all" else 3)
    assert denominators == [1]
    assert counts.sum() == (3 if scope == "all" else 2)
    np.testing.assert_array_equal(frequencies, counts)
    if scope == "all":
        assert counts[edges.index((2, 0)), 0] == 1
    # Omitting an explicit scope reads the saved scope for both old and new data.
    assert _load_trial(curve, selection, 100, 1) == summary
    np.testing.assert_array_equal(roc._load_seed_path(curve, 100)[1], fpr)
    assert collect_frequencies(tmp_path)[0] == edges


@pytest.mark.parametrize("scope", ["all", "upper"])
def test_analysis_accepts_unrestricted_supports(tmp_path, scope):
    curve, selection = save_trial(tmp_path, scope, nested_supports=False)
    summary = _load_trial(curve, selection, 100, 1, support_scope=scope)
    assert summary.best_on_path_selection
    dimensions, fpr, tpr, _, path_edges = roc._load_seed_path(curve, 100, scope)
    assert not path_edges[1] <= path_edges[2]
    assert fpr[1] == pytest.approx(1 / (4 if scope == "all" else 1))
    assert tpr[1] == 0
    assert roc._load_selected_index(selection, dimensions, path_edges) == summary.selected_dimension - 1
    # A path declared nested must still satisfy that constraint.
    with np.load(curve, allow_pickle=False) as data:
        saved = dict(data)
    saved["nested_supports"] = True
    np.savez(curve, **saved)
    with pytest.raises(ValueError, match="not nested"):
        roc._load_seed_path(curve, 100, scope)


@pytest.mark.parametrize("saved_scope, requested_scope", [("all", "upper"), ("upper", "all")])
def test_analysis_rejects_scope_mismatch(tmp_path, saved_scope, requested_scope):
    curve, selection = save_trial(tmp_path, saved_scope)
    with pytest.raises(ValueError, match="expected"):
        _load_trial(curve, selection, 100, 1, support_scope=requested_scope)
    with pytest.raises(ValueError, match="expected"):
        roc._load_seed_path(curve, 100, requested_scope)
    with pytest.raises(ValueError, match="expected"):
        collect_frequencies(tmp_path, requested_scope)


def test_analysis_rejects_mixed_saved_scopes(tmp_path, monkeypatch):
    save_trial(tmp_path, "all", seed=1)
    save_trial(tmp_path, "upper", seed=2)
    with pytest.raises(ValueError, match="inconsistent"):
        collect_frequencies(tmp_path)
    with pytest.raises(ValueError, match="expected"):
        collect_trials(tmp_path, (100,), 2)
    monkeypatch.setattr(roc, "EXPECTED_SEED_COUNT", 2)
    with pytest.raises(ValueError, match="expected"):
        roc.collect_mean_roc(tmp_path)


@pytest.mark.parametrize("scope", ["all", "upper"])
@pytest.mark.parametrize("nested_supports", [True, False])
def test_both_scopes_render_all_plots(tmp_path, monkeypatch, scope, nested_supports):
    monkeypatch.setattr(roc, "EXPECTED_SEED_COUNT", 1)
    for samples in roc.SAMPLE_SIZES:
        save_trial(tmp_path, scope, samples=samples, nested_supports=nested_supports)
    outputs = [tmp_path / name for name in ("heatmap.png", "roc.png", "final.png")]
    plot_heatmap(tmp_path, outputs[0], scope)
    roc.plot_mean_roc(tmp_path, outputs[1], scope)
    roc.plot_final_selections(tmp_path, outputs[2], scope)
    assert all(path.stat().st_size > 0 for path in outputs)
