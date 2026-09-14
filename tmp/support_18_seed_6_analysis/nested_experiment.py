"""Force the true support at D=4, then greedily add one upper edge per step."""
from pathlib import Path
import contextlib
import csv
import io
import json
import sys
from types import SimpleNamespace
from dataclasses import asdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from optimizers.support_search import solve_support_with_restarts, update_best_support
from experiments.select_scaling_parameter import load_selection_inputs, report_selection
from scaling_selection import build_dimension_path, select_minimal_scale

SOURCE = ROOT / 'experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18/seed_6/objective_curve_sigma_hat.npz'
OUT = Path(__file__).resolve().parent / 'forced_true_nested'


def edges(mask):
    return (np.argwhere(mask & ~np.eye(4, dtype=bool)) + 1).tolist()


def main():
    OUT.mkdir(exist_ok=True)
    with np.load(SOURCE) as data:
        arrays = {key: data[key].copy() for key in data.files}
    original = arrays['objective_values'].copy()
    sigma = arrays['Sigma']
    mask = arrays['Lambda_star'] != 0
    omega = float(arrays['omega_ref'])
    settings = dict(beta=1.0, max_iter=800, tol=1e-7, zero_tol=1e-5,
                    max_restarts=10, min_omega=0.0, omega_fixed=omega,
                    omega_upper=float(np.linalg.eigvalsh(sigma)[0] - 1e-6),
                    init_strategy='halton')
    steps = []
    for dim in range(4, 8):
        candidates = []
        if dim == 4:
            candidates.append((None, mask.copy()))
        else:
            for i, j in np.argwhere(np.triu(np.ones((4, 4), dtype=bool), 1) & ~mask):
                extension = mask.copy()
                extension[i, j] = True
                candidates.append(([int(i + 1), int(j + 1)], extension))
        best = (None, None, np.inf, None)
        candidate_records = []
        for edge, candidate in candidates:
            fitted = solve_support_with_restarts(sigma, candidate, **settings)
            if fitted is None:
                raise RuntimeError('No finite fit for candidate support')
            candidate_records.append(dict(added_edge=edge, edges=edges(candidate),
                                          objective=float(fitted[2]), Lambda=fitted[0].tolist()))
            best = update_best_support(*best, candidate, fitted, obj_tol=1e-8)
        lam, fitted_omega, objective, mask = best
        index = int(np.flatnonzero(arrays['d_m_values'] == dim)[0])
        arrays['objective_values'][index] = objective
        arrays['selected_support_masks'][index] = mask
        arrays['selected_support_valid'][index] = True
        step = dict(dimension=dim, objective=float(objective), edges=edges(mask),
                    Lambda=lam.tolist(), omega=float(fitted_omega), candidates=candidate_records)
        steps.append(step)
        print(f'D={dim}: objective={objective:.12g}; edges={edges(mask)}', flush=True)
    assert np.array_equal(arrays['objective_values'][:3], original[:3])
    assert np.array_equal(arrays['selected_support_masks'][3], arrays['Lambda_star'] != 0)
    for index in range(4, 7):
        before, after = arrays['selected_support_masks'][index - 1:index + 1]
        assert np.all(~before | after) and int(after.sum() - before.sum()) == 1
    assert np.all(np.diff(arrays['objective_values']) <= 0)
    arrays['experiment_description'] = np.array('Original D=1..3; true support at D=4; best one-edge upper-triangular extension at D=5..7')
    arrays['source_curve'] = np.array(str(SOURCE))
    target = OUT / 'objective_curve_sigma_hat.npz'
    np.savez_compressed(target, **arrays)
    args = SimpleNamespace(num_samples=None, r=1.0, Lm=1.0, L=1.0, xi=10.0,
                           objective_floor=1e-8)
    selections = {}
    for label, path in [('original', SOURCE), ('forced_true_nested', target)]:
        data = load_selection_inputs(path, args)
        dimension_path = build_dimension_path(data['d_m_values'], data['objective_values'], data['penalty_values'])
        selections[label] = {'dimension_path': asdict(dimension_path)}
        for method in ('plateau', 'window'):
            result = select_minimal_scale(data['d_m_values'], data['objective_values'],
                                          data['penalty_values'], method=method, eta=None,
                                          recommendation_factor=2.0)
            selections[label][method] = asdict(result)
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                report_selection(path, data, result)
            (OUT / f'{label}_{method}.log').write_text(log.getvalue())
            print(f'{label} {method}: dimension={result.selected_dimension}; scale={result.recommended_scale:.12g}', flush=True)
    summary = dict(source=str(SOURCE), solver_settings=settings, obj_tol=1e-8,
                   objective_floor=1e-8, recommendation_factor=2.0,
                   original_objectives=original.tolist(), nested_objectives=arrays['objective_values'].tolist(),
                   steps=steps, selections=selections)
    (OUT / 'results.json').write_text(json.dumps(summary, indent=2))
    with (OUT / 'objective_comparison.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dimension', 'original_objective', 'forced_true_nested_objective'])
        writer.writerows(zip(arrays['d_m_values'], original, arrays['objective_values']))


if __name__ == '__main__':
    main()
