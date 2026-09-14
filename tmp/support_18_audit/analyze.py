"""Audit saved support-18 results and explicit true-support interventions."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.select_scaling_parameter import load_selection_inputs, floor_objective_values
from scaling_selection import build_dimension_path, select_minimal_scale
from optimizers.support_search import solve_support_with_restarts, update_best_support

SOURCE = ROOT / 'experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18'
OUT = Path(__file__).resolve().parent
ARGS = SimpleNamespace(num_samples=None, r=1., Lm=1., L=1., xi=10., objective_floor=1e-8)


def selections(dims, raw, penalties, floor=1e-8):
    q, _ = floor_objective_values(raw, floor)
    path = build_dimension_path(dims, q, penalties)
    intervals = []
    for i, d in enumerate(path.dimensions):
        intervals.append(dict(dimension=d, left=path.breakpoints[i],
                              right=path.breakpoints[i + 1] if i + 1 < len(path.dimensions) else None))
    result = dict(intervals=intervals, true_interval=next((r for r in intervals if r['dimension'] == 4), None))
    for method in ('plateau', 'window'):
        try:
            result[method] = asdict(select_minimal_scale(dims, q, penalties, method=method, eta=None, recommendation_factor=2.))
        except ValueError as e:
            result[method] = dict(selected_dimension=None, error=str(e))
    return result


def fit(sigma, mask):
    settings = dict(beta=1., max_iter=800, tol=1e-7, zero_tol=1e-5,
                    max_restarts=10, min_omega=0., omega_fixed=1.,
                    omega_upper=float(np.linalg.eigvalsh(sigma)[0] - 1e-6), init_strategy='halton')
    result = solve_support_with_restarts(sigma, mask, **settings)
    if result is None:
        raise RuntimeError('No feasible fit')
    return result


def main():
    rows = []
    for directory in sorted(SOURCE.glob('seed_*'), key=lambda p: int(p.name[5:])):
        source = directory / 'objective_curve_sigma_hat.npz'
        with np.load(source) as data:
            a = {k: data[k].copy() for k in data.files}
        saved = next(csv.DictReader((directory / 'result.csv').open()))
        dims, raw, sigma = a['d_m_values'], a['objective_values'], a['Sigma']
        true = a['Lambda_star'] != 0
        idx = int(np.flatnonzero(dims == 4)[0])
        assert np.array_equal(dims, np.arange(1, 8))
        assert np.all(a['selected_support_valid'])
        assert float(a['omega_ref']) == float(a['omega_star']) == 1.
        assert np.array_equal(np.argwhere(true & ~np.eye(4, dtype=bool)), [[0, 3], [1, 3], [2, 3]])
        inputs = load_selection_inputs(source, ARGS)
        penalties = inputs['penalty_values']
        original = selections(dims, raw, penalties)
        for method in ('plateau', 'window'):
            expected = None if saved[f'{method}_dimension'] == 'NA' else int(saved[f'{method}_dimension'])
            assert original[method]['selected_dimension'] == expected, (directory, method)
            if expected is not None:
                log = (directory / f'selection_{method}.log').read_text()
                logged = float(next(s.split(': ', 1)[1] for s in log.splitlines() if s.startswith('Recommended scale:')))
                assert np.isclose(logged, original[method]['recommended_scale'], rtol=1e-10, atol=0)

        c = float(raw[idx])
        # If a true-support fit has Q <= c, each diagonal residual has magnitude
        # <= sqrt(c), hence |Lambda_ii| <= b_i for i=1,2,3. Its first 3x3 block
        # then implies Q >= sum_{i<j<=3} 2 Sigma_ij^2 (1-b_i*b_j)_+^2.
        b_squared = (np.diag(sigma)[:3] - 1. + np.sqrt(c)) / np.diag(sigma)[:3]
        b = np.sqrt(np.maximum(0., b_squared))
        bound = float(sum(2 * sigma[i, j]**2 * max(0., 1. - b[i] * b[j])**2
                          for i in range(3) for j in range(i + 1, 3)))
        certified = bool(np.any(b_squared < 0.) or bound > c)
        support_correct = bool(np.array_equal(a['selected_support_masks'][idx], true))
        assert support_correct != certified
        lam, omega, true_obj = fit(sigma, true)
        replacement = raw.copy()
        replacement[idx] = true_obj
        nested = replacement.copy()
        mask = true.copy()
        steps = [dict(dimension=4, objective=true_obj, edges=(np.argwhere(mask & ~np.eye(4, dtype=bool)) + 1).tolist())]
        for d in range(5, 8):
            best = (None, None, np.inf, None)
            for i, j in np.argwhere(np.triu(np.ones((4, 4), dtype=bool), 1) & ~mask):
                candidate = mask.copy()
                candidate[i, j] = True
                best = update_best_support(*best, candidate, fit(sigma, candidate), obj_tol=1e-8)
            _, _, objective, mask = best
            nested[d - 1] = objective
            steps.append(dict(dimension=d, objective=objective, edges=(np.argwhere(mask & ~np.eye(4, dtype=bool)) + 1).tolist()))
        assert np.all(np.diff(nested) <= 1e-8)
        row = dict(seed=int(directory.name[5:]), support_correct_at_4=support_correct,
                   selected_edges_at_4=(np.argwhere(a['selected_support_masks'][idx] & ~np.eye(4, dtype=bool)) + 1).tolist(),
                   ranking_failure_certified=certified, saved_q4=c, conditional_true_lower_bound=bound,
                   true_support_refit_objective=true_obj, true_support_refit_Lambda=lam.tolist(),
                   original_objectives=raw.tolist(), original=original,
                   true4_only_objectives=replacement.tolist(), true4_only=selections(dims, replacement, penalties),
                   true_nested_objectives=nested.tolist(), true_nested=selections(dims, nested, penalties),
                   nested_steps=steps,
                   floor_sensitivity={str(f): selections(dims, raw, penalties, f) for f in (0., 1e-10, 1e-8, 1e-7, 1e-6)})
        rows.append(row)
        (OUT / 'results.json').write_text(json.dumps(rows, indent=2))
        print(directory.name, 'correct_support', support_correct, 'certificate', certified,
              'original', [original[m]['selected_dimension'] for m in ('plateau', 'window')],
              'true4_only', [row['true4_only'][m]['selected_dimension'] for m in ('plateau', 'window')],
              'nested', [row['true_nested'][m]['selected_dimension'] for m in ('plateau', 'window')], flush=True)

    columns = ['seed', 'support_correct_at_4', 'ranking_failure_certified', 'saved_q4', 'conditional_true_lower_bound', 'true_support_refit_objective']
    columns += [f'{label}_{method}' for label in ('original', 'true4_only', 'true_nested') for method in ('plateau', 'window')]
    with (OUT / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in columns if k in row}
            flat.update({f'{label}_{method}': row[label][method]['selected_dimension'] for label in ('original', 'true4_only', 'true_nested') for method in ('plateau', 'window')})
            writer.writerow(flat)


if __name__ == '__main__':
    main()
