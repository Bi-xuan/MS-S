"""Reproducible diagnostics of the four fixed-support ROC studies.

This script reads the original results and writes only into this directory.
The diagnostic least-squares fitter optimizes only allowed Lambda entries.
"""
from pathlib import Path
import csv
import itertools
import json
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from admm import halton_point, admm_solve

OUT = Path(__file__).resolve().parent
NAMES = {
    'upper_forward': 'fixed_support_scaling_n4_upper_scope_fit_omega_ref_true',
    'all_exhaustive': 'fixed_support_scaling_n4_all_scope_fit_omega_ref_true_nested_false',
    'all_fb': 'fixed_support_scaling_n4_all_scope_fit_omega_ref_true_fbnested',
    'all_forward': 'fixed_support_scaling_n4_all_scope_fit_omega_ref_true',
}
EDGES = tuple((i, j) for i in range(4) for j in range(4) if i != j)
TRUTH = frozenset(((0, 3), (1, 3), (2, 3)))


def mask_for(edges):
    mask = np.eye(4, dtype=bool)
    for edge in edges:
        mask[edge] = True
    return mask


def edges_for(mask):
    return frozenset((int(i), int(j)) for i,j in np.argwhere(mask & ~np.eye(4, dtype=bool)))


def load(label, size, seed):
    p = ROOT / 'experiments/output' / NAMES[label] / f'num_samples_{size}' / f'seed_{seed}'
    with np.load(p / 'objective_curve_sigma_hat.npz') as d:
        return {k: d[k].copy() for k in d.files}, p


def fit(sigma, omega, edges, starts=10, extra=(), sign_starts=False):
    mask = mask_for(edges)
    rows, cols = np.where(mask)
    n = len(sigma)
    target = sigma - omega * np.eye(n)

    def unpack(x):
        a = np.zeros((n, n))
        a[rows, cols] = x
        return a

    def residual(x):
        a = unpack(x)
        return (target - a.T @ sigma @ a).ravel()

    def jacobian(x):
        sl = sigma @ unpack(x)
        j = np.zeros((n, n, len(rows)))
        for k, (a, b) in enumerate(zip(rows, cols)):
            j[b, :, k] -= sl[a, :]
            j[:, b, k] -= sl[a, :]
        return j.reshape(n*n, -1)

    points = [(2 * (halton_point(i + 1, n*n).reshape(n, n) - .5))[mask]
              for i in range(starts)]
    points += [a[mask] for a in extra]
    if sign_starts:
        diagonal = np.sqrt(np.maximum(1-omega/np.diag(sigma), 1e-4))
        for signs in itertools.product((-1,1), repeat=n-1):
            # Fix the last sign, since Lambda and -Lambda have identical loss.
            a = np.diag(diagonal*np.asarray((*signs,1)))
            points.append(a[mask])
    best = None
    for x in points:
        r = least_squares(residual, x, jac=jacobian, max_nfev=500,
                          ftol=1e-11, xtol=1e-11, gtol=1e-11)
        q = float(r.fun @ r.fun)
        if best is None or q < best['objective']:
            a = unpack(r.x)
            best = dict(objective=q, Lambda=a.tolist(), optimality=float(r.optimality),
                        spectral_radius=float(max(abs(np.linalg.eigvals(a)))),
                        nfev=int(r.nfev), success=bool(r.success))
        if q < 1e-22:
            break
    return best


def collect():
    trials, summaries = [], []
    for label in NAMES:
        for size in (100, 1000, 10000, 1000000):
            group = []
            sample_dir = ROOT / 'experiments/output' / NAMES[label] / f'num_samples_{size}'
            for directory in sorted(sample_dir.glob('seed_*')):
                seed = int(directory.name[5:])
                d, p = load(label, size, seed)
                reference, _ = load('all_forward', size, seed)
                assert np.allclose(d['Sigma'], reference['Sigma'], atol=1e-12, rtol=1e-12)
                assert np.array_equal(d['Lambda_star'], reference['Lambda_star'])
                assert abs(float(d['omega_ref']) - float(reference['omega_ref'])) < 1e-12
                path = [edges_for(m) for m in d['selected_support_masks']]
                selection = json.loads((p / 'selection_plateau_bootstrap.json').read_text())
                final = frozenset(map(tuple, selection['selected_edges']))
                assert final == path[int(selection['selected_dimension']) - 1]
                fp = np.array([len(e - TRUTH) for e in path])
                tp = np.array([len(e & TRUTH) for e in path])
                nested = all(a <= b for a,b in zip(path, path[1:]))
                denom = 3 if label == 'upper_forward' else 9
                auc = float(np.trapz(tp/3, fp/denom)) if nested else None
                state = json.loads(str(d['search_state_json'])) if 'search_state_json' in d else {}
                row = dict(method=label, num_samples=size, seed=seed,
                           tpr_k3=float(tp[3]/3), fp_k3=int(fp[3]),
                           true_on_path=TRUTH in path, exact_final=final == TRUTH,
                           final_tpr=len(final & TRUTH)/3, final_fpr=len(final-TRUTH)/denom,
                           final_edges=len(final), auc=auc, nested=nested,
                           backtracks=int(np.count_nonzero(np.diff(fp)<0) + np.count_nonzero(np.diff(tp)<0)),
                           corrections=state.get('corrections'),
                           first_edge=[list(e) for e in sorted(path[1])],
                           k3_edges=[list(e) for e in sorted(path[3])],
                           q3=float(d['objective_values'][3]), omega_ref=float(d['omega_ref']))
                trials.append(row)
                group.append(row)
            avg = lambda k: float(np.mean([r[k] for r in group]))
            summary = dict(method=label, num_samples=size, n=len(group),
                           tpr_k3=avg('tpr_k3'), fp_k3=avg('fp_k3'),
                           true_on_path=sum(r['true_on_path'] for r in group),
                           exact_final=sum(r['exact_final'] for r in group),
                           final_tpr=avg('final_tpr'), final_fpr=avg('final_fpr'),
                           final_edges=avg('final_edges'),
                           auc=avg('auc') if all(r['auc'] is not None for r in group) else None,
                           backtracks=sum(r['backtracks'] for r in group),
                           corrections=sum(r['corrections'] or 0 for r in group))
            summaries.append(summary)
    (OUT/'saved_trials.json').write_text(json.dumps(trials, indent=2))
    with (OUT/'saved_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for r in summaries:
        print(r, flush=True)


def focused():
    d, _ = load('all_forward', 1000000, 2)
    sigma_pop = d['Sigma_population']
    population_omega = .93 * np.linalg.eigvalsh(sigma_pop)[0]
    results = {}
    for case, sigma, omega in [('population_fitted_omega', sigma_pop, population_omega),
                               ('population_true_omega', sigma_pop, 1.),
                               ('seed2_fitted_omega', d['Sigma'], float(d['omega_ref']))]:
        fits = []
        supports = [TRUTH, frozenset(((0,3),(1,3),(3,2))),
                    frozenset(((2,3),)), frozenset(((3,2),))]
        for edges in supports:
            lm = fit(sigma, omega, edges)
            a, _ = admm_solve(sigma, mask_for(edges), omega_fixed=omega,
                             max_iter=800, tol=1e-7, max_restarts=10)
            r = sigma - a.T @ sigma @ a - omega*np.eye(4)
            sl = sigma @ a
            grad = (-4 * sl @ r)[mask_for(edges)]
            lm.update(edges=[list(e) for e in sorted(edges)],
                      admm_objective=float(np.sum(r*r)),
                      admm_gradient_inf=float(max(abs(grad))),
                      admm_Lambda=a.tolist())
            fits.append(lm)
            print(case, sorted(edges), 'LS',lm['objective'],'ADMM',lm['admm_objective'], flush=True)
        results[case] = dict(omega=omega, fits=fits)
    (OUT/'focused_fits.json').write_text(json.dumps(results, indent=2))


def population_search(sign_starts=False):
    d, _ = load('all_forward', 1000000, 2)
    sigma = d['Sigma_population']
    omega = .93 * np.linalg.eigvalsh(sigma)[0]
    results = []
    for k in (1, 2, 3):
        rows = []
        for edges in itertools.combinations(EDGES, k):
            r = fit(sigma, omega, edges, starts=10, sign_starts=sign_starts)
            r['edges'] = [list(e) for e in edges]
            rows.append(r)
        rows.sort(key=lambda r:r['objective'])
        results.append(dict(k=k, fits=rows))
        print('population k', k, 'best', rows[:3], flush=True)
        name = 'population_exhaustive_signs.json' if sign_starts else 'population_exhaustive.json'
        (OUT/name).write_text(json.dumps(results, indent=2))


def lookahead_path(sigma, omega, width=10, depth=3):
    """Keep several prefixes, choose a depth-3 endpoint, then rebuild its prefix.

    The depth is a diagnostic tuning parameter, not inferred true sparsity.
    All 12 directed positions remain eligible at every step.
    """
    cache = {}
    def get(edges, polish=False):
        edges = frozenset(edges)
        if edges not in cache:
            cache[edges] = fit(sigma, omega, edges, starts=3)
        if polish and not cache[edges].get('polished'):
            initial = np.asarray(cache[edges]['Lambda'])
            r = fit(sigma, omega, edges, starts=10, extra=(initial,))
            r['polished'] = True
            cache[edges] = r
        return cache[edges]
    key = lambda e: (get(e)['objective'], sorted(e))
    beam = [frozenset()]
    for k in range(1, depth+1):
        candidates = {e | {edge} for e in beam for edge in EDGES if edge not in e}
        ordered = sorted(candidates, key=key)
        for e in ordered[:max(width, 20)]:
            get(e, polish=True)
        beam = sorted(candidates, key=key)[:width]
    endpoint = beam[0]
    prefix = [endpoint]
    while prefix[-1]:
        e = prefix[-1]
        candidates = [e - {edge} for edge in e]
        for candidate in candidates:
            get(candidate, polish=True)
        prefix.append(min(candidates, key=key))
    path = prefix[::-1]
    while len(path) <= len(EDGES):
        candidates = [path[-1] | {edge} for edge in EDGES if edge not in path[-1]]
        for candidate in candidates:
            get(candidate, polish=True)
        path.append(min(candidates, key=key))
    # Fit accuracy alone, with the original greedy-forward search policy.
    forward = [frozenset()]
    while len(forward) <= len(EDGES):
        candidates = [forward[-1] | {edge} for edge in EDGES if edge not in forward[-1]]
        for candidate in candidates:
            get(candidate, polish=True)
        forward.append(min(candidates, key=key))
    return path, forward, cache


def path_metrics(path):
    fp = np.asarray([len(e-TRUTH) for e in path])
    tp = np.asarray([len(e&TRUTH) for e in path])
    return dict(auc=float(np.trapz(tp/3,fp/9)), tpr_k3=float(tp[3]/3),
                true_on_path=TRUTH in path,
                edges=[[list(e) for e in sorted(s)] for s in path])


def lookahead_experiment():
    rows = []
    d, _ = load('all_forward', 1000000, 2)
    cases = [('population', None, d['Sigma_population'], .93*np.linalg.eigvalsh(d['Sigma_population'])[0])]
    for size in (1000000, 10000, 1000, 100):
        sample_dir = ROOT/'experiments/output'/NAMES['all_forward']/f'num_samples_{size}'
        for p in sorted(sample_dir.glob('seed_*')):
            seed = int(p.name[5:])
            d, _ = load('all_forward', size, seed)
            cases.append((size, seed, d['Sigma'], float(d['omega_ref'])))
    for size, seed, sigma, omega in cases:
        t = time.monotonic()
        path, forward, cache = lookahead_path(sigma, omega)
        row = dict(num_samples=size, seed=seed, lookahead=path_metrics(path),
                   polished_forward=path_metrics(forward),
                   q_k3=cache[path[3]]['objective'],
                   candidate_fits=len(cache), elapsed_seconds=time.monotonic()-t,
                   maximum_path_spectral_radius=max(cache[e]['spectral_radius'] for e in path))
        rows.append(row)
        (OUT/'lookahead_results.json').write_text(json.dumps(rows, indent=2))
        print(size, seed, 'lookahead',row['lookahead']['auc'],row['lookahead']['true_on_path'],
              'polished forward',row['polished_forward']['auc'], 'seconds',row['elapsed_seconds'],flush=True)


def true_support_comparisons():
    rows = []
    lookahead = json.loads((OUT/'lookahead_results.json').read_text())
    for row in lookahead:
        if row['num_samples'] == 'population':
            continue
        size, seed = row['num_samples'], row['seed']
        d, _ = load('all_forward', size, seed)
        sigma, omega = d['Sigma'], float(d['omega_ref'])
        true_fit = fit(sigma, omega, TRUTH)
        exhaustive, _ = load('all_exhaustive', size, seed)
        edges = edges_for(exhaustive['selected_support_masks'][3])
        a, _ = admm_solve(sigma, mask_for(edges), omega_fixed=omega,
                         max_iter=800, tol=1e-7, max_restarts=10)
        previous_fit = fit(sigma, omega, edges, extra=(a,), sign_starts=True)
        row = dict(num_samples=size, seed=seed, true_q=true_fit['objective'],
                   lookahead_q=row['q_k3'], old_exhaustive_q=float(exhaustive['objective_values'][3]),
                   polished_old_exhaustive_q=previous_fit['objective'],
                   true_better_than_lookahead=true_fit['objective']<row['q_k3']-1e-10)
        rows.append(row)
    (OUT/'true_support_comparisons.json').write_text(json.dumps(rows, indent=2))
    for size in (100,1000,10000,1000000):
        group=[r for r in rows if r['num_samples']==size]
        print(size,'true beats lookahead',sum(r['true_better_than_lookahead'] for r in group),
              'mean true q',np.mean([r['true_q'] for r in group]),flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'collect'
    {'collect': collect, 'focused': focused, 'population': population_search,
     'population_signs':lambda:population_search(sign_starts=True),
     'lookahead': lookahead_experiment, 'compare_truth':true_support_comparisons}[mode]()
