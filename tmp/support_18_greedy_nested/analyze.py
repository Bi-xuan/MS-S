"""Reconstruct a forward-nested support path without changing production code.

Run from the repository root:
    .venv/bin/python tmp/support_18_greedy_nested/analyze.py
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from objective import frobenius_objective
from optimizers.support_search import solve_support_with_restarts

SOURCE = ROOT / "experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18"
OUT = Path(__file__).resolve().parent
SETTINGS = dict(beta=1., max_iter=800, tol=1e-7, zero_tol=1e-5,
                max_restarts=10, min_omega=0., init_strategy="halton")


def edges(mask):
    return (np.argwhere(np.triu(mask, 1)) + 1).tolist()


def run_seed(source):
    with np.load(source) as data:
        a = {k: data[k].copy() for k in data.files}
    seed = int(a["random_seed"])
    sigma = a["Sigma"]
    true = a["Lambda_star"] != 0
    n = int(a["n"])
    assert n == 4 and int(a["num_samples"]) == 10000
    assert str(a["support_scope"]) == "upper"
    assert edges(true) == [[1, 4], [2, 4], [3, 4]]
    assert np.array_equal(a["d_m_values"], np.arange(1, 8))
    settings = dict(SETTINGS, omega_fixed=float(a["omega_ref"]),
                    omega_upper=float(np.linalg.eigvalsh(sigma)[0] - 1e-6))
    np.random.seed(int(a["solve_seed"]))
    mask = np.eye(n, dtype=bool)
    steps = []
    for dim in range(1, 8):
        candidates = [mask.copy()] if dim == 1 else []
        if dim > 1:
            for i, j in np.argwhere(np.triu(np.ones((n, n), dtype=bool), 1) & ~mask):
                candidate = mask.copy()
                candidate[i, j] = True
                candidates.append(candidate)
        assert len(candidates) == (1 if dim == 1 else 8 - dim)
        fits = []
        for candidate in candidates:
            result = solve_support_with_restarts(sigma, candidate, **settings)
            if result is None:
                raise RuntimeError(f"seed {seed}, D={dim}: no finite fit")
            lam, omega, objective = result
            assert np.all(lam[~candidate] == 0)
            assert np.isclose(objective, frobenius_objective(sigma, lam, omega), rtol=1e-12, atol=1e-15)
            fits.append(dict(edges=edges(candidate), objective=float(objective),
                             Lambda=lam.tolist(), omega=float(omega)))
        # Strict numerical minimum; exact ties retain row-major candidate order.
        winner = min(range(len(fits)), key=lambda i: fits[i]["objective"])
        previous = mask
        mask = candidates[winner]
        assert np.all(~previous | mask)
        assert np.count_nonzero(np.triu(mask, 1)) == dim - 1
        ordered = sorted(fit["objective"] for fit in fits)
        steps.append(dict(dimension=dim, **fits[winner], candidates=fits,
                          runner_up_gap=ordered[1] - ordered[0] if len(ordered) > 1 else None,
                          exact_true_support=bool(np.array_equal(mask, true)),
                          contains_true_support=bool(np.all(~true | mask))))
    assert np.all(np.diff([s["objective"] for s in steps]) <= 1e-8)
    # The first two dimensions have the same candidate sets as the saved exhaustive search.
    assert np.allclose([s["objective"] for s in steps[:2]], a["objective_values"][:2], rtol=1e-6, atol=1e-8)
    assert steps[1]["edges"] == edges(a["selected_support_masks"][1])
    assert steps[-1]["contains_true_support"]
    hit = [s["dimension"] for s in steps if s["exact_true_support"]]
    return dict(seed=seed, source=str(source.relative_to(ROOT)), solver_settings=settings,
                true_edges=edges(true), passes_true_support=bool(hit), exact_hit_dimensions=hit,
                first_contains_true_dimension=next(s["dimension"] for s in steps if s["contains_true_support"]),
                original_passes_true_support=bool(np.array_equal(a["selected_support_masks"][3], true)),
                original_d4_edges=edges(a["selected_support_masks"][3]), steps=steps)


def digest(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    protected = sorted(SOURCE.rglob("*"))
    protected = [p for p in protected if p.is_file()]
    protected += sorted(p for p in ROOT.rglob("*.py") if not any(part in {"tmp", ".venv", ".git"} for part in p.relative_to(ROOT).parts))
    before = digest(protected)
    results = []
    sources = sorted(SOURCE.glob("seed_*/objective_curve_sigma_hat.npz"))
    assert len(sources) == 10
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_seed, source) for source in sources]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Seed {result['seed']}: passes={result['passes_true_support']}; D4={result['steps'][3]['edges']}", flush=True)
            (OUT / f"seed_{result['seed']}.json").write_text(json.dumps(result, indent=2) + "\n")
    results.sort(key=lambda r: r["seed"])
    assert before == digest(protected), "Source experiment or production Python files changed"
    report = dict(strategy="Start diagonal-only at D=1; enumerate every one-edge upper-triangular extension of the preceding selected support, refit, and take the strict minimum objective.",
                  comparison="Exact support equality, rather than containment at a larger dimension.",
                  passed=sum(r["passes_true_support"] for r in results), total=len(results),
                  protected_file_sha256=before, results=results)
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "passes_true_support", "D4_edges", "D4_objective", "original_passes_true_support", "first_contains_true_dimension", "edge_addition_order"])
        for r in results:
            steps = r["steps"]
            additions = [next(e for e in curr["edges"] if e not in prev["edges"]) for prev, curr in zip(steps, steps[1:])]
            writer.writerow([r["seed"], r["passes_true_support"], json.dumps(steps[3]["edges"]), steps[3]["objective"], r["original_passes_true_support"], r["first_contains_true_dimension"], json.dumps(additions)])
    print(f"Exact true-support hits: {report['passed']}/{report['total']}. Protected inputs unchanged.", flush=True)


if __name__ == "__main__":
    main()
