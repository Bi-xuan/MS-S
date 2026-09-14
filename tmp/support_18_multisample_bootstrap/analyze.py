"""Nested recovery and top-three sequential plateau bootstrap at three N values."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import wishart, beta as beta_distribution

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from batched_solver import fit_batch
from experiments.compute_objective_curve import covariance_from_lambda_star
from experiments.select_scaling_parameter import floor_objective_values
from objective import frobenius_objective
from scaling_selection import build_dimension_path

HERE = Path(__file__).resolve().parent
LABELS = ("1e6", "100", "1000")
B, ALPHA, MASTER_SEED = 199, .05, 20260913
DIMS = np.arange(1, 8)


def edges(mask):
    return (np.argwhere(np.triu(mask, 1)) + 1).tolist()


def mask_from_edges(values):
    mask = np.eye(4, dtype=bool)
    for i, j in values:
        mask[i - 1, j - 1] = True
    return mask


def reconstruct(source):
    with np.load(source) as data:
        a = {k: data[k].copy() for k in data.files}
    n_samples, seed = int(a["num_samples"]), int(a["random_seed"])
    sigma = a["Sigma"]
    true_mask = a["Lambda_star"] != 0
    assert edges(true_mask) == [[1, 4], [2, 4], [3, 4]]
    assert int(a["n"]) == 4 and float(a["omega_ref"]) == 1.
    assert str(a["support_scope"]) == "upper"
    assert np.array_equal(a["d_m_values"], DIMS)
    mask = np.eye(4, dtype=bool)
    steps = []
    for dim in DIMS:
        masks = [mask.copy()] if dim == 1 else []
        if dim > 1:
            for i, j in np.argwhere(np.triu(np.ones((4, 4), dtype=bool), 1) & ~mask):
                m = mask.copy()
                m[i, j] = True
                masks.append(m)
        assert len(masks) == (1 if dim == 1 else 8 - dim)
        lambdas, objectives = fit_batch(np.repeat(sigma[None], len(masks), axis=0), np.array(masks))
        best = int(np.argmin(objectives))
        previous, mask = mask, masks[best]
        assert np.all(~previous | mask) and len(edges(mask)) == dim - 1
        lam, objective = lambdas[best], float(objectives[best])
        np.testing.assert_allclose(objective, frobenius_objective(sigma, lam, 1.), rtol=1e-12, atol=1e-15)
        steps.append(dict(dimension=int(dim), edges=edges(mask), Lambda=lam.tolist(), objective=objective,
                          candidate_objectives=objectives.tolist(), candidate_edges=[edges(m) for m in masks],
                          exact_true_support=bool(np.array_equal(mask, true_mask))))
    raw = np.array([s["objective"] for s in steps])
    floored, _ = floor_objective_values(raw, 1e-8)
    path = build_dimension_path(DIMS, floored, np.sqrt(DIMS))
    ranked = sorted([dict(dimension=int(path.dimensions[i]),
                          log_width=float(np.log(path.breakpoints[i + 1]) - np.log(path.breakpoints[i])))
                     for i in range(1, len(path.dimensions) - 1)], key=lambda r: (-r["log_width"], r["dimension"]))
    shortlist = sorted([r["dimension"] for r in ranked[:3]], reverse=True)
    return dict(key=f"{n_samples}:{seed}", source=str(source.relative_to(ROOT)),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), num_samples=n_samples, seed=seed,
                Sigma=sigma.tolist(), true_edges=edges(true_mask), steps=steps,
                ranked_plateaus=ranked, candidates_descending=shortlist,
                path_passes_true_support=any(s["exact_true_support"] for s in steps),
                objective_increases=int(np.count_nonzero(np.diff(raw) > 1e-8)),
                saved_first_two_max_objective_difference=float(np.max(np.abs(raw[:2] - a["objective_values"][:2]))))


def make_pair(trial, stage):
    large_dim, small_dim = trial["candidates_descending"][stage - 1:stage + 1]
    small, large = trial["steps"][small_dim - 1], trial["steps"][large_dim - 1]
    assert np.all(~mask_from_edges(small["edges"]) | mask_from_edges(large["edges"]))
    lam = np.array(small["Lambda"])
    rho = float(np.max(np.abs(np.linalg.eigvals(lam))))
    if rho >= 1:
        raise ValueError(f"Unstable null model for {trial['key']} stage {stage}: {rho}")
    sigma_null = covariance_from_lambda_star(lam, 1.)
    assert np.linalg.eigvalsh(sigma_null)[0] > 0
    assert frobenius_objective(sigma_null, lam, 1.) < 1e-23
    np.testing.assert_allclose(wishart.mean(df=trial["num_samples"], scale=sigma_null / trial["num_samples"]), sigma_null, rtol=1e-14)
    gain = small["objective"] - large["objective"]
    assert gain > 0
    return dict(key=f"{trial['key']}:{small_dim}:{large_dim}", trial_key=trial["key"], stage=stage,
                num_samples=trial["num_samples"], seed=trial["seed"], small_dimension=small_dim, large_dimension=large_dim,
                small_edges=small["edges"], large_edges=large["edges"], observed_gain=gain,
                null_covariance=sigma_null.tolist(), null_Lambda=lam.tolist(), null_spectral_radius=rho)


def bootstrap_batch(candidate, indices):
    n_samples = candidate["num_samples"]
    population = np.array(candidate["null_covariance"])
    sigmas = []
    for index in indices:
        rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, n_samples, candidate["seed"], index]))
        # Exactly the law of X.T @ X / N for N independent N(0,population) rows.
        sigmas.append(wishart.rvs(df=n_samples, scale=population, random_state=rng) / n_samples)
    sigmas = np.array(sigmas)
    masks = np.array([mask_from_edges(candidate["small_edges"]), mask_from_edges(candidate["large_edges"])])
    lambdas, q = fit_batch(np.repeat(sigmas, 2, axis=0), np.tile(masks, (len(indices), 1, 1)))
    lambdas, q = lambdas.reshape(-1, 2, 4, 4), q.reshape(-1, 2)
    records = []
    for row, index in enumerate(indices):
        for j in range(2):
            assert np.all(lambdas[row, j][~masks[j]] == 0)
            np.testing.assert_allclose(q[row, j], frobenius_objective(sigmas[row], lambdas[row, j], 1.), rtol=1e-12, atol=1e-15)
        raw_gain = float(q[row, 0] - q[row, 1])
        gain = max(raw_gain, 0.)  # smaller fit is feasible on the larger support
        assert (gain >= candidate["observed_gain"]) == (raw_gain >= candidate["observed_gain"])
        records.append(dict(pair_key=candidate["key"], replicate=index, Sigma=sigmas[row].tolist(),
                            Lambda_small=lambdas[row, 0].tolist(), Lambda_large=lambdas[row, 1].tolist(),
                            objective_small=float(q[row, 0]), objective_large=float(q[row, 1]),
                            raw_gain=raw_gain, gain=gain,
                            min_sample_eigenvalue=float(np.linalg.eigvalsh(sigmas[row])[0])))
    return records


def summarize(candidate, records):
    assert sorted(r["replicate"] for r in records) == list(range(B))
    count = sum(r["gain"] >= candidate["observed_gain"] for r in records)
    p = (1 + count) / (B + 1)
    lower = float(beta_distribution.ppf(.025, count, B - count + 1)) if count else 0.
    upper = float(beta_distribution.ppf(.975, count + 1, B - count)) if count < B else 1.
    return dict(**candidate, exceedances=count, B=B, p_value=p, retain_large=p < ALPHA,
                selected_dimension=candidate["large_dimension"] if p < ALPHA else candidate["small_dimension"],
                mc_interval=[lower, upper], mc_interval_crosses_alpha=lower <= ALPHA <= upper,
                feasible_fallback_count=sum(r["raw_gain"] < 0 for r in records),
                omega_cap_rejection_count=sum(r["min_sample_eigenvalue"] < 1.000001 for r in records))


def main():
    roots = [ROOT / f"experiments/output/upp_scal_n4_dm4_nsm{label}_omega1_minabs02_10seeds_PlaAbs/support_18" for label in LABELS]
    sources = [p for root in roots for p in sorted(root.glob("seed_*/objective_curve_sigma_hat.npz"), key=lambda p: int(p.parent.name[5:]))]
    assert len(sources) == 30
    protected_files = [p for root in roots for p in root.rglob("*") if p.is_file()]
    protected_files += [p for p in ROOT.rglob("*.py") if not any(x in {"tmp", ".venv", ".git"} for x in p.relative_to(ROOT).parts)]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_files}
    config = dict(sample_sizes=[1000000, 100, 1000], Lm=1, top_plateaus=3, B=B, alpha_per_comparison=ALPHA,
                  objective_floor=1e-8, bootstrap_screening=False, multiplicity_adjustment="none", master_seed=MASTER_SEED,
                  bootstrap_sampler="Wishart(df=N,scale=fitted_null_covariance)/N; exact Gaussian X.T X/N distribution",
                  settings=dict(beta=1, max_iter=800, tol=1e-7, zero_tol=1e-5, max_restarts=10, omega_fixed=1, omega_upper=None, init_strategy="halton"),
                  sources=[str(p.relative_to(ROOT)) for p in sources], protected_sha256=hashes)
    config_file = HERE / "config.json"
    if config_file.exists():
        assert json.loads(config_file.read_text()) == config
    else:
        config_file.write_text(json.dumps(config, indent=2) + "\n")
    path_file = HERE / "nested_paths.json"
    if path_file.exists():
        trials = json.loads(path_file.read_text())
        assert all(hashlib.sha256((ROOT / t["source"]).read_bytes()).hexdigest() == t["source_sha256"] for t in trials)
    else:
        trials = []
        with ProcessPoolExecutor(max_workers=6) as pool:
            for future in as_completed([pool.submit(reconstruct, source) for source in sources]):
                trial = future.result()
                trials.append(trial)
                print(f"Nested {trial['key']}: true_support={trial['path_passes_true_support']}; candidates={trial['candidates_descending']}", flush=True)
        trials.sort(key=lambda t: ([1000000, 100, 1000].index(t["num_samples"]), t["seed"]))
        path_file.write_text(json.dumps(trials, indent=2) + "\n")
    cache_file = HERE / "bootstrap.jsonl"
    cache = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            r = json.loads(line)
            cache[(r["pair_key"], r["replicate"])] = r
    tests, selected, statuses = {}, {}, {}
    for trial in trials:
        dims = trial["candidates_descending"]
        if not dims:
            statuses[trial["key"]] = "No bounded plateaus"
        elif len(dims) == 1:
            selected[trial["key"]] = dims[0]
    for stage in (1, 2):
        active = [make_pair(t, stage) for t in trials if t["key"] not in selected and t["key"] not in statuses and len(t["candidates_descending"]) > stage]
        tasks = []
        for c in active:
            pending = [i for i in range(B) if (c["key"], i) not in cache]
            tasks.extend((c, pending[i:i + 20]) for i in range(0, len(pending), 20))
        tasks.sort(key=lambda x: (x[1][0], x[0]["num_samples"], x[0]["seed"]))
        total = sum(len(indices) for _, indices in tasks)
        completed, started = 0, time.perf_counter()
        print(f"Stage {stage}: {len(active)} comparisons; {total} new bootstrap replicates", flush=True)
        with cache_file.open("a") as stream, ProcessPoolExecutor(max_workers=6) as pool:
            for future in as_completed([pool.submit(bootstrap_batch, c, indices) for c, indices in tasks]):
                records = future.result()
                for r in records:
                    cache[(r["pair_key"], r["replicate"])] = r
                    stream.write(json.dumps(r) + "\n")
                stream.flush()
                completed += len(records)
                if completed % 200 == 0 or completed == total:
                    print(f"Stage {stage}: {completed}/{total} complete; elapsed={time.perf_counter()-started:.1f}s", flush=True)
        for c in active:
            test = summarize(c, [cache[(c["key"], i)] for i in range(B)])
            tests[c["key"]] = test
            trial = next(t for t in trials if t["key"] == c["trial_key"])
            if test["retain_large"] or stage == len(trial["candidates_descending"]) - 1:
                selected[c["trial_key"]] = test["selected_dimension"]
            print(f"Test {c['key']}: p={test['p_value']:.3f}; {'final' if c['trial_key'] in selected else 'continue'}", flush=True)
    assert len(selected) + len(statuses) == 30
    rows = []
    for trial in trials:
        ts = sorted([t for t in tests.values() if t["trial_key"] == trial["key"]], key=lambda t: t["stage"])
        dim = selected.get(trial["key"])
        found = set(map(tuple, trial["steps"][dim - 1]["edges"])) if dim is not None else set()
        true = set(map(tuple, trial["true_edges"]))
        tp, count = len(found & true), len(found)
        assert dim is None or count == dim - 1
        rows.append(dict(num_samples=trial["num_samples"], seed=trial["seed"],
                         candidates_descending=trial["candidates_descending"],
                         first_p=ts[0]["p_value"] if ts else None, second_p=ts[1]["p_value"] if len(ts) > 1 else None,
                         selected_dimension=dim, true_positive_edges=tp, selected_edge_count=count,
                         precision=tp/count if count else None, exact_support_recovery=found == true,
                         path_passes_true_support=trial["path_passes_true_support"], status=statuses.get(trial["key"], "ok")))
    summary = []
    for n in (1000000, 100, 1000):
        group = [r for r in rows if r["num_samples"] == n]
        defined = [r["precision"] for r in group if r["precision"] is not None]
        summary.append(dict(num_samples=n, seeds=len(group), successful_selections=sum(r["status"] == "ok" for r in group),
                            correct_dimension_count=sum(r["selected_dimension"] == 4 for r in group),
                            exact_support_count=sum(r["exact_support_recovery"] for r in group),
                            path_passes_true_support_count=sum(r["path_passes_true_support"] for r in group),
                            mean_precision=float(np.mean(defined)) if defined else None))
    assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_files}
    (HERE / "results.json").write_text(json.dumps(dict(config=config, rows=rows, tests=tests, summary=summary,
                                                       solver_validation=json.loads((HERE / "solver_validation.json").read_text())), indent=2) + "\n")
    with (HERE / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(rows=rows, summary=summary), indent=2), flush=True)
    print("All original experiment files and production Python files unchanged.", flush=True)


if __name__ == "__main__":
    main()
