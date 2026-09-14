"""Fixed-support parametric bootstrap for the two longest nested-path plateaus.

This standalone experiment imports, but does not edit, production functions.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import beta as beta_distribution

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.compute_objective_curve import covariance_from_lambda_star
from experiments.select_scaling_parameter import floor_objective_values
from objective import frobenius_objective
from optimizers.support_search import solve_support_with_restarts
from scaling_selection import build_dimension_path

HERE = Path(__file__).resolve().parent
NESTED = ROOT / "tmp/support_18_greedy_nested/results.json"
SETTINGS = dict(beta=1., max_iter=800, tol=1e-7, zero_tol=1e-5,
                max_restarts=10, min_omega=0., omega_fixed=1.,
                omega_upper=None, init_strategy="halton")
BOOTSTRAP_SEED = 20260913


def mask_from_edges(edges):
    mask = np.eye(4, dtype=bool)
    for i, j in edges:
        mask[i - 1, j - 1] = True
    return mask


def screen(trial):
    raw = np.array([s["objective"] for s in trial["steps"]])
    values, _ = floor_objective_values(raw, 1e-8)
    # With Lm=1, the theorem penalty is C*sqrt(D), C>0. Removing C
    # translates log scale, preserving every plateau width and its ranking.
    path = build_dimension_path(np.arange(1, 8), values, np.sqrt(np.arange(1, 8)))
    plateaus = [dict(dimension=int(path.dimensions[i]),
                     log_width=float(np.log(path.breakpoints[i + 1]) - np.log(path.breakpoints[i])))
                for i in range(1, len(path.dimensions) - 1)]
    ranked = sorted(plateaus, key=lambda p: (-p["log_width"], p["dimension"]))
    assert len(ranked) >= 2
    small_dim, large_dim = sorted(p["dimension"] for p in ranked[:2])
    small = trial["steps"][small_dim - 1]
    large = trial["steps"][large_dim - 1]
    small_mask, large_mask = map(mask_from_edges, (small["edges"], large["edges"]))
    assert np.all(~small_mask | large_mask) and small_dim < large_dim
    lam = np.array(small["Lambda"])
    rho = float(np.max(np.abs(np.linalg.eigvals(lam))))
    assert rho < 1 and small["omega"] == 1.
    sigma_null = covariance_from_lambda_star(lam, 1.)
    assert np.linalg.eigvalsh(sigma_null)[0] > 0
    assert frobenius_objective(sigma_null, lam, 1.) < 1e-24
    with np.load(ROOT / trial["source"]) as source:
        assert int(source["num_samples"]) == 10000 and float(source["omega_ref"]) == 1.
        for step in (small, large):
            np.testing.assert_allclose(step["objective"], frobenius_objective(source["Sigma"], np.array(step["Lambda"]), 1.), rtol=1e-12, atol=1e-15)
    return dict(seed=trial["seed"], ranked_plateaus=ranked,
                small_dimension=small_dim, large_dimension=large_dim,
                small_edges=small["edges"], large_edges=large["edges"],
                small_Lambda=small["Lambda"], large_Lambda=large["Lambda"],
                small_objective=small["objective"], large_objective=large["objective"],
                observed_gain=small["objective"] - large["objective"],
                null_covariance=sigma_null.tolist(), null_spectral_radius=rho,
                true_edges=trial["true_edges"])


def bootstrap_one(candidate, replicate):
    started = time.perf_counter()
    seed = candidate["seed"]
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, seed, replicate]))
    x = rng.multivariate_normal(np.zeros(4), np.array(candidate["null_covariance"]), size=10000)
    sigma = x.T @ x / 10000
    fits = []
    for edges in (candidate["small_edges"], candidate["large_edges"]):
        mask = mask_from_edges(edges)
        fit = solve_support_with_restarts(sigma, mask, **SETTINGS)
        if fit is None:
            raise RuntimeError(f"No finite fit for seed={seed}, replicate={replicate}")
        lam, omega, objective = fit
        assert omega == 1. and np.all(lam[~mask] == 0)
        np.testing.assert_allclose(objective, frobenius_objective(sigma, lam, omega), rtol=1e-12, atol=1e-15)
        fits.append(dict(objective=float(objective), Lambda=lam.tolist()))
    return dict(seed=seed, replicate=replicate, Sigma=sigma.tolist(),
                min_sample_eigenvalue=float(np.linalg.eigvalsh(sigma)[0]),
                small_fit=fits[0], large_fit=fits[1],
                gain=fits[0]["objective"] - fits[1]["objective"],
                elapsed_seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=199)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=HERE / "run_b199")
    args = parser.parse_args()
    assert args.replicates > 0 and 0 < args.alpha < 1
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    nested_bytes = NESTED.read_bytes()
    nested = json.loads(nested_bytes)
    protected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in nested["protected_file_sha256"]}
    candidates = [screen(trial) for trial in nested["results"]]
    assert len(candidates) == 10
    config = dict(replicates=args.replicates, alpha=args.alpha, bootstrap_seed=BOOTSTRAP_SEED,
                  sample_size=10000, Lm=1, objective_floor=1e-8, settings=SETTINGS,
                  statistic="Raw objective on original smaller support minus raw objective on original larger support; both refitted on each bootstrap sample.",
                  reconstruction_in_bootstrap=False, nested_input_sha256=hashlib.sha256(nested_bytes).hexdigest(),
                  candidates=candidates)
    config_path = out / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "Resume configuration differs"
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    records_path = out / "replicates.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    done = {(r["seed"], r["replicate"]) for r in records}
    assert len(done) == len(records)
    tasks = [(c, b) for b in range(args.replicates) for c in candidates if (c["seed"], b) not in done]
    started = time.perf_counter()
    print(f"Starting {len(tasks)} remaining bootstrap replicates; {len(records)} cached; {args.workers} workers", flush=True)
    with records_path.open("a") as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(bootstrap_one, c, b) for c, b in tasks]
        for future in as_completed(futures):
            record = future.result()
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records.append(record)
            if len(records) % 50 == 0 or len(records) == 10 * args.replicates:
                print(f"Completed {len(records)}/{10 * args.replicates}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    rows = []
    for c in candidates:
        draws = sorted([r for r in records if r["seed"] == c["seed"]], key=lambda r: r["replicate"])
        assert [r["replicate"] for r in draws] == list(range(args.replicates))
        gains = np.array([r["gain"] for r in draws])
        exceedances = int(np.count_nonzero(gains >= c["observed_gain"]))
        p = (1 + exceedances) / (args.replicates + 1)
        # Strict convention: retain the larger model only when p < alpha.
        use_large = p < args.alpha
        selected_dim = c["large_dimension"] if use_large else c["small_dimension"]
        edges = c["large_edges"] if use_large else c["small_edges"]
        truth, selected = set(map(tuple, c["true_edges"])), set(map(tuple, edges))
        tp, count = len(truth & selected), len(selected)
        assert count == selected_dim - 1 and len(truth) == 3
        # Exact binomial interval quantifies Monte Carlo uncertainty in the
        # underlying bootstrap exceedance probability, not a model-validity guarantee.
        lower = float(beta_distribution.ppf(.025, exceedances, args.replicates - exceedances + 1)) if exceedances else 0.
        upper = float(beta_distribution.ppf(.975, exceedances + 1, args.replicates - exceedances)) if exceedances < args.replicates else 1.
        row = dict(seed=c["seed"], small_dimension=c["small_dimension"], large_dimension=c["large_dimension"],
                   observed_gain=c["observed_gain"], bootstrap_exceedances=exceedances,
                   bootstrap_replicates=args.replicates, p_value=p, alpha=args.alpha,
                   selected_dimension=selected_dim, true_positive_edges=tp, selected_edges=count,
                   precision=tp / count, exact_support_recovery=selected == truth,
                   mc_interval_lower=lower, mc_interval_upper=upper,
                   bootstrap_gain_q95=float(np.quantile(gains, .95)),
                   negative_bootstrap_gains=int(np.count_nonzero(gains < -1e-8)),
                   draws_below_original_omega_cap=sum(r["min_sample_eigenvalue"] < 1. + 1e-6 for r in draws))
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert NESTED.read_bytes() == nested_bytes
    assert protected == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in protected}
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(config=config, rows=rows, mean_precision=float(np.mean([r["precision"] for r in rows])),
                  correct_dimension_count=sum(r["selected_dimension"] == 4 for r in rows),
                  exact_support_recovery_count=sum(r["exact_support_recovery"] for r in rows),
                  protected_file_sha256=protected)
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Mean precision={report['mean_precision']:.6f}; correct dimension={report['correct_dimension_count']}/10; exact support={report['exact_support_recovery_count']}/10; protected files unchanged", flush=True)


if __name__ == "__main__":
    main()
