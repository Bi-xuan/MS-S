"""Sequential fixed-support bootstrap over the three longest plateaus.

Import the existing two-candidate diagnostic and reuse identical saved tests.
Production code and previous experiment outputs are read-only.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import beta as beta_distribution

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "tmp/support_18_plateau_bootstrap/run_b199"
spec = importlib.util.spec_from_file_location("two_plateau_bootstrap", PREVIOUS.parent / "analyze.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

B = 199
ALPHA = .05


def pair_key(candidate):
    return f"{candidate['seed']}:{candidate['small_dimension']}:{candidate['large_dimension']}"


def make_pair(trial, small_dim, large_dim):
    small, large = trial["steps"][small_dim - 1], trial["steps"][large_dim - 1]
    assert np.all(~base.mask_from_edges(small["edges"]) | base.mask_from_edges(large["edges"]))
    lam = np.array(small["Lambda"])
    rho = float(max(abs(np.linalg.eigvals(lam))))
    assert rho < 1.
    covariance = base.covariance_from_lambda_star(lam, 1.)
    assert np.linalg.eigvalsh(covariance)[0] > 0
    assert base.frobenius_objective(covariance, lam, 1.) < 1e-24
    gain = small["objective"] - large["objective"]
    assert gain > 0
    return dict(seed=trial["seed"], small_dimension=small_dim, large_dimension=large_dim,
                small_edges=small["edges"], large_edges=large["edges"],
                small_Lambda=small["Lambda"], large_Lambda=large["Lambda"],
                small_objective=small["objective"], large_objective=large["objective"],
                observed_gain=gain, null_covariance=covariance.tolist(),
                null_spectral_radius=rho, true_edges=trial["true_edges"])


def run_one(candidate, replicate, cached_small):
    if cached_small is None:
        result = base.bootstrap_one(candidate, replicate)
        result["reuse"] = "none"
    else:
        # Same fitted null and SeedSequence -> exactly the same bootstrap data.
        # Reuse the smaller-support fit and only fit the new larger support.
        sigma = np.array(cached_small["Sigma"])
        mask = base.mask_from_edges(candidate["large_edges"])
        fit = base.solve_support_with_restarts(sigma, mask, **base.SETTINGS)
        if fit is None:
            raise RuntimeError(f"No fit for {pair_key(candidate)}, replicate {replicate}")
        lam, omega, objective = fit
        assert omega == 1. and np.all(lam[~mask] == 0)
        result = dict(seed=candidate["seed"], replicate=replicate,
                      Sigma=cached_small["Sigma"],
                      min_sample_eigenvalue=cached_small["min_sample_eigenvalue"],
                      small_fit=cached_small["small_fit"],
                      large_fit=dict(Lambda=lam.tolist(), objective=float(objective)),
                      gain=cached_small["small_fit"]["objective"] - float(objective),
                      reuse="same_null_smaller_fit")
    result["pair"] = pair_key(candidate)
    return result


def verify_and_summarize(candidate, draws):
    assert sorted(r["replicate"] for r in draws) == list(range(B))
    raw_gains = []
    for draw in draws:
        assert draw["pair"] == pair_key(candidate)
        sigma = np.array(draw["Sigma"])
        for name, edges in (("small_fit", candidate["small_edges"]), ("large_fit", candidate["large_edges"])):
            lam = np.array(draw[name]["Lambda"])
            assert np.all(lam[~base.mask_from_edges(edges)] == 0)
            np.testing.assert_allclose(base.frobenius_objective(sigma, lam, 1.), draw[name]["objective"], rtol=1e-12, atol=1e-15)
        raw_gain = draw["small_fit"]["objective"] - draw["large_fit"]["objective"]
        assert raw_gain == draw["gain"]
        raw_gains.append(raw_gain)
    raw_gains = np.array(raw_gains)
    # The smaller fitted matrix is feasible on the larger nested support.
    gains = np.maximum(raw_gains, 0.)
    exceedances = int(sum(gains >= candidate["observed_gain"]))
    assert exceedances == int(sum(raw_gains >= candidate["observed_gain"]))
    p_value = (1 + exceedances) / (B + 1)
    lo = float(beta_distribution.ppf(.025, exceedances, B - exceedances + 1)) if exceedances else 0.
    hi = float(beta_distribution.ppf(.975, exceedances + 1, B - exceedances)) if exceedances < B else 1.
    return dict(pair=pair_key(candidate), seed=candidate["seed"],
                small_dimension=candidate["small_dimension"], large_dimension=candidate["large_dimension"],
                observed_gain=candidate["observed_gain"], exceedances=exceedances, B=B,
                p_value=p_value, retain_large=p_value < ALPHA,
                selected_dimension=candidate["large_dimension"] if p_value < ALPHA else candidate["small_dimension"],
                mc_interval=[lo, hi], mc_interval_crosses_alpha=lo <= ALPHA <= hi,
                feasible_fallback_count=int(sum(raw_gains < 0)),
                original_omega_cap_rejection_count=sum(r["min_sample_eigenvalue"] < 1.000001 for r in draws))


def main():
    nested_bytes = base.NESTED.read_bytes()
    nested = json.loads(nested_bytes)
    previous_bytes = (PREVIOUS / "replicates.jsonl").read_bytes()
    previous_config_bytes = (PREVIOUS / "config.json").read_bytes()
    old_config = json.loads(previous_config_bytes)
    assert old_config["replicates"] == B and old_config["alpha"] == ALPHA
    assert old_config["settings"] == base.SETTINGS
    assert old_config["bootstrap_seed"] == base.BOOTSTRAP_SEED
    old_candidates = {c["seed"]: c for c in old_config["candidates"]}
    old_draws = {(r["seed"], r["replicate"]): r for r in map(json.loads, previous_bytes.splitlines())}
    trials = {r["seed"]: r for r in nested["results"]}
    shortlist = {}
    candidates = {}
    for seed, trial in trials.items():
        ranked = base.screen(trial)["ranked_plateaus"]
        assert ranked == old_candidates[seed]["ranked_plateaus"]
        assert len(ranked) >= 3
        dimensions = sorted([r["dimension"] for r in ranked[:3]], reverse=True)
        shortlist[seed] = dimensions
        for large, small in zip(dimensions, dimensions[1:]):
            candidate = make_pair(trial, small, large)
            candidates[pair_key(candidate)] = candidate
    protected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in nested["protected_file_sha256"]}
    config = dict(B=B, alpha_per_comparison=ALPHA, Lm=1, objective_floor=1e-8,
                  bootstrap_seed=base.BOOTSTRAP_SEED, settings=base.SETTINGS,
                  procedure="Test largest vs middle; if p<alpha select largest and stop, otherwise test middle vs smallest. At the second test select middle if p<alpha, else smallest.",
                  multiplicity_adjustment="none", bootstrap_screening=False,
                  shortlist=shortlist, candidates=candidates,
                  nested_sha256=hashlib.sha256(nested_bytes).hexdigest(),
                  previous_replicates_sha256=hashlib.sha256(previous_bytes).hexdigest())
    config_file = HERE / "config.json"
    serialized_config = json.dumps(config, indent=2) + "\n"
    if config_file.exists():
        assert config_file.read_text() == serialized_config
    else:
        config_file.write_text(serialized_config)
    cache_file = HERE / "replicates.jsonl"
    draws = [json.loads(line) for line in cache_file.read_text().splitlines()] if cache_file.exists() else []
    cache = {(r["pair"], r["replicate"]): r for r in draws}
    assert len(cache) == len(draws)
    tests = {}
    selected = {}
    reuse_counts = dict(whole_pair_draws=0, smaller_fit_draws=0, newly_generated_draws=0)
    for stage in (1, 2):
        active = []
        for seed, dimensions in shortlist.items():
            if stage == 2 and seed in selected:
                continue
            large, small = dimensions[stage - 1:stage + 1]
            active.append(candidates[f"{seed}:{small}:{large}"])
        tasks = []
        with cache_file.open("a") as stream:
            for c in active:
                old = old_candidates[c["seed"]]
                identical_null = c["small_dimension"] == old["small_dimension"]
                if identical_null:
                    assert c["small_edges"] == old["small_edges"]
                    np.testing.assert_array_equal(c["small_Lambda"], old["small_Lambda"])
                    np.testing.assert_array_equal(c["null_covariance"], old["null_covariance"])
                identical_pair = identical_null and c["large_dimension"] == old["large_dimension"]
                for b in range(B):
                    key = (pair_key(c), b)
                    if key in cache:
                        continue
                    old_record = old_draws[(c["seed"], b)] if identical_null else None
                    if identical_pair:
                        assert c["large_edges"] == old["large_edges"]
                        record = dict(old_record, pair=pair_key(c), reuse="whole_pair")
                        stream.write(json.dumps(record) + "\n")
                        cache[key] = record
                    else:
                        tasks.append((c, b, old_record))
            stream.flush()
            # Interleave seeds for representative progress throughout each stage.
            tasks.sort(key=lambda item: (item[1], item[0]["seed"]))
            print(f"Stage {stage}: {len(active)} comparisons, {len(tasks)} new replicate computations", flush=True)
            started = time.perf_counter()
            with ProcessPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(run_one, *task) for task in tasks]
                for i, future in enumerate(as_completed(futures), 1):
                    record = future.result()
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    cache[(record["pair"], record["replicate"])] = record
                    if i % 50 == 0 or i == len(tasks):
                        print(f"Stage {stage}: {i}/{len(tasks)} new replicates complete; {time.perf_counter()-started:.1f}s", flush=True)
        for c in active:
            records = [cache[(pair_key(c), b)] for b in range(B)]
            result = verify_and_summarize(c, records)
            result["stage"] = stage
            tests[pair_key(c)] = result
            if result["retain_large"] or stage == 2:
                selected[c["seed"]] = result["selected_dimension"]
            print(f"Seed {c['seed']}, stage {stage}, {c['large_dimension']} vs {c['small_dimension']}: p={result['p_value']:.3f}, {'final' if c['seed'] in selected else 'continue'}", flush=True)
    assert len(selected) == 10
    rows = []
    previous_results = {r["seed"]: r for r in json.loads((PREVIOUS / "results.json").read_text())["rows"]}
    for seed, trial in trials.items():
        dims = shortlist[seed]
        first = tests[f"{seed}:{dims[1]}:{dims[0]}"]
        second = tests.get(f"{seed}:{dims[2]}:{dims[1]}")
        dim = selected[seed]
        selected_edges = set(map(tuple, trial["steps"][dim - 1]["edges"]))
        true_edges = set(map(tuple, trial["true_edges"]))
        tp, count = len(selected_edges & true_edges), len(selected_edges)
        assert count == dim - 1 and len(true_edges) == 3
        rows.append(dict(seed=seed, candidates_descending=dims,
                         first_p_value=first["p_value"], second_p_value=second["p_value"] if second else None,
                         selected_dimension=dim, true_positive_edges=tp, selected_edge_count=count,
                         precision=tp / count, exact_support_recovery=selected_edges == true_edges,
                         prior_top2_dimension=previous_results[seed]["selected_dimension"],
                         prior_top2_precision=previous_results[seed]["precision"]))
    for record in cache.values():
        reuse_counts[{"whole_pair": "whole_pair_draws", "same_null_smaller_fit": "smaller_fit_draws", "none": "newly_generated_draws"}[record["reuse"]]] += 1
    assert base.NESTED.read_bytes() == nested_bytes
    assert (PREVIOUS / "replicates.jsonl").read_bytes() == previous_bytes
    assert (PREVIOUS / "config.json").read_bytes() == previous_config_bytes
    assert protected == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in protected}
    summary = dict(mean_precision=float(np.mean([r["precision"] for r in rows])),
                   correct_dimension_count=sum(r["selected_dimension"] == 4 for r in rows),
                   exact_support_recovery_count=sum(r["exact_support_recovery"] for r in rows),
                   total_comparisons=len(tests), reuse_counts=reuse_counts,
                   feasible_fallback_count=sum(t["feasible_fallback_count"] for t in tests.values()))
    (HERE / "results.json").write_text(json.dumps(dict(config=config, rows=rows, tests=tests, summary=summary,
                                                       protected_file_sha256=protected), indent=2) + "\n")
    with (HERE / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Top-three plateau candidates: descending bootstrap selection", "",
             "Use the previously reconstructed nested support paths for support_18, N=10000. Settings remain L_m=1, B=199 per pair, alpha=0.05 per comparison, and objective floor 1e-8 for plateau screening. Retain the three bounded plateaus with largest log-width and sort their dimensions descending (d1>d2>d3).", "",
             "Test the original support at d1 against the original support at d2, generating bootstrap samples under the fitted d2 model. If p<0.05, select d1 and stop. Otherwise test d2 against d3, using a new null covariance fitted at d3; select d2 if p<0.05, otherwise select d3. Each replicate refits both fixed supports. No support reconstruction or plateau screening occurs inside the bootstrap. No multiplicity adjustment is applied, retaining the previous per-comparison threshold.", "",
             "| Seed | Candidates, descending | First p | Second p | Selected dimension | Precision |", "|---|---|---|---|---|---|"]
    for row in rows:
        second = f"{row['second_p_value']:.3f}" if row["second_p_value"] is not None else "Not reached"
        lines.append(f"| {row['seed']} | {', '.join(map(str, row['candidates_descending']))} | {row['first_p_value']:.3f} | {second} | {row['selected_dimension']} | {row['true_positive_edges']}/{row['selected_edge_count']} = {100*row['precision']:.2f}% |")
    lines += ["", f"Mean precision: {100*summary['mean_precision']:.2f}%. Correct dimension: {summary['correct_dimension_count']}/10. Exact support recovery: {summary['exact_support_recovery_count']}/10.", "",
              "Precision excludes diagonal entries: TP / selected off-diagonal edges. The true support has three edges, (1,4), (2,4), (3,4), and true dimension 4. Seed 21's top-three shortlist still excludes dimension 4. Seeds 6 and 12 have the wrong support at dimension 4.", "",
              "## Bootstrap and verification", "",
              "The unmodified ADMM solver uses beta=1, 800 iterations, tol=1e-7, 10 Halton starts, zero_tol=1e-5, omega_fixed=1, min_omega=0, omega_upper=None. The empirical eigenvalue cap is omitted exactly as in the top-two experiment, retaining all generated samples. Each null covariance solves Sigma=Lambda.T Sigma Lambda+I; positive definiteness and stability are checked. No new optimization of the observed paths is performed.", "",
              "The statistic is the smaller fit's raw objective minus the larger fit's raw objective, with p=(1+number of bootstrap gains >= observed gain)/200. When the larger numerical fit is worse, the smaller fitted matrix is retained as a feasible candidate on the larger support. Clipping such negative gains to zero leaves every observed tail count unchanged because the observed gains are positive. Fits remain numerical approximations, not global-optimality certificates.", "",
              "Identical pairs reuse the previous 199 bootstrap replicates. When the fitted smaller model is identical but the larger support changes, the same samples and smaller fits are reused and the new larger support is refitted. Independent replicate streams use SeedSequence([20260913, original_seed, replicate_index]); changing the null covariance reuses the random stream as common random numbers across comparisons. This does not make successive test statistics independent. The fixed-candidate bootstrap and unadjusted sequential rule do not assert familywise or post-selection error control.", "",
              f"Computed {summary['total_comparisons']} reached pairwise comparisons. Reuse counts: {reuse_counts}. Feasible larger-fit fallbacks: {summary['feasible_fallback_count']}. All stored fit objectives and support constraints were independently recomputed. Hash checks confirm original experiments, production Python files, nested-path input, and reused bootstrap inputs are unchanged.", "",
              "## Monte Carlo uncertainty", "", "95% binomial intervals for the bootstrap exceedance probability (not model-validity confidence intervals):", "",
              "| Seed | Stage | Larger vs smaller | p | Interval |", "|---|---|---|---|---|"]
    for test in sorted(tests.values(), key=lambda t: (t["seed"], t["stage"])):
        lo, hi = test["mc_interval"]
        lines.append(f"| {test['seed']} | {test['stage']} | {test['large_dimension']} vs {test['small_dimension']} | {test['p_value']:.3f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ["", "## Reproduction", "", "From the repository root:", "", "```sh",
              "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python tmp/support_18_plateau_bootstrap_top3/analyze.py",
              "```", "", "The script resumes from `replicates.jsonl`. `config.json` records all candidate pairs and parameters; `results.json` records reached tests, tail counts, Monte Carlo intervals, selections, and verification hashes; `summary.csv` contains the per-seed comparison with top-two selection.", ""]
    (HERE / "README.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
