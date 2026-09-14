"""Check native fits nearest each observed bootstrap cutoff and write tables."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from optimizers.support_search import solve_support_with_restarts

HERE = Path(__file__).resolve().parent
SETTINGS = dict(beta=1., max_iter=800, tol=1e-7, zero_tol=1e-5,
                max_restarts=10, min_omega=0., omega_fixed=1., omega_upper=None, init_strategy="halton")


def check_native(task):
    test, record = task
    sigma = np.array(record["Sigma"])
    objectives = []
    differences = []
    for name in ("small", "large"):
        mask = np.eye(4, dtype=bool)
        for i, j in test[f"{name}_edges"]:
            mask[i - 1, j - 1] = True
        fit = solve_support_with_restarts(sigma, mask, **SETTINGS)
        assert fit is not None
        q = float(fit[2])
        np.testing.assert_allclose(q, record[f"objective_{name}"], rtol=1e-8, atol=1e-12)
        objectives.append(q)
        differences.append(abs(q - record[f"objective_{name}"]))
    gain = max(objectives[0] - objectives[1], 0.)
    assert (gain >= test["observed_gain"]) == (record["gain"] >= test["observed_gain"])
    return dict(pair=test["key"], replicate=record["replicate"], max_difference=max(differences), tail_indicator_matches=True)


def main():
    result = json.loads((HERE / "results.json").read_text())
    paths = json.loads((HERE / "nested_paths.json").read_text())
    records = [json.loads(line) for line in (HERE / "bootstrap.jsonl").read_text().splitlines()]
    tests = result["tests"]
    assert len(records) == 199 * len(tests)
    tasks = []
    for key, test in tests.items():
        draws = [r for r in records if r["pair_key"] == key]
        assert sorted(r["replicate"] for r in draws) == list(range(199))
        assert sum(r["gain"] >= test["observed_gain"] for r in draws) == test["exceedances"]
        # The closest replicate is the most sensitive to numerical changes.
        record = min(draws, key=lambda r: abs(r["gain"] - test["observed_gain"]))
        tasks.append((test, record))
    with ProcessPoolExecutor(max_workers=6) as pool:
        validation = list(pool.map(check_native, tasks))
    verification = dict(pair_count=len(tests), bootstrap_draws=len(records),
                        native_fits_checked=2*len(validation),
                        max_native_objective_difference=max(r["max_difference"] for r in validation),
                        all_nearest_cutoff_indicators_match=True, checks=validation)
    (HERE / "native_validation.json").write_text(json.dumps(verification, indent=2) + "\n")
    source_hashes = result["config"]["protected_sha256"]
    assert source_hashes == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_hashes}
    lines = ["# Nested support recovery and top-three plateau bootstrap across sample sizes", "",
             "Sources are the three explicitly requested support_18 folders, including the folder named nsm1e6 (not the separately named nsm1000000 folder). Each saved covariance is reused without resampling for the observed reconstruction.", "",
             "Settings: L_m=1, B=199 per pair, alpha=0.05 per comparison, objective floor=1e-8 for screening. Start with the diagonal-only support at D=1, enumerate every upper-triangular one-edge extension at each dimension through 7, fit all candidates with 10 Halton starts, and retain the strict minimum objective. Rank bounded plateaus by absolute log-width and keep the top three. All 30 curves have at least three bounded plateaus.", "",
             "Order candidates d1>d2>d3. Test d1 against d2 under the fitted d2 null. If p<0.05, select d1 and stop. Otherwise test d2 against d3 under the fitted d3 null; select d2 if p<0.05, else d3. The original pair of supports is fixed and both are refitted in every bootstrap replicate. No bootstrap support reconstruction, no bootstrap screening, and no multiplicity adjustment are performed. At p=0.050 exactly, the smaller support is selected.", "",
             "The statistic uses raw objective gains and p=(1 + number of bootstrap gains >= observed gain)/200. Precision is the fraction of selected off-diagonal edges that belong to the true support. True edges are (1,4), (2,4), (3,4); true dimension is 4. Means below are arithmetic means across the ten seeds.", ""]
    for n in (1000000, 100, 1000):
        lines += [f"## N={n}", "", "| Seed | Candidates descending | First p | Second p | Selected dimension | Precision |", "|---|---|---|---|---|---|"]
        for row in result["rows"]:
            if row["num_samples"] != n:
                continue
            p1 = f"{row['first_p']:.3f}" if row["first_p"] is not None else "—"
            p2 = f"{row['second_p']:.3f}" if row["second_p"] is not None else "Not reached"
            precision = f"{row['true_positive_edges']}/{row['selected_edge_count']} = {100*row['precision']:.2f}%" if row["precision"] is not None else "Undefined"
            lines.append(f"| {row['seed']} | {', '.join(map(str,row['candidates_descending']))} | {p1} | {p2} | {row['selected_dimension']} | {precision} |")
        s = next(s for s in result["summary"] if s["num_samples"] == n)
        lines += ["", f"Mean precision: {100*s['mean_precision']:.2f}%. Correct dimension: {s['correct_dimension_count']}/10. Exact support recovery: {s['exact_support_count']}/10. Observed nested path passes through the true support: {s['path_passes_true_support_count']}/10.", ""]
    lines += ["## Interpretation and Monte Carlo uncertainty", "",
              "At N=1000, precision is 100% for every seed, but most selected models omit true edges; exact support recovery occurs in only 2/10 seeds. At N=100, four selections have the true dimension but all four have incorrect supports. At N=1000000, seed 3 retains dimension 5 rather than 4 (first-stage p=0.030).", "",
              "With B=199, p-values have resolution 0.005. The following reached tests have 95% binomial Monte Carlo intervals for their bootstrap exceedance probability that include 0.05:", "",
              "| N | Seed | Stage | Larger vs smaller | p | Monte Carlo interval |", "|---|---|---|---|---|---|"]
    for test in tests.values():
        if test["mc_interval_crosses_alpha"]:
            lo, hi = test["mc_interval"]
            lines.append(f"| {test['num_samples']} | {test['seed']} | {test['stage']} | {test['large_dimension']} vs {test['small_dimension']} | {test['p_value']:.3f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ["", "These are finite-bootstrap outcomes. The intervals describe simulation uncertainty, not model-validity confidence intervals. The fixed, data-selected candidate pairs and unadjusted sequential testing do not claim post-selection or familywise error control.", "",
              "## Computation and verification", "",
              "Under the mean-zero Gaussian model, X.T @ X has the Wishart(N, Sigma_null) distribution. Each bootstrap covariance is drawn as Wishart(df=N, scale=Sigma_null)/N. This is the exact finite-sample covariance law of the original sampling scheme, not an asymptotic covariance approximation. It avoids allocating a million-row dataset for each large-N replicate. The original code uses the uncentered covariance X.T X/N, hence df=N rather than N-1. [SciPy Wishart sampler documentation](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.stats.wishart.html).", "",
              "Random streams use SeedSequence([20260913, N, seed, replicate_index]). Generating covariances solve Sigma_null=Lambda.T Sigma_null Lambda+I; stability, positive definiteness, and the Wishart mean are verified. Omega is fixed at 1. As in the earlier bootstrap runs, the empirical eigenvalue cap is omitted to retain all generated samples.", "",
              "The diagnostic batches the same ADMM matrix updates, per-start convergence checks, 800-iteration cap, 10 Halton starts, beta=1, tol=1e-7, and zero_tol=1e-5. A preflight comparison of 82 fits against the unmodified native solver agreed to maximum objective error 1.0408340855860843e-17. All 30 observed D=1,2 objectives reproduce the saved unrestricted objectives to less than 7e-16, and all 30 reconstructed objective curves are nonincreasing within 1e-8.", "",
              f"Post-run verification refitted the bootstrap sample nearest the observed cutoff for each of {len(tests)} reached tests using the original solver ({verification['native_fits_checked']} native fits). Maximum objective difference: {verification['max_native_objective_difference']:.4g}; every tail indicator matched. All {len(records)} bootstrap sample pairs have checked support constraints and independently recomputed objective values. No generated sample is discarded.", "",
              f"There were {sum(t['feasible_fallback_count'] for t in tests.values())} negative raw numerical gains. The smaller fitted coefficient matrix is feasible on the larger nested support, so it is retained as a fallback and the gain becomes zero. Every observed gain is positive; the fallback leaves all tail counts and selections unchanged. Numerical ADMM fits are not certificates of global minima.", "",
              "SHA-256 checks confirm the three original experiment directories and production Python files are unchanged. All new artifacts are in this diagnostic directory.", "",
              "## Reproduction", "", "From the repository root:", "", "```sh",
              "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python tmp/support_18_multisample_bootstrap/analyze.py",
              "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python tmp/support_18_multisample_bootstrap/report.py", "```", "",
              "The runner resumes from nested_paths.json and bootstrap.jsonl. results.json contains all selected dimensions, precisions, test parameters, p-values, and Monte Carlo intervals; summary.csv provides a compact per-seed table. The two validation JSON files record equivalence checks against the original solver.", ""]
    (HERE / "README.md").write_text("\n".join(lines))
    print(json.dumps(verification, indent=2))
    for row in result["rows"]:
        print(row["num_samples"], row["seed"], row["selected_dimension"], row["precision"])
    print("SUMMARY", result["summary"])


if __name__ == "__main__":
    main()
