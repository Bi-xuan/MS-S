"""Verify the saved bootstrap run and produce the report.

The fitted smaller coefficient matrix is also a feasible larger-model fit.
Include it as a fallback whenever the larger numerical fit is worse.
This makes the gain nonnegative and, since all observed gains are positive,
does not change any bootstrap tail count in this experiment.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from objective import frobenius_objective

HERE = Path(__file__).resolve().parent
RUN = HERE / "run_b199"


def main():
    result = json.loads((RUN / "results.json").read_text())
    records = [json.loads(line) for line in (RUN / "replicates.jsonl").read_text().splitlines()]
    config = result["config"]
    candidates = {c["seed"]: c for c in config["candidates"]}
    corrected = []
    for record in records:
        c = candidates[record["seed"]]
        assert c["observed_gain"] > 0
        sigma = np.array(record["Sigma"])
        objectives = []
        for name, edges in (("small_fit", c["small_edges"]), ("large_fit", c["large_edges"])):
            lam = np.array(record[name]["Lambda"])
            mask = np.eye(4, dtype=bool)
            for i, j in edges:
                mask[i - 1, j - 1] = True
            assert np.all(lam[~mask] == 0)
            objective = frobenius_objective(sigma, lam, 1.)
            np.testing.assert_allclose(objective, record[name]["objective"], rtol=1e-12, atol=1e-15)
            objectives.append(objective)
        record["larger_feasible_fallback"] = bool(objectives[1] > objectives[0])
        record["feasible_large_objective"] = float(min(objectives))
        record["feasible_gain"] = float(objectives[0] - min(objectives))
        assert (record["feasible_gain"] >= c["observed_gain"]) == (record["gain"] >= c["observed_gain"])
        corrected.append(record)
    assert len(corrected) == 10 * config["replicates"]
    with (RUN / "replicates_verified.jsonl").open("w") as stream:
        for record in sorted(corrected, key=lambda r: (r["seed"], r["replicate"])):
            stream.write(json.dumps(record) + "\n")
    for row in result["rows"]:
        draws = [r for r in corrected if r["seed"] == row["seed"]]
        row["larger_feasible_fallback_count"] = sum(r["larger_feasible_fallback"] for r in draws)
        row["monte_carlo_interval_crosses_alpha"] = row["mc_interval_lower"] <= row["alpha"] <= row["mc_interval_upper"]
    result["verification"] = dict(all_3980_fit_objectives_recomputed=True,
                                  feasible_fallback_tail_counts_unchanged=True,
                                  larger_feasible_fallback_count=sum(r["larger_feasible_fallback"] for r in corrected))
    (RUN / "results_verified.json").write_text(json.dumps(result, indent=2) + "\n")
    with (RUN / "summary_verified.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    lines = ["# Two-plateau bootstrap selection on nested support paths", "",
             "Source: `experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18`.", "",
             "The observed support paths and fitted coefficients are taken from `tmp/support_18_greedy_nested/results.json`. At L_m=1, select the two bounded plateaus with greatest absolute log-width and order their supports by dimension. The unbounded end plateaus are excluded, as in the existing selector. Plateaus are screened with objective floor 1e-8; the test uses raw objective differences. There is no subsequent factor-of-two scale adjustment because this procedure directly selects a candidate support.", "",
             "For each seed, simulate B=199 independent, mean-zero Gaussian datasets of N=10000 observations from the smaller fitted model's stationary covariance. Omega is fixed at 1. Refit the ORIGINAL two candidate supports on each replicate; do not reconstruct or re-screen supports inside the bootstrap. This follows the user's clarified fixed-support test.", "",
             "Compute p=(1 + number of bootstrap gains >= observed gain)/(B+1). Retain the larger support when p<0.05; otherwise choose the smaller support. Precision is TP divided by the number of selected off-diagonal edges. The true dimension is 4 and the true edges are (1,4), (2,4), (3,4).", "",
             "| Seed | Candidate dimensions (small, large) | Observed gain | Exceedances / 199 | p | Selected dimension | Precision |", "|---|---|---|---|---|---|---|"]
    for row in result["rows"]:
        lines.append(f"| {row['seed']} | {row['small_dimension']}, {row['large_dimension']} | {row['observed_gain']:.8g} | {row['bootstrap_exceedances']} / 199 | {row['p_value']:.3f} | {row['selected_dimension']} | {row['true_positive_edges']}/{row['selected_edges']} = {100*row['precision']:.2f}% |")
    lines += ["", f"Mean per-seed precision: {100*result['mean_precision']:.2f}%. Correct dimension: {result['correct_dimension_count']}/10. Exact support recovery: {result['exact_support_recovery_count']}/10.", "",
              "Dimension 4 is absent from the two-candidate shortlist for seeds 3 and 21. Precision alone does not establish complete support recovery.", "", "## Numerical and statistical details", "",
              "Fits call the unmodified production `solve_support_with_restarts` with beta=1, max_iter=800, tol=1e-7, zero_tol=1e-5, 10 Halton starts, min_omega=0, omega_fixed=1. These are numerical fits, not certified global minima.", "",
              "For bootstrap refits, omega_upper=None: sample covariance eigenvalues can fall below the known omega even when sampling from a valid stationary model. The original empirical eigenvalue cap would reject those samples entirely. No generated samples are discarded or redrawn. Original observed fits are unchanged; every generating covariance is positive definite and its fitted Lambda has spectral radius below 1.", "",
              f"The larger numerical fit was worse than the smaller fit in {result['verification']['larger_feasible_fallback_count']} replicates. Since the models are nested, the smaller fitted matrix is a feasible candidate for the larger model. Including it changes these gains to zero. All observed gains are positive, so every exceedance count, p-value, and selected dimension is unchanged by this numerical fallback. Raw and verified records are both retained.", "",
              "These are plug-in parametric-bootstrap tail estimates for the fixed, data-selected candidate pair; they are not a guarantee of post-selection type-I error control. B=199 gives p-values on a 0.005 grid. Monte Carlo binomial intervals describe uncertainty in the bootstrap exceedance probability, not uncertainty in model validity.", "",
              "| Seed | 95% Monte Carlo interval | Contains alpha=0.05? |", "|---|---|---|"]
    for row in result["rows"]:
        lines.append(f"| {row['seed']} | [{row['mc_interval_lower']:.4f}, {row['mc_interval_upper']:.4f}] | {'Yes' if row['monte_carlo_interval_crosses_alpha'] else 'No'} |")
    lines += ["", "## Reproduction", "", "From the repository root:", "", "```sh",
              "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python tmp/support_18_plateau_bootstrap/analyze.py",
              ".venv/bin/python tmp/support_18_plateau_bootstrap/report.py", "```", "",
              "The simulation uses independent deterministic NumPy SeedSequence([20260913, original_seed, replicate_index]) streams. The runner resumes from completed JSONL records and checks that its configuration matches. All 3980 stored fit objectives and support constraints are independently recomputed and verified. SHA-256 checks confirm original experiment files, nested reconstruction results, and production Python files were unchanged during the run.", "",
              "`run_b199/config.json` records candidates, fitted null parameters, generating covariances, and settings. `run_b199/replicates.jsonl` records all raw fits and empirical covariance matrices. `run_b199/replicates_verified.jsonl` additionally records the feasible fallback. `run_b199/results_verified.json` and `run_b199/summary_verified.csv` contain final results.", ""]
    (HERE / "README.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
