"""Read-only benchmark diagnostics for the specified saved 40-trial study.

The correlation baseline is exploratory: an approximate Fisher-z test applied
to the saved known-zero-mean second moments, with Bonferroni alpha .05 over
the six eligible upper-triangle pairs. It uses neither truth nor tuned cutoffs
for selection. It is not a general estimator of discrete Lyapunov support.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/output/fixed_support_scaling_n4_upper_scope_fit_omega_ref_true_greedy"
OUTPUT = Path(__file__).resolve().parent
ALPHA = .05
PAIRS = 6
CUTOFF = float(norm.isf(ALPHA / (2 * PAIRS)))


def metrics(selected, truth):
    tp = int(np.count_nonzero(selected & truth))
    fp = int(np.count_nonzero(selected & ~truth))
    fn = int(np.count_nonzero(~selected & truth))
    return dict(exact=int(fp + fn == 0), tp=tp, fp=fp, fn=fn,
                recall=tp / int(truth.sum()), hamming=fp + fn)


rows = []
for path in sorted(ROOT.glob("num_samples_*/seed_*/objective_curve_sigma_hat.npz")):
    report = json.loads((path.parent / "selection_plateau_bootstrap.json").read_text())
    with np.load(path) as data:
        sample_size = int(data["num_samples"])
        p = int(data["n"])
        assert p == 4 and str(data["support_scope"].item()) == "upper"
        truth = np.triu(np.abs(data["Lambda_star"]) > 1e-10, 1)
        dims = data["d_m_values"]
        masks = np.triu(data["selected_support_masks"], 1)
        valid = data["selected_support_valid"]
        k = int(truth.sum())
        known_size_index = int(np.flatnonzero(dims == k + 1)[0])
        chosen_index = int(np.flatnonzero(dims == report["selected_dimension"])[0])
        assert valid[known_size_index] and valid[chosen_index]
        current = metrics(masks[chosen_index], truth)
        known_size = metrics(masks[known_size_index], truth)
        path_loss = min(metrics(m, truth)["hamming"]
                        for m, is_valid in zip(masks, valid) if is_valid)
        sigma = data["Sigma"]
        correlation = sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))
        z_score = np.arctanh(np.clip(correlation, -.999999999, .999999999)) * np.sqrt(sample_size - 3)
        baseline = metrics(np.triu(np.abs(z_score) > CUTOFF, 1), truth)
        row = dict(num_samples=sample_size, seed=int(data["random_seed"]),
                   selected_dimension=int(report["selected_dimension"]),
                   true_dimension=k + 1,
                   true_size_screened_in=int(k + 1 in report["candidate_dimensions"]),
                   path_oracle_hamming=path_loss,
                   omega_ref=float(data["omega_ref"]))
        for name, values in [("proposed", current), ("known_size", known_size), ("correlation", baseline)]:
            row.update({f"{name}_{key}": value for key, value in values.items()})
        rows.append(row)
assert len(rows) == 40
with (OUTPUT / "trials.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = []
for sample_size in sorted({row["num_samples"] for row in rows}):
    group = [row for row in rows if row["num_samples"] == sample_size]
    item = dict(num_samples=sample_size, trials=len(group))
    for name in ["proposed", "known_size", "correlation"]:
        item[f"{name}_exact_count"] = sum(row[f"{name}_exact"] for row in group)
        for key in ["recall", "hamming", "fp", "fn"]:
            item[f"{name}_mean_{key}"] = float(np.mean([row[f"{name}_{key}"] for row in group]))
    item["path_oracle_exact_count"] = sum(row["path_oracle_hamming"] == 0 for row in group)
    item["path_oracle_mean_hamming"] = float(np.mean([row["path_oracle_hamming"] for row in group]))
    item["true_size_screened_in_count"] = sum(row["true_size_screened_in"] for row in group)
    summary.append(item)
result = dict(input=str(ROOT), alpha=ALPHA, eligible_pairs=PAIRS,
              fisher_z_cutoff=CUTOFF, baseline_is_approximate=True,
              known_size_is_saved_candidate_not_global_optimum=True,
              summary=summary)
(OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
