"""Compare three L_m rules on the saved forward-nested support paths."""
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.select_scaling_parameter import load_selection_inputs, floor_objective_values
from penalty import pen_n, theorem_constants
from scaling_selection import select_minimal_scale, build_dimension_path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dimension_selection"
RULES = {"1": 0, "D_m": 1, "D_m^8": 8}
ARGS = SimpleNamespace(num_samples=None, r=1., Lm=1., L=1., xi=10., objective_floor=1e-8)


def cell(row):
    if row["status"] != "ok":
        return "Failed / —"
    precision = row["support_precision"]
    return f"{row['selected_dimension']} / {100 * precision:.2f}%" if precision is not None else f"{row['selected_dimension']} / undefined"


def main():
    OUT.mkdir(exist_ok=True)
    input_path = HERE / "results.json"
    original = input_path.read_bytes()
    nested = json.loads(original)
    protected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in nested["protected_file_sha256"]}
    rows, diagnostics = [], []
    for trial in nested["results"]:
        inputs = load_selection_inputs(ROOT / trial["source"], ARGS)
        constants = inputs["constants"]
        derived = theorem_constants(constants)
        steps = trial["steps"]
        dims = np.array([s["dimension"] for s in steps])
        raw = np.array([s["objective"] for s in steps])
        values, floored = floor_objective_values(raw, ARGS.objective_floor)
        assert np.array_equal(dims, inputs["d_m_values"])
        assert constants.num_samples == 10000
        truth = set(map(tuple, trial["true_edges"]))
        for rule, power in RULES.items():
            penalties = np.array([pen_n(float(d), constants, Lm=float(d)**power) for d in dims])
            expected = derived["K"] * np.sqrt(dims) + np.sqrt(2 * derived["v"]) * dims.astype(float)**((power + 1) / 2)
            np.testing.assert_allclose(penalties, expected, rtol=1e-14)
            if power == 0:
                np.testing.assert_array_equal(penalties, inputs["penalty_values"])
            path = build_dimension_path(dims, values, penalties)
            for method in ("plateau", "window"):
                row = dict(seed=trial["seed"], Lm=rule, method=method, status="ok",
                           selected_dimension=None, true_positive_edges=None,
                           selected_edge_count=None, support_precision=None,
                           dimension_correct=False, exact_support_recovery=False,
                           minimal_scale=None, recommended_scale=None, error=None)
                diagnostic = dict(seed=trial["seed"], Lm=rule, method=method,
                                  constants=asdict(constants), derived_constants=derived,
                                  dimensions=dims.tolist(), raw_objectives=raw.tolist(),
                                  floored_objectives=values.tolist(), penalty_values=penalties.tolist(),
                                  dimension_path=asdict(path))
                try:
                    result = select_minimal_scale(dims, values, penalties, method=method,
                                                  eta=None, recommendation_factor=2.)
                except ValueError as exc:
                    if not str(exc).startswith(("Window selection failed:", "Plateau selection failed:")):
                        raise
                    row.update(status="failed", error=str(exc))
                else:
                    dim = result.selected_dimension
                    index = int(np.flatnonzero(dims == dim)[0])
                    # Independently minimize the penalized criterion at the recommended scale.
                    criterion = values + result.recommended_scale * penalties
                    assert int(dims[np.argmin(criterion)]) == dim
                    selected = set(map(tuple, steps[index]["edges"]))
                    tp, count = len(selected & truth), len(selected)
                    assert count == dim - 1 and len(truth) == 3
                    row.update(selected_dimension=dim, true_positive_edges=tp,
                               selected_edge_count=count, support_precision=tp/count if count else None,
                               dimension_correct=dim == 4, exact_support_recovery=selected == truth,
                               minimal_scale=result.minimal_scale, recommended_scale=result.recommended_scale)
                    diagnostic.update(selection=asdict(result), selected_edges=steps[index]["edges"],
                                      penalized_criterion=criterion.tolist())
                diagnostic["outcome"] = row
                rows.append(row)
                diagnostics.append(diagnostic)
    assert len(rows) == 60
    summary = []
    for method in ("plateau", "window"):
        for rule in RULES:
            group = [r for r in rows if r["method"] == method and r["Lm"] == rule]
            precision = [r["support_precision"] for r in group if r["support_precision"] is not None]
            summary.append(dict(method=method, Lm=rule, successful_selections=sum(r["status"] == "ok" for r in group),
                                mean_precision=float(np.mean(precision)) if precision else None,
                                precision_denominator=len(precision),
                                correct_dimension_count=sum(r["dimension_correct"] for r in group),
                                exact_support_recovery_count=sum(r["exact_support_recovery"] for r in group), total=10))
    assert original == input_path.read_bytes()
    assert protected == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in protected}
    with (OUT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "results.json").write_text(json.dumps(dict(rows=rows, summary=summary, diagnostics=diagnostics), indent=2) + "\n")
    report = ["# Dimension selection on the forward-nested support paths", "",
              "Source paths and objectives: `../results.json`, reconstructed from the support_18 N=10000 experiment.", "",
              "Each table entry is selected dimension / support precision. Precision is TP divided by selected off-diagonal edges; diagonal entries and the dimension offset are excluded. Failed selections have undefined precision. The true dimension is 4 with three true edges.", "",
              "Settings: objective floor 1e-8, recommendation factor 2, automatic window width, r=L=1, xi=10, and covariance-derived constants from each original NPZ. The penalty is sqrt(D_m) * (K + sqrt(2*v*L_m)), evaluated with L_m=1, D_m, or D_m^8. The existing selectors are called without modification.", ""]
    for method in ("plateau", "window"):
        report += [f"## {method.capitalize()}", "", "| Seed | L_m=1 | L_m=D_m | L_m=D_m^8 |", "|---|---|---|---|"]
        for seed in sorted({r["seed"] for r in rows}):
            group = [next(r for r in rows if r["seed"] == seed and r["method"] == method and r["Lm"] == rule) for rule in RULES]
            report.append(f"| {seed} | " + " | ".join(map(cell, group)) + " |")
        report += [""]
    report += ["## Aggregate results", "", "Mean precision is the arithmetic mean over defined results, not pooled edge counts. Failures count as unsuccessful for dimension and exact-support recovery.", "",
               "| Method | L_m | Returned | Mean precision | Correct dimension | Exact support |", "|---|---|---|---|---|---|"]
    for s in summary:
        report.append(f"| {s['method']} | {s['Lm']} | {s['successful_selections']}/10 | {100*s['mean_precision']:.2f}% ({s['precision_denominator']} seeds) | {s['correct_dimension_count']}/10 | {s['exact_support_recovery_count']}/10 |")
    report += ["", "Precision of 100% at dimension 2 or 3 still omits true edges; it does not imply exact recovery.", "",
               "## Reproduction and verification", "", "Run `.venv/bin/python tmp/support_18_greedy_nested/select_dimensions.py` from the repository root.", "",
               "The script checks all three penalty formulas, directly minimizes the penalized objective for every returned dimension, and verifies support edge counts. `comparison.csv` records dimensions, TP counts, precisions, scales, and failure reasons; `results.json` additionally records all selection diagnostics and objective/penalty arrays. Hash checks confirm the nested reconstruction results, original experiment files, and production Python files are unchanged.", ""]
    (OUT / "README.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
