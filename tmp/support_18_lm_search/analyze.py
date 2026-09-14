"""Empirical Lm shape search on saved support_18 curves; no objective refits."""
from argparse import Namespace
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.select_scaling_parameter import load_selection_inputs
from penalty import pen_n, theorem_constants
from scaling_selection import select_minimal_scale

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18"
ARGS = Namespace(num_samples=None, r=1., Lm=1., L=1., xi=10., objective_floor=1e-8)


def candidates():
    for p in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32):
        yield f"D^{p}", lambda d, p=p: d**p
    for a in (.25, .5, 1., 1.5, 2., 3., 4., 6., 8.):
        yield f"exp({a:g}*(D-1))", lambda d, a=a: np.exp(a * (d - 1))
    for a in (.1, .25, .5, 1., 1.5, 2.):
        yield f"exp({a:g}*(D^2-1))", lambda d, a=a: np.exp(a * (d*d - 1))


def load_curves(source):
    curves = []
    for folder in sorted(source.glob("seed_*"), key=lambda p: int(p.name[5:])):
        path = folder / "objective_curve_sigma_hat.npz"
        inputs = load_selection_inputs(path, ARGS)
        with np.load(path) as data:
            off_diag = ~np.eye(int(data["n"]), dtype=bool)
            true = (data["Lambda_star"] != 0) & off_diag
            masks = data["selected_support_masks"] & off_diag
            assert np.all(data["selected_support_valid"])
            dims = inputs["d_m_values"]
            counts = masks.sum(axis=(1, 2))
            np.testing.assert_array_equal(counts, dims - 1)
            tp = (masks & true).sum(axis=(1, 2))
        inputs.update(seed=int(folder.name[5:]), counts=counts, tp=tp,
                      true_count=int(true.sum()), derived=theorem_constants(inputs["constants"]))
        curves.append(inputs)
    return curves


def evaluate(curves, name, rule):
    rows = []
    for curve in curves:
        dims = curve["d_m_values"].astype(float)
        lm = rule(dims)
        k, v = curve["derived"]["K"], curve["derived"]["v"]
        penalties = np.sqrt(dims) * (k + np.sqrt(2*v*lm))
        # Check optimized evaluation against the actual penalty implementation.
        np.testing.assert_allclose(penalties, [pen_n(d, curve["constants"], Lm=float(l)) for d, l in zip(dims, lm)], rtol=1e-14)
        for method in ("plateau", "window"):
            row = dict(form=name, seed=curve["seed"], method=method, dimension=None,
                       precision=None, recall=None, tp=None, edges=None,
                       recommended_scale=None, error=None)
            try:
                result = select_minimal_scale(dims, curve["objective_values"], penalties,
                                              method=method, eta=None, recommendation_factor=2.)
            except ValueError as exc:
                row["error"] = str(exc)
            else:
                idx = int(np.flatnonzero(dims == result.selected_dimension)[0])
                criterion = curve["objective_values"] + result.recommended_scale * penalties
                assert dims[np.argmin(criterion)] == result.selected_dimension
                count, tp = int(curve["counts"][idx]), int(curve["tp"][idx])
                row.update(dimension=result.selected_dimension, precision=tp/count if count else None,
                           recall=tp/curve["true_count"], tp=tp, edges=count,
                           recommended_scale=result.recommended_scale)
            rows.append(row)
    return rows


def summarize(rows):
    summaries = []
    for name in dict.fromkeys(r["form"] for r in rows):
        for method in ("plateau", "window"):
            group = [r for r in rows if r["form"] == name and r["method"] == method]
            success = [r for r in group if r["dimension"] is not None]
            defined = [r for r in success if r["precision"] is not None]
            summaries.append(dict(form=name, method=method, successful=len(success),
                                  precision_n=len(defined),
                                  mean_precision=float(np.mean([r["precision"] for r in defined])) if defined else None,
                                  mean_recall=float(np.mean([r["recall"] for r in success])) if success else None,
                                  mean_dimension=float(np.mean([r["dimension"] for r in success])) if success else None,
                                  dimensions=[r["dimension"] for r in group]))
    return summaries


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    curves = load_curves(SOURCE)
    print("K / sqrt(2v):", [c["derived"]["K"] / np.sqrt(2*c["derived"]["v"]) for c in curves], flush=True)
    rows = []
    for name, rule in candidates():
        result = evaluate(curves, name, rule)
        rows.extend(result)
        print(json.dumps(summarize(result)), flush=True)
    write_csv(OUT / "sweep.csv", rows)
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
