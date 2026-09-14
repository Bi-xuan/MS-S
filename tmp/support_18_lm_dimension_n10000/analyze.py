"""Replay the Lm comparison at N=10000 and measure selected-support precision."""

from contextlib import redirect_stdout
import csv
import importlib.util
from io import StringIO
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18"


def main():
    spec = importlib.util.spec_from_file_location(
        "lm_replay", ROOT / "tmp/support_18_lm_dimension/analyze.py"
    )
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    replay.SOURCE = SOURCE
    replay.OUT = OUT
    with redirect_stdout(StringIO()):
        replay.main()
    rows = json.loads((OUT / "results.json").read_text())
    for row in rows:
        row.update(dimension_correct=False, true_positive_edges=None,
                   selected_edge_count=None, support_precision=None)
        if row["status"] != "ok":
            continue
        with np.load(SOURCE / f"seed_{row['seed']}" / "objective_curve_sigma_hat.npz") as data:
            assert int(data["num_samples"]) == 10000
            idx = int(np.flatnonzero(data["d_m_values"] == row["selected_dimension"])[0])
            assert data["selected_support_valid"][idx]
            off_diag = ~np.eye(int(data["n"]), dtype=bool)
            selected = data["selected_support_masks"][idx] & off_diag
            true = (data["Lambda_star"] != 0) & off_diag
            selected_count = int(selected.sum())
            tp = int((selected & true).sum())
            assert selected_count == row["selected_dimension"] - 1
            assert true.sum() == row["true_dimension"] - 1 == 3
            row.update(dimension_correct=row["selected_dimension"] == row["true_dimension"],
                       true_positive_edges=tp, selected_edge_count=selected_count,
                       support_precision=tp / selected_count if selected_count else None)
    summary = []
    for mode in ("constant", "dimension"):
        for method in ("plateau", "window"):
            group = [r for r in rows if r["mode"] == mode and r["method"] == method]
            success = [r for r in group if r["status"] == "ok"]
            precisions = [r["support_precision"] for r in success if r["support_precision"] is not None]
            summary.append(dict(mode=mode, method=method, seeds=len(group),
                                successful_selections=len(success),
                                mean_selected_dimension=float(np.mean([r["selected_dimension"] for r in success])),
                                dimension_accuracy=sum(r["dimension_correct"] for r in group) / len(group),
                                mean_support_precision=float(np.mean(precisions)),
                                precision_denominator=len(precisions)))
    with (OUT / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in rows:
        if row["mode"] == "dimension":
            print(row)
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
