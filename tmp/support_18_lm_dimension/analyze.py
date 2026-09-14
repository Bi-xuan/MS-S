"""Replay support_18 at N=1e6 with constant and dimension-dependent Lm."""

from argparse import Namespace
from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import select_scaling_parameter as selection
from penalty import pen_n, theorem_constants

SOURCE = ROOT / "experiments/output/upp_scal_n4_dm4_nsm1e6_omega1_minabs02_10seeds_PlaAbs/support_18"
OUT = Path(__file__).resolve().parent


def dimension_penalty(d_m, constants):
    return pen_n(d_m, constants, Lm=float(d_m))


def main():
    rows = []
    for folder in sorted(SOURCE.glob("seed_*"), key=lambda p: int(p.name[5:])):
        with (folder / "result.csv").open() as handle:
            saved = next(csv.DictReader(handle))
        for mode in ("constant", "dimension"):
            for method in ("plateau", "window"):
                args = Namespace(
                    input=str(folder / "objective_curve_sigma_hat.npz"),
                    method=method, eta=None, recommendation_factor=2.0,
                    objective_floor=1e-8, num_samples=None, r=1.0,
                    Lm=1.0, L=1.0, xi=10.0, validate_only=False,
                )
                row = dict(seed=int(folder.name[5:]), mode=mode, method=method,
                           true_dimension=int(saved["true_dimension"]), status="ok",
                           selected_dimension=None, minimal_scale=None,
                           recommended_scale=None, error=None)
                log = StringIO()
                evaluator = pen_n if mode == "constant" else dimension_penalty
                with patch.object(selection, "pen_n", evaluator):
                    inputs = selection.load_selection_inputs(args.input, args)
                    derived = theorem_constants(inputs["constants"])
                    dims = inputs["d_m_values"]
                    expected = np.sqrt(dims) * derived["K"] + np.sqrt(2 * derived["v"]) * (
                        np.sqrt(dims) if mode == "constant" else dims
                    )
                    np.testing.assert_allclose(inputs["penalty_values"], expected, rtol=1e-14)
                    try:
                        with redirect_stdout(log):
                            result = selection.run(args)
                        row.update(selected_dimension=result.selected_dimension,
                                   minimal_scale=result.minimal_scale,
                                   recommended_scale=result.recommended_scale)
                        # Independently verify the penalized-criterion minimizer.
                        criterion = inputs["objective_values"] + result.recommended_scale * inputs["penalty_values"]
                        assert dims[np.argmin(criterion)] == result.selected_dimension
                    except ValueError as exc:
                        if not str(exc).startswith("Window selection failed:"):
                            raise
                        row.update(status="failed", error=str(exc))
                        log.write(str(exc) + "\n")
                if mode == "constant":
                    assert row["status"] == saved[f"{method}_status"]
                    if row["status"] == "ok":
                        assert row["selected_dimension"] == int(saved[f"{method}_dimension"])
                        original_log = (folder / f"selection_{method}.log").read_text()
                        original_scale = float(re.search(r"^Recommended scale: (.+)$", original_log, re.M)[1])
                        np.testing.assert_allclose(row["recommended_scale"], original_scale, rtol=1e-10, atol=0)
                log_path = OUT / folder.name / f"{mode}_{method}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(f"Lm rule: {'1' if mode == 'constant' else 'D_m'}\n" + log.getvalue())
                rows.append(row)
    with (OUT / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
