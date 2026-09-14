"""Replay selected powers through the experiment script's actual run function."""
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
import json
from unittest.mock import patch

import numpy as np

from analyze import OUT, SOURCE, pen_n, write_csv
from experiments import select_scaling_parameter as selection


def main():
    rows = json.loads((OUT / "sweep.json").read_text())
    rows += json.loads((OUT / "power_sensitivity.json").read_text())
    selected = {(r["form"], r["seed"], r["method"]): r for r in rows
                if r["form"] in ("D^1", "D^2", "D^8", "D^32", "D^48")}
    baseline = json.loads((OUT.parent / "support_18_lm_dimension_n10000/results.json").read_text())
    for old in baseline:
        if old["mode"] != "dimension":
            continue
        new = selected[("D^1", old["seed"], old["method"])]
        assert old["selected_dimension"] == new["dimension"]
        assert old["support_precision"] == new["precision"]
        assert old["error"] == new["error"]
    for row in selected.values():
        p = int(row["form"].split("^")[1])
        args = Namespace(input=str(SOURCE / f"seed_{row['seed']}" / "objective_curve_sigma_hat.npz"),
                         method=row["method"], eta=None, recommendation_factor=2.,
                         objective_floor=1e-8, num_samples=None, r=1., Lm=1., L=1., xi=10.,
                         validate_only=False)
        output = StringIO()
        def evaluator(d, constants):
            return pen_n(d, constants, Lm=d**p)
        with patch.object(selection, "pen_n", evaluator), redirect_stdout(output):
            try:
                result = selection.run(args)
            except ValueError as exc:
                assert str(exc) == row["error"]
                output.write(str(exc) + "\n")
            else:
                assert result.selected_dimension == row["dimension"]
                np.testing.assert_allclose(result.recommended_scale, row["recommended_scale"], rtol=1e-12, atol=0)
        path = OUT / "finalist_logs" / f"power_{p}" / f"seed_{row['seed']}_{row['method']}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Lm = D_m^{p}\n" + output.getvalue())
    write_csv(OUT / "finalists.csv", list(selected.values()))
    print(f"Verified {len(selected)} finalist selections through selection.run; baseline matches the prior replay.")


if __name__ == "__main__":
    main()
