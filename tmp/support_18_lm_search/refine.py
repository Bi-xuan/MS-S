"""Follow-up search for forms improving precision without more failures."""
import json
import numpy as np
from analyze import OUT, SOURCE, evaluate, load_curves, summarize, write_csv


def candidates():
    for p in (1, 2, 3, 4, 6, 8, 10, 12):
        for c in (.01, .1, 1., 10., 100., 10000.):
            yield f"1+{c:g}*(D-1)^{p}", lambda d, c=c, p=p: 1 + c*(d-1)**p
    for p in (1, 2, 3, 4, 6, 8):
        for c in (.01, .1, 10., 100., 10000.):
            yield f"{c:g}*D^{p}", lambda d, c=c, p=p: c*d**p


def main():
    curves = load_curves(SOURCE)
    rows = []
    for name, rule in candidates():
        results = evaluate(curves, name, rule)
        rows.extend(results)
        s = summarize(results)
        if s[0]["mean_precision"] > .7217 and s[1]["successful"] >= 6:
            print(json.dumps(s), flush=True)
    write_csv(OUT / "refined.csv", rows)
    (OUT / "refined.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "refined_summary.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
