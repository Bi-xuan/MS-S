"""Power-law sensitivity and a separate N=1e6 transfer check."""
import json
from analyze import ROOT, OUT, SOURCE, evaluate, load_curves, summarize, write_csv


def main():
    curves = load_curves(SOURCE)
    rows = []
    for p in (7, 8, 9, 28, 30, 32, 34, 36, 40, 48, 64):
        results = evaluate(curves, f"D^{p}", lambda d, p=p: d**p)
        rows.extend(results)
        print(json.dumps(summarize(results)), flush=True)
    write_csv(OUT / "power_sensitivity.csv", rows)
    (OUT / "power_sensitivity.json").write_text(json.dumps(rows, indent=2) + "\n")
    other = ROOT / "experiments/output/upp_scal_n4_dm4_nsm1e6_omega1_minabs02_10seeds_PlaAbs/support_18"
    curves = load_curves(other)
    rows = []
    for p in (1, 8, 32, 64):
        results = evaluate(curves, f"D^{p}", lambda d, p=p: d**p)
        rows.extend(results)
        print("N=1e6", json.dumps(summarize(results)), flush=True)
    write_csv(OUT / "transfer_n1e6.csv", rows)
    (OUT / "transfer_n1e6_summary.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
