"""Render the completed quick L1 benchmark and numerical checks."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ms-s-l1-matplotlib")
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
result=json.loads((OUT/"results.json").read_text())
rows=result["summary"]
methods=[("proposed","Proposed","#153b63","-"),
         ("full_l1_cv","Full L1: holdout","#d46532","-"),
         ("offdiagonal_l1_cv","Off-diagonal L1: holdout","#258274","-"),
         ("full_l1_oracle","Full L1: best on path","#d46532","--"),
         ("offdiagonal_l1_oracle","Off-diagonal L1: best on path","#258274","--")]
fig,axes=plt.subplots(1,2,figsize=(11.6,4.6))
for method,label,color,style in methods:
    group=sorted([r for r in rows if r["method"]==method],key=lambda r:r["num_samples"])
    x=[r["num_samples"] for r in group]
    axes[0].plot(x,[r["exact_count"]/r["trials"] for r in group],marker="o",ls=style,color=color,label=label,linewidth=2)
    axes[1].plot(x,[r["mean_hamming"] for r in group],marker="o",ls=style,color=color,linewidth=2)
for ax in axes:
    ax.set_xscale("log");ax.set_xticks([100,1000,10000,1000000]);ax.set_xticklabels(["100","1,000","10,000","1,000,000"])
    ax.set_xlabel("Number of independent observations");ax.grid(alpha=.18);ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Exact support recovery rate");axes[0].set_ylim(-.04,1.04)
axes[1].set_ylabel("Mean number of wrong edges (FP + FN)");axes[1].set_ylim(bottom=-.08)
fig.suptitle("Matched discrete Lyapunov model: proposed selector versus L1 penalties",fontsize=13)
fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,fontsize=9)
fig.text(.5,.125,"10 paired trials per sample size · known upper-triangular order · dashed lines use simulation truth",ha="center",fontsize=9,color="#555555")
fig.tight_layout(rect=[0,.17,1,.94]);fig.savefig(OUT/"comparison.png",dpi=180);plt.close(fig)

checks={"threshold_sensitivity":{},"max_selected_kkt":{},"cv_endpoint_counts":{}}
trials=[json.loads(p.read_text()) for p in sorted(OUT.glob("trial_n*_seed*.json"))]
for variant in ("full_l1","offdiagonal_l1"):
    checks["max_selected_kkt"][variant]=max(t["variants"][variant]["full_details"][t["variants"][variant]["cv_index"]]["projected_gradient"] for t in trials)
    checks["cv_endpoint_counts"][variant]={"unpenalized":sum(t["variants"][variant]["cv_index"]==0 for t in trials),"maximum_penalty":sum(t["variants"][variant]["cv_index"]==len(result["ratios"])-1 for t in trials)}
    for threshold in (1e-6,1e-5,1e-4):
        counts={}
        for t in trials:
            n=t["num_samples"];seed=t["seed"]
            with np.load(OUT/f"path_n{n}_seed{seed}.npz") as z:
                a=z[f"{variant}_full_path"][t["variants"][variant]["cv_index"]]
                exact=bool(np.array_equal(np.triu(abs(a)>threshold,1),z["truth"]))
            counts[n]=counts.get(n,0)+exact
        checks["threshold_sensitivity"][f"{variant}_{threshold:g}"]=counts
(OUT/"verification.json").write_text(json.dumps(checks,indent=2)+"\n")
print(json.dumps(checks,indent=2))
for row in rows:
    print(row["num_samples"],row["method"],"exact",row["exact_count"],"Hamming",round(row["mean_hamming"],3),"precision",round(row["mean_precision"],3),"recall",round(row["mean_recall"],3))
