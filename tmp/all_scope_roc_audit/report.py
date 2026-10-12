"""Plot all-direction ROC comparisons and summarize the diagnostic prototype."""
import csv
import json
import os
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import load, edges_for, TRUTH

rows = json.loads((OUT/'lookahead_results.json').read_text())
saved = list(csv.DictReader((OUT/'saved_summary.csv').open()))
summaries = []
figure, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
for ax, size in zip(axes.flat, (100, 1000, 10000, 1000000)):
    group = [r for r in rows if r['num_samples']==size]
    for label, color, style, name in [('all_forward','#0072B2','-', 'Saved forward'),
                                     ('all_fb','#009E73','--', 'Saved forward/backward')]:
        paths = []
        for r in group:
            d, _ = load(label, size, r['seed'])
            paths.append([edges_for(m) for m in d['selected_support_masks']])
        fpr = np.mean([[len(e-TRUTH)/9 for e in p] for p in paths],axis=0)
        tpr = np.mean([[len(e&TRUTH)/3 for e in p] for p in paths],axis=0)
        ax.plot(fpr,tpr,color=color,ls=style,lw=2,label=name)
    for key, color, style, name in [('lookahead','#D55E00','-', 'Beam + rebuilt prefix'),
                                  ('polished_forward','#777777',':','Least-squares forward')]:
        paths = [[frozenset(map(tuple,e)) for e in r[key]['edges']] for r in group]
        fpr = np.mean([[len(e-TRUTH)/9 for e in p] for p in paths],axis=0)
        tpr = np.mean([[len(e&TRUTH)/3 for e in p] for p in paths],axis=0)
        ax.plot(fpr,tpr,color=color,ls=style,lw=2,label=name)
        baseline = next(r for r in saved if r['method']=='all_forward' and int(r['num_samples'])==size)
        summaries.append(dict(num_samples=size,method=key,
                              mean_auc=float(np.mean([r[key]['auc'] for r in group])),
                              saved_forward_mean_auc=float(baseline['auc']),
                              mean_tpr_k3=float(np.mean([r[key]['tpr_k3'] for r in group])),
                              true_on_path=sum(r[key]['true_on_path'] for r in group),n=len(group)))
    ax.plot([0,1],[0,1],color='.8',lw=1,zorder=0)
    ax.set_title(f'N = {size:,}')
    ax.set_xlim(-.02,1.02)
    ax.set_ylim(-.02,1.02)
    ax.set_aspect('equal')
    ax.grid(color='.9',lw=.6)
for ax in axes[-1,:]: ax.set_xlabel('False positive rate (9 false directed edges)')
for ax in axes[:,0]: ax.set_ylabel('True positive rate (3 true directed edges)')
handles, labels = axes[0,0].get_legend_handles_labels()
figure.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.995))
figure.suptitle('All directions eligible: mean support-path ROC over 10 matched seeds',y=.92,fontsize=13)
figure.text(.5,.025,'Diagnostic beam: width 10, endpoint at 3 edges; generic Halton least-squares starts.\n'
            'The horizon matches this experiment’s true sparsity; it needs validation for unknown sparsity.',
            ha='center',fontsize=9,color='.3')
figure.tight_layout(rect=(0,.065,1,.92))
figure.savefig(OUT/'roc_comparison.png',dpi=180)
plt.close(figure)
with (OUT/'prototype_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)
print(json.dumps(summaries,indent=2))
