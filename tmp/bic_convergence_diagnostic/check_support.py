import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from optimizers import bic

root=Path('experiments/output/upp_scal_n4_dm4_upper_scope_fit_omega_true_greedy_oracle_lasso/support_01')
original=bic.minimize
records=[]
current=[]
def monitored(fun,x0,**kwargs):
    result=original(fun,x0,**kwargs)
    value,g=fun(result.x,*kwargs['args'])
    pg=g.copy()
    for i,(lo,hi) in enumerate(kwargs['bounds']):
        if lo is not None:
            pg[i]=result.x[i]-np.clip(result.x[i]-g[i],lo,hi)
    current.append(dict(objective=float(value),gradient=float(np.max(np.abs(pg))),
                        nit=int(result.nit),message=str(result.message),
                        diagonal=np.tanh(result.x[:4]).tolist(),coordinates=result.x.tolist()))
    return result
bic.minimize=monitored
mask=np.eye(4,dtype=bool)
mask[0,1]=mask[0,2]=True
for curve in sorted(root.glob('num_samples_*/seed_*/objective_curve_sigma_hat.npz')):
    current=[]
    with np.load(curve,allow_pickle=False) as data:
        sigma=data['Sigma']
    start=time.monotonic()
    try:
        result=bic.fit_support(sigma,mask)
        status='ok'
    except ValueError as e:
        status=str(e)
    record=dict(path=str(curve),status=status,seconds=time.monotonic()-start,
                condition=float(np.linalg.cond(sigma)),restarts=current)
    records.append(record)
    print(curve.parent.parent.name,curve.parent.name,status,
          'best_pg',min(x['gradient'] for x in current),
          'max_nit',max(x['nit'] for x in current),flush=True)
    Path('tmp/bic_convergence_diagnostic/results.json').write_text(json.dumps(records,indent=2))
