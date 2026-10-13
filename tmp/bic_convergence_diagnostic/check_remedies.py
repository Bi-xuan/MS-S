import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from optimizers import bic
p=Path('experiments/output/upp_scal_n4_dm4_upper_scope_fit_omega_true_greedy_oracle_lasso/support_02/num_samples_100/seed_124/objective_curve_sigma_hat.npz')
with np.load(p) as data:
    sigma=data['Sigma']
mask=np.eye(4,dtype=bool); mask[0,1]=mask[0,2]=True
for name,options in [('more_iterations',dict(max_iter=5000)),('all_unique_starts',dict(max_restarts=24)),('looser_gradient',dict(tol=1e-5)),('bound_9999',dict(diagonal_bound=.9999)),('bound_999',dict(diagonal_bound=.999)),('bound_99',dict(diagonal_bound=.99))]:
    try:
        f=bic.fit_support(sigma,mask,**options)
        print(name,'q',f['objective_value'],'pg',f['projected_gradient'],'omega',f['fitted_omega'],'diagonals',np.diag(f['fitted_lambda']).tolist(),'valid_starts',f['valid_restarts'],flush=True)
    except ValueError as e:
        print(name,str(e),flush=True)
rows,cols=np.where(np.triu(mask,1))
records=json.loads(Path('tmp/bic_convergence_diagnostic/support02_results.json').read_text())[0]['restarts']
bound=np.arctanh(.999999)
bounds=[(-bound,bound)]*4+[(None,None)]*2
for i,record in enumerate(sorted(records,key=lambda s:s['objective'])[:3]):
    y=np.array(record['coordinates'])
    for retry in range(3):
        r=bic.minimize(bic._optimization_gradient,y,args=(sigma,rows,cols),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=1500,ftol=1e-14,gtol=1e-7,maxls=50))
        q,g=bic._optimization_gradient(r.x,sigma,rows,cols)
        g[:4]=r.x[:4]-np.clip(r.x[:4]-g[:4],-bound,bound)
        print('polish',i,retry,'q',q,'pg',np.max(np.abs(g)),'nit',r.nit,'stop',r.message,flush=True)
        y=r.x
