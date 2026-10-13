from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from optimizers import bic
p=Path('experiments/output/upp_scal_n4_dm4_upper_scope_fit_omega_true_greedy_oracle_lasso/support_02/num_samples_100/seed_124/objective_curve_sigma_hat.npz')
with np.load(p) as d:
    sigma=d['Sigma']; ref=float(d['omega_ref'])
original=bic._profile
mask=np.eye(4,dtype=bool);mask[0,1]=mask[0,2]=True
for fixed in (1.,ref,float(np.linalg.eigvalsh(sigma)[0])):
    def fixed_profile(matrix,sigma):
        _,_,unit,inverse=original(matrix,sigma)
        logdet=np.linalg.slogdet(unit)[1]
        value=float(logdet+len(matrix)*np.log(fixed)+np.sum(inverse*sigma)/fixed)
        return value,fixed,unit,inverse
    bic._profile=fixed_profile
    f=bic.fit_support(sigma,mask)
    print('fixed omega',fixed,'support q',f['objective_value'],'pg',f['projected_gradient'],'max |diag|',np.max(np.abs(np.diag(f['fitted_lambda']))),'valid restarts',f['valid_restarts'],flush=True)
    if fixed==ref:
        r=bic.select_bic(sigma,100)
        print('all 64 fixed-reference supports converged; selected',r['support_edges'].tolist(),'max pg',r['candidate_projected_gradients'].max(),'max |diag|',np.max(np.abs(np.diagonal(r['candidate_fitted_lambdas'],axis1=1,axis2=2))),flush=True)
        np.savez_compressed('tmp/bic_convergence_diagnostic/fixed_omega_result.npz',**r)
