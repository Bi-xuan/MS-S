from pathlib import Path
import sys, json
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from admm import covariance_from_lambda_star
from objective import frobenius_objective
from optimizers.support_search import solve_support_with_restarts
from supports.exact import get_upper_triangular_supports
np.set_printoptions(precision=10,suppress=True)
p=ROOT/'experiments/output/upp_scal_n4_dm4_nsm10000_omega1_minabs02_10seeds_PlaAbs/support_18/seed_6/objective_curve_sigma_hat.npz'
a=np.load(p)
S=a['Sigma']; Lstar=a['Lambda_star']; pop=covariance_from_lambda_star(Lstar,1.)
true=Lstar!=0; selected=a['selected_support_masks'][3]
rng=np.random.default_rng(20260910)
results={}
for name,T in [('empirical',S),('population',pop)]:
 rows=[]
 for index,mask in enumerate(get_upper_triangular_supports(4,3)):
  positions=np.argwhere(mask)
  def unpack(x):
   L=np.zeros((4,4)); L[mask]=x; return L
  def fun(x):
   L=unpack(x); return (T-L.T@T@L-np.eye(4)).ravel()
  def jac(x):
   L=unpack(x); TL=T@L
   J=[]
   for i,j in positions:
    d=np.zeros((4,4)); d[j,:]=TL[i,:]; d[:,j]+=TL[i,:]
    J.append(-d.ravel())
   return np.array(J).T
  starts=[Lstar[mask]]+[rng.uniform(-1,1,mask.sum()) for _ in range(31)]
  fits=[least_squares(fun,x,jac=jac,ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=3000) for x in starts]
  best=min(fits,key=lambda f: np.dot(f.fun,f.fun))
  row=dict(index=index,edges=np.argwhere(mask & ~np.eye(4,dtype=bool)).tolist(),objective=float(best.fun@best.fun),Lambda=unpack(best.x).tolist(),gradient=float(np.linalg.norm(2*best.jac.T@best.fun)),best_count=int(sum(abs(f.fun@f.fun-best.fun@best.fun)<1e-10 for f in fits)))
  rows.append(row)
 rows.sort(key=lambda r:r['objective'])
 results[name]=rows
 print(name,'best supports',[(r['index'],r['objective']) for r in rows[:5]],flush=True)
 for label,mask in [('true',true),('selected',selected)]:
  row=next(r for r in rows if np.array_equal(np.array(r['edges']),np.argwhere(mask & ~np.eye(4,dtype=bool))))
  print(label,json.dumps(row),flush=True)
  sol=solve_support_with_restarts(T,mask,beta=1.,max_iter=800,tol=1e-7,zero_tol=1e-5,max_restarts=10,min_omega=0.,omega_fixed=1.,omega_upper=np.linalg.eigvalsh(T)[0]-1e-6)
  print('ADMM',label,sol[2], '\n',sol[0],flush=True)
  results[name+'_'+label+'_admm']={'objective':sol[2],'Lambda':sol[0].tolist()}
print('plug-in objective',frobenius_objective(S,Lstar,1.))
print('sample eigenvalues',np.linalg.eigvalsh(S))
C=float(a['objective_values'][3]); t=np.sqrt(C)
b1=np.sqrt((S[0,0]-1+t)/S[0,0]); b2=np.sqrt((S[1,1]-1+t)/S[1,1]); lower=2*S[0,1]**2*(1-b1*b2)**2
print('certificate',{'candidate':C,'conditional_true_lower_bound':lower,'diag_bounds':[b1,b2]})
results['certificate']={'candidate':C,'conditional_true_lower_bound':lower,'diag_bounds':[b1,b2]}
(ROOT/'tmp/support_18_seed_6_analysis/results.json').write_text(json.dumps(results,indent=2))
