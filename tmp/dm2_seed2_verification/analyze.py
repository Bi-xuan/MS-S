"""Independently refit every Dm=2 support at the saved fixed omega."""
from pathlib import Path
import sys, json, csv
import numpy as np
from scipy.optimize import least_squares, differential_evolution
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from admm import covariance_from_lambda_star
from optimizers.support_search import solve_support_with_restarts
SOURCE = ROOT/'experiments/output/fixed_support_scaling_n4_all_scope_fit_omega_ref_true_nested_false/num_samples_1000000/seed_2/objective_curve_sigma_hat.npz'
OUT = Path(__file__).resolve().parent
saved = np.load(SOURCE)
S = saved['Sigma']; omega = float(saved['omega_ref']); n = len(S)
truth = {tuple(e) for e in saved['lambda_star_support_edges']}
settings = json.loads(str(saved['fit_settings_json']))
rows = []
rng = np.random.default_rng(20261012)
for p in range(n):
 for q in range(n):
  if p == q: continue
  mask = np.eye(n, dtype=bool); mask[p,q] = True
  positions = np.argwhere(mask)
  def unpack(x):
   L = np.zeros((n,n)); L[mask] = x; return L
  def residual(x):
   L = unpack(x); return (S-L.T@S@L-omega*np.eye(n)).ravel()
  def jac(x):
   SL = S@unpack(x); J = []
   for i,j in positions:
    d = np.zeros((n,n)); d[j,:] = SL[i,:]; d[:,j] += SL[i,:]; J.append(-d.ravel())
   return np.array(J).T
  native = solve_support_with_restarts(S, mask, omega_upper=float(np.linalg.eigvalsh(S)[0]), **settings)
  starts = [native[0][mask], saved['Lambda_star'][mask]]
  starts += [rng.uniform(-1.5,1.5,len(positions)) for _ in range(126)]
  fits = [least_squares(residual,x,jac=jac,ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=2000) for x in starts]
  best = min(fits,key=lambda f: f.fun@f.fun)
  C = float(best.fun@best.fun)
  # Every point beating C must lie in this box, by its diagonal residuals.
  bounds = []
  sq = float(S[q,q]-omega+np.sqrt(C))
  principal = S[np.ix_([p,q],[p,q])]
  for i,j in positions:
   if j != q: b = np.sqrt((S[j,j]-omega+np.sqrt(C))/S[j,j])
   else:
    idx = 0 if i == p else 1
    b = np.sqrt(sq*np.linalg.inv(principal)[idx,idx])
   bounds.append((-float(b),float(b)))
  de = differential_evolution(lambda x: float(residual(x)@residual(x)), bounds, seed=20261012, popsize=25,maxiter=1500,tol=1e-10,atol=1e-13,polish=False)
  refined = least_squares(residual,de.x,jac=jac,ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=2000)
  if refined.fun@refined.fun < C: best = refined; C = float(best.fun@best.fun)
  row = dict(edge=[p,q],true_edge=(p,q) in truth,objective=C,native_admm_objective=float(native[2]),native_gradient_norm=float(np.linalg.norm(2*jac(native[0][mask]).T@residual(native[0][mask]))),gradient_norm=float(np.linalg.norm(2*best.jac.T@best.fun)),multistart_best_count=int(sum(abs(f.fun@f.fun-C)<1e-10 for f in fits)),num_starts=len(fits),de_objective=float(refined.fun@refined.fun),de_success=bool(de.success),Lambda=unpack(best.x).tolist(),native_Lambda=native[0].tolist(),bounds=bounds)
  rows.append(row)
  print(row['edge'], 'LS',C,'ADMM',native[2],'DE',row['de_objective'],'best starts',row['multistart_best_count'],flush=True)
rows.sort(key=lambda r:r['objective'])
result = dict(source=str(SOURCE),omega_ref=omega,saved_Dm2_objective=float(saved['objective_values'][1]),selected_edge=np.argwhere(saved['selected_support_masks'][1]&~np.eye(n,dtype=bool)).tolist(),settings=settings,rows=rows)
(OUT/'results.json').write_text(json.dumps(result,indent=2))
with (OUT/'comparison.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['edge','true_edge','objective','native_admm_objective','gradient_norm','multistart_best_count','de_objective'])
 writer.writeheader(); writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
print('Saved',OUT/'results.json',flush=True)
