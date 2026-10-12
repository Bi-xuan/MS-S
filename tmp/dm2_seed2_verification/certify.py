"""Global objective lower bounds using a PSD quadratic dual certificate.

For arbitrary symmetric Y and residual R(L)=S-L.T S L-wI:
  f(L) = ||R(L)-Y||_F^2 + x.T H(Y) x + gamma(Y),
where L=sum x_k E_k, H_kl=-2 tr(Y E_k.T S E_l),
and gamma=2 <Y,S-wI>-||Y||_F^2. If H is PSD then f>=gamma
for ALL real coefficients x on the support, independently of local fitting.
"""
from pathlib import Path
import json, numpy as np
OUT=Path(__file__).resolve().parent
result=json.loads((OUT/'results.json').read_text())
S=np.load(result['source'])['Sigma']; n=len(S); w=result['omega_ref']
T=S-w*np.eye(n); smin=float(np.linalg.eigvalsh(S)[0])
certificates=[]
for row in result['rows']:
 mask=np.eye(n,dtype=bool); mask[tuple(row['edge'])]=True
 basis=[]
 for i,j in np.argwhere(mask):
  E=np.zeros((n,n)); E[i,j]=1.; basis.append(E)
 def quadratic(Y):
  H=np.array([[-2*np.trace(Y@E.T@S@F) for F in basis] for E in basis])
  return (H+H.T)/2
 L=np.array(row['Lambda']); R=T-L.T@S@L
 min_eig=float(np.linalg.eigvalsh(quadratic(R))[0])
 # Ensure strict positive definiteness with a margin far above rounding noise.
 epsilon=max(1e-9,(-min_eig+2e-9)/(2*smin))
 Y=R-epsilon*np.eye(n); H=quadratic(Y)
 eig=np.linalg.eigvalsh(H); chol=np.linalg.cholesky(H)
 gamma=float(2*np.sum(Y*T)-np.sum(Y*Y))
 error=float(np.linalg.norm(H-chol@chol.T,2))
 certificate=dict(edge=row['edge'],objective_upper_bound=row['objective'],objective_lower_bound=gamma,gap=row['objective']-gamma,min_eigenvalue=float(eig[0]),cholesky_error=error,epsilon=epsilon,Y=Y.tolist(),H=H.tolist())
 certificates.append(certificate)
 print(row['edge'], 'lower',gamma, 'upper',row['objective'],'gap',certificate['gap'],'mineig',eig[0],flush=True)
result['global_lower_bound_certificates']=certificates
(OUT/'results.json').write_text(json.dumps(result,indent=2))
with (OUT/'README.md').open('w') as f:
 f.write('# Dm=2, seed 2: reverse-edge verification\n\n')
 f.write('All edge indices are zero-based. Dm=2 allows all four diagonal coefficients and one directed off-diagonal coefficient. The fixed omega is '+format(w,'.16g')+'.\n\n')
 f.write('| Edge | True edge | Saved-settings ADMM | Independent refit | Global lower bound |\n|---|---|---:|---:|---:|\n')
 for row,c in zip(result['rows'],certificates):
  f.write(f"| {tuple(row['edge'])} | {row['true_edge']} | {row['native_admm_objective']:.12g} | {row['objective']:.12g} | {c['objective_lower_bound']:.12g} |\n")
 f.write('\nThe saved support (3,2) wins. Refitting improves its objective from 0.008735911473584467 to 0.008633679508414347. Each true-edge support has a certified lower bound above even the original saved objective. The smallest other-support lower bound exceeds the winning feasible objective, so numerical inaccuracy cannot explain the support choice. The winning-support upper/lower gap is approximately 6e-9 (see results.json for the exact gap).\n\n')
 f.write('## Independent optimization\n\nEach support used 128 least-squares starts with an analytic Jacobian (tolerances 1e-13), plus differential evolution over a box guaranteed to contain every point beating the best fit. The box follows from bounding every diagonal residual by sqrt(best objective). Differential evolution can miss individual local optima; its result is recorded separately.\n\n')
 f.write('## Global lower bound\n\nFor T=Sigma-omega I, any symmetric Y, and L=sum_k x_k E_k,\n\n')
 f.write('`||T-L^T Sigma L||_F^2 = ||T-L^T Sigma L-Y||_F^2 + x^T H(Y) x + gamma(Y)`,\n\n')
 f.write('where `H_kl=-2 trace(Y E_k^T Sigma E_l)` and `gamma(Y)=2 <Y,T>-||Y||_F^2`. A positive semidefinite H therefore gives a lower bound gamma over every possible coefficient vector, with no local-optimization assumption. We choose Y as the fitted residual minus epsilon I. This raises H by a positive definite matrix and provides a strictly positive eigenvalue margin. The complete Y and H matrices, eigenvalues, and Cholesky residuals are saved in results.json. These certificates were checked in float64, not formal interval/exact arithmetic.\n\n')
 f.write('Run `python tmp/dm2_seed2_verification/analyze.py` followed by `python tmp/dm2_seed2_verification/certify.py` from the project root. The original experiment artifacts are unchanged.\n')
