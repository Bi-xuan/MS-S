# Dm=2, seed 2: reverse-edge verification

All edge indices are zero-based. Dm=2 allows all four diagonal coefficients and one directed off-diagonal coefficient. The fixed omega is 0.9358870179996589.

| Edge | True edge | Saved-settings ADMM | Independent refit | Global lower bound |
|---|---|---:|---:|---:|
| (3, 2) | False | 0.00873591147358 | 0.00863367950841 | 0.0086336751681 |
| (2, 3) | True | 0.00917593491353 | 0.00908116841567 | 0.00908116408655 |
| (3, 0) | False | 0.0637767725333 | 0.0586340992358 | 0.0586340947888 |
| (3, 1) | False | 0.064633415144 | 0.0594662764242 | 0.0594662719719 |
| (0, 3) | True | 0.0646616352343 | 0.0595001385557 | 0.0595001340982 |
| (1, 3) | True | 0.0651470199107 | 0.0599868177601 | 0.0599868133038 |
| (0, 2) | False | 0.0683141883009 | 0.0623672002062 | 0.0623671957487 |
| (1, 2) | False | 0.0685758776786 | 0.0627383171876 | 0.0627383127267 |
| (2, 0) | False | 0.0685994147493 | 0.0627591378188 | 0.0627591333571 |
| (2, 1) | False | 0.068677636867 | 0.0629109050251 | 0.0629109005626 |
| (0, 1) | False | 0.0685731556029 | 0.063198584787 | 0.0617086623217 |
| (1, 0) | False | 0.0685902364055 | 0.0632378284075 | 0.0632378239436 |

The saved support (3,2) wins. Refitting improves its objective from 0.008735911473584467 to 0.008633679508414347. Each true-edge support has a certified lower bound above even the original saved objective. The smallest other-support lower bound exceeds the winning feasible objective, so numerical inaccuracy cannot explain the support choice. The winning-support upper/lower gap is approximately 4.4e-9 (see results.json for the exact gap).

## Independent optimization

Each support used 128 least-squares starts with an analytic Jacobian (tolerances 1e-13), plus differential evolution over a box guaranteed to contain every point beating the best fit. The box follows from bounding every diagonal residual by sqrt(best objective). Differential evolution can miss individual local optima; its result is recorded separately.

## Global lower bound

For T=Sigma-omega I, any symmetric Y, and L=sum_k x_k E_k,

`||T-L^T Sigma L||_F^2 = ||T-L^T Sigma L-Y||_F^2 + x^T H(Y) x + gamma(Y)`,

where `H_kl=-2 trace(Y E_k^T Sigma E_l)` and `gamma(Y)=2 <Y,T>-||Y||_F^2`. A positive semidefinite H therefore gives a lower bound gamma over every possible coefficient vector, with no local-optimization assumption. We choose Y as the fitted residual minus epsilon I. This raises H by a positive definite matrix and provides a strictly positive eigenvalue margin. The complete Y and H matrices, eigenvalues, and Cholesky residuals are saved in results.json. These certificates were checked in float64, not formal interval/exact arithmetic.

Run `python tmp/dm2_seed2_verification/analyze.py` followed by `python tmp/dm2_seed2_verification/certify.py` from the project root. The original experiment artifacts are unchanged.
