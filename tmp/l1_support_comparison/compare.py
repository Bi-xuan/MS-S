"""Quick matched-model L1 comparison; never changes the saved experiments."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import itertools
import json
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_lyapunov

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "experiments/output/fixed_support_scaling_n4_upper_scope_fit_omega_ref_true_greedy"
OUT = Path(__file__).resolve().parent
P = 4
ER, EC = np.triu_indices(P, 1)
SIGNS = [np.array((*s, 1.)) for s in itertools.product((-1., 1.), repeat=P-1)]
RATIOS = np.r_[0., np.logspace(-7, 0, 29)]
ZERO_TOL = 1e-5


def encode(a):
    edges = a[ER, EC]
    return np.r_[np.abs(np.diag(a)), np.maximum(edges, 0), np.maximum(-edges, 0)]


def decode(x, signs):
    a = np.diag(signs * x[:P])
    a[ER, EC] = x[P:P+6] - x[P+6:]
    return a


def fun_grad(x, sigma, omega, signs, ratio, penalize_diagonal, scale):
    a = decode(x, signs)
    residual = sigma - a.T @ sigma @ a - omega * np.eye(P)
    loss = np.sum(residual**2) / scale
    gradient = -4 * sigma @ a @ residual / scale
    weights = np.r_[np.full(P, int(penalize_diagonal)), np.ones(12)]
    g = np.r_[signs * np.diag(gradient), gradient[ER, EC], -gradient[ER, EC]]
    return loss + ratio * np.dot(weights, x), g + ratio * weights


def projected_gradient(x, gradient):
    g = gradient.copy()
    g[(x <= 1e-10) & (g > 0)] = 0
    g[:P][(x[:P] >= .999999-1e-10) & (g[:P] < 0)] = 0
    return float(np.max(np.abs(g)))


def fit_path(sigma, omega, penalize_diagonal):
    scale = float(np.linalg.norm(sigma, 2)**2)
    r = np.linalg.cholesky(sigma).T
    c = np.linalg.cholesky(sigma - omega*np.eye(P)).T
    initial = [encode(np.linalg.solve(r, np.diag(signs) @ c)) for signs in SIGNS]
    warm = [x.copy() for x in initial]
    bounds = [(0., .999999)]*P + [(0., None)]*12
    matrices, details = [], []
    for ratio in RATIOS:
        candidates = []
        for j, signs in enumerate(SIGNS):
            if ratio == 0:
                x = initial[j].copy()
                value, gradient = fun_grad(x, sigma, omega, signs, ratio, penalize_diagonal, scale)
                candidates.append((value, x, signs, True, projected_gradient(x, gradient), 0))
                continue
            best = None
            seeds = [warm[j], initial[j]]
            for seed in seeds:
                result = minimize(fun_grad, seed, args=(sigma, omega, signs, ratio, penalize_diagonal, scale),
                                  jac=True, method="L-BFGS-B", bounds=bounds,
                                  options=dict(maxiter=1500, ftol=1e-14, gtol=2e-9, maxls=50))
                value, gradient = fun_grad(result.x, sigma, omega, signs, ratio, penalize_diagonal, scale)
                candidate = (value, result.x, signs, bool(result.success), projected_gradient(result.x, gradient), int(result.nit))
                if best is None or value < best[0]:
                    best = candidate
            warm[j] = best[1].copy()
            candidates.append(best)
        value, x, signs, success, pg, nit = min(candidates, key=lambda item: item[0])
        a = decode(x, signs)
        matrices.append(a)
        details.append(dict(ratio=float(ratio), alpha=float(ratio*scale), normalized_penalized_objective=float(value),
                            projected_gradient=pg, optimizer_success=success, iterations=nit,
                            spectral_radius=float(np.max(abs(np.diag(a))))))
    return np.array(matrices), details


def metric(mask, truth):
    tp = int(np.sum(mask & truth)); fp = int(np.sum(mask & ~truth)); fn = int(np.sum(~mask & truth))
    return dict(exact=int(fp+fn == 0), precision=tp/(tp+fp) if tp+fp else 0., recall=tp/int(truth.sum()),
                f1=2*tp/(2*tp+fp+fn), hamming=fp+fn, edges=int(mask.sum()), fp=fp, fn=fn)


def trial(path):
    start = time.perf_counter()
    with np.load(path) as saved:
        data = {key:saved[key].copy() for key in saved.files}
    n = int(data["num_samples"]); seed = int(data["random_seed"]); kappa = float(data["kappa"])
    truth = np.triu(abs(data["Lambda_star"]) > 1e-10, 1)
    # Truth is used only to regenerate this simulator's data and evaluate results.
    population = solve_discrete_lyapunov(data["Lambda_star"].T, float(data["omega_star"])*np.eye(P))
    samples = np.random.default_rng(seed+2).multivariate_normal(np.zeros(P), population, size=n)
    sigma_full = samples.T @ samples/n
    if not np.allclose(sigma_full, data["Sigma"], atol=2e-12, rtol=2e-12):
        raise RuntimeError(f"Regenerated sample covariance does not match saved data: {path}")
    cut = int(.8*n)
    train, validation = samples[:cut], samples[cut:]
    sigma_train = train.T @ train/len(train)
    sigma_validation = validation.T @ validation/len(validation)
    omega_train = kappa*np.linalg.eigvalsh(sigma_train)[0]
    omega_full = float(data["omega_ref"])
    proposed_report = json.loads((path.parent/"selection_plateau_bootstrap.json").read_text())
    proposed_mask = np.zeros((P,P),dtype=bool)
    for i,j in proposed_report["selected_edges"]:
        proposed_mask[i,j] = True
    result = dict(num_samples=n, seed=seed, proposed=metric(proposed_mask, truth), variants={})
    saved_paths = {}
    for variant, penalize_diagonal in [("full_l1", True), ("offdiagonal_l1", False)]:
        train_path, train_details = fit_path(sigma_train, omega_train, penalize_diagonal)
        full_path, full_details = fit_path(sigma_full, omega_full, penalize_diagonal)
        saved_paths[f"{variant}_full_path"] = full_path
        saved_paths[f"{variant}_train_path"] = train_path
        scores, standard_errors = [], []
        for a in train_path:
            fitted_cov = solve_discrete_lyapunov(a.T, omega_train*np.eye(P))
            sign, logdet = np.linalg.slogdet(fitted_cov)
            if sign != 1:
                raise RuntimeError("Non-positive covariance")
            inverse = np.linalg.inv(fitted_cov)
            scores.append(float(logdet + np.trace(inverse @ sigma_validation)))
            values = np.einsum("ij,jk,ik->i",validation,inverse,validation)
            standard_errors.append(float(values.std(ddof=1)/np.sqrt(len(values))))
        scores = np.array(scores)
        cv = int(np.argmin(scores))
        one_se = int(np.flatnonzero(scores <= scores[cv]+standard_errors[cv])[-1])
        masks = np.triu(abs(full_path)>ZERO_TOL, 1)
        path_metrics = [metric(mask, truth) for mask in masks]
        oracle = min(range(len(RATIOS)), key=lambda i: (path_metrics[i]["hamming"], -RATIOS[i]))
        known_size_indices = [i for i,item in enumerate(path_metrics) if item["edges"]==int(truth.sum())]
        # This diagnostic still uses truth to select the best candidate at that size.
        known_size_best = min((path_metrics[i]["hamming"] for i in known_size_indices), default=None)
        result["variants"][variant] = dict(cv_index=cv, one_se_index=one_se, oracle_index=oracle,
            cv=path_metrics[cv], one_se=path_metrics[one_se], oracle=path_metrics[oracle],
            true_support_on_path=any(item["exact"] for item in path_metrics),
            known_size_best_hamming=known_size_best, validation_scores=scores.tolist(),
            validation_standard_errors=standard_errors, train_details=train_details,
            full_details=full_details, path_metrics=path_metrics,
            cv_selected_edges=np.argwhere(masks[cv]).tolist(), cv_lambda=full_path[cv].tolist())
    np.savez_compressed(OUT/f"path_n{n}_seed{seed}.npz", ratios=RATIOS, truth=truth,
                        **saved_paths,
                        **{f"{variant}_cv_lambda":np.array(item["cv_lambda"]) for variant,item in result["variants"].items()})
    result["elapsed_seconds"] = time.perf_counter()-start
    (OUT/f"trial_n{n}_seed{seed}.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def verify_gradient():
    rng = np.random.default_rng(42)
    b=rng.normal(size=(P,P)); sigma=b.T@b+np.eye(P)
    x=rng.uniform(.1,.4,16); signs=SIGNS[2]
    for diagonal in (True, False):
        _, analytic=fun_grad(x,sigma,.5,signs,.03,diagonal,4.)
        numerical=[]
        for j in range(len(x)):
            direction=np.zeros(len(x)); direction[j]=1e-6
            numerical.append((fun_grad(x+direction,sigma,.5,signs,.03,diagonal,4.)[0]-fun_grad(x-direction,sigma,.5,signs,.03,diagonal,4.)[0])/2e-6)
        error=float(np.max(abs(analytic-numerical)))
        assert error < 1e-7,error
        print(f"Gradient verified, penalize diagonal={diagonal}: max error {error:.3g}",flush=True)


def summarize(results):
    rows=[]
    for n in sorted({item["num_samples"] for item in results}):
        group=[item for item in results if item["num_samples"]==n]
        methods={"proposed":[item["proposed"] for item in group]}
        for variant in ("full_l1","offdiagonal_l1"):
            for selector in ("cv","one_se","oracle"):
                methods[f"{variant}_{selector}"]=[item["variants"][variant][selector] for item in group]
        for method, values in methods.items():
            row=dict(num_samples=n, method=method, trials=len(values), exact_count=sum(v["exact"] for v in values))
            row.update({f"mean_{key}":float(np.mean([v[key] for v in values])) for key in ("precision","recall","f1","hamming","edges","fp","fn")})
            rows.append(row)
    with (OUT/"summary.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    settings=dict(input=str(INPUT), ratios=RATIOS.tolist(), zero_tol=ZERO_TOL, split="first 80% train, last 20% validation", variants=["full_l1","offdiagonal_l1"],
                  tuning="minimum held-out Gaussian negative twice log likelihood per observation; transfer alpha/||Sigma_hat||_op^2 to full-data refit",
                  solver="L-BFGS-B with positive/negative variable splitting, eight diagonal sign branches, warm and dense exact starts", summary=rows,
                  max_selected_projected_gradient=max(d["projected_gradient"] for r in results for v in r["variants"].values() for d in v["full_details"]+v["train_details"]))
    (OUT/"results.json").write_text(json.dumps(settings,indent=2)+"\n")
    print(json.dumps(settings,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=4);parser.add_argument("--limit",type=int);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();verify_gradient()
    paths=sorted(INPUT.glob("num_samples_*/seed_*/objective_curve_sigma_hat.npz"))
    if args.limit:paths=paths[:args.limit]
    results=[];pending=[]
    for path in paths:
        n=int(path.parent.parent.name.split("_")[-1]);seed=int(path.parent.name.split("_")[-1]);cached=OUT/f"trial_n{n}_seed{seed}.json"
        if args.resume and cached.exists():results.append(json.loads(cached.read_text()))
        else:pending.append(path)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(trial,path):path for path in pending}
        for future in as_completed(jobs):
            result=future.result();results.append(result)
            print(f"Completed N={result['num_samples']}, seed={result['seed']}, {result['elapsed_seconds']:.1f}s",flush=True)
    summarize(results)
