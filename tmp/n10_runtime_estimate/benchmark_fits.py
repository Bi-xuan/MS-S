"""Small timing pilot; no changes to the estimator or experiment outputs."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import json
import platform
import sys
import time
from pathlib import Path
sys.path.insert(0, "/Users/liubixuan/Documents/LPSM/model_selection/lambda-support-recovery/src")
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from lambda_support_recovery.optimizers.support_search import solve_support_with_restarts

n = 10
truth = np.diag(np.linspace(.1, .55, n))
values = np.linspace(.60, -.45, n-1)
truth[:-1, -1] = np.copysign(np.clip(np.abs(values), .2, .6), values)
population = solve_discrete_lyapunov(truth.T, np.eye(n))
edges = [(i, j) for i in range(n) for j in range(i+1, n)]
rng_masks = np.random.default_rng(42)
order = rng_masks.permutation(len(edges))
records = []
print(json.dumps({"python": sys.version, "architecture": platform.machine(), "numpy": np.__version__, "cpus": os.cpu_count()}), flush=True)
for samples in (100, 1000, 10000, 1000000):
    start = time.perf_counter()
    x = np.random.default_rng(126).multivariate_normal(np.zeros(n), population, size=samples)
    sigma = x.T @ x / samples
    del x
    sampling_seconds = time.perf_counter() - start
    omega = .93 * np.linalg.eigvalsh(sigma)[0]
    for count in (0, 1, 9, 22, 36, 45):
        mask = np.eye(n, dtype=bool)
        for edge_id in order[:count]:
            mask[edges[edge_id]] = True
        start = time.perf_counter()
        fit = solve_support_with_restarts(sigma, mask, beta=1., max_iter=800, tol=1e-7,
                                          zero_tol=1e-5, max_restarts=10, omega_fixed=omega)
        seconds = time.perf_counter() - start
        record = dict(num_samples=samples, edges=count, seconds=seconds,
                      objective=None if fit is None else float(fit[2]), sampling_seconds=sampling_seconds)
        records.append(record)
        print(json.dumps(record), flush=True)
Path(__file__).with_name("fit_timings.json").write_text(json.dumps(records, indent=2))
