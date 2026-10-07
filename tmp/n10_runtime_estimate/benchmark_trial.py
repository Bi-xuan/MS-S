"""One end-to-end timing trial with default statistical settings."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, "/Users/liubixuan/Documents/LPSM/model_selection/lambda-support-recovery/src")
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from lambda_support_recovery import select_support

def main():
    truth = np.diag(np.linspace(.1, .55, 10))
    values = np.linspace(.60, -.45, 9)
    truth[:-1, -1] = np.copysign(np.clip(np.abs(values), .2, .6), values)
    population = solve_discrete_lyapunov(truth.T, np.eye(10))
    x = np.random.default_rng(126).multivariate_normal(np.zeros(10), population, size=100)
    sigma = x.T @ x / 100
    start = time.perf_counter()
    events = []
    def progress(message):
        event = dict(seconds=time.perf_counter() - start, message=message)
        events.append(event)
        print(json.dumps(event), flush=True)
    result = select_support(sigma, num_samples=100, support_scope="upper", n_jobs=5,
                            return_result=True, progress=progress)
    record = dict(num_samples=100, n_jobs=5, total_seconds=time.perf_counter()-start,
                  selected_dimension=result.selected_dimension,
                  comparisons=len(result.bootstrap_comparisons), events=events)
    Path(__file__).with_name("trial_timing.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k:v for k,v in record.items() if k != "events"}), flush=True)

if __name__ == "__main__":
    main()
