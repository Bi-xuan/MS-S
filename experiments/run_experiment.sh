#!/usr/bin/env bash
# Run the default synthetic experiment with single-threaded BLAS settings.
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"
N="${N:-4}"
REFINE_AFTER_FIXED_OMEGA="${REFINE_AFTER_FIXED_OMEGA:-false}"
OMEGA_REF="${OMEGA_REF:-1.0}"

python experiments/compute_objective_curve.py \
    --n-jobs "${N_JOBS}" \
    --random-seed "${RANDOM_SEED}" \
    --max-restarts "${MAX_RESTARTS}" \
    --refine-after-fixed-omega "${REFINE_AFTER_FIXED_OMEGA}" \
    --omega-ref "${OMEGA_REF}" \
    --lambda-star-dims "${N}"
# python experiments/run_synthetic.py \
#     --n-jobs "${N_JOBS}" \
#     --random-seed "${RANDOM_SEED}" \
#     --max-restarts "${MAX_RESTARTS}" \
#     --refine-after-fixed-omega "${REFINE_AFTER_FIXED_OMEGA}" \
#     --omega-ref "${OMEGA_REF}"
