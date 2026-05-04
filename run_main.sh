#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
RANDOM_SEED="${RANDOM_SEED:-42}"
N="${N:-4}"

# python main.py --n-jobs "${N_JOBS}" --random-seed "${RANDOM_SEED}"
# python compute_objective_curve.py --n-jobs "${N_JOBS}" --random-seed "${RANDOM_SEED}" --lambda-star-dims "${N}"
python main.py --n-jobs "${N_JOBS}" --random-seed "${RANDOM_SEED}"