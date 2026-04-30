#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_JOBS="${N_JOBS:-4}"
RANDOM_SEED="${RANDOM_SEED:-42}"

# python main.py --n-jobs "${N_JOBS}" --random-seed "${RANDOM_SEED}"
python plot_objective_vs_dm.py --n-jobs "${N_JOBS}" --random-seed "${RANDOM_SEED}"
