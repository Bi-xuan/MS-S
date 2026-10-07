#!/usr/bin/env bash
# Fixed-support study using lambda_support_recovery's default estimator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

PYTHON="${PYTHON:-python}"
N="${N:-10}"
SUPPORT_SCOPE="${SUPPORT_SCOPE:-upper}"
TOTAL_CPUS="${TOTAL_CPUS:-${SLURM_CPUS_PER_TASK:-${NSLOTS:-100}}}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-5}"
LAMBDA_STAR_OFFDIAG_ABS_MIN="${LAMBDA_STAR_OFFDIAG_ABS_MIN:-0.20}"
LAMBDA_STAR_OFFDIAG_ABS_MAX="${LAMBDA_STAR_OFFDIAG_ABS_MAX:-0.60}"
OMEGA_STAR="${OMEGA_STAR:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/output/fixed_support_scaling_n${N}_package_fit_omega_ref_true}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/experiments/output/.matplotlib}"
COMMAND="${1:-run-all}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [run-all | summarize [OUTPUT_FOLDER]]

  run-all    Run 40 package-based trials, then write the summary and plots.
  summarize  Regenerate the summary and plots from completed package trials.

Defaults: N=10, SUPPORT_SCOPE=upper, TOTAL_CPUS=100, CPUS_PER_TRIAL=5.
Sample sizes: 100, 1000, 10000, 1000000; ten seeds per sample size.
Set PYTHON to a Python 3.10+ executable with requirements.txt installed.
OUTPUT_ROOT defaults to experiments/output/fixed_support_scaling_n\${N}_package_fit_omega_ref_true.
Data-generation options: OMEGA_STAR (1), LAMBDA_STAR_OFFDIAG_ABS_MIN (0.20),
LAMBDA_STAR_OFFDIAG_ABS_MAX (0.60). These do not override estimator defaults.

Only support_scope, n_jobs, return_result, and progress are configured in
select_support(). All statistical settings, including estimated fixed omega,
10 restarts, 800 iterations, 199 bootstrap replicates, and RNG seeds, retain
the installed package's defaults. Trial seeds control observation sampling.
Existing trial files in OUTPUT_ROOT are recomputed by run-all.
EOF
}

case "${COMMAND}" in
    run-all) (($# <= 1)) || { usage >&2; exit 2; } ;;
    summarize)
        (($# <= 2)) || { usage >&2; exit 2; }
        OUTPUT_ROOT="${2:-${OUTPUT_ROOT}}"
        ;;
    -h|--help|help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [[ "${SUPPORT_SCOPE}" != "upper" ]]; then
    echo "Error: this summary/plot workflow requires SUPPORT_SCOPE=upper." >&2
    exit 2
fi

if [[ "${COMMAND}" == "run-all" ]]; then
    if [[ ! "${N}" =~ ^[1-9][0-9]*$ ]] || ((N < 3)); then
        echo "Error: N must be at least 3 so the study has true and false upper edges." >&2
        exit 2
    fi
    if [[ ! "${TOTAL_CPUS}" =~ ^[1-9][0-9]*$ || ! "${CPUS_PER_TRIAL}" =~ ^[1-9][0-9]*$ ]] \
        || ((TOTAL_CPUS < CPUS_PER_TRIAL)); then
        echo "Error: TOTAL_CPUS >= CPUS_PER_TRIAL >= 1 is required (integer values)." >&2
        exit 2
    fi
    # Check dependencies and data-generation settings before launching workers.
    "${PYTHON}" experiments/run_fixed_support_scaling_package_trial.py \
        --n "${N}" --num-samples 100 --seed 124 --n-jobs "${CPUS_PER_TRIAL}" \
        --offdiag-abs-min "${LAMBDA_STAR_OFFDIAG_ABS_MIN}" \
        --offdiag-abs-max "${LAMBDA_STAR_OFFDIAG_ABS_MAX}" \
        --omega-star "${OMEGA_STAR}" --check-only
fi

NUM_SAMPLE_VALUES=(100 1000 10000 1000000)
task_num_samples=()
task_seeds=()
for num_samples in "${NUM_SAMPLE_VALUES[@]}"; do
    case "${num_samples}" in
        100) seeds=(124 139 141 147 153 156 173 192 237 243) ;;
        1000) seeds=(6 15 23 46 51 53 59 61 74 85) ;;
        10000) seeds=(3 4 6 8 12 13 18 21 22 27) ;;
        1000000) seeds=(2 3 124 139 147 153 156 173 192 237) ;;
    esac
    for random_seed in "${seeds[@]}"; do
        task_num_samples+=("${num_samples}")
        task_seeds+=("${random_seed}")
    done
done

run_trial() {
    local num_samples="$1" random_seed="$2"
    local trial_dir="${OUTPUT_ROOT}/num_samples_${num_samples}/seed_${random_seed}"
    mkdir -p "${trial_dir}"
    echo "=== package trial: n=${N}; num_samples=${num_samples}; seed=${random_seed}; workers=${CPUS_PER_TRIAL} ==="
    "${PYTHON}" -u experiments/run_fixed_support_scaling_package_trial.py \
        --n "${N}" --num-samples "${num_samples}" --seed "${random_seed}" \
        --n-jobs "${CPUS_PER_TRIAL}" --output-dir "${trial_dir}" \
        --offdiag-abs-min "${LAMBDA_STAR_OFFDIAG_ABS_MIN}" \
        --offdiag-abs-max "${LAMBDA_STAR_OFFDIAG_ABS_MAX}" \
        --omega-star "${OMEGA_STAR}" \
        2>&1 | tee "${trial_dir}/selection_plateau_bootstrap.log"
}

if [[ "${COMMAND}" == "run-all" ]]; then
    mkdir -p "${OUTPUT_ROOT}"
    worker_count=$((TOTAL_CPUS / CPUS_PER_TRIAL))
    if ((worker_count > ${#task_seeds[@]})); then
        worker_count="${#task_seeds[@]}"
    fi
    echo "Running ${#task_seeds[@]} trials: ${worker_count} concurrent trials x ${CPUS_PER_TRIAL} workers."
    worker_pids=()
    for ((worker = 0; worker < worker_count; worker++)); do
        (
            for ((task = worker; task < ${#task_seeds[@]}; task += worker_count)); do
                run_trial "${task_num_samples[task]}" "${task_seeds[task]}"
            done
        ) &
        worker_pids+=("$!")
    done
    worker_failed=0
    for worker_pid in "${worker_pids[@]}"; do
        if ! wait "${worker_pid}"; then
            worker_failed=1
        fi
    done
    if ((worker_failed)); then
        echo "Error: at least one package trial failed; summary and plots were skipped. See per-trial logs." >&2
        exit 1
    fi
fi

"${PYTHON}" experiments/summarize_fixed_support_scaling.py \
    "${OUTPUT_ROOT}" --sample-sizes "${NUM_SAMPLE_VALUES[@]}" --expected-seeds 10
"${PYTHON}" experiments/plot_edge_recovery_heatmap.py --input-dir "${OUTPUT_ROOT}"
"${PYTHON}" experiments/plot_roc_curve.py --input-dir "${OUTPUT_ROOT}"
echo "Results, summary, and plots: ${OUTPUT_ROOT}"
