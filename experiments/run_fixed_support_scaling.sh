#!/usr/bin/env bash
# Evaluate one fixed Lambda_star across sample sizes with nested support recovery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N="${N:-4}"
LAMBDA_STAR_OFFDIAG_ABS_MIN="${LAMBDA_STAR_OFFDIAG_ABS_MIN:-0.20}"
LAMBDA_STAR_OFFDIAG_ABS_MAX="${LAMBDA_STAR_OFFDIAG_ABS_MAX:-0.60}"
TOTAL_CPUS="${TOTAL_CPUS:-${SLURM_CPUS_PER_TASK:-${NSLOTS:-100}}}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-5}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"
REFINE_AFTER_FIXED_OMEGA="${REFINE_AFTER_FIXED_OMEGA:-false}"
OMEGA_STAR="${OMEGA_STAR:-1}"
OMEGA_REF="${OMEGA_REF:-1}"
FIT_OMEGA_REF="${FIT_OMEGA_REF:-false}"
KAPPA="${KAPPA:-0.93}"
SUPPORT_SCOPE="${SUPPORT_SCOPE:-upper}"
NESTED_SUPPORTS="${NESTED_SUPPORTS:-true}"
OBJECTIVE_FLOOR="${OBJECTIVE_FLOOR:-1e-8}"
TOP_PLATEAUS="${TOP_PLATEAUS:-3}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-199}"
BOOTSTRAP_ALPHA="${BOOTSTRAP_ALPHA:-0.05}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260913}"
COMMAND="${1:-run-all}"
if [[ "${COMMAND}" == "reselect" ]]; then
    OUTPUT_ROOT="${2:-${OUTPUT_ROOT:-experiments/output/fixed_support_scaling_n4_omega_ref_eq_star}}"
else
    OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/output/fixed_support_scaling_n${N}}"
fi
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-${NSLOTS:-8}}}"
RESELECT_ROOT="${OUTPUT_ROOT}/reselect_lm_support_count_plateau"
if [[ "${COMMAND}" == "reselect" ]]; then
    TRIALS_PATH="${RESELECT_ROOT}/selection_trials.csv"
    TABLE_PATH="${RESELECT_ROOT}/selection_summary.md"
else
    TRIALS_PATH="${OUTPUT_ROOT}/selection_trials.csv"
    TABLE_PATH="${OUTPUT_ROOT}/selection_summary.md"
fi
NUM_SAMPLE_VALUES=(100 1000 10000 1000000)

if [[ ! "${N}" =~ ^[0-9]+$ ]] || ((N < 2)); then
    echo "Error: N must be an integer of at least 2." >&2
    exit 2
fi
if [[ "${COMMAND}" != "reselect" ]] &&
    { [[ ! "${TOTAL_CPUS}" =~ ^[0-9]+$ || ! "${CPUS_PER_TRIAL}" =~ ^[0-9]+$ ]] \
    || ((TOTAL_CPUS < CPUS_PER_TRIAL || CPUS_PER_TRIAL < 1)); }; then
    echo "Error: TOTAL_CPUS and CPUS_PER_TRIAL must be positive integers, with TOTAL_CPUS >= CPUS_PER_TRIAL." >&2
    exit 2
fi

usage() {
    cat <<EOF
Usage: $(basename "$0") [run-all | reselect [OUTPUT_FOLDER]]

  Set LAMBDA_STAR_OFFDIAG_ABS_MIN and LAMBDA_STAR_OFFDIAG_ABS_MAX
  to bound the absolute values of nonzero Lambda_star off-diagonal entries.
  Defaults: 0.20 and 0.60. Example:
    LAMBDA_STAR_OFFDIAG_ABS_MIN=0.30 LAMBDA_STAR_OFFDIAG_ABS_MAX=0.50 OUTPUT_ROOT=experiments/output/custom_bounds bash experiments/run_fixed_support_scaling.sh

  run-all   Compute curves, select dimensions, and summarize (default).
  reselect  Discover saved num_samples_*/seed_* curves under OUTPUT_ROOT,
            select with Lm equal to the number of available supports at each
            dimension, and summarize. Uses up to N_JOBS concurrent trials,
            with one curve-only selection per trial. Does not recompute curves.
            Writes selections and summaries under OUTPUT_ROOT/reselect_lm_support_count_plateau.
            Default folder: experiments/output/fixed_support_scaling_n4_omega_ref_eq_star
EOF
}

if [[ "${COMMAND}" == "run-all" ]]; then
    mkdir -p "${OUTPUT_ROOT}"
fi

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
    local num_samples="$1"
    local random_seed="$2"
    local selection_only="${3:-false}"
    local trial_dir="${OUTPUT_ROOT}/num_samples_${num_samples}/seed_${random_seed}"
    local result_dir="${trial_dir}"
    local curve_path="${trial_dir}/objective_curve_sigma_hat.npz"
    local selection_log="${trial_dir}/selection_plateau_bootstrap.log"
    local selection_json="${trial_dir}/selection_plateau_bootstrap.json"
    local selection_method="plateau-bootstrap"
    local selected_dimension
    local precision
    local exact=false
    local selection_options=()
    local selection_jobs="${CPUS_PER_TRIAL}"

    if [[ "${selection_only}" == "true" ]]; then
        [[ -f "${curve_path}" ]] || { echo "Missing saved curve: ${curve_path}" >&2; return 1; }
        result_dir="${RESELECT_ROOT}/num_samples_${num_samples}/seed_${random_seed}"
        mkdir -p "${result_dir}"
        selection_options+=(--lm-mode support-count)
        selection_log="${result_dir}/selection_plateau.log"
        selection_json="${result_dir}/selection_plateau.json"
        selection_method="plateau"
        selection_jobs=1
        echo "Reusing saved objective curve: ${curve_path}"
    else
        mkdir -p "${trial_dir}"
        echo
        echo "=== n=${N}; num_samples=${num_samples}; seed=${random_seed}; CPUs=${CPUS_PER_TRIAL} ==="
        python experiments/compute_objective_curve.py \
            --curve sigma_hat \
            --sigma-hat-output "${curve_path}" \
            --lambda-star-dims "${N}" \
            --lambda-star-offdiag-abs-min "${LAMBDA_STAR_OFFDIAG_ABS_MIN}" \
            --lambda-star-offdiag-abs-max "${LAMBDA_STAR_OFFDIAG_ABS_MAX}" \
            --nested-supports "${NESTED_SUPPORTS}" \
            --support-scope "${SUPPORT_SCOPE}" \
            --num-samples "${num_samples}" \
            --n-jobs "${CPUS_PER_TRIAL}" \
            --random-seed "${random_seed}" \
            --max-restarts "${MAX_RESTARTS}" \
            --refine-after-fixed-omega "${REFINE_AFTER_FIXED_OMEGA}" \
            --omega-star "${OMEGA_STAR}" \
            --omega-ref "${OMEGA_REF}" \
            --fit-omega-ref "${FIT_OMEGA_REF}" \
            --kappa "${KAPPA}" \
            2>&1 | tee "${trial_dir}/compute.log"
        selection_options+=(--top-plateaus "${TOP_PLATEAUS}"
            --bootstrap-replicates "${BOOTSTRAP_REPLICATES}"
            --bootstrap-alpha "${BOOTSTRAP_ALPHA}"
            --bootstrap-seed "${BOOTSTRAP_SEED}")
    fi

    python experiments/select_scaling_parameter.py \
            "${curve_path}" \
            --method "${selection_method}" \
            --objective-floor "${OBJECTIVE_FLOOR}" \
            --n-jobs "${selection_jobs}" \
            "${selection_options[@]}" \
            --output-json "${selection_json}" \
            2>&1 | tee "${selection_log}"

    selected_dimension="$(awk -F ': ' '$1 == "Selected dimension" || $1 == "Selected dimension at recommended scale" {print $2}' "${selection_log}" | tail -n 1)"
    if [[ ! "${selected_dimension}" =~ ^[0-9]+$ ]]; then
        echo "Error: could not parse selected dimension from ${selection_log}" >&2
        return 1
    fi
    precision="$(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["precision"] if p["precision"] is not None else 0)' "${selection_json}")"
    precision="${precision:-0}"
    if [[ "${selected_dimension}" == "${N}" ]] && awk -v p="${precision}" 'BEGIN {exit !(p == 1)}'; then
        exact=true
    fi
    printf '%s,%s,%s,%s,%s\n' \
        "${num_samples}" "${random_seed}" "${selected_dimension}" "${precision}" "${exact}" \
        > "${result_dir}/result.csv"
}

reselect_all() {
    local trial_dir sample_dir num_samples random_seed worker_count worker task worker_pid
    local worker_failed=0 expected_seeds=0
    local trial_dirs=() sample_sizes=() worker_pids=()

    if [[ ! "${N_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: N_JOBS must be a positive integer." >&2
        return 2
    fi
    for sample_dir in "${OUTPUT_ROOT}"/num_samples_*; do
        [[ -d "${sample_dir}" ]] || continue
        num_samples="${sample_dir##*/num_samples_}"
        if [[ ! "${num_samples}" =~ ^[0-9]+$ ]]; then
            echo "Error: invalid sample directory: ${sample_dir}" >&2
            return 1
        fi
        local seed_count=0
        for trial_dir in "${sample_dir}"/seed_*; do
            [[ -d "${trial_dir}" ]] || continue
            random_seed="${trial_dir##*/seed_}"
            if [[ ! "${random_seed}" =~ ^[0-9]+$ || ! -f "${trial_dir}/objective_curve_sigma_hat.npz" ]]; then
                echo "Error: invalid trial or missing saved curve: ${trial_dir}" >&2
                return 1
            fi
            trial_dirs+=("${trial_dir}")
            seed_count=$((seed_count + 1))
        done
        if ((seed_count > 0)); then
            sample_sizes+=("${num_samples}")
            if ((expected_seeds == 0)); then
                expected_seeds="${seed_count}"
            elif ((seed_count != expected_seeds)); then
                echo "Error: ${sample_dir} has ${seed_count} seeds; expected ${expected_seeds}." >&2
                return 1
            fi
        fi
    done
    if ((${#trial_dirs[@]} == 0)); then
        echo "Error: no saved trials found under ${OUTPUT_ROOT}." >&2
        return 1
    fi

    worker_count="${N_JOBS}"
    if ((worker_count > ${#trial_dirs[@]})); then
        worker_count="${#trial_dirs[@]}"
    fi
    echo "Reselecting ${#trial_dirs[@]} saved curves with ${worker_count} concurrent workers."
    for ((worker = 0; worker < worker_count; worker++)); do
        (
            for ((task = worker; task < ${#trial_dirs[@]}; task += worker_count)); do
                trial_dir="${trial_dirs[task]}"
                sample_dir="${trial_dir%/*}"
                run_trial "${sample_dir##*/num_samples_}" "${trial_dir##*/seed_}" true
            done
        ) &
        worker_pids+=("$!")
    done
    for worker_pid in "${worker_pids[@]}"; do
        if ! wait "${worker_pid}"; then
            worker_failed=1
        fi
    done
    if ((worker_failed)); then
        echo "Error: a reselect worker failed; summary generation was skipped." >&2
        return 1
    fi
    python experiments/summarize_fixed_support_scaling.py \
        "${OUTPUT_ROOT}" --sample-sizes "${sample_sizes[@]}" \
        --expected-seeds "${expected_seeds}" \
        --selection-root "${RESELECT_ROOT}" \
        --selection-file selection_plateau.json \
        --trials-output "${TRIALS_PATH}" --summary-output "${TABLE_PATH}"
    echo "Trial results: ${TRIALS_PATH}"
    echo "Summary table: ${TABLE_PATH}"
}

case "${COMMAND}" in
    reselect)
        (($# <= 2)) || { usage >&2; exit 2; }
        reselect_all
        exit $?
        ;;
    run-all)
        (($# <= 1)) || { usage >&2; exit 2; }
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if ! python -c 'import math, sys; lo, hi = map(float, sys.argv[1:]); sys.exit(0 if math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi else 1)' \
    "${LAMBDA_STAR_OFFDIAG_ABS_MIN}" "${LAMBDA_STAR_OFFDIAG_ABS_MAX}" 2>/dev/null; then
    echo "Error: Lambda_star off-diagonal absolute-value bounds must satisfy 0 < MIN <= MAX and be finite." >&2
    exit 2
fi

worker_count="$((TOTAL_CPUS / CPUS_PER_TRIAL))"
if ((worker_count > ${#task_seeds[@]})); then
    worker_count="${#task_seeds[@]}"
fi
echo "Running ${#task_seeds[@]} trials with ${worker_count} concurrent trials x ${CPUS_PER_TRIAL} CPUs."
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
    echo "Error: at least one trial failed; summary generation was skipped." >&2
    exit 1
fi

python experiments/summarize_fixed_support_scaling.py \
    "${OUTPUT_ROOT}" \
    --sample-sizes "${NUM_SAMPLE_VALUES[@]}" \
    --expected-seeds 10 \
    --trials-output "${TRIALS_PATH}" \
    --summary-output "${TABLE_PATH}"

echo
echo "Trial results: ${TRIALS_PATH}"
echo "Summary table: ${TABLE_PATH}"
