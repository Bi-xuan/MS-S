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
TOTAL_CPUS="${TOTAL_CPUS:-${SLURM_CPUS_PER_TASK:-${NSLOTS:-100}}}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-5}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"
REFINE_AFTER_FIXED_OMEGA="${REFINE_AFTER_FIXED_OMEGA:-false}"
OMEGA_STAR="${OMEGA_STAR:-1}"
OMEGA_REF="${OMEGA_REF:-1}"
SUPPORT_SCOPE="${SUPPORT_SCOPE:-all}"
NESTED_SUPPORTS="${NESTED_SUPPORTS:-true}"
OBJECTIVE_FLOOR="${OBJECTIVE_FLOOR:-1e-8}"
TOP_PLATEAUS="${TOP_PLATEAUS:-3}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-199}"
BOOTSTRAP_ALPHA="${BOOTSTRAP_ALPHA:-0.05}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260913}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/output/fixed_support_scaling_n${N}}"
TRIALS_PATH="${OUTPUT_ROOT}/selection_trials.csv"
TABLE_PATH="${OUTPUT_ROOT}/selection_summary.md"
NUM_SAMPLE_VALUES=(100 1000 10000 1000000)

if [[ ! "${N}" =~ ^[0-9]+$ ]] || ((N < 2)); then
    echo "Error: N must be an integer of at least 2." >&2
    exit 2
fi
if [[ ! "${TOTAL_CPUS}" =~ ^[0-9]+$ || ! "${CPUS_PER_TRIAL}" =~ ^[0-9]+$ ]] \
    || ((TOTAL_CPUS < CPUS_PER_TRIAL || CPUS_PER_TRIAL < 1)); then
    echo "Error: TOTAL_CPUS and CPUS_PER_TRIAL must be positive integers, with TOTAL_CPUS >= CPUS_PER_TRIAL." >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

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
    local trial_dir="${OUTPUT_ROOT}/num_samples_${num_samples}/seed_${random_seed}"
    local curve_path="${trial_dir}/objective_curve_sigma_hat.npz"
    local selection_log="${trial_dir}/selection_plateau_bootstrap.log"
    local selected_dimension
    local precision
    local exact=false
    mkdir -p "${trial_dir}"

    echo
    echo "=== n=${N}; num_samples=${num_samples}; seed=${random_seed}; CPUs=${CPUS_PER_TRIAL} ==="
    python experiments/compute_objective_curve.py \
            --curve sigma_hat \
            --sigma-hat-output "${curve_path}" \
            --lambda-star-dims "${N}" \
            --nested-supports "${NESTED_SUPPORTS}" \
            --support-scope "${SUPPORT_SCOPE}" \
            --num-samples "${num_samples}" \
            --n-jobs "${CPUS_PER_TRIAL}" \
            --random-seed "${random_seed}" \
            --max-restarts "${MAX_RESTARTS}" \
            --refine-after-fixed-omega "${REFINE_AFTER_FIXED_OMEGA}" \
            --omega-star "${OMEGA_STAR}" \
            --omega-ref "${OMEGA_REF}" \
            2>&1 | tee "${trial_dir}/compute.log"

    python experiments/select_scaling_parameter.py \
            "${curve_path}" \
            --method plateau-bootstrap \
            --objective-floor "${OBJECTIVE_FLOOR}" \
            --top-plateaus "${TOP_PLATEAUS}" \
            --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
            --bootstrap-alpha "${BOOTSTRAP_ALPHA}" \
            --bootstrap-seed "${BOOTSTRAP_SEED}" \
            --n-jobs "${CPUS_PER_TRIAL}" \
            --output-json "${trial_dir}/selection_plateau_bootstrap.json" \
            2>&1 | tee "${selection_log}"

    selected_dimension="$(awk -F ': ' '$1 == "Selected dimension" {print $2}' "${selection_log}" | tail -n 1)"
    precision="$(awk -F ': ' '$1 == "Selected support precision" {print $2}' "${selection_log}" | tail -n 1)"
    precision="${precision:-0}"
    if [[ "${selected_dimension}" == "${N}" ]] && awk -v p="${precision}" 'BEGIN {exit !(p == 1)}'; then
        exact=true
    fi
    printf '%s,%s,%s,%s,%s\n' \
        "${num_samples}" "${random_seed}" "${selected_dimension}" "${precision}" "${exact}" \
        > "${trial_dir}/result.csv"
}

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

printf '%s\n' 'num_samples,random_seed,selected_dimension,precision,exact_support_recovery' > "${TRIALS_PATH}"
for ((task = 0; task < ${#task_seeds[@]}; task++)); do
    result_path="${OUTPUT_ROOT}/num_samples_${task_num_samples[task]}/seed_${task_seeds[task]}/result.csv"
    if [[ ! -f "${result_path}" ]]; then
        echo "Error: missing trial result: ${result_path}" >&2
        exit 1
    fi
    tail -n 1 "${result_path}" >> "${TRIALS_PATH}"
done

awk -F ',' '
    NR > 1 {count[$1]++; precision[$1] += $4; exact[$1] += ($5 == "true")}
    END {
        print "| Number of samples | Average precision over 10 seeds | Exact support recovery (x/10) |"
        print "|---:|---:|---:|"
        for (i = 1; i <= 4; i++) {
            n = values[i]
            printf "| %d | %.6f | %d/10 |\n", n, precision[n] / count[n], exact[n]
        }
    }
    BEGIN {values[1] = 100; values[2] = 1000; values[3] = 10000; values[4] = 1000000}
' "${TRIALS_PATH}" | tee "${TABLE_PATH}"

echo
echo "Trial results: ${TRIALS_PATH}"
echo "Summary table: ${TABLE_PATH}"
