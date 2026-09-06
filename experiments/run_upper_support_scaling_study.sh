#!/usr/bin/env bash
# Test scaling-parameter dimension selection over a configurable number of the
# 20 strict-upper true supports for n=4 and true D_m=4. When invoked by a
# Slurm array, each task runs one support/seed pair. Outside a Slurm array, the
# default remains a sequential run of every pair.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N=4
TRUE_DIMENSION=4
NUM_SAMPLES=100
# SEEDS=(124 17 43 89 733 2027 4093 211 347 509)
SEEDS=(124 139 141 147 153 156 173 192 237 243)

N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
NUM_SUPPORTS="${NUM_SUPPORTS:-20}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"
REFINE_AFTER_FIXED_OMEGA="${REFINE_AFTER_FIXED_OMEGA:-false}"
OMEGA_STAR="${OMEGA_STAR:-1}"
OMEGA_REF="${OMEGA_REF:-1}"
RECOMMENDATION_FACTOR="${RECOMMENDATION_FACTOR:-2.0}"
OBJECTIVE_FLOOR="${OBJECTIVE_FLOOR:-1e-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/output/upper_support_scaling_n4_dm4_nsm100_omega1_minabs02_10seeds}"

if [[ ! "${NUM_SUPPORTS}" =~ ^[1-9][0-9]*$ ]] || ((NUM_SUPPORTS > 20)); then
    echo "Error: NUM_SUPPORTS must be an integer between 1 and 20." >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
SUMMARY_PATH="${OUTPUT_ROOT}/selection_summary.csv"
COVERAGE_PLOT_PATH="${OUTPUT_ROOT}/selection_coverage.png"
COVERAGE_PDF_PATH="${OUTPUT_ROOT}/selection_coverage.pdf"
CSV_HEADER="support_index,random_seed,true_dimension,scaling_input_status,window_status,window_dimension,window_correct,plateau_status,plateau_dimension,plateau_correct,curve_file,scaling_input_log"
SEED_COUNT="${#SEEDS[@]}"
TOTAL_TRIALS="$((NUM_SUPPORTS * SEED_COUNT))"

usage() {
    cat <<EOF
Usage: $(basename "$0") [run-all | trial [TASK_ID] | aggregate]

  no command  Run one trial when SLURM_ARRAY_TASK_ID is set; otherwise run all
              ${TOTAL_TRIALS} trials sequentially and aggregate their results.
  run-all     Run all trials sequentially, then aggregate results.
  trial       Run one trial. TASK_ID defaults to SLURM_ARRAY_TASK_ID and maps
              from 0 to $((TOTAL_TRIALS - 1)) in support-major, seed-minor order.
  aggregate   Validate all per-trial result files, create selection_summary.csv,
              and draw the PNG and PDF coverage plots.

Example for 100 CPUs (20 concurrent trials x 5 CPUs):
  array_job=\$(sbatch --parsable --array=0-$((TOTAL_TRIALS - 1))%20 \\
      --cpus-per-task=5 "$0" trial)
  sbatch --dependency=afterok:\${array_job} --cpus-per-task=1 "$0" aggregate
EOF
}

run_selection() {
    local curve_path="$1"
    local method="$2"
    local log_path="$3"
    local selected_dimension

    if python experiments/select_scaling_parameter.py \
        "${curve_path}" \
        --method "${method}" \
        --recommendation-factor "${RECOMMENDATION_FACTOR}" \
        --objective-floor "${OBJECTIVE_FLOOR}" \
        2>&1 | tee "${log_path}"; then
        selected_dimension="$(awk -F ': ' \
            '$1 == "Selected dimension at recommended scale" {print $2}' \
            "${log_path}" | tail -n 1)"
        if [[ "${selected_dimension}" =~ ^[0-9]+$ ]]; then
            SELECTION_STATUS="ok"
            SELECTION_DIMENSION="${selected_dimension}"
            if [[ "${selected_dimension}" -eq "${TRUE_DIMENSION}" ]]; then
                SELECTION_CORRECT="true"
            else
                SELECTION_CORRECT="false"
            fi
        else
            SELECTION_STATUS="parse_error"
            SELECTION_DIMENSION="NA"
            SELECTION_CORRECT="NA"
        fi
    else
        SELECTION_STATUS="failed"
        SELECTION_DIMENSION="NA"
        SELECTION_CORRECT="NA"
    fi
}

trial_result_path() {
    local support_index="$1"
    local random_seed="$2"
    printf '%s/support_%02d/seed_%s/result.csv' \
        "${OUTPUT_ROOT}" "${support_index}" "${random_seed}"
}

run_trial() {
    local support_index="$1"
    local random_seed="$2"
    local support_dir
    local scenario_dir
    local curve_path
    local compute_log
    local result_path
    local result_tmp
    local validation_log
    local scaling_input_status
    local window_status
    local window_dimension
    local window_correct
    local plateau_status
    local plateau_dimension
    local plateau_correct

    support_dir="${OUTPUT_ROOT}/support_$(printf '%02d' "${support_index}")"
    scenario_dir="${support_dir}/seed_${random_seed}"
    mkdir -p "${scenario_dir}"

    curve_path="${scenario_dir}/objective_curve_sigma_hat.npz"
    compute_log="${scenario_dir}/compute.log"
    validation_log="${scenario_dir}/scaling_input_validation.log"
    result_path="$(trial_result_path "${support_index}" "${random_seed}")"
    result_tmp="${result_path}.tmp.${SLURM_JOB_ID:-local_$$}"

    echo
    echo "=== Support $((support_index + 1))/${NUM_SUPPORTS}; seed ${random_seed}; N_JOBS=${N_JOBS} ==="
    python experiments/compute_objective_curve.py \
        --curve sigma_hat \
        --sigma-hat-output "${curve_path}" \
        --given-output "${scenario_dir}/sigma_hat.npz" \
        --lambda-star-dims "${N}" \
        --lambda-star-dimension "${TRUE_DIMENSION}" \
        --lambda-star-support-index "${support_index}" \
        --support-scope upper \
        --num-samples "${NUM_SAMPLES}" \
        --n-jobs "${N_JOBS}" \
        --random-seed "${random_seed}" \
        --max-restarts "${MAX_RESTARTS}" \
        --refine-after-fixed-omega "${REFINE_AFTER_FIXED_OMEGA}" \
        --omega-star "${OMEGA_STAR}" \
        --omega-ref "${OMEGA_REF}" \
        2>&1 | tee "${compute_log}"

    if python experiments/select_scaling_parameter.py \
        "${curve_path}" \
        --validate-only \
        --objective-floor "${OBJECTIVE_FLOOR}" \
        2>&1 | tee "${validation_log}"; then
        scaling_input_status="valid"

        run_selection "${curve_path}" window "${scenario_dir}/selection_window.log"
        window_status="${SELECTION_STATUS}"
        window_dimension="${SELECTION_DIMENSION}"
        window_correct="${SELECTION_CORRECT}"

        run_selection "${curve_path}" plateau "${scenario_dir}/selection_plateau.log"
        plateau_status="${SELECTION_STATUS}"
        plateau_dimension="${SELECTION_DIMENSION}"
        plateau_correct="${SELECTION_CORRECT}"
    else
        scaling_input_status="invalid"
        window_status="invalid_input"
        window_dimension="NA"
        window_correct="NA"
        plateau_status="invalid_input"
        plateau_dimension="NA"
        plateau_correct="NA"
        printf '%s\n' \
            "Selection skipped because the scaling input is invalid." \
            > "${scenario_dir}/selection_window.log"
        printf '%s\n' \
            "Selection skipped because the scaling input is invalid." \
            > "${scenario_dir}/selection_plateau.log"
    fi

    {
        printf '%s\n' "${CSV_HEADER}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${support_index}" "${random_seed}" "${TRUE_DIMENSION}" "${scaling_input_status}" \
            "${window_status}" "${window_dimension}" "${window_correct}" \
            "${plateau_status}" "${plateau_dimension}" "${plateau_correct}" \
            "${curve_path}" "${validation_log}"
    } > "${result_tmp}"
    mv -f "${result_tmp}" "${result_path}"

    echo "Trial result: ${result_path}"
}

run_trial_by_task_id() {
    local task_id="$1"
    local support_index
    local seed_index

    if [[ ! "${task_id}" =~ ^[0-9]+$ ]] || ((task_id >= TOTAL_TRIALS)); then
        echo "Error: TASK_ID must be an integer between 0 and $((TOTAL_TRIALS - 1))." >&2
        exit 2
    fi

    support_index="$((task_id / SEED_COUNT))"
    seed_index="$((task_id % SEED_COUNT))"
    run_trial "${support_index}" "${SEEDS[seed_index]}"
}

aggregate_results() {
    local support_index
    local random_seed
    local result_path
    local result_header
    local result_row
    local row_support
    local row_seed
    local summary_tmp
    local missing_count=0
    local result_paths=()

    for ((support_index = 0; support_index < NUM_SUPPORTS; support_index++)); do
        for random_seed in "${SEEDS[@]}"; do
            result_path="$(trial_result_path "${support_index}" "${random_seed}")"
            if [[ ! -f "${result_path}" ]]; then
                echo "Missing trial result: ${result_path}" >&2
                missing_count="$((missing_count + 1))"
                continue
            fi

            IFS= read -r result_header < "${result_path}"
            result_header="${result_header%$'\r'}"
            if [[ "${result_header}" != "${CSV_HEADER}" ]]; then
                echo "Invalid CSV header: ${result_path}" >&2
                exit 1
            fi
            result_row="$(tail -n 1 "${result_path}")"
            IFS=',' read -r row_support row_seed _ <<< "${result_row}"
            if [[ "${row_support}" != "${support_index}" || "${row_seed}" != "${random_seed}" ]]; then
                echo "Mismatched support or seed in ${result_path}" >&2
                exit 1
            fi
            result_paths+=("${result_path}")
        done
    done

    if ((missing_count > 0)); then
        echo "Error: cannot aggregate; ${missing_count} of ${TOTAL_TRIALS} trial results are missing." >&2
        exit 1
    fi

    summary_tmp="${SUMMARY_PATH}.tmp.${SLURM_JOB_ID:-local_$$}"
    trap '[[ -z "${summary_tmp:-}" ]] || rm -f -- "${summary_tmp}"' EXIT
    printf '%s\n' "${CSV_HEADER}" > "${summary_tmp}"
    for result_path in "${result_paths[@]}"; do
        tail -n 1 "${result_path}" >> "${summary_tmp}"
    done
    mv -f "${summary_tmp}" "${SUMMARY_PATH}"
    trap - EXIT

    python experiments/plot_upper_support_coverage.py \
        "${SUMMARY_PATH}" \
        --output "${COVERAGE_PLOT_PATH}" \
        --pdf-output "${COVERAGE_PDF_PATH}"

    echo
    echo "Aggregated ${TOTAL_TRIALS} trials across ${NUM_SUPPORTS} supports."
    echo "Selection summary: ${SUMMARY_PATH}"
    echo "Coverage plot: ${COVERAGE_PLOT_PATH}"
    echo "Coverage PDF: ${COVERAGE_PDF_PATH}"
    awk -F ',' 'NR > 1 && $4 == "valid" {
        valid++
        if ($7 == "true" || $10 == "true") count++
    }
    NR > 1 && $4 != "valid" {invalid++}
    END {
        if (valid > 0) {
            printf "Valid trials with at least one correct selection: %d/%d (%.1f%%)\n", count + 0, valid, 100 * count / valid
        } else {
            print "Valid trials with at least one correct selection: 0/0 (N/A)"
        }
        printf "Invalid trials excluded from coverage: %d\n", invalid + 0
    }' "${SUMMARY_PATH}"
}

run_all() {
    local support_index
    local random_seed

    for ((support_index = 0; support_index < NUM_SUPPORTS; support_index++)); do
        for random_seed in "${SEEDS[@]}"; do
            run_trial "${support_index}" "${random_seed}"
        done
    done
    aggregate_results
}

command="${1:-auto}"
case "${command}" in
    auto)
        if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
            run_trial_by_task_id "${SLURM_ARRAY_TASK_ID}"
        else
            run_all
        fi
        ;;
    run-all)
        if (($# != 1)); then
            usage >&2
            exit 2
        fi
        run_all
        ;;
    trial)
        if (($# > 2)); then
            usage >&2
            exit 2
        fi
        task_id="${2:-${SLURM_ARRAY_TASK_ID:-}}"
        if [[ -z "${task_id}" ]]; then
            echo "Error: trial requires TASK_ID or SLURM_ARRAY_TASK_ID." >&2
            exit 2
        fi
        run_trial_by_task_id "${task_id}"
        ;;
    aggregate)
        if (($# != 1)); then
            usage >&2
            exit 2
        fi
        aggregate_results
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Error: unknown command: ${command}" >&2
        usage >&2
        exit 2
        ;;
esac
