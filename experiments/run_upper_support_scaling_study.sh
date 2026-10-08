#!/usr/bin/env bash
# Run the fixed-support sample-size study for all 20 strict-upper supports of
# n=4 and true D_m=4. Each support receives its own trial CSV and summary table.
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
TRUE_DIMENSION="${TRUE_DIMENSION:-${N}}"
NUM_SUPPORTS="${NUM_SUPPORTS:-20}"
LAMBDA_STAR_OFFDIAG_ABS_MIN="${LAMBDA_STAR_OFFDIAG_ABS_MIN:-0.20}"
LAMBDA_STAR_OFFDIAG_ABS_MAX="${LAMBDA_STAR_OFFDIAG_ABS_MAX:-0.60}"
TOTAL_CPUS="${TOTAL_CPUS:-${SLURM_CPUS_PER_TASK:-${NSLOTS:-100}}}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-5}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-${NSLOTS:-8}}}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"
REFINE_AFTER_FIXED_OMEGA="${REFINE_AFTER_FIXED_OMEGA:-false}"
OMEGA_STAR="${OMEGA_STAR:-1}"
OMEGA_REF="${OMEGA_REF:-1}"
FIT_OMEGA_REF="${FIT_OMEGA_REF:-false}"
KAPPA="${KAPPA:-0.93}"
SUPPORT_SCOPE="${SUPPORT_SCOPE:-upper}"
NESTED_SUPPORTS="${NESTED_SUPPORTS:-true}"
OBJECTIVE_FLOOR="${OBJECTIVE_FLOOR:-1e-8}"
LM_WEIGHT="${LM_WEIGHT:-0.1}"
SCORE_A="${SCORE_A:-0.5}"
TOP_PLATEAUS="${TOP_PLATEAUS:-3}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-199}"
BOOTSTRAP_ALPHA="${BOOTSTRAP_ALPHA:-0.05}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260913}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/output/upper_support_scaling_n${N}_dm${TRUE_DIMENSION}}"
RESELECT_SUBDIR="reselect_lm_support_count_plateau"
NUM_SAMPLE_VALUES=(100 1000 10000 1000000)

usage() {
    cat <<EOF_USAGE
Usage: $(basename "$0") [run-all | trial [TASK_ID] | aggregate | reselect [OUTPUT_FOLDER]]

  no command  Run one trial when SLURM_ARRAY_TASK_ID is set; otherwise run-all.
  run-all     Run all supports, four sample sizes, and ten seeds per sample size,
              then write selection_trials.csv and selection_summary.md per support.
              Defaults: 20 supports, 800 trials, TOTAL_CPUS=100, CPUS_PER_TRIAL=5.
  trial       Run one trial. TASK_ID defaults to SLURM_ARRAY_TASK_ID and maps
              from 0 to $((TOTAL_TRIALS - 1)) in support/sample-size/seed order.
  aggregate   Summarize all configured supports from their saved curves and selections.
              Also write OUTPUT_ROOT/selection_summary.md with scores over all trials.
  reselect    Discover support_*/num_samples_*/seed_* saved curves and select with
              Lm = LM_WEIGHT times the number of available supports (default 0.1).
              Use up to N_JOBS concurrent trials, with one CPU per selection.
              Write selections and per-support summaries under each support's
              ${RESELECT_SUBDIR}/ directory. Do not recompute curves.
              OUTPUT_FOLDER overrides OUTPUT_ROOT for this command.

  NUM_SUPPORTS defaults to all 20 supports for N=4, TRUE_DIMENSION=4.
  LAMBDA_STAR_OFFDIAG_ABS_MIN/MAX bound nonzero coefficients (defaults 0.20/0.60).
  SCORE_A configures the maximal-class score weight (default 0.5; 0 < SCORE_A < 1).
  All fitting, support-search, objective-floor, and bootstrap settings from
  run_fixed_support_scaling.sh are configurable through the same environment variables.

Example for 100 CPUs (20 concurrent trials x 5 CPUs):
  array_job=\$(sbatch --parsable --array=0-$((TOTAL_TRIALS - 1))%20 \\
      --cpus-per-task=5 "$0" trial)
  sbatch --dependency=afterok:\${array_job} --cpus-per-task=1 "$0" aggregate
EOF_USAGE
}

if [[ ! "${N}" =~ ^[1-9][0-9]*$ ]] || ((N < 2)); then
    echo "Error: N must be an integer of at least 2." >&2
    exit 2
fi
if [[ ! "${TRUE_DIMENSION}" =~ ^[1-9][0-9]*$ ]] ||
    ((TRUE_DIMENSION < 2 || TRUE_DIMENSION > N * (N - 1) / 2)); then
    echo "Error: TRUE_DIMENSION must leave at least one true and one false upper edge." >&2
    exit 2
fi
if [[ ! "${NUM_SUPPORTS}" =~ ^[1-9][0-9]*$ ]] || ((NUM_SUPPORTS > 20)); then
    echo "Error: NUM_SUPPORTS must be an integer between 1 and 20." >&2
    exit 2
fi
# Count C(n * (n - 1) / 2, TRUE_DIMENSION - 1) without starting Python.
num_edges="$((N * (N - 1) / 2))"
choose_edges="$((TRUE_DIMENSION - 1))"
if ((choose_edges > num_edges - choose_edges)); then
    choose_edges="$((num_edges - choose_edges))"
fi
available_supports=1
for ((edge = 1; edge <= choose_edges; edge++)); do
    available_supports="$((available_supports * (num_edges - edge + 1) / edge))"
    # Only need to know whether at least NUM_SUPPORTS supports are available.
    ((available_supports < NUM_SUPPORTS)) || break
done
if ((NUM_SUPPORTS > available_supports)); then
    echo "Error: only ${available_supports} supports exist for N=${N}, TRUE_DIMENSION=${TRUE_DIMENSION}; reduce NUM_SUPPORTS." >&2
    exit 2
fi

task_support_indices=()
task_num_samples=()
task_seeds=()
for ((support_index = 0; support_index < NUM_SUPPORTS; support_index++)); do
    for num_samples in "${NUM_SAMPLE_VALUES[@]}"; do
        case "${num_samples}" in
            100) seeds=(124 139 141 147 153 156 173 192 237 243) ;;
            1000) seeds=(6 15 23 46 51 53 59 61 74 85) ;;
            10000) seeds=(3 4 6 8 12 13 18 21 22 27) ;;
            1000000) seeds=(2 3 124 139 147 153 156 173 192 237) ;;
        esac
        for random_seed in "${seeds[@]}"; do
            task_support_indices+=("${support_index}")
            task_num_samples+=("${num_samples}")
            task_seeds+=("${random_seed}")
        done
    done
done
TOTAL_TRIALS="${#task_seeds[@]}"

run_trial() {
    local support_index="$1"
    local num_samples="$2"
    local random_seed="$3"
    local selection_only="${4:-false}"
    local support_dir="${OUTPUT_ROOT}/support_$(printf '%02d' "${support_index}")"
    local trial_dir="${support_dir}/num_samples_${num_samples}/seed_${random_seed}"
    local result_dir="${trial_dir}"
    local curve_path="${trial_dir}/objective_curve_sigma_hat.npz"
    local selection_log="${trial_dir}/selection_plateau_bootstrap.log"
    local selection_json="${trial_dir}/selection_plateau_bootstrap.json"
    local selection_method="plateau-bootstrap"
    local selection_jobs="${CPUS_PER_TRIAL}"
    local selection_options=()
    local selected_dimension precision
    local exact=false

    if [[ "${selection_only}" == "true" ]]; then
        [[ -f "${curve_path}" ]] || { echo "Missing saved curve: ${curve_path}" >&2; return 1; }
        result_dir="${support_dir}/${RESELECT_SUBDIR}/num_samples_${num_samples}/seed_${random_seed}"
        mkdir -p "${result_dir}"
        selection_options+=(--lm-mode support-count --Lm "${LM_WEIGHT}")
        selection_log="${result_dir}/selection_plateau.log"
        selection_json="${result_dir}/selection_plateau.json"
        selection_method="plateau"
        selection_jobs=1
        echo "Reusing saved objective curve: ${curve_path}"
    else
        mkdir -p "${trial_dir}"
        echo
        echo "=== Support ${support_index}; n=${N}; num_samples=${num_samples}; seed=${random_seed}; CPUs=${CPUS_PER_TRIAL} ==="
        python experiments/compute_objective_curve.py \
            --curve sigma_hat \
            --sigma-hat-output "${curve_path}" \
            --given-output "${trial_dir}/sigma_hat.npz" \
            --lambda-star-dims "${N}" \
            --lambda-star-dimension "${TRUE_DIMENSION}" \
            --lambda-star-support-index "${support_index}" \
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
    if [[ "${selected_dimension}" == "${TRUE_DIMENSION}" ]] && awk -v p="${precision}" 'BEGIN {exit !(p == 1)}'; then
        exact=true
    fi
    printf '%s,%s,%s,%s,%s\n' \
        "${num_samples}" "${random_seed}" "${selected_dimension}" "${precision}" "${exact}" \
        > "${result_dir}/result.csv"
}

run_trial_by_task_id() {
    local task_id="$1"
    if [[ ! "${task_id}" =~ ^[0-9]+$ ]] || ((task_id >= TOTAL_TRIALS)); then
        echo "Error: TASK_ID must be an integer between 0 and $((TOTAL_TRIALS - 1))." >&2
        return 2
    fi
    run_trial "${task_support_indices[task_id]}" "${task_num_samples[task_id]}" "${task_seeds[task_id]}"
}

aggregate_results() {
    local selection_only="${1:-false}"
    if (($# > 0)); then shift; fi
    local support_index support_dir sample_dir trial_dir num_samples
    local summary_dir expected_seeds seed_count overall_summary_dir="${OUTPUT_ROOT}"
    local support_dirs=() sample_sizes=() selection_options=() trial_csv_paths=()
    if [[ "${selection_only}" == "true" ]]; then
        overall_summary_dir="${OUTPUT_ROOT}/${RESELECT_SUBDIR}"
    fi
    if (($# > 0)); then
        support_dirs=("$@")
    else
        for ((support_index = 0; support_index < NUM_SUPPORTS; support_index++)); do
            support_dirs+=("${OUTPUT_ROOT}/support_$(printf '%02d' "${support_index}")")
        done
    fi
    for support_dir in "${support_dirs[@]}"; do
        summary_dir="${support_dir}"
        sample_sizes=("${NUM_SAMPLE_VALUES[@]}")
        expected_seeds=10
        selection_options=(--selection-file selection_plateau_bootstrap.json)
        if [[ "${selection_only}" == "true" ]]; then
            summary_dir="${support_dir}/${RESELECT_SUBDIR}"
            selection_options=(--selection-root "${summary_dir}" --selection-file selection_plateau.json)
            sample_sizes=()
            expected_seeds=0
            for sample_dir in "${support_dir}"/num_samples_*; do
                [[ -d "${sample_dir}" ]] || continue
                num_samples="${sample_dir##*/num_samples_}"
                seed_count=0
                for trial_dir in "${sample_dir}"/seed_*; do
                    [[ -d "${trial_dir}" ]] || continue
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
        fi
        python experiments/summarize_fixed_support_scaling.py \
            "${support_dir}" --sample-sizes "${sample_sizes[@]}" \
            --expected-seeds "${expected_seeds}" \
            --score-a "${SCORE_A}" \
            "${selection_options[@]}" \
            --trials-output "${summary_dir}/selection_trials.csv" \
            --summary-output "${summary_dir}/selection_summary.md"
        trial_csv_paths+=("${summary_dir}/selection_trials.csv")
        echo "Trial results: ${summary_dir}/selection_trials.csv"
        echo "Summary table: ${summary_dir}/selection_summary.md"
    done
    python experiments/summarize_fixed_support_scaling.py \
        --overall-trials "${trial_csv_paths[@]}" \
        --summary-output "${overall_summary_dir}/selection_summary.md"
    echo "Overall score table: ${overall_summary_dir}/selection_summary.md"
}

reselect_all() {
    local support_dir sample_dir trial_dir support_index num_samples random_seed
    local worker_count worker task worker_pid
    local worker_failed=0 expected_seeds seed_count support_trial_count
    local support_dirs=() support_indices=() sample_sizes=() seeds=() worker_pids=()

    if [[ ! "${N_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: reselect requires N_JOBS to be a positive integer." >&2
        return 2
    fi
    # Preflight all saved trials before writing any selection outputs.
    for support_dir in "${OUTPUT_ROOT}"/support_*; do
        [[ -d "${support_dir}" ]] || continue
        support_index="${support_dir##*/support_}"
        if [[ ! "${support_index}" =~ ^[0-9]+$ || "${support_dir##*/}" != "$(printf 'support_%02d' "$((10#${support_index}))")" ]]; then
            echo "Error: invalid support directory: ${support_dir}" >&2
            return 1
        fi
        support_index="$((10#${support_index}))"
        support_trial_count=0
        expected_seeds=0
        for sample_dir in "${support_dir}"/num_samples_*; do
            [[ -d "${sample_dir}" ]] || continue
            num_samples="${sample_dir##*/num_samples_}"
            if [[ ! "${num_samples}" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: invalid sample directory: ${sample_dir}" >&2
                return 1
            fi
            seed_count=0
            for trial_dir in "${sample_dir}"/seed_*; do
                [[ -d "${trial_dir}" ]] || continue
                random_seed="${trial_dir##*/seed_}"
                if [[ ! "${random_seed}" =~ ^[0-9]+$ || ! -f "${trial_dir}/objective_curve_sigma_hat.npz" ]]; then
                    echo "Error: invalid trial or missing saved curve: ${trial_dir}" >&2
                    return 1
                fi
                support_indices+=("${support_index}")
                sample_sizes+=("${num_samples}")
                seeds+=("${random_seed}")
                seed_count=$((seed_count + 1))
                support_trial_count=$((support_trial_count + 1))
            done
            if ((seed_count > 0)); then
                if ((expected_seeds == 0)); then
                    expected_seeds="${seed_count}"
                elif ((seed_count != expected_seeds)); then
                    echo "Error: ${sample_dir} has ${seed_count} seeds; expected ${expected_seeds}." >&2
                    return 1
                fi
            fi
        done
        if ((support_trial_count > 0)); then
            support_dirs+=("${support_dir}")
        fi
    done
    if ((${#seeds[@]} == 0)); then
        echo "Error: no saved support/sample-size/seed trials found under ${OUTPUT_ROOT}." >&2
        return 1
    fi

    worker_count="${N_JOBS}"
    if ((worker_count > ${#seeds[@]})); then
        worker_count="${#seeds[@]}"
    fi
    echo "Reselecting ${#seeds[@]} saved curves with ${worker_count} concurrent workers."
    for ((worker = 0; worker < worker_count; worker++)); do
        (
            # Disjoint trial lists also work on Bash versions without wait -n.
            for ((task = worker; task < ${#seeds[@]}; task += worker_count)); do
                run_trial "${support_indices[task]}" "${sample_sizes[task]}" "${seeds[task]}" true
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
        echo "Error: a reselect worker failed; aggregation was skipped." >&2
        return 1
    fi
    aggregate_results true "${support_dirs[@]}"
}

run_all() {
    local worker_count worker task worker_pid
    local worker_failed=0
    local worker_pids=()
    worker_count="$((TOTAL_CPUS / CPUS_PER_TRIAL))"
    if ((worker_count > TOTAL_TRIALS)); then
        worker_count="${TOTAL_TRIALS}"
    fi
    echo "Running ${TOTAL_TRIALS} trials with ${worker_count} concurrent trials x ${CPUS_PER_TRIAL} CPUs."
    for ((worker = 0; worker < worker_count; worker++)); do
        (
            for ((task = worker; task < TOTAL_TRIALS; task += worker_count)); do
                run_trial_by_task_id "${task}"
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
        echo "Error: at least one trial failed; summary generation was skipped." >&2
        return 1
    fi
    aggregate_results
}

command="${1:-auto}"
case "${command}" in
    auto)
        if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then command=trial; else command=run-all; fi
        ;;
    run-all|aggregate)
        (($# == 1)) || { usage >&2; exit 2; }
        ;;
    trial|reselect)
        (($# <= 2)) || { usage >&2; exit 2; }
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Error: unknown command: ${command}" >&2
        usage >&2
        exit 2
        ;;
esac

if ! python -c 'import math, sys; a = float(sys.argv[1]); sys.exit(0 if math.isfinite(a) and 0 < a < 1 else 1)' \
    "${SCORE_A}" 2>/dev/null; then
    echo "Error: SCORE_A must be finite and satisfy 0 < SCORE_A < 1." >&2
    exit 2
fi

if [[ "${command}" == "reselect" ]]; then
    OUTPUT_ROOT="${2:-${OUTPUT_ROOT}}"
    reselect_all
elif [[ "${command}" == "aggregate" ]]; then
    aggregate_results
else
    if [[ ! "${TOTAL_CPUS}" =~ ^[1-9][0-9]*$ || ! "${CPUS_PER_TRIAL}" =~ ^[1-9][0-9]*$ ]] ||
        ((TOTAL_CPUS < CPUS_PER_TRIAL)); then
        echo "Error: TOTAL_CPUS and CPUS_PER_TRIAL must be positive integers, with TOTAL_CPUS >= CPUS_PER_TRIAL." >&2
        exit 2
    fi
    if ! python -c 'import math, sys; lo, hi = map(float, sys.argv[1:]); sys.exit(0 if math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi else 1)' \
        "${LAMBDA_STAR_OFFDIAG_ABS_MIN}" "${LAMBDA_STAR_OFFDIAG_ABS_MAX}" 2>/dev/null; then
        echo "Error: Lambda_star off-diagonal absolute-value bounds must satisfy 0 < MIN <= MAX and be finite." >&2
        exit 2
    fi
    if [[ "${command}" == "trial" ]]; then
        task_id="${2:-${SLURM_ARRAY_TASK_ID:-}}"
        [[ -n "${task_id}" ]] || { echo "Error: trial requires TASK_ID or SLURM_ARRAY_TASK_ID." >&2; exit 2; }
        run_trial_by_task_id "${task_id}"
    else
        run_all
    fi
fi
