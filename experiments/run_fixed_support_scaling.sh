#!/usr/bin/env bash
# Evaluate one fixed Lambda_star across sample sizes with support recovery.
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
SUPPORT_SCOPE="${SUPPORT_SCOPE:-all}"
SEARCH_OPTION="${SEARCH_OPTION:-forward}"
OBJECTIVE_FLOOR="${OBJECTIVE_FLOOR:-1e-8}"
LM_WEIGHT="${LM_WEIGHT:-0.1}"
LM_CONSTANT="${LM_CONSTANT:-1}"
COMMAND="${1:-run-all}"
TOP_PLATEAUS="${TOP_PLATEAUS:-2}"
BOOTSTRAP_ALPHA="${BOOTSTRAP_ALPHA:-0.1}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-199}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260913}"
if [[ "${COMMAND}" == "reselect" ]]; then
    OUTPUT_ROOT="${2:-${OUTPUT_ROOT:-experiments/output/fixed_support_scaling_n4_omega_ref_eq_star}}"
else
    default_output_root="experiments/output/fixed_support_scaling_n${N}"
    if [[ "${SEARCH_OPTION}" == "forward-backward-nested" ]]; then
        default_output_root+="_forward_backward_nested"
    elif [[ "${SEARCH_OPTION}" == "greedy" ]]; then
        default_output_root+="_greedy"
    fi
    OUTPUT_ROOT="${OUTPUT_ROOT:-${default_output_root}}"
    if [[ "${COMMAND}" == "oracle" || "${COMMAND}" == "lasso" || "${COMMAND}" == "BIC" || "${COMMAND}" == "bic" ]]; then
        OUTPUT_ROOT="${2:-${OUTPUT_ROOT}}"
    fi
fi
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-${NSLOTS:-8}}}"
TRIALS_PATH="${OUTPUT_ROOT}/selection_trials.csv"
TABLE_PATH="${OUTPUT_ROOT}/selection_summary.md"
NUM_SAMPLE_VALUES=(100 1000 10000 1000000)

if [[ "${COMMAND}" == "run-all" ]]; then
    case "${SEARCH_OPTION}" in
        forward|forward-backward-nested) NESTED_SUPPORTS="${NESTED_SUPPORTS:-true}" ;;
        greedy) NESTED_SUPPORTS=false ;;
        *)
            echo "Error: SEARCH_OPTION must be forward, forward-backward-nested, or greedy." >&2
            exit 2
            ;;
    esac
    if [[ "${SEARCH_OPTION}" != "greedy" && "${NESTED_SUPPORTS}" != "true" ]]; then
        echo "Error: ${SEARCH_OPTION} requires nested supports; use SEARCH_OPTION=greedy for exhaustive search." >&2
        exit 2
    fi
fi

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
Usage: $(basename "$0") [run-all | reselect [OUTPUT_FOLDER] | oracle [OUTPUT_FOLDER] | lasso [OUTPUT_FOLDER] | BIC [OUTPUT_FOLDER]]

  Set LAMBDA_STAR_OFFDIAG_ABS_MIN and LAMBDA_STAR_OFFDIAG_ABS_MAX
  to bound the absolute values of nonzero Lambda_star off-diagonal entries.
  Defaults: 0.20 and 0.60. Example:
    LAMBDA_STAR_OFFDIAG_ABS_MIN=0.30 LAMBDA_STAR_OFFDIAG_ABS_MAX=0.50 OUTPUT_ROOT=experiments/output/custom_bounds bash experiments/run_fixed_support_scaling.sh

  run-all   Compute curves, select dimensions, and summarize (default).
            SEARCH_OPTION=forward (default) uses the existing nested forward path.
            SEARCH_OPTION=forward-backward-nested adds backward corrections and
            reconstructs earlier nested supports. Both modes require nesting.
            SEARCH_OPTION=greedy searches all supports independently at each
            dimension, using SUPPORT_SCOPE=all (default) or upper, without nesting.
            The backward and greedy modes default to output folders ending in
            _forward_backward_nested and _greedy; OUTPUT_ROOT overrides the folder.
  reselect  Discover saved num_samples_*/seed_* curves under OUTPUT_ROOT,
            then run all four combinations of plateau/plateau-bootstrap and
            constant/support-count Lm using the saved curves and current penalty.
            Constant Lm = LM_CONSTANT (default 1); support-count Lm = LM_WEIGHT
            times the number of available supports (LM_WEIGHT defaults to 0.1).
            Bootstrap uses TOP_PLATEAUS (default 2), BOOTSTRAP_REPLICATES (199),
            BOOTSTRAP_ALPHA (0.1), and BOOTSTRAP_SEED (20260913), with the trial's
            saved fitting and nesting settings. Use up to N_JOBS concurrent
            trials, with one CPU per selection. Do not recompute curves.
            Create a fresh reselect_<run-id>/ folder under OUTPUT_ROOT, with
            four experiment subfolders containing per-trial selection.json,
            selection.log, result.csv, selection_trials.csv, selection_summary.md,
            and methodological_parameters.json. Preserve existing results.
            OUTPUT_FOLDER overrides OUTPUT_ROOT for this command.
            Default folder: experiments/output/fixed_support_scaling_n4_omega_ref_eq_star
  oracle    Fit all supports independently at each saved lambda_star_dimension,
            using the saved covariance, scope, fitting settings, and solver seed.
            Save oracle fields in each trial NPZ and refresh summaries with
            oracle exact recovery, EC recovery, and average EC score.
            Uses TOTAL_CPUS / CPUS_PER_TRIAL concurrent trials.
  lasso     Fit five-fold-CV upper-triangular lasso on saved observations.
            Save lasso fields in NPZs and add exact/EC recovery and EC score
            to summaries. Configure LASSO_GAMMA_MIN/MAX/COUNT and
            LASSO_GAMMA_SPACING (log or linear); zero is always included.
  BIC       Fit Gaussian maximum likelihood on all upper supports (2 <= N <= 4).
            Use fixed saved omega_ref, save BIC fields in NPZs, and refresh summaries.
            Parallel support fits: BIC_SUPPORT_WORKERS (default CPUS_PER_TRIAL),
            from 1 to CPUS_PER_TRIAL; TOTAL_CPUS / CPUS_PER_TRIAL concurrent trials.
            Requires saved SUPPORT_SCOPE=upper; accepts lowercase bic too.
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
        result_dir="${RESELECT_ROOT}/${RESELECT_VARIANT}/num_samples_${num_samples}/seed_${random_seed}"
        mkdir -p "${result_dir}"
        selection_options+=(--lm-mode "${RESELECT_LM_MODE}" --Lm "${RESELECT_LM_VALUE}")
        selection_log="${result_dir}/selection.log"
        selection_json="${result_dir}/selection.json"
        selection_method="${RESELECT_METHOD}"
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
            --search-option "${SEARCH_OPTION}" \
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
    fi
    if [[ "${selection_method}" == "plateau-bootstrap" ]]; then
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
    local variant_index summary_dir selection_subdir
    local RESELECT_ROOT RESELECT_VARIANT RESELECT_METHOD RESELECT_LM_MODE RESELECT_LM_VALUE
    local worker_failed=0 expected_seeds=0
    local trial_dirs=() sample_sizes=() worker_pids=()
    local variants=(penalty_plateau_lm_constant penalty_plateau_lm_support_count
                    penalty_plateau_bootstrap_lm_constant penalty_plateau_bootstrap_lm_support_count)
    local methods=(plateau plateau plateau-bootstrap plateau-bootstrap)
    local lm_modes=(constant support-count constant support-count)

    if [[ ! "${N_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: reselect requires N_JOBS to be a positive integer." >&2
        return 2
    fi
    if ! python -c 'import math,sys; values=map(float,sys.argv[1:]); sys.exit(0 if all(math.isfinite(v) and v > 0 for v in values) else 1)' \
        "${LM_CONSTANT}" "${LM_WEIGHT}" 2>/dev/null; then
        echo "Error: LM_CONSTANT and LM_WEIGHT must be finite and positive." >&2
        return 2
    fi
    # Preflight all saved trials before writing any selection outputs.
    for sample_dir in "${OUTPUT_ROOT}"/num_samples_*; do
        [[ -d "${sample_dir}" ]] || continue
        num_samples="${sample_dir##*/num_samples_}"
        if [[ ! "${num_samples}" =~ ^[1-9][0-9]*$ ]]; then
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
    RESELECT_ROOT="$(mktemp -d "${OUTPUT_ROOT}/reselect_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
    cat > "${RESELECT_ROOT}/README.md" <<EOF_COMPARISON
# Penalty selection comparison

Saved curves: ${OUTPUT_ROOT}
Each experiment has its own selection_summary.md and methodological_parameters.json.
Per-trial selections are under EXPERIMENT/num_samples_*/seed_*/.

| Experiment | Selection method | Lm |
|---|---|---|
| ${variants[0]} | plateau | ${LM_CONSTANT} |
| ${variants[1]} | plateau | ${LM_WEIGHT} * support count |
| ${variants[2]} | plateau-bootstrap | ${LM_CONSTANT} |
| ${variants[3]} | plateau-bootstrap | ${LM_WEIGHT} * support count |
EOF_COMPARISON
    echo "Comparison output: ${RESELECT_ROOT}"
    for ((variant_index = 0; variant_index < ${#variants[@]}; variant_index++)); do
        RESELECT_VARIANT="${variants[variant_index]}"
        RESELECT_METHOD="${methods[variant_index]}"
        RESELECT_LM_MODE="${lm_modes[variant_index]}"
        RESELECT_LM_VALUE="${LM_CONSTANT}"
        if [[ "${RESELECT_LM_MODE}" == "support-count" ]]; then
            RESELECT_LM_VALUE="${LM_WEIGHT}"
        fi
        summary_dir="${RESELECT_ROOT}/${RESELECT_VARIANT}"
        selection_subdir="${RESELECT_ROOT##*/}/${RESELECT_VARIANT}"
        mkdir "${summary_dir}"
        worker_failed=0
        worker_pids=()
        echo "Experiment ${RESELECT_VARIANT}: ${#trial_dirs[@]} saved curves with ${worker_count} concurrent workers."
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
            echo "Error: a reselect worker failed in ${RESELECT_VARIANT}; aggregation was skipped." >&2
            return 1
        fi
        python experiments/summarize_fixed_support_scaling.py \
            "${OUTPUT_ROOT}" --sample-sizes "${sample_sizes[@]}" \
            --expected-seeds "${expected_seeds}" \
            --selection-root "${summary_dir}" --selection-file selection.json \
            --trials-output "${summary_dir}/selection_trials.csv" \
            --summary-output "${summary_dir}/selection_summary.md"
        echo "Trial results: ${summary_dir}/selection_trials.csv"
        echo "Summary table: ${summary_dir}/selection_summary.md"
        (
            export N LAMBDA_STAR_OFFDIAG_ABS_MIN LAMBDA_STAR_OFFDIAG_ABS_MAX
            export TOTAL_CPUS CPUS_PER_TRIAL N_JOBS MAX_RESTARTS REFINE_AFTER_FIXED_OMEGA
            export OMEGA_STAR OMEGA_REF FIT_OMEGA_REF KAPPA SUPPORT_SCOPE SEARCH_OPTION
            export STUDY_SAMPLE_SIZES="${sample_sizes[*]}" TOTAL_TRIALS="${#trial_dirs[@]}"
            python "${SCRIPT_DIR}/write_upper_study_parameters.py" "${summary_dir}" \
                --command "${COMMAND}" --source-root "${OUTPUT_ROOT}" \
                --selection-subdir "${selection_subdir}" --selection-file selection.json
        )
    done
    echo "Completed all four experiments: ${RESELECT_ROOT}"
}

case "${COMMAND}" in
    BIC|bic)
        (($# <= 2)) || { usage >&2; exit 2; }
        "${PYTHON:-python}" experiments/run_bic_support_scaling.py "${OUTPUT_ROOT}" \
            --total-cpus "${TOTAL_CPUS}" --cpus-per-trial "${CPUS_PER_TRIAL}" \
            --support-workers "${BIC_SUPPORT_WORKERS:-${CPUS_PER_TRIAL}}" \
            --diagonal-bound "${BIC_DIAGONAL_BOUND:-0.999999}" \
            --max-restarts "${BIC_MAX_RESTARTS:-10}" \
            --max-iter "${BIC_MAX_ITER:-1500}" --tol "${BIC_TOL:-1e-6}"
        exit $?
        ;;
    lasso)
        (($# <= 2)) || { usage >&2; exit 2; }
        "${PYTHON:-python}" experiments/run_lasso_support_scaling.py "${OUTPUT_ROOT}" \
            --total-cpus "${TOTAL_CPUS}" --cpus-per-trial "${CPUS_PER_TRIAL}" \
            --gamma-min "${LASSO_GAMMA_MIN:-1e-7}" \
            --gamma-max "${LASSO_GAMMA_MAX:-1}" \
            --gamma-count "${LASSO_GAMMA_COUNT:-29}" \
            --gamma-spacing "${LASSO_GAMMA_SPACING:-log}" \
            --fold-seed "${LASSO_CV_SEED:-20261013}" \
            --kappa "${LASSO_KAPPA:-0.93}" \
            --diagonal-bound "${LASSO_DIAGONAL_BOUND:-0.999999}" \
            --zero-tol "${LASSO_ZERO_TOL:-1e-5}" \
            --max-restarts "${LASSO_MAX_RESTARTS:-10}" \
            --max-iter "${LASSO_MAX_ITER:-1500}" \
            --tol "${LASSO_TOL:-1e-6}"
        exit $?
        ;;
    oracle)
        (($# <= 2)) || { usage >&2; exit 2; }
        "${PYTHON:-python}" experiments/run_oracle_support_scaling.py "${OUTPUT_ROOT}" \
            --total-cpus "${TOTAL_CPUS}" --cpus-per-trial "${CPUS_PER_TRIAL}"
        exit $?
        ;;
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
echo "Running ${#task_seeds[@]} trials with ${worker_count} concurrent trials x ${CPUS_PER_TRIAL} CPUs; search=${SEARCH_OPTION}."
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
