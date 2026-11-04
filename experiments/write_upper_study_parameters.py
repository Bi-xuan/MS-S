"""Write an atomic methodology snapshot beside the upper-study summaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np


PARAMETER_NAMES = (
    "N", "TRUE_DIMENSION", "NUM_SUPPORTS", "LAMBDA_STAR_OFFDIAG_ABS_MIN",
    "LAMBDA_STAR_OFFDIAG_ABS_MAX", "TOTAL_CPUS", "CPUS_PER_TRIAL", "N_JOBS",
    "MAX_RESTARTS", "REFINE_AFTER_FIXED_OMEGA", "OMEGA_STAR", "OMEGA_REF",
    "FIT_OMEGA_REF", "KAPPA", "SUPPORT_SCOPE", "SEARCH_OPTION", "NESTED_SUPPORTS",
)
CURVE_FIELDS = {
    "study": ("n", "lambda_star_dimension"),
    "search": ("search_option", "support_scope", "nested_supports", "preselect_k",
               "preselect_direction_policy", "stop_obj_threshold"),
    "data_generation": ("lambda_star_min_edge_magnitude", "lambda_star_max_edge_magnitude",
                        "omega_star", "fit_omega_ref", "kappa"),
}
SELECTION_FIELDS = ("method", "objective_floor", "Lm_mode", "Lm")
METHOD_FIELDS = {
    "plateau": ("recommendation_factor",),
    "plateau-bootstrap": ("top_plateaus", "bootstrap_replicates", "alpha", "seed", "nested"),
}


def _parameter(value):
    if value.lower() in ("true", "false", "none"):
        return {"true": True, "false": False, "none": None}[value.lower()]
    try:
        return int(value)
    except ValueError:
        try:
            number = float(value)
            return number if math.isfinite(number) else value
        except ValueError:
            return value


def _summarize(values):
    unique = {json.dumps(value, sort_keys=True, allow_nan=False): value for value in values}
    ordered = [unique[key] for key in sorted(unique)]
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) > 20 and all(type(value) in (int, float) for value in ordered):
        return {"min": min(ordered), "max": max(ordered), "distinct_count": len(ordered)}
    return {"values": ordered, "distinct_count": len(ordered)}


def write_parameters(root, command, *, environ=None, task_id=None,
                     source_root=None, selection_subdir=None,
                     selection_file="selection_plateau_bootstrap.json"):
    """Record requested settings separately from the actual saved methodology.

    A concurrent trial may still be writing its report; incomplete files are
    counted explicitly and a later aggregate invocation refreshes the snapshot.
    Atomic replacement prevents concurrent array jobs from corrupting the JSON.
    source_root can be the study root or one support. selection_subdir locates
    an experiment relative to each support, so pooled and per-support metadata
    read the same experiment's reports without changing the original results.
    """
    root = Path(root).resolve()
    source_root = Path(source_root).resolve() if source_root is not None else root
    environ = os.environ if environ is None else environ
    configured = {key: _parameter(environ[key]) for key in PARAMETER_NAMES if key in environ}
    configured.update({key: _parameter(value) for key, value in environ.items()
                       if key.startswith(("BIC_", "LASSO_"))})
    observed = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    support_names = set()
    missing_selections = 0
    errors = []
    curves = sorted(source_root.glob("support_*/num_samples_*/seed_*/objective_curve_sigma_hat.npz"))
    if not curves:
        curves = sorted(source_root.glob("num_samples_*/seed_*/objective_curve_sigma_hat.npz"))
    for curve in curves:
        support_names.add(curve.parents[2].name)
        samples = int(curve.parent.parent.name.removeprefix("num_samples_"))
        seeds[samples].add(int(curve.parent.name.removeprefix("seed_")))
        try:
            with np.load(curve, allow_pickle=False) as data:
                for group, keys in CURVE_FIELDS.items():
                    for key in keys:
                        if key in data:
                            observed[group][key].append(data[key].item())
                if "d_m_values" in data:
                    observed["search"]["candidate_dimensions"].append(data["d_m_values"].tolist())
                if "omega_ref" in data:
                    # Legacy free-omega files can encode None in an object scalar.
                    try:
                        omega_ref = data["omega_ref"].item()
                    except ValueError:
                        omega_ref = None
                    observed["fitting"]["resolved_omega_ref"].append(omega_ref)
                if "solve_seed" in data and "random_seed" in data:
                    observed["data_generation"]["solve_seed_offset"].append(
                        int(data["solve_seed"].item()) - int(data["random_seed"].item()))
                if "fallback_seed" in data and "random_seed" in data:
                    observed["data_generation"]["fallback_seed_offset"].append(
                        int(data["fallback_seed"].item()) - int(data["random_seed"].item()))
                if "lasso_gamma_grid" in data:
                    observed["lasso"]["gamma_grid"].append(data["lasso_gamma_grid"].tolist())
                for group, key in (("fitting", "fit_settings_json"),
                                   ("oracle", "oracle_fit_options_json"),
                                   ("lasso", "lasso_fit_options_json"),
                                   ("bic", "bic_fit_options_json")):
                    if key in data:
                        for setting, value in json.loads(data[key].item()).items():
                            observed[group][setting].append(value)
        except (OSError, ValueError, KeyError) as error:
            errors.append({"file": str(curve.relative_to(source_root)), "error": str(error)})
        if selection_subdir is None:
            selection = curve.with_name(selection_file)
        else:
            support = curve.parents[2]
            selection = support / selection_subdir / curve.parent.relative_to(support) / selection_file
        if not selection.is_file():
            missing_selections += 1
            continue
        try:
            report = json.loads(selection.read_text())
            method = report.get("method")
            for key in SELECTION_FIELDS + METHOD_FIELDS.get(method, ()):
                if key in report:
                    observed["selection"][key].append(report[key])
            if method == "plateau" and "recommendation_factor" not in report:
                log = selection.with_suffix(".log")
                if log.is_file():
                    match = re.search(r"^Recommendation factor: ([^\s]+)$", log.read_text(), re.MULTILINE)
                    if match:
                        observed["selection"]["recommendation_factor"].append(float(match.group(1)))
        except (OSError, ValueError) as error:
            errors.append({"file": str(selection.relative_to(source_root)), "error": str(error)})

    saved_results = {group: {key: _summarize(values) for key, values in settings.items()}
                     for group, settings in observed.items()}
    saved_results.update(
        source="Saved curve NPZs and this summary's selection JSONs; these describe the stored results.",
        curve_count=len(curves), support_count=len(support_names),
        sample_sizes=sorted(seeds),
        random_seeds_by_sample_size={str(size): sorted(values) for size, values in sorted(seeds.items())},
        missing_selection_reports=missing_selections, metadata_read_errors=errors,
    )
    payload = {
        "schema_version": 2,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(root),
        "source_root": str(source_root),
        "selection_subdir": selection_subdir,
        "selection_file": selection_file,
        "last_command": command,
        "task_id": task_id,
        "configured_parameters": configured,
        "configured_sample_sizes": [int(value) for value in environ.get("STUDY_SAMPLE_SIZES", "").split()],
        "configured_trial_count": int(environ["TOTAL_TRIALS"]) if "TOTAL_TRIALS" in environ else None,
        "configured_parameters_note": (
            "Resolved shell settings for this invocation. Reselect reuses the saved search and fitting "
            "settings; aggregate/oracle/lasso/BIC retain the saved selection method. "
            "Selection-method options are omitted here; saved_results.selection records only "
            "parameters belonging to the methods used for the stored summaries."
        ),
        "saved_results": saved_results,
    }
    root.mkdir(parents=True, exist_ok=True)
    target = root / "methodological_parameters.json"
    descriptor, temporary = tempfile.mkstemp(prefix=".methodological_parameters_", suffix=".json", dir=root)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--command", required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--selection-subdir")
    parser.add_argument("--selection-file", default="selection_plateau_bootstrap.json")
    args = parser.parse_args()
    path = write_parameters(args.output_root, args.command, task_id=args.task_id,
                            source_root=args.source_root, selection_subdir=args.selection_subdir,
                            selection_file=args.selection_file)
    print(f"Methodological parameters: {path}")
