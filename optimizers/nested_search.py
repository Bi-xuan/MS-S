"""Forward/backward support search with reconstruction of one nested curve."""

from dataclasses import dataclass
import hashlib

import numpy as np

from optimizers import support_search as search
from supports.common import validate_support_mask


SEARCH_OPTIONS = ("forward", "forward-backward-nested", "greedy")


class NestedCurveResult(tuple):
    """Keep the legacy six curve arrays, with separate checkpoint history."""

    def __new__(cls, arrays, search_state):
        result = super().__new__(cls, arrays)
        result.search_state = search_state
        return result


@dataclass
class SupportFit:
    mask: np.ndarray
    Lambda: np.ndarray
    omega: float
    objective: float


class CachedSupportFitter:
    """Fit each mask once; starts and ties are independent of search order."""

    def __init__(self, Sigma, *, beta, max_iter, tol, zero_tol, max_restarts,
                 min_omega, omega_ref, n_jobs, random_seed, init_strategy,
                 refine_after_fixed_omega):
        if n_jobs is not None and n_jobs < 1:
            raise ValueError("n_jobs must be positive or None.")
        if refine_after_fixed_omega and omega_ref is None:
            raise ValueError("refine_after_fixed_omega=True requires omega_ref to be set.")
        self.Sigma = Sigma
        self.settings = (beta, max_iter, tol, zero_tol, max_restarts, min_omega)
        self.omega_ref = omega_ref
        self.omega_upper = float(np.linalg.eigvalsh(Sigma)[0])
        if omega_ref is None:
            self.omega_upper -= 1e-6
        self.n_jobs = n_jobs
        self.random_seed = 0 if random_seed is None else random_seed
        self.init_strategy = init_strategy
        self.refine = refine_after_fixed_omega
        self.cache = {}

    def _task(self, index, mask, refine=False):
        seed = None
        if self.random_seed is not None:
            digest = hashlib.sha256(mask.tobytes()).digest()
            seed = (int(self.random_seed) + int.from_bytes(digest[:4], "little")) % 2**32
        omega = None if refine else self.omega_ref
        upper = self.omega_upper - 1e-6 if refine else self.omega_upper
        return (index, self.Sigma, mask, *self.settings, omega, upper,
                seed, self.init_strategy)

    def _run(self, masks, refine=False):
        tasks = [self._task(index, mask, refine) for index, mask in enumerate(masks)]
        if self.n_jobs == 1 or len(tasks) <= 1:
            return map(search.solve_support_worker, tasks)
        return search.iter_parallel_support_results(tasks, self.n_jobs)

    def best(self, masks):
        masks = [np.asarray(mask, dtype=bool).copy() for mask in masks]
        missing = {mask.tobytes(): mask for mask in masks if mask.tobytes() not in self.cache}
        feasible = self.omega_upper >= self.settings[-1]
        feasible &= self.omega_ref is None or self.omega_ref <= self.omega_upper
        if self.refine:
            feasible &= self.omega_upper - 1e-6 >= self.settings[-1]
        if not feasible:
            self.cache.update({key: None for key in missing})
        else:
            for _, mask, fit in self._run(list(missing.values())):
                self.cache[mask.tobytes()] = (
                    None if fit is None else SupportFit(mask.copy(), *fit)
                )
            if self.refine:
                valid_masks = [mask for mask in missing.values() if self.cache[mask.tobytes()] is not None]
                for _, mask, fit in self._run(valid_masks, refine=True):
                    self.cache[mask.tobytes()] = (
                        None if fit is None else SupportFit(mask.copy(), *fit)
                    )
        fits = [self.cache[mask.tobytes()] for mask in masks]
        # Enumeration order is canonical in the caller, including parallel fits.
        return min((fit for fit in fits if fit is not None),
                   key=lambda fit: (fit.objective, search.support_edges_from_mask(fit.mask)),
                   default=None)


def compute_forward_backward_nested_curve(
    Sigma, allowed_edges, *, obj_tol, initial_curve_result=None,
    save_callback=None, **fit_settings,
):
    """Correct endpoints, rebuild their prefixes, and resume forward expansion.

    Historical best objectives are guards against cycling, not the reported
    curve. Reconstructed earlier models may have worse objectives than before.
    """
    if not np.isfinite(obj_tol) or obj_tol < 0:
        raise ValueError("obj_tol must be finite and nonnegative.")
    n = len(Sigma)
    allowed_edges = sorted(allowed_edges)
    max_edges = len(allowed_edges)
    fitter = CachedSupportFitter(Sigma, **fit_settings)
    best_seen = [np.inf] * (max_edges + 1)
    corrections = 0

    def deletion_masks(mask):
        for edge in allowed_edges:
            if mask[edge]:
                candidate = mask.copy()
                candidate[edge] = False
                yield candidate

    def rebuild(endpoint):
        prefix = [endpoint]
        while np.count_nonzero(prefix[-1].mask) > n:
            smaller = fitter.best(deletion_masks(prefix[-1].mask))
            if smaller is None:
                return None
            prefix.append(smaller)
        return prefix[::-1]

    def result(path, complete=False):
        objectives = [np.inf if fit is None else fit.objective for fit in path]
        masks = [np.zeros((n, n), dtype=bool) if fit is None else fit.mask.copy() for fit in path]
        state = dict(version=1, complete=complete, corrections=corrections,
                     best_seen=[None if not np.isfinite(value) else float(value) for value in best_seen])
        return NestedCurveResult((
            np.arange(1, len(path) + 1), np.asarray(objectives),
            np.array([], dtype=int), np.array([], dtype=float),
            np.asarray(masks, dtype=bool), np.asarray([fit is not None for fit in path]),
        ), state)

    def checkpoint(path, complete=False):
        value = result(path, complete)
        if save_callback is not None:
            save_callback(value)
        return value

    state = getattr(initial_curve_result, "search_state", None)
    if state is not None:
        if state.get("version") != 1 or len(state.get("best_seen", [])) != max_edges + 1:
            raise ValueError("Invalid forward-backward-nested checkpoint history.")
        best_seen = [np.inf if value is None else float(value) for value in state["best_seen"]]
        if any(np.isnan(value) or value < 0 for value in best_seen):
            raise ValueError("Invalid historical support objectives.")
        corrections = int(state.get("corrections", 0))
        if state.get("complete"):
            if len(initial_curve_result[0]) != max_edges + 1:
                raise ValueError("Completed search checkpoint must contain every dimension.")
            return initial_curve_result
        path = []
        for index, (mask, objective) in enumerate(zip(initial_curve_result[4], initial_curve_result[1])):
            mask = validate_support_mask(mask, n, index, allowed_edges)
            fit = fitter.best([mask])
            if fit is None or not np.isclose(fit.objective, objective, rtol=1e-7, atol=obj_tol):
                raise ValueError("Saved nested support does not reproduce its checkpoint objective.")
            path.append(fit)
            best_seen[index] = min(best_seen[index], objective)
    else:
        if initial_curve_result is not None:
            print("No backward-search history saved; restarting the search from D_m=1.")
        path = []

    if not path:
        print("Solving for D_m = 1...")
        fit = fitter.best([np.eye(n, dtype=bool)])
        if fit is None:
            return checkpoint([None] * (max_edges + 1), complete=True)
        path = [fit]
        best_seen[0] = fit.objective
        print(f"  Objective = {fit.objective:.6f}")
        checkpoint(path, complete=max_edges == 0)

    while len(path) <= max_edges:
        k = len(path) - 1
        print(f"Solving for D_m = {k + 2}...")
        extensions = []
        for edge in allowed_edges:
            if not path[-1].mask[edge]:
                mask = path[-1].mask.copy()
                mask[edge] = True
                extensions.append(mask)
        expanded = fitter.best(extensions)
        if expanded is None:
            print("  No finite extension; recording remaining objectives = Inf.")
            return checkpoint(path + [None] * (max_edges + 1 - len(path)), complete=True)

        corrected = fitter.best(deletion_masks(expanded.mask)) if k else None
        threshold = min(path[-1].objective, best_seen[k])
        if corrected is not None and corrected.objective < threshold - obj_tol:
            prefix = rebuild(corrected)
            if prefix is not None:
                removed = search.support_edges_from_mask(expanded.mask & ~corrected.mask)
                print(f"  Backward correction at D_m={k + 1}: removed {removed}; "
                      f"objective {path[-1].objective:.9g} -> {corrected.objective:.9g}.")
                print(f"  Rebuilding nested prefix D_m=1..{k + 1}.")
                path = prefix
                corrections += 1
                for index, fit in enumerate(path):
                    best_seen[index] = min(best_seen[index], fit.objective)
                checkpoint(path)
                continue
            print("  Correction rejected: no finite nested prefix could be rebuilt.")

        path.append(expanded)
        best_seen[k + 1] = min(best_seen[k + 1], expanded.objective)
        print(f"  Objective = {expanded.objective:.6f}")
        checkpoint(path, complete=len(path) == max_edges + 1)

    return result(path, complete=True)
