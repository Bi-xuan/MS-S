"""Search over support masks and optimize Lambda for each candidate support."""

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from math import comb

import numpy as np

from admm import admm_solve
from objective import frobenius_objective
from supports.exact import get_all_supports


def threshold_lambda(Lambda, zero_tol):
    Lambda_thr = Lambda.copy()
    Lambda_thr[np.abs(Lambda_thr) < zero_tol] = 0.0
    return Lambda_thr


def is_finite_candidate(Lambda, omega, obj):
    return (
        np.all(np.isfinite(Lambda))
        and np.isfinite(omega)
        and np.isfinite(obj)
    )


def satisfies_hard_constraints(
    omega,
    min_omega,
    omega_upper=None,
):
    if omega < min_omega:
        return False

    if omega_upper is not None and omega > omega_upper:
        return False

    return True


def print_optimization_result(Lambda, omega, obj):
    if Lambda is None or omega is None or not np.isfinite(obj):
        print("No candidate satisfied the hard constraints.")
        return

    print(f"Best objective: {obj:.6f}")
    print(f"Best omega:     {omega:.6f}")
    print(f"Best Lambda:\n{Lambda}")


def solve_support_with_restarts(
    Sigma,
    mask,
    beta,
    max_iter,
    tol,
    zero_tol,
    max_restarts,
    min_omega,
    omega_fixed=None,
    omega_upper=None,
    init_strategy="halton",
):
    runs = []

    for restart_index in range(max_restarts):
        Lambda, omega = admm_solve(
            Sigma,
            mask,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            max_restarts=1,
            omega_fixed=omega_fixed,
            omega_upper=omega_upper,
            init_strategy=init_strategy,
            init_offset=restart_index,
        )
        Lambda_thr = threshold_lambda(Lambda, zero_tol)
        obj = frobenius_objective(Sigma, Lambda_thr, omega)

        if not is_finite_candidate(Lambda_thr, omega, obj):
            continue

        if not satisfies_hard_constraints(
            omega,
            min_omega,
            omega_upper=omega_upper,
        ):
            continue

        runs.append((Lambda_thr, omega, obj))

    if not runs:
        return None

    return min(runs, key=lambda run: run[2])


def solve_support_worker(task):
    (
        support_index,
        Sigma,
        mask,
        beta,
        max_iter,
        tol,
        zero_tol,
        max_restarts,
        min_omega,
        omega_fixed,
        omega_upper,
        seed,
        init_strategy,
    ) = task

    if seed is not None:
        np.random.seed(seed)

    result = solve_support_with_restarts(
        Sigma,
        mask,
        beta=beta,
        max_iter=max_iter,
        tol=tol,
        zero_tol=zero_tol,
        max_restarts=max_restarts,
        min_omega=min_omega,
        omega_fixed=omega_fixed,
        omega_upper=omega_upper,
        init_strategy=init_strategy,
    )

    return support_index, mask, result


def iter_parallel_support_results(tasks, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        task_iter = iter(tasks)
        pending = set()

        for _ in range(max_workers):
            try:
                pending.add(executor.submit(solve_support_worker, next(task_iter)))
            except StopIteration:
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)

            for future in done:
                yield future.result()

                try:
                    pending.add(executor.submit(solve_support_worker, next(task_iter)))
                except StopIteration:
                    pass


def update_best_support(
    best_Lambda,
    best_omega,
    best_obj,
    best_mask,
    mask,
    result,
    obj_tol,
):
    if result is None:
        return best_Lambda, best_omega, best_obj, best_mask

    Lambda, omega, obj = result

    if best_Lambda is None or obj < best_obj - obj_tol:
        return Lambda.copy(), omega, obj, mask.copy()

    return best_Lambda, best_omega, best_obj, best_mask


def optimize_lambda(
    Sigma,
    D_m,
    beta=1.0,
    max_iter=500,
    tol=1e-6,
    zero_tol=1e-5,
    obj_tol=1e-8,
    max_restarts=3,
    min_omega=1e-8,
    omega_upper_gap=1e-3,
    omega_ref=None,
    support_iterator=None,
    n_jobs=1,
    random_seed=None,
    init_strategy="halton",
    refine_after_fixed_omega=False,
):
    """
    Optimize Lambda over a support iterator.

    By default this uses exact exhaustive support enumeration. For larger
    problems, pass either a support_iterator(n, n_edge) callable or an iterable
    that yields support masks.
    """
    n = Sigma.shape[0]
    n_edge = D_m - 1
    lambda_min_sigma = np.min(np.linalg.eigvalsh(Sigma))
    omega_upper = lambda_min_sigma - omega_upper_gap

    if refine_after_fixed_omega and omega_ref is None:
        raise ValueError(
            "refine_after_fixed_omega=True requires omega_ref to be set. "
            "Set omega_ref to a fixed value, or set "
            "refine_after_fixed_omega=False when running the first stage "
            "with omega_ref=None."
        )

    if omega_upper < min_omega:
        return None, None, np.inf

    if omega_ref is not None and omega_ref > omega_upper:
        return None, None, np.inf

    best_obj = np.inf
    best_Lambda = None
    best_omega = None
    best_mask = None
    if support_iterator is None:
        support_iterator = get_all_supports

    def iter_support_masks():
        if callable(support_iterator):
            yield from support_iterator(n, n_edge)
        else:
            yield from support_iterator

    def iter_support_tasks():
        for support_index, mask in enumerate(iter_support_masks()):
            yield (
                support_index,
                Sigma,
                mask,
                beta,
                max_iter,
                tol,
                zero_tol,
                max_restarts,
                min_omega,
                omega_ref,
                omega_upper,
                None if random_seed is None else random_seed + support_index,
                init_strategy,
            )

    if n_jobs is None:
        max_workers = os.cpu_count() or 1
    else:
        max_workers = n_jobs

    if max_workers is not None and max_workers < 1:
        raise ValueError("n_jobs must be positive or None.")

    support_count = None
    if support_iterator is get_all_supports:
        support_count = comb(n * (n - 1), n_edge)

    if max_workers == 1 or support_count == 1:
        support_results = map(solve_support_worker, iter_support_tasks())
    else:
        support_results = iter_parallel_support_results(
            iter_support_tasks(),
            max_workers,
        )

    for _, mask, result in support_results:
        best_Lambda, best_omega, best_obj, best_mask = update_best_support(
            best_Lambda,
            best_omega,
            best_obj,
            best_mask,
            mask,
            result,
            obj_tol,
        )

    if (
        refine_after_fixed_omega
        and omega_ref is not None
        and best_mask is not None
    ):
        if random_seed is not None:
            final_seed_offset = support_count if support_count is not None else 0
            np.random.seed(random_seed + final_seed_offset)

        final_result = solve_support_with_restarts(
            Sigma,
            best_mask,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            zero_tol=zero_tol,
            max_restarts=max_restarts,
            min_omega=min_omega,
            omega_fixed=None,
            omega_upper=omega_upper,
            init_strategy=init_strategy,
        )

        if final_result is None:
            return None, None, np.inf

        best_Lambda, best_omega, best_obj = final_result

    return best_Lambda, best_omega, best_obj
