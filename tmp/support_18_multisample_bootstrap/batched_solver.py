"""Batch the existing fixed-omega ADMM updates without changing their math."""
import numpy as np
from admm import initialize_admm_state


def fit_batch(sigmas, masks, max_iter=800, tol=1e-7, zero_tol=1e-5, restarts=10):
    count, n, _ = sigmas.shape
    assert masks.shape == sigmas.shape and n == 4
    full = np.ones((n, n), dtype=bool)
    initial = np.array([initialize_admm_state(n, full, i, "halton")[0] for i in range(restarts)])
    sigma = np.repeat(sigmas, restarts, axis=0)
    mask = np.repeat(masks, restarts, axis=0)
    l1 = np.tile(initial, (count, 1, 1)) * mask
    l2 = l1.copy()
    dual = np.zeros_like(l1)
    identity = np.eye(n)
    active = np.arange(len(l1))
    for _ in range(max_iter):
        if not len(active):
            break
        s = sigma[active]
        old = l1[active].copy()
        second = l2[active]
        a = dual[active]
        m = mask[active]
        sl2 = s @ second
        a1 = 2 * sl2 @ sl2.swapaxes(-1, -2) + identity
        b1 = 2 * sl2 @ (s - identity) - a + second
        first = np.linalg.solve(a1, b1)
        first[~m] = 0.
        sl1 = s @ first
        a2 = 2 * sl1 @ sl1.swapaxes(-1, -2) + identity
        b2 = 2 * sl1 @ (s - identity) + a + first
        second = np.linalg.solve(a2, b2)
        second[~m] = 0.
        a = a + (first - second)
        finite = np.all(np.isfinite(first) & np.isfinite(second) & np.isfinite(a), axis=(1, 2))
        l1[active], l2[active], dual[active] = first, second, a
        l1[active[~finite]] = np.nan
        converged = np.linalg.norm(first - old, axis=(1, 2)) < tol
        active = active[finite & ~converged]
    l1[np.abs(l1) < zero_tol] = 0.
    residual = sigma - l1.swapaxes(-1, -2) @ sigma @ l1 - identity
    objectives = np.linalg.norm(residual, axis=(1, 2)) ** 2
    objectives[~np.isfinite(objectives)] = np.inf
    objectives = objectives.reshape(count, restarts)
    winners = objectives.argmin(axis=1)
    best = objectives[np.arange(count), winners]
    if not np.all(np.isfinite(best)):
        raise RuntimeError("No finite batched fit")
    return l1.reshape(count, restarts, n, n)[np.arange(count), winners], best
