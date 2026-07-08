"""Core for Experiment A (gauge diffusion across training checkpoints) and
Experiment C (intervention transfer across checkpoints/seeds up to gauge
alignment), for LLMs with cyclic vocabularies.

Regime logic (why checkpoints, not fine-tuning deltas -- Pred 2's failure):
directed gradient steps avoid flat directions; long stochastic training
diffuses along them. Post-convergence checkpoint-to-checkpoint change should
therefore be commutant-tangent above null (A), and any intervention keyed to
a specific orientation should decay in transferability with checkpoint gap
unless gauge-aligned (C).

All statistics reuse the calibrated machinery from Pred 1/2:
- tangent fraction of a change (commutant_change_decomposition)
- empirical nulls from random changes of matched size
- permutation placebo (analysis under shuffled token order)
- Procrustes gauge alignment with spectral floor
"""
import numpy as np


def fourier_modes(n):
    j = np.arange(n)
    modes = [(0, np.ones((n, 1)) / np.sqrt(n))]
    for k in range(1, n // 2 + 1):
        c = np.cos(2 * np.pi * k * j / n)
        s = np.sin(2 * np.pi * k * j / n)
        M = np.column_stack([c, s]) if (2 * k != n) else c[:, None]
        Q, _ = np.linalg.qr(M)
        modes.append((k, Q))
    assert sum(B.shape[1] for _, B in modes) == n
    return modes


def circle_energy_pvalue(X, modes, B_rot=499, rng=None):
    """Calibrated structure gate: is mode-1 energy above the spectrum-preserving
    rotation null? Returns (energy_fraction, p)."""
    rng = rng or np.random.default_rng(0)
    P1 = modes[1][1] @ modes[1][1].T
    Xc = X - X.mean(axis=0, keepdims=True)
    def frac(Y):
        return np.linalg.norm(P1 @ Y) ** 2 / np.linalg.norm(Y) ** 2
    f_obs = frac(Xc)
    n = X.shape[0]
    count = 0
    for _ in range(B_rot):
        A = rng.standard_normal((n, n))
        Q, R = np.linalg.qr(A)
        Q = Q * np.sign(np.diag(R))
        if frac(Q @ Xc) >= f_obs:
            count += 1
    return float(f_obs), (1 + count) / (B_rot + 1)


def tangent_fraction(X_from, X_to, modes, k):
    """Fraction of the mode-k change lying in the commutant tangent of X_from."""
    B = dict(modes)[k]
    Cb, Cn = B.T @ X_from, B.T @ X_to
    D = Cn - Cb
    if Cb.shape[0] != 2:
        c, d = Cb[0], D[0]
        nc, nd = np.linalg.norm(c), np.linalg.norm(d)
        if nc < 1e-12 or nd < 1e-12:
            return np.nan
        return float((c @ d) ** 2 / (nc ** 2 * nd ** 2))
    z = Cb[0] + 1j * Cb[1]
    d = D[0] + 1j * D[1]
    nz, nd = np.linalg.norm(z), np.linalg.norm(d)
    if nz < 1e-12 or nd < 1e-12:
        return np.nan
    u = z / nz
    return float(np.abs(np.vdot(u, d)) ** 2 / nd ** 2)


def circle_phase(X, modes, ref_u=None):
    """Phase of the mode-1 content relative to a reference complex direction.
    Returns (phase, u) where u can be reused as the reference for a trajectory."""
    B = dict(modes)[1]
    C = B.T @ X
    z = C[0] + 1j * C[1]
    if ref_u is None:
        ref_u = z / np.linalg.norm(z)
        return 0.0, ref_u
    w = np.vdot(ref_u, z)
    return float(np.angle(w)), ref_u


def mode_energies(X, modes):
    return {k: float(np.linalg.norm(B.T @ X) ** 2) for k, B in modes}


def empirical_null_tangent(X_base, modes, k, delta_norm, n_draws, rng):
    fr = []
    for _ in range(n_draws):
        D = rng.standard_normal(X_base.shape)
        D *= delta_norm / np.linalg.norm(D)
        fr.append(tangent_fraction(X_base, X_base + D, modes, k))
    return np.array(fr)


def placebo_tangent(X_from, X_to, modes, k, n_perm, rng):
    n = X_from.shape[0]
    fr = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        fr.append(tangent_fraction(X_from[perm], X_to[perm], modes, k))
    return np.array(fr)


def procrustes_align(XA, XB):
    U, s, Vt = np.linalg.svd(XA.T @ XB)
    Q = U @ Vt
    return Q, float(np.linalg.norm(XA @ Q - XB) ** 2)


def circle_plane(X, modes):
    """Orthonormal basis (d x 2) of the model-space plane carrying mode 1."""
    B = dict(modes)[1]
    C = B.T @ X
    Qp, _ = np.linalg.qr(C.T)
    return Qp


def ablate_plane(X, V):
    return X - (X @ V) @ V.T


def transfer_efficiency(damage_transferred, damage_self):
    if abs(damage_self) < 1e-9:
        return np.nan
    return float(damage_transferred / damage_self)
