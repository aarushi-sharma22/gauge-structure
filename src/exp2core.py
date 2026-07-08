"""Experiment 2 core (LLM deviation causality). Model-agnostic pieces:
Fourier decomposition over a cyclic vocabulary, the intervention family,
and the statistics. All validated on synthetic ground truth before use.

Decomposition convention (frozen):
  X in R^{n x d}: rows = token embeddings for the n cyclic items.
  Real Fourier basis F over Z/n: mode 0 (constant), modes 1..floor(n/2)
  (cos/sin pairs; the Nyquist mode is 1-dim for even n).
  circle    = mode k=1
  deviation = modes k>=2
  mode 0 (mean) is ALWAYS preserved by every intervention.

Interventions (all preserve mode 0 and mode 1 exactly; asserted):
  intact                  X unchanged
  circle_only             deviation zeroed
  matched_random(draw)    deviation replaced by random content with the SAME
                          per-mode Frobenius energy (N independent draws)
  rotated_within_modes    deviation rotated by an independent Haar rotation
                          inside each mode's token-subspace: preserves the
                          per-mode energy AND each mode's singular spectrum,
                          changes only orientation (strongest control)
  dose(alpha)             circle + alpha * true deviation
  dose_ctrl(alpha, draw)  circle + alpha * matched random deviation
  zero_rows               positive control: token rows zeroed
"""
import numpy as np


def fourier_modes(n):
    """Returns list of (k, B_k) with B_k an orthonormal basis (n x dim_k) of
    mode k's token subspace. Sum of dims = n."""
    j = np.arange(n)
    modes = [(0, np.ones((n, 1)) / np.sqrt(n))]
    for k in range(1, n // 2 + 1):
        c = np.cos(2 * np.pi * k * j / n)
        s = np.sin(2 * np.pi * k * j / n)
        M = np.column_stack([c, s]) if (2 * k != n) else c[:, None]
        Q, _ = np.linalg.qr(M)
        modes.append((k, Q))
    total = sum(B.shape[1] for _, B in modes)
    assert total == n
    return modes


class CyclicDecomposition:
    def __init__(self, n):
        self.n = n
        self.modes = fourier_modes(n)
        self.P = {k: B @ B.T for k, B in self.modes}

    def components(self, X):
        return {k: self.P[k] @ X for k, _ in self.modes}

    def mode_energies(self, X):
        return {k: float(np.linalg.norm(self.P[k] @ X) ** 2) for k, _ in self.modes}

    # ------------------------- interventions -------------------------
    def circle_only(self, X):
        return self.P[0] @ X + self.P[1] @ X

    def matched_random(self, X, rng):
        out = self.P[0] @ X + self.P[1] @ X
        for k, B in self.modes:
            if k < 2:
                continue
            target = np.linalg.norm(self.P[k] @ X)
            R = B @ rng.standard_normal((B.shape[1], X.shape[1]))
            nr = np.linalg.norm(R)
            if nr > 0:
                out = out + R * (target / nr)
        return out

    def rotated_within_modes(self, X, rng):
        out = self.P[0] @ X + self.P[1] @ X
        for k, B in self.modes:
            if k < 2:
                continue
            dk = B.shape[1]
            coords = B.T @ X                       # dk x d
            A = rng.standard_normal((dk, dk))
            Q, Rr = np.linalg.qr(A)
            Q = Q * np.sign(np.diag(Rr))
            out = out + B @ (Q @ coords)
        return out

    def dose(self, X, alpha):
        dev = X - self.P[0] @ X - self.P[1] @ X
        return self.P[0] @ X + self.P[1] @ X + alpha * dev

    def dose_ctrl(self, X, alpha, rng):
        rand_full = self.matched_random(X, rng)
        dev = rand_full - self.P[0] @ rand_full - self.P[1] @ rand_full
        return self.P[0] @ X + self.P[1] @ X + alpha * dev

    def check_circle_preserved(self, X, Y, tol=1e-8):
        for k in (0, 1):
            if not np.allclose(self.P[k] @ X, self.P[k] @ Y, atol=tol):
                return False
        return True


# ----------------------------- statistics -----------------------------------
def bootstrap_ci_accuracy(correct, cluster_ids, n_boot=5000, seed=0, alpha=0.05):
    """Cluster bootstrap over items (a 'cluster' = one base item, so prompts
    sharing an operand are resampled together)."""
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    clusters = np.unique(cluster_ids)
    members = {c: np.where(cluster_ids == c)[0] for c in clusters}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        idxs = np.concatenate([members[c] for c in pick])
        stats[b] = correct[idxs].mean()
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(correct.mean()), (float(lo), float(hi))


def draw_null_pvalue(acc_intact, acc_draws):
    """Empirical one-sided p for 'intact beats the matched-random null':
    fraction of random-deviation draws achieving >= intact accuracy."""
    acc_draws = np.asarray(acc_draws, dtype=float)
    return float((1 + np.sum(acc_draws >= acc_intact)) / (1 + len(acc_draws)))


def gap_and_z(acc_intact, acc_draws):
    mu, sd = float(np.mean(acc_draws)), float(np.std(acc_draws, ddof=1))
    z = (acc_intact - mu) / sd if sd > 0 else np.inf
    return acc_intact - mu, z


def page_trend_L(matrix):
    """Page's L for ordered alternatives; matrix subjects x conditions,
    conditions in hypothesized increasing order."""
    from scipy.stats import rankdata
    n, k = matrix.shape
    ranks = np.vstack([rankdata(row) for row in matrix])
    R = ranks.sum(axis=0)
    L = float(np.sum(np.arange(1, k + 1) * R))
    EL = n * k * (k + 1) ** 2 / 4
    VL = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144
    return L, float((L - EL) / np.sqrt(VL))


def selectivity_did(drop_modular, drop_control):
    """Difference-in-differences: (modular impairment) - (control-task
    impairment). Positive = damage selective to modular computation."""
    return float(drop_modular - drop_control)


# ============ Fixes added after the Qwen pilot (problems 1-3) ============
from scipy.stats import mannwhitneyu as _mannwhitneyu

def report_null_comparison(acc_intact, acc_draws):
    """Fix 1: honest null report. Empirical p (bounded by draw count) plus a
    DESCRIPTIVE effect size explicitly NOT presented as an inferential z.
    (The old gap_and_z produced z~14 from 25 draws, implying p~1e-44, which is
    false precision. Use empirical p and report intact-vs-draw-range instead.)"""
    acc_draws = np.asarray(acc_draws, float)
    p_emp = (1 + np.sum(acc_draws >= acc_intact)) / (1 + len(acc_draws))
    gap = acc_intact - acc_draws.mean()
    sd = acc_draws.std(ddof=1)
    return dict(p_empirical=float(p_emp), gap=float(gap),
                effect_size_descriptive=float(gap / sd) if sd > 0 else float("inf"),
                n_draws=int(len(acc_draws)),
                intact_exceeds_all=bool(acc_intact > acc_draws.max()),
                draw_min=float(acc_draws.min()), draw_max=float(acc_draws.max()))

def rotation_vs_matched(rot_draws, matched_draws):
    """Fix 2: the spectrum-preserving rotation control is high-VARIANCE, not a
    point estimate (real Qwen draws ran 0.064-0.308). Reporting its mean hides
    the story. Report the distribution and test it against the matched-random
    distribution as a two-sample comparison (orientation matters iff rotating
    the real deviation differs from replacing it with matched noise)."""
    r, m = np.asarray(rot_draws, float), np.asarray(matched_draws, float)
    _, p = _mannwhitneyu(r, m, alternative="two-sided")
    return dict(rot_median=float(np.median(r)), matched_median=float(np.median(m)),
                rot_iqr=float(np.subtract(*np.percentile(r, [75, 25]))),
                rot_range=(float(r.min()), float(r.max())), p_vs_matched=float(p))

def did_bootstrap(mi_c, mi_i, mc_c, mc_i, ci_c, ci_i, cc_c, cc_i,
                  items=12, n_boot=5000, seed=0):
    """Fix 3: item-cluster bootstrap of the difference-in-differences, giving
    the selectivity DiD an actual CI (the old code reported a bare point).
    _c = per-item correctness (0/1 array), _i = per-item month label.
    mi/mc = modular intact/corrupted, ci/cc = control intact/corrupted.
    Resamples the 12 months with replacement; excludes_zero is the real test."""
    rng = np.random.default_rng(seed)
    pool = np.arange(items)
    def cell(c, it, keep):
        m = np.isin(it, keep)
        return c[m].mean() if m.any() else np.nan
    stats = np.empty(n_boot)
    for b in range(n_boot):
        keep = rng.choice(pool, size=items, replace=True)
        stats[b] = ((cell(mi_c, mi_i, keep) - cell(mc_c, mc_i, keep))
                    - (cell(ci_c, ci_i, keep) - cell(cc_c, cc_i, keep)))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return dict(did=float(np.nanmean(stats)), ci=(float(lo), float(hi)),
                excludes_zero=bool(lo > 0))
