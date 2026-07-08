"""Experiment engine for Exp 1 (gauge vs dimension).

Outcome definitions (frozen; see pre-registration cell in notebook):

- Planted weights: W0 = orthonormal basis of the target subspace (n x r),
  noise model W_sigma = W0 + sigma * E, E_ij ~ N(0, 1/sqrt(n)) so that the
  expected column norm of sigma*E is sigma (noise scale comparable across n).

- Detection: isotypic energy fraction f_P(W) = ||P W||_F^2 / ||W||_F^2 against
  the spectrum-preserving rotation null f_P(Q W), Q ~ Haar(O(n)).
  p = (1 + #{f_P(QW) >= f_P(W)}) / (B + 1).  Detected iff p <= alpha.

- sigma*_det (per seed): first sigma on the grid where detection fails,
  after isotonic cleanup (a single spurious re-detection at higher sigma does
  not resurrect); summary = median over seeds.

- Invariance defect delta(W): per the paper's Eq. 3, computed on the
  orthonormalized column space, averaged over GENERATORS (not all g) for
  speed -- verified to be a monotone proxy for the all-g version.

- Copy-fidelity decomposition (Step 2): planted copy U (one d-dim invariant
  copy inside a multiplicity-m isotypic block):
    block_containment = ||P_block Uhat||_F^2 / d      (stayed in the block)
    copy_fidelity     = ||U^T Uhat||_F^2 / d          (stayed the SAME copy)
    gauge_drift       = block_containment - copy_fidelity
  gauge_drift is energy that remained in the isotypic block but moved to a
  different (gauge-equivalent) copy: "rotated within its commutant".
"""
import numpy as np


def haar_orthogonal(n, rng):
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))


def energy_fraction(P, W):
    num = np.linalg.norm(P @ W) ** 2
    den = np.linalg.norm(W) ** 2
    return num / den


def rotation_null_pvalue(P, W, B, rng):
    f_obs = energy_fraction(P, W)
    count = 0
    n = W.shape[0]
    for _ in range(B):
        Q = haar_orthogonal(n, rng)
        if energy_fraction(P, Q @ W) >= f_obs:
            count += 1
    return (1 + count) / (B + 1)


def invariance_defect(W, rep_mats):
    """delta over the supplied representation matrices (use generators)."""
    Q, _ = np.linalg.qr(W)
    Pr = Q @ Q.T
    nrm = np.linalg.norm(Pr)
    vals = [np.linalg.norm(L @ Pr - Pr @ L) / nrm for L in rep_mats]
    return float(np.mean(vals))


def orthonormal_basis(P):
    r = int(np.round(np.trace(P)))
    U, s, _ = np.linalg.svd(P)
    return U[:, :r]


def sigma_star_detection(P, W0, sigma_grid, alpha, B, n_seeds, rng):
    """Per-seed sigma* then median. Returns (median_sigma_star, per_seed array,
    detection_rate_per_sigma)."""
    n = W0.shape[0]
    det = np.zeros((n_seeds, len(sigma_grid)), dtype=bool)
    for s in range(n_seeds):
        E = rng.standard_normal(W0.shape) / np.sqrt(n)
        for j, sig in enumerate(sigma_grid):
            W = W0 + sig * E
            p = rotation_null_pvalue(P, W, B, rng)
            det[s, j] = p <= alpha
    # isotonic cleanup: once failed, stays failed
    stars = np.empty(n_seeds)
    for s in range(n_seeds):
        ok = det[s].copy()
        fail = np.where(~ok)[0]
        if len(fail) == 0:
            stars[s] = sigma_grid[-1]          # right-censored
        else:
            stars[s] = sigma_grid[fail[0]]     # first failure
    return float(np.median(stars)), stars, det.mean(axis=0)


# ---------------- Step 2: multiplicity construction for S_n standard irrep ----
def standard_irrep_matrices(group, n_points):
    """S_n acting on n_points via its natural permutation action, restricted
    to the sum-zero subspace -> the (n_points-1)-dim standard irrep, as
    explicit orthogonal matrices for every group element."""
    # group.elems are permutation tuples of range(n_points)
    ones = np.ones(n_points) / np.sqrt(n_points)
    # orthonormal basis of sum-zero subspace
    Bz = np.linalg.svd(np.eye(n_points) - np.outer(ones, ones))[0][:, :n_points - 1]
    mats = []
    for p in group.elems:
        Pm = np.zeros((n_points, n_points))
        for i in range(n_points):
            Pm[p[i], i] = 1.0
        mats.append(Bz.T @ Pm @ Bz)
    return mats  # list of (n_points-1) x (n_points-1) orthogonal matrices


def multiplicity_rep(std_mats, m, ambient, rng):
    """Representation on R^ambient containing the standard irrep with
    multiplicity exactly m: rho = std^{oplus m} oplus trivial^{oplus rest},
    conjugated by a fixed random orthogonal basis so nothing is axis-aligned.
    Returns (rep matrices per element, block projector P_std, one-copy basis U)."""
    d = std_mats[0].shape[0]
    assert m * d <= ambient
    Qamb = haar_orthogonal(ambient, rng)
    reps, P = [], np.zeros((ambient, ambient))
    for M in std_mats:
        R = np.eye(ambient)
        for c in range(m):
            sl = slice(c * d, (c + 1) * d)
            R[sl, sl] = M
        reps.append(Qamb @ R @ Qamb.T)
    P[:m * d, :m * d] = np.eye(m * d)
    P = Qamb @ P @ Qamb.T
    U = Qamb[:, :d]                       # one specific copy
    return reps, P, U


def copy_decomposition(U, W, P_block):
    """Given noisy W spanning ~the planted copy U (n x d), decompose where the
    energy went."""
    Q, _ = np.linalg.qr(W)
    d = U.shape[1]
    block = np.linalg.norm(P_block @ Q) ** 2 / d
    copy = np.linalg.norm(U.T @ Q) ** 2 / d
    return dict(block_containment=block, copy_fidelity=copy,
                gauge_drift=block - copy)


# ---------------- statistics -------------------------------------------------
def partial_spearman(y, x, z):
    """Spearman partial correlation of y with x controlling z (all 1-d)."""
    from scipy.stats import rankdata
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ey, ex = resid(ry, rz), resid(rx, rz)
    denom = (np.linalg.norm(ey) * np.linalg.norm(ex))
    if denom == 0:
        return 0.0
    return float(ey @ ex / denom)


def cluster_bootstrap_ci(df_rows, stat_fn, cluster_key, n_boot=2000, seed=0,
                         alpha=0.05):
    """Nonparametric cluster bootstrap: resample clusters with replacement."""
    rng = np.random.default_rng(seed)
    clusters = sorted({r[cluster_key] for r in df_rows})
    by_c = {c: [r for r in df_rows if r[cluster_key] == c] for c in clusters}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [r for c in pick for r in by_c[c]]
        try:
            stats.append(stat_fn(sample))
        except Exception:
            continue
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(stats)), (float(lo), float(hi))


def page_trend_L(matrix):
    """Page's L statistic for ordered alternatives. matrix: subjects x
    conditions, conditions in hypothesized increasing order. Returns (L, z)."""
    from scipy.stats import rankdata
    n, k = matrix.shape
    ranks = np.vstack([rankdata(row) for row in matrix])
    R = ranks.sum(axis=0)
    L = float(np.sum(np.arange(1, k + 1) * R))
    EL = n * k * (k + 1) ** 2 / 4
    VL = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144
    z = (L - EL) / np.sqrt(VL)
    return L, float(z)
