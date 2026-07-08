"""Core for Predictions 1 & 2: seed-gauge analysis of learned modular-arithmetic
embeddings, and commutant decomposition of fine-tuning changes.

Definitions (frozen):

- Embedding X in R^{p x N}: row t = model-space vector of token t.
- Real Fourier modes B_k over Z/p (from fourier_modes): C_k = B_k^T X, the
  mode-k content (2 x N, or 1 x N for the Nyquist mode).
- Complex form of a 2-row mode content: z = C[0] + i C[1] in C^N. The token-side
  commutant of the Z/p action on mode k is C^* acting as z -> w z
  (rotation+scale). The commutant ORBIT TANGENT at z is span_R{z, iz}.

GAUGE-INVARIANT per mode: energy ||C_k||_F^2 and singular values of C_k.
GAUGE-DEPENDENT: everything else (phase, model-space plane).

Alignment test (Prediction 1): the full gauge group between two seeds is
(model-space orthogonal Q) x (token-side per-mode commutant). With large N the
model-space Q absorbs token-side phases, so alignment = orthogonal Procrustes
X_A Q ~ X_B. No gauge map can change per-mode singular values, so
    residual^2 >= spectral_floor := sum_k sum_i (s_i^A(k) - s_i^B(k))^2.
Prediction: residual^2 ~ spectral_floor << raw difference. Falsifier: a large
gap between residual and floor = genuine non-gauge structural difference.

Change decomposition (Prediction 2): for small change D_k = C_k' - C_k, the
commutant-tangent fraction is ||proj_{span_R{z,iz}} d||^2 / ||d||^2 where
d = D_k in complex form. Null level for an unstructured change: 1/N.
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


# ---------------- gauge-invariant / gauge-dependent inventory ----------------
def mode_table(X, modes):
    """Per-mode: energy, singular values, complex content z (or None for 1-dim
    modes)."""
    rows = []
    for k, B in modes:
        C = B.T @ X
        s = np.linalg.svd(C, compute_uv=False)
        z = (C[0] + 1j * C[1]) if C.shape[0] == 2 else None
        rows.append(dict(k=k, dim=C.shape[0], energy=float(np.sum(C ** 2)),
                         svals=s, z=z, C=C))
    return rows

def active_modes(table, factor=3.0):
    """Modes whose energy exceeds `factor` x the uniform share (excl. mode 0)."""
    tot = sum(r['energy'] for r in table if r['k'] > 0)
    n_modes = sum(1 for r in table if r['k'] > 0)
    thr = factor * tot / n_modes
    return sorted(r['k'] for r in table if r['k'] > 0 and r['energy'] > thr)


# ---------------- Prediction 1: alignment with spectral floor -----------------
def procrustes_align(XA, XB):
    """Q = argmin_{Q in O(N)} ||XA Q - XB||_F. Returns Q and residual^2."""
    U, s, Vt = np.linalg.svd(XA.T @ XB)
    Q = U @ Vt
    res2 = float(np.linalg.norm(XA @ Q - XB) ** 2)
    return Q, res2

def spectral_floor(tabA, tabB):
    """sum over modes of squared singular-value mismatch (gauge cannot fix)."""
    tot = 0.0
    for a, b in zip(tabA, tabB):
        sa, sb = a['svals'], b['svals']
        m = max(len(sa), len(sb))
        sa = np.pad(sa, (0, m - len(sa))); sb = np.pad(sb, (0, m - len(sb)))
        tot += float(np.sum((sa - sb) ** 2))
    return tot

def seed_pair_report(XA, XB, modes):
    tabA, tabB = mode_table(XA, modes), mode_table(XB, modes)
    raw = float(np.linalg.norm(XA - XB) ** 2)
    Q, res2 = procrustes_align(XA, XB)
    floor = spectral_floor(tabA, tabB)
    scale = float(np.linalg.norm(XA) ** 2 + np.linalg.norm(XB) ** 2) / 2
    return dict(raw=raw, aligned_residual=res2, spectral_floor=floor,
                raw_frac=raw / scale, aligned_frac=res2 / scale,
                floor_frac=floor / scale, Q=Q,
                active_A=active_modes(tabA), active_B=active_modes(tabB))


# ---------------- Prediction 2: commutant-tangent change decomposition -------
def commutant_change_decomposition(X_base, X_new, modes):
    """Per mode: split the CHANGE into commutant-tangent part (rotation+scale
    of the base content) vs orthogonal residual. Null level = 1/N."""
    out = []
    for k, B in modes:
        Cb, Cn = B.T @ X_base, B.T @ X_new
        D = Cn - Cb
        dE = float(np.sum(D ** 2))
        if Cb.shape[0] == 2:
            z = Cb[0] + 1j * Cb[1]
            d = D[0] + 1j * D[1]
            nz = np.linalg.norm(z)
            if nz < 1e-12 or dE < 1e-18:
                frac = np.nan
            else:
                u = z / nz
                # proj of d onto span_R{u, iu} = <u,d> u (complex inner product)
                coef = np.vdot(u, d)
                frac = float(np.abs(coef) ** 2 / np.sum(np.abs(d) ** 2))
        elif k == 0:
            frac = np.nan
        else:  # Nyquist: commutant is R^*, tangent = span_R{c}
            c = Cb[0]; d = D[0]
            nc = np.linalg.norm(c)
            frac = float((c @ d) ** 2 / (nc ** 2 * (d @ d))) if nc > 1e-12 and dE > 1e-18 else np.nan
        out.append(dict(k=k, change_energy=dE, commutant_frac=frac,
                        energy_base=float(np.sum(Cb ** 2)),
                        energy_new=float(np.sum(Cn ** 2))))
    return out


# ---------------- grokking trainer (modular addition, quadratic MLP) ---------
def train_grokking(p, seed, width=512, train_frac=0.6, lr=1e-3, wd=0.5,
                   max_steps=60000, target_acc=0.99, log_every=2000,
                   device="cpu", verbose=True):
    """Gromov-style 2-layer quadratic MLP on (a+b) mod p. Returns dict with
    embeddings XA (a-side, p x width), XB (b-side), test acc trace.
    Grokks reliably for p<=97 with defaults; if stuck at max_steps with high
    train / low test acc, increase wd or train_frac."""
    import torch
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    pairs = torch.tensor([(a, b) for a in range(p) for b in range(p)])
    perm = torch.randperm(len(pairs), generator=g)
    n_tr = int(train_frac * len(pairs))
    tr, te = pairs[perm[:n_tr]], pairs[perm[n_tr:]]
    y_tr, y_te = (tr[:, 0] + tr[:, 1]) % p, (te[:, 0] + te[:, 1]) % p

    Ea = torch.nn.Parameter(torch.randn(p, width, generator=g) / np.sqrt(width))
    Eb = torch.nn.Parameter(torch.randn(p, width, generator=g) / np.sqrt(width))
    W2 = torch.nn.Parameter(torch.randn(width, p, generator=g) / np.sqrt(width))
    params = [Ea, Eb, W2]
    for prm in params: prm.data = prm.data.to(device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    onehot = torch.eye(p, device=device)
    tr, te, y_tr, y_te = [t.to(device) for t in (tr, te, y_tr, y_te)]

    def forward(idx):
        h = (Ea[idx[:, 0]] + Eb[idx[:, 1]]) ** 2
        return h @ W2

    trace = []
    for step in range(max_steps):
        opt.zero_grad()
        logits = forward(tr)
        loss = torch.mean((logits - onehot[y_tr]) ** 2)
        loss.backward(); opt.step()
        if step % log_every == 0 or step == max_steps - 1:
            with torch.no_grad():
                acc_tr = (forward(tr).argmax(1) == y_tr).float().mean().item()
                acc_te = (forward(te).argmax(1) == y_te).float().mean().item()
            trace.append((step, loss.item(), acc_tr, acc_te))
            if verbose:
                print(f"  seed {seed} step {step:6d} loss {loss.item():.5f} "
                      f"train {acc_tr:.3f} test {acc_te:.3f}")
            if acc_te >= target_acc and acc_tr >= target_acc:
                break
    return dict(XA=Ea.detach().cpu().numpy(), XB=Eb.detach().cpu().numpy(),
                trace=trace, test_acc=trace[-1][3], seed=seed, p=p,
                _torch=dict(Ea=Ea, Eb=Eb, W2=W2, te=te, y_te=y_te,
                            forward=forward))


# ---------------- Part C: transfer of a model-space ablation ------------------
def mode_plane(X, modes, k):
    """Orthonormal basis (N x 2) of the model-space plane carrying mode k."""
    B = dict(modes)[k]
    C = B.T @ X                      # 2 x N
    Qp, _ = np.linalg.qr(C.T)        # N x 2
    return Qp

def ablate_plane(X, V):
    """Project model-space rows of X off the plane spanned by V (N x r)."""
    return X - (X @ V) @ V.T
