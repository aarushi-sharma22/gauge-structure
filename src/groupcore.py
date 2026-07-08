"""Core machinery for Experiment 1: groups, regular representations,
numeric character tables (Dixon-style), isotypic projectors, commutant dims.

Everything is verified numerically (group axioms, projector completeness,
orthogonality relations) so errors surface immediately rather than as wrong
science downstream.
"""
import itertools
import numpy as np

# ---------------------------------------------------------------------------
# Group construction: each group is a list of hashable elements + mult function
# ---------------------------------------------------------------------------

def cyclic(n):
    elems = list(range(n))
    mult = lambda a, b: (a + b) % n
    gens = [1]
    return elems, mult, gens, f"Z/{n}"

def dihedral(n):
    # elements (k, f): rotation by k, flip flag f. r=(1,0), s=(0,1)
    elems = [(k, f) for f in (0, 1) for k in range(n)]
    def mult(a, b):
        k1, f1 = a; k2, f2 = b
        if f1 == 0:
            return ((k1 + k2) % n, f2)
        else:
            return ((k1 - k2) % n, 1 - f2)
    gens = [(1, 0), (0, 1)]
    return elems, mult, gens, f"D{n}"

def symmetric(n):
    elems = [tuple(p) for p in itertools.permutations(range(n))]
    def mult(a, b):  # (a*b)(x) = a(b(x))
        return tuple(a[b[i]] for i in range(n))
    gens = [tuple([1, 0] + list(range(2, n)))]
    if n > 2:
        gens.append(tuple(list(range(1, n)) + [0]))  # n-cycle
    return elems, mult, gens, f"S{n}"

def _parity(p):
    p = list(p); par = 0; seen = [False]*len(p)
    for i in range(len(p)):
        if seen[i]: continue
        j, clen = i, 0
        while not seen[j]:
            seen[j] = True; j = p[j]; clen += 1
        par ^= (clen - 1) & 1
    return par

def alternating(n):
    elems = [tuple(p) for p in itertools.permutations(range(n)) if _parity(p) == 0]
    def mult(a, b):
        return tuple(a[b[i]] for i in range(n))
    # generators: 3-cycles (0 1 2) and (0 1 2 ... ) pattern; use (012) and (0,1)(2,..n-1) style
    g1 = tuple([1, 2, 0] + list(range(3, n)))
    if n % 2 == 1:
        g2 = tuple(list(range(1, n)) + [0])           # n-cycle (even perm for odd n)
    else:
        g2 = tuple([0] + list(range(2, n)) + [1])     # (n-1)-cycle on 1..n-1
    return elems, mult, [g1, g2], f"A{n}"

def quaternion8():
    # elements (s, u): s in {+1,-1}, u in {'1','i','j','k'}
    units = ['1', 'i', 'j', 'k']
    tab = {  # unit multiplication: (u,v) -> (sign, unit)
        ('1','1'):(1,'1'),('1','i'):(1,'i'),('1','j'):(1,'j'),('1','k'):(1,'k'),
        ('i','1'):(1,'i'),('i','i'):(-1,'1'),('i','j'):(1,'k'),('i','k'):(-1,'j'),
        ('j','1'):(1,'j'),('j','i'):(-1,'k'),('j','j'):(-1,'1'),('j','k'):(1,'i'),
        ('k','1'):(1,'k'),('k','i'):(1,'j'),('k','j'):(-1,'i'),('k','k'):(-1,'1'),
    }
    elems = [(s, u) for u in units for s in (1, -1)]
    def mult(a, b):
        s1, u1 = a; s2, u2 = b
        s3, u3 = tab[(u1, u2)]
        return (s1 * s2 * s3, u3)
    gens = [(1, 'i'), (1, 'j')]
    return elems, mult, gens, "Q8"


class Group:
    """Finite group with Cayley table, regular rep, classes, characters."""
    def __init__(self, elems, mult, gens, name):
        self.name = name
        self.elems = list(elems)
        self.n = len(self.elems)
        self.idx = {e: i for i, e in enumerate(self.elems)}
        self._verify_and_build(mult, gens)

    def _verify_and_build(self, mult, gens):
        n, idx, elems = self.n, self.idx, self.elems
        # Cayley table
        C = np.empty((n, n), dtype=np.int32)
        for i, a in enumerate(elems):
            for j, b in enumerate(elems):
                C[i, j] = idx[mult(a, b)]
        self.cayley = C
        # identity
        eid = [i for i in range(n) if all(C[i, j] == j for j in range(n))]
        assert len(eid) == 1, f"{self.name}: identity not unique"
        self.e = eid[0]
        # inverses + closure implicit in table construction; associativity spot check
        rng = np.random.default_rng(0)
        for _ in range(200):
            a, b, c = rng.integers(0, n, 3)
            assert C[C[a, b], c] == C[a, C[b, c]], f"{self.name}: not associative"
        inv = np.empty(n, dtype=np.int32)
        for i in range(n):
            j = np.where(C[i] == self.e)[0]
            assert len(j) == 1
            inv[i] = j[0]
        self.inv = inv
        # generators generate the whole group
        gset = {self.e}
        frontier = [self.e]
        gidx = [idx[g] for g in gens]
        while frontier:
            new = []
            for x in frontier:
                for g in gidx:
                    y = C[g, x]
                    if y not in gset:
                        gset.add(y); new.append(y)
            frontier = new
        assert len(gset) == n, f"{self.name}: generators don't generate group"
        self.gens = gidx
        # conjugacy classes
        assigned = np.full(n, -1, dtype=np.int32)
        classes = []
        for i in range(n):
            if assigned[i] >= 0: continue
            orb = set()
            for x in range(n):
                orb.add(C[C[x, i], inv[x]])
            k = len(classes)
            for j in orb: assigned[j] = k
            classes.append(sorted(orb))
        self.classes = classes
        self.class_of = assigned
        self.r = len(classes)

    # -------------------- regular representation --------------------
    def regular_rep(self, g):
        """L(g): permutation matrix, L(g) e_h = e_{gh}."""
        n = self.n
        M = np.zeros((n, n))
        M[self.cayley[g, np.arange(n)], np.arange(n)] = 1.0
        return M

    # -------------------- numeric character table (Dixon) --------------------
    def character_table(self, seed=1, tol=1e-8):
        """Returns chars: (r x r) complex array, chars[t, i] = chi_t(class i),
        verified against orthogonality relations."""
        n, r, classes = self.n, self.r, self.classes
        reps = [c[0] for c in classes]
        sizes = np.array([len(c) for c in classes])
        # structure constants a[i][j][k]: #{(x,y) in C_i x C_j : xy = rep_k}
        A = np.zeros((r, r, r))
        Cay = self.cayley
        for i in range(r):
            for x in classes[i]:
                row = Cay[x]            # x*y for all y
                for j in range(r):
                    prods = row[classes[j]]
                    for k in range(r):
                        A[i, j, k] += np.count_nonzero(prods == reps[k])
        # eigenvector method: w (per character) satisfies M_i w = w_i w, M_i = A[i]
        rng = np.random.default_rng(seed)
        for attempt in range(20):
            c = rng.standard_normal(r)
            M = np.tensordot(c, A, axes=(0, 0))
            evals, evecs = np.linalg.eig(M)
            if np.min(np.abs(evals[:, None] - evals[None, :]) + np.eye(r)) > 1e-6:
                break
        chars = np.zeros((r, r), dtype=complex)
        for t in range(r):
            v = evecs[:, t]
            # identity class index
            id_cls = self.class_of[self.e]
            v = v / v[id_cls]                      # w_identity = 1
            denom = np.sum(np.abs(v) ** 2 / sizes)
            d = np.sqrt(n / denom)
            d = np.round(d.real) if abs(d - np.round(d.real)) < 1e-6 else d
            chars[t] = d * v / sizes
        # sort by degree then verify orthogonality
        order = np.argsort([c[self.class_of[self.e]].real for c in chars])
        chars = chars[order]
        G_inner = (chars * sizes) @ chars.conj().T / n
        assert np.allclose(G_inner, np.eye(r), atol=1e-7), \
            f"{self.name}: character orthogonality failed"
        degs = chars[:, self.class_of[self.e]].real
        assert abs(np.sum(degs ** 2) - n) < 1e-6, f"{self.name}: sum d^2 != |G|"
        self.chars = chars
        self.degrees = np.round(degs).astype(int)
        self.class_sizes = sizes
        return chars

    def fs_indicators(self):
        """Frobenius-Schur indicator per complex irrep: +1 real, 0 complex, -1 quaternionic."""
        sq_class = np.array([self.class_of[self.cayley[g, g]] for g in range(self.n)])
        fs = np.array([np.sum(self.chars[t][sq_class]).real / self.n
                       for t in range(self.r)])
        fs = np.round(fs).astype(int)
        assert np.all(np.isin(fs, [-1, 0, 1])), f"{self.name}: bad FS indicators"
        self.fs = fs
        return fs

    # -------------------- real isotypic blocks of the regular rep -------------
    def real_isotypic_blocks(self):
        """Group complex irreps into real blocks (complex-conjugate pairs merged).
        Returns list of dicts with projector, dims, multiplicity, gauge (numeric)."""
        if not hasattr(self, 'chars'): self.character_table()
        if not hasattr(self, 'fs'): self.fs_indicators()
        n, r = self.n, self.r
        Ls = [self.regular_rep(g) for g in range(n)]
        char_by_elem = self.chars[:, self.class_of]          # r x n
        used = np.zeros(r, dtype=bool)
        blocks = []
        for t in range(r):
            if used[t]: continue
            partners = [t]
            if self.fs[t] == 0:
                # find conjugate partner
                for s in range(r):
                    if s != t and not used[s] and \
                       np.allclose(self.chars[s], self.chars[t].conj(), atol=1e-7):
                        partners.append(s); break
                assert len(partners) == 2, f"{self.name}: conj partner missing"
            for s in partners: used[s] = True
            P = np.zeros((n, n))
            for s in partners:
                d = self.degrees[s]
                coef = (d / n) * char_by_elem[s].conj()
                P += np.real(sum(coef[g] * Ls[g] for g in range(n)))
            # verify projector
            assert np.allclose(P @ P, P, atol=1e-9), f"{self.name}: block not idempotent"
            rank = int(np.round(np.trace(P)))
            d_complex = int(self.degrees[t])
            fs = int(self.fs[t])
            blocks.append(dict(group=self.name, irrep_dim=d_complex, fs=fs,
                               partners=partners, P=P, block_rank=rank))
        # completeness
        Ptot = sum(b['P'] for b in blocks)
        assert np.allclose(Ptot, np.eye(n), atol=1e-8), f"{self.name}: blocks incomplete"
        self.blocks = blocks
        return blocks

    def commutant_dim(self, P, tol=1e-7):
        """Numeric real commutant dimension of the regular action restricted to
        range(P): dim{ M : M rho(g) = rho(g) M for generators }."""
        rank = int(np.round(np.trace(P)))
        # orthonormal basis of range(P)
        U, s, _ = np.linalg.svd(P)
        B = U[:, :rank]
        rows = []
        for g in self.gens:
            R = B.T @ self.regular_rep(g) @ B          # rank x rank
            # constraint (R^T kron I - I kron R) vec(M) = 0
            K = np.kron(R.T, np.eye(rank)) - np.kron(np.eye(rank), R)
            rows.append(K)
        K = np.vstack(rows)
        sv = np.linalg.svd(K, compute_uv=False)
        return int(np.sum(sv < tol * max(1.0, sv[0]))), B

    def multiplicity_in_regular(self, t):
        return int(self.degrees[t])   # regular rep: m = d (complex)


def all_eleven_groups():
    specs = [cyclic(6), cyclic(12), dihedral(3), dihedral(4), quaternion8(),
             dihedral(5), dihedral(6), alternating(4), symmetric(4),
             alternating(5), symmetric(5)]
    # note: S3 ~= D3; use D3 construction labelled S3 to match the paper's list
    out = []
    for elems, mult, gens, name in specs:
        if name == "D3": name = "S3"
        out.append(Group(elems, mult, gens, name))
    return out
