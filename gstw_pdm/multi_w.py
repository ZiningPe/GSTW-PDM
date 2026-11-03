"""
multi_w.py
----------
Several weight matrices, each split into same-period and cross-period blocks,
on a time-major panel (rows ordered year by year, cities within year).

Model with M modes (for example rail and road):

    Y = sum_m ( rho_{m0} W^m_0 Y + rho_{m1} W^m_1 Y ) + ...

W^m_0 keeps the diagonal (same-year) blocks of W^m, W^m_1 keeps the strictly
lower (earlier-year) blocks. With recursive weights (no block above the
diagonal) the matrix S = I - sum rho W is block lower triangular, so

    ln|det S| = sum_t ln|det( I_n - sum_m rho_{m0} W^m_{tt} )|

(paper, Lemma 2) and every solve with S is a block forward substitution.
Traces that the adjusted quasi-score needs, tr(F G_j) and tr(F G_j G_l)
with F = C_T kron I_n and G_j = W_j S^{-1}, are computed from n-column
solves instead of the dense inverse.

DenseNetwork (any block pattern) and KroneckerNetwork (W_0 = I_T kron W_S,
W_1 = M kron W_S, eq. (2.5)) have the same interface, so every estimator
takes any of them; make_network picks the recursive form when it applies.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse, linalg
from typing import Sequence


# ---------------------------------------------------------------------------
# block split and demeaning
# ---------------------------------------------------------------------------

def split_same_cross(W, n: int, T: int):
    """
    Split a time-major (nT x nT) weight matrix into same-year and earlier-year parts.

    Returns (W0, W1) as CSR matrices with W0 + W1 == W. Raises if W has a
    block above the diagonal (a later year feeding an earlier one).
    """
    W = sparse.coo_matrix(W)
    year = np.repeat(np.arange(T), n)
    r, c = year[W.row], year[W.col]
    if (c > r).any():
        raise ValueError("W links a later year to an earlier one; not recursive")
    same = r == c
    W0 = sparse.csr_matrix((W.data[same], (W.row[same], W.col[same])), shape=W.shape)
    W1 = sparse.csr_matrix((W.data[~same], (W.row[~same], W.col[~same])), shape=W.shape)
    return W0, W1


def city_demean(a: np.ndarray, n: int, T: int) -> np.ndarray:
    """Apply F = (C_T kron I_n): subtract each city's time mean (time-major rows)."""
    a = np.asarray(a, float)
    one = a.ndim == 1
    b = a.reshape(T, n, -1)
    b = b - b.mean(0, keepdims=True)
    b = b.reshape(T * n, -1)
    return b.ravel() if one else b


# ---------------------------------------------------------------------------
# recursive network
# ---------------------------------------------------------------------------

class RecursiveNetwork:
    """
    S(rho) = I - sum_j rho_j W_j on a block lower-triangular set of lag matrices.

    Parameters
    ----------
    lags : sequence of (name, W, is_cross)
        W is (nT x nT) time-major (sparse or dense) restricted to the
        estimation years. is_cross=True marks a strictly lower block matrix
        (it does not enter the determinant).
    n, T : panel dimensions of the estimation sample.
    """

    def __init__(self, lags: Sequence[tuple], n: int, T: int):
        self.n, self.T, self.N = n, T, n * T
        self.names = [l[0] for l in lags]
        self.W = [sparse.csr_matrix(l[1]) for l in lags]
        self.is_cross = [bool(l[2]) for l in lags]
        self.p = len(lags)
        # dense diagonal blocks of the same-period matrices
        self.diag = {}
        for j, (W, cross) in enumerate(zip(self.W, self.is_cross)):
            if not cross:
                self.diag[j] = [W[t * n:(t + 1) * n, t * n:(t + 1) * n].toarray() for t in range(T)]
        self.rho = None

    # -- factorisation ------------------------------------------------------
    def factor(self, rho: Sequence[float]) -> float:
        """Factor the diagonal blocks; return ln|det S| or -inf if singular."""
        rho = np.asarray(rho, float)
        n, T = self.n, self.T
        self.lu, self.Dinv = [], []
        ld = 0.0
        for t in range(T):
            D = np.eye(n)
            for j in self.diag:
                D -= rho[j] * self.diag[j][t]
            sgn, l = np.linalg.slogdet(D)
            if sgn <= 0:
                self.rho = None
                return -np.inf
            ld += l
            self.lu.append(linalg.lu_factor(D))
            self.Dinv.append(linalg.inv(D))
        cross = [j for j in range(self.p) if self.is_cross[j]]
        self.L = None
        if cross:
            self.L = sum(rho[j] * self.W[j] for j in cross).tocsr()
            self.Lrows = [self.L[t * n:(t + 1) * n] for t in range(T)]
        self.rho = rho
        self.logdet = ld
        return ld

    def solve(self, B: np.ndarray) -> np.ndarray:
        """X = S^{-1} B by block forward substitution."""
        if self.rho is None:
            raise RuntimeError("call factor(rho) first")
        n, T = self.n, self.T
        B = np.asarray(B, float)
        one = B.ndim == 1
        if one:
            B = B[:, None]
        X = np.zeros_like(B)
        for t in range(T):
            rhs = B[t * n:(t + 1) * n]
            if self.L is not None and t > 0:
                rhs = rhs + self.Lrows[t] @ X
            X[t * n:(t + 1) * n] = linalg.lu_solve(self.lu[t], rhs)
        return X.ravel() if one else X

    def inverse(self) -> np.ndarray:
        """Dense S^{-1} (nT x nT); used only for the variance at the solution."""
        return self.solve(np.eye(self.N))

    # -- traces -----------------------------------------------------------
    def tr_G(self) -> np.ndarray:
        """tr(G_j), G_j = W_j S^{-1}: zero for cross-period lags."""
        out = np.zeros(self.p)
        for j in self.diag:
            out[j] = sum(np.trace(self.diag[j][t] @ self.Dinv[t]) for t in range(self.T))
        return out

    def tr_GG(self) -> np.ndarray:
        """tr(G_j G_l): only same-period pairs are non-zero."""
        out = np.zeros((self.p, self.p))
        for a in self.diag:
            for b in self.diag:
                out[a, b] = sum(np.trace(self.diag[a][t] @ self.Dinv[t] @ self.diag[b][t] @ self.Dinv[t])
                                for t in range(self.T))
        return out

    def _B(self) -> np.ndarray:
        return np.tile(np.eye(self.n), (self.T, 1))

    def tr_FG(self):
        """tr(F G_j) = tr(G_j) - (1/T) tr(B' W_j S^{-1} B), B = iota_T kron I_n. Returns (traces, S^{-1}B)."""
        SB = self.solve(self._B())
        out = self.tr_G()
        for j in range(self.p):
            GB = self.W[j] @ SB
            out[j] -= np.trace(GB.reshape(self.T, self.n, self.n).sum(0)) / self.T
        return out, SB

    def tr_FGG(self, SB: np.ndarray) -> np.ndarray:
        """tr(F G_j G_l) = tr(G_j G_l) - (1/T) tr(B' G_j G_l B)."""
        out = self.tr_GG()
        for b in range(self.p):
            SGlB = self.solve(self.W[b] @ SB)
            for a in range(self.p):
                GGB = self.W[a] @ SGlB
                out[a, b] -= np.trace(GGB.reshape(self.T, self.n, self.n).sum(0)) / self.T
        return out

    def tr_PG(self, Q: np.ndarray):
        """tr(Q' G_j Q) for a matrix Q with orthonormal columns (projection P = QQ'). Returns (traces, S^{-1}Q)."""
        SQ = self.solve(Q)
        return np.array([np.sum(Q * (self.W[j] @ SQ)) for j in range(self.p)]), SQ

    def tr_PGG(self, Q: np.ndarray, SQ: np.ndarray) -> np.ndarray:
        """tr(Q' G_j G_l Q)."""
        out = np.zeros((self.p, self.p))
        for b in range(self.p):
            SGQ = self.solve(self.W[b] @ SQ)
            for a in range(self.p):
                out[a, b] = np.sum(Q * (self.W[a] @ SGQ))
        return out

    # -- diagnostics ------------------------------------------------------
    def multiplier_row_sums(self) -> np.ndarray:
        """Row sums of S^{-1} averaged within each year: the long-run response to a unit shock everywhere."""
        return self.solve(np.ones(self.N)).reshape(self.T, self.n).mean(1)


class TiedNetwork:
    """
    A RecursiveNetwork whose coefficients are tied by rho_full = A rho.

    Example: one coefficient per mode on the same- and cross-period blocks,
    A = [[1, 0], [1, 0], [0, 1], [0, 1]] for lags (rail0, rail1, road0, road1).
    It has the interface the estimators in fixed_effects use.
    """

    def __init__(self, base: RecursiveNetwork, A, names: Sequence[str]):
        self.base, self.A = base, np.asarray(A, float)
        self.names = list(names)
        self.p = self.A.shape[1]
        self.n, self.T, self.N = base.n, base.T, base.N
        self.W = [sum(self.A[j, k] * base.W[j] for j in range(base.p)).tocsr() for k in range(self.p)]

    def factor(self, rho):
        return self.base.factor(self.A @ np.asarray(rho, float))

    def solve(self, B):
        return self.base.solve(B)

    def inverse(self):
        return self.base.inverse()

    def tr_FG(self):
        t, SB = self.base.tr_FG()
        return self.A.T @ t, SB

    def tr_FGG(self, SB):
        return self.A.T @ self.base.tr_FGG(SB) @ self.A

    def tr_PG(self, Q):
        t, SQ = self.base.tr_PG(Q)
        return self.A.T @ t, SQ

    def tr_PGG(self, Q, SQ):
        return self.A.T @ self.base.tr_PGG(Q, SQ) @ self.A

    def multiplier_row_sums(self):
        return self.base.multiplier_row_sums()


# ---------------------------------------------------------------------------
# general and Kronecker networks (same interface as RecursiveNetwork)
# ---------------------------------------------------------------------------

class DenseNetwork:
    """
    S(rho) = I - sum_j rho_j W_j for weights of any block pattern (paper, Section 2.1).

    The log-determinant is an N x N LU factorisation. Use RecursiveNetwork when
    no block lies above the diagonal and KroneckerNetwork for M kron W_S.

    Parameters
    ----------
    lags : sequence of (name, W) or (name, W, is_cross); W is (nT x nT) time-major.
    n, T : panel dimensions.
    """

    def __init__(self, lags: Sequence[tuple], n: int, T: int):
        self.n, self.T, self.N = n, T, n * T
        self.names = [l[0] for l in lags]
        self.W = [sparse.csr_matrix(l[1]) for l in lags]
        self.is_cross = [bool(l[2]) if len(l) > 2 else False for l in lags]
        self.p = len(lags)
        self.rho = None

    def S(self, rho) -> np.ndarray:
        S = np.eye(self.N)
        for r, W in zip(np.asarray(rho, float), self.W):
            S -= r * W.toarray()
        return S

    def factor(self, rho) -> float:
        """Factor S(rho); return ln|det S| or -inf if singular."""
        rho = np.asarray(rho, float)
        S = self.S(rho)
        self.lu = linalg.lu_factor(S, check_finite=False)
        d = np.abs(np.diag(self.lu[0]))
        if not np.all(d > 1e-12):
            self.rho = None
            return -np.inf
        self.rho = rho
        self.logdet = float(np.sum(np.log(d)))
        return self.logdet

    def solve(self, B: np.ndarray) -> np.ndarray:
        if self.rho is None:
            raise RuntimeError("call factor(rho) first")
        return linalg.lu_solve(self.lu, np.asarray(B, float))

    def inverse(self) -> np.ndarray:
        return self.solve(np.eye(self.N))

    def multiplier_row_sums(self) -> np.ndarray:
        return self.solve(np.ones(self.N)).reshape(self.T, self.n).mean(1)


class KroneckerNetwork(DenseNetwork):
    """
    Kronecker weights W_0 = I_T kron W_S, W_1 = M kron W_S (paper, Section 2.2).

    The log-determinant uses the eigenvalues of M and W_S,

        ln|det S(rho)| = sum_j sum_i ln|1 - (rho_0 + rho_1 mu_j) lambda_i|,

    so the likelihood never factors an N x N matrix; S(rho) is factored only
    when a solve is requested (variance, effects).
    """

    def __init__(self, M: np.ndarray, W_S: np.ndarray, names: Sequence[str] = ("rho0", "rho1")):
        M, W_S = np.asarray(M, float), np.asarray(W_S, float)
        n, T = W_S.shape[0], M.shape[0]
        super().__init__([(names[0], np.kron(np.eye(T), W_S), False),
                          (names[1], np.kron(M, W_S), True)], n, T)
        self.M, self.W_S = M, W_S
        self.mu = linalg.eigvals(M)
        self.lam = linalg.eigvals(W_S)
        self.lu = None

    def factor(self, rho) -> float:
        rho = np.asarray(rho, float)
        z = 1 - np.outer(rho[0] + rho[1] * self.mu, self.lam)
        a = np.abs(z)
        if not np.all(a > 1e-12):
            self.rho = None
            return -np.inf
        self.rho = rho
        self.lu = None
        self.logdet = float(np.sum(np.log(a)))
        return self.logdet

    def solve(self, B: np.ndarray) -> np.ndarray:
        if self.rho is None:
            raise RuntimeError("call factor(rho) first")
        if self.lu is None:
            self.lu = linalg.lu_factor(self.S(self.rho), check_finite=False)
        return linalg.lu_solve(self.lu, np.asarray(B, float))


def make_network(lags: Sequence[tuple], n: int, T: int):
    """RecursiveNetwork if no lag matrix has a block above the diagonal, DenseNetwork otherwise.

    lags holds (name, W) pairs; each W is split into its same-year part and its
    earlier-year part only if the caller does so (see split_same_cross).
    """
    year = np.repeat(np.arange(T), n)
    out, recursive = [], True
    for l in lags:
        W = sparse.coo_matrix(l[1])
        r, c = year[W.row], year[W.col]
        nz = W.data != 0
        if (c[nz] > r[nz]).any():
            recursive = False
        cross = bool((c[nz] != r[nz]).any())
        if cross and (c[nz] == r[nz]).any():
            recursive = False          # a lag that mixes same-year and cross-year links
        out.append((l[0], l[1], cross))
    if recursive:
        return RecursiveNetwork(out, n, T)
    return DenseNetwork(out, n, T)
