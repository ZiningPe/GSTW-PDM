"""
weight_construction.py
----------------------
Build the weight blocks W_0 = I_T ⊗ W_S, W_1 = M ⊗ W_S and the temporal weight
matrix (TWM) M from data-driven statistics (Moran's I, Geary's C, Getis-Ord G,
Spatial Gini, or parametric decay).

Stacking convention (time-major)
---------------------------------
Panel vector Y of length N = n·T is ordered as

    Y = [y_{1,1}, …, y_{n,1},   ← period 1
         y_{1,2}, …, y_{n,2},   ← period 2
         …
         y_{1,T}, …, y_{n,T}]   ← period T

so that np.kron(M, W_S) gives the correct N×N weight matrix.

References
----------
Moran, P. A. P. (1950). Biometrika, 37(1/2), 17–23.
Geary, R. C. (1954). The Incorporated Statistician, 5(3), 115–145.
Getis, A. & Ord, J. K. (1992). Geographical Analysis, 24(3), 189–206.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg
from typing import Literal, Sequence


# ---------------------------------------------------------------------------
# 1.  Spatial weight matrices
# ---------------------------------------------------------------------------

def rook_weights(n_side: int) -> np.ndarray:
    """
    Row-normalised rook-contiguity weight matrix for an n_side×n_side grid.

    Parameters
    ----------
    n_side : side length of the square grid  (n = n_side²)

    Returns
    -------
    W_S : (n, n) row-normalised weight matrix
    """
    n = n_side ** 2
    W = np.zeros((n, n))
    for i in range(n):
        r, c = divmod(i, n_side)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_side and 0 <= nc < n_side:
                W[i, nr * n_side + nc] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


def queen_weights(n_side: int) -> np.ndarray:
    """
    Row-normalised queen-contiguity weight matrix for an n_side×n_side grid.
    """
    n = n_side ** 2
    W = np.zeros((n, n))
    for i in range(n):
        r, c = divmod(i, n_side)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_side and 0 <= nc < n_side:
                    W[i, nr * n_side + nc] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


def inverse_distance_weights(coords: np.ndarray,
                              power: float = 1.0,
                              row_standardize: bool = True) -> np.ndarray:
    """
    Inverse-distance weight matrix.

    Parameters
    ----------
    coords         : (n, 2) array of (lon, lat) or (x, y) coordinates
    power          : distance decay exponent
    row_standardize: whether to row-normalise

    Returns
    -------
    W : (n, n) weight matrix (zero diagonal)
    """
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]          # (n, n, 2)
    dist = np.sqrt((diff ** 2).sum(axis=-1))                  # (n, n)
    np.fill_diagonal(dist, np.inf)
    W = 1.0 / dist ** power
    np.fill_diagonal(W, 0.0)
    if row_standardize:
        rs = W.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = W / rs
    return W


def knn_weights(coords: np.ndarray, k: int,
                row_standardize: bool = True) -> np.ndarray:
    """
    k-nearest-neighbour binary weight matrix.
    """
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    W = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(dist[i])[:k]
        W[i, idx] = 1.0
    if row_standardize:
        rs = W.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = W / rs
    return W


# ---------------------------------------------------------------------------
# 2.  Temporal statistics used to build M (TWM)
# ---------------------------------------------------------------------------

def compute_morans_i(y: np.ndarray, W_S: np.ndarray) -> float:
    """
    Global Moran's I statistic for a single cross-section.

    I = (n / S_0) · (y'Wy) / (y'y),   S_0 = sum of all weights.

    Parameters
    ----------
    y   : (n,) cross-sectional observations (mean-demeaned internally)
    W_S : (n, n) spatial weight matrix

    Returns
    -------
    Moran's I statistic
    """
    y = np.asarray(y, float)
    z = y - y.mean()
    n = len(z)
    S0 = W_S.sum()
    if S0 == 0 or z @ z == 0:
        return 0.0
    return float(n / S0 * (z @ W_S @ z) / (z @ z))


def compute_gearys_c(y: np.ndarray, W_S: np.ndarray) -> float:
    """Geary's C statistic for a single cross-section."""
    y = np.asarray(y, float)
    n = len(y)
    S0 = W_S.sum()
    var_y = np.var(y)
    if S0 == 0 or var_y == 0:
        return 1.0
    num = 0.0
    for i in range(n):
        for j in range(n):
            num += W_S[i, j] * (y[i] - y[j]) ** 2
    return float((n - 1) * num / (2 * S0 * n * var_y))


def compute_getis_ord_g(y: np.ndarray, W_S: np.ndarray) -> float:
    """Global Getis-Ord G statistic (assumes non-negative y)."""
    y = np.asarray(y, float)
    n = len(y)
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                num += W_S[i, j] * y[i] * y[j]
                den += y[i] * y[j]
    return float(num / den) if den != 0 else 0.0


def compute_spatial_gini(y: np.ndarray, W_S: np.ndarray) -> float:
    """
    Spatial Gini coefficient measuring regional inequality with spatial structure.
    """
    y = np.asarray(y, float)
    n = len(y)
    mean_y = y.mean()
    if mean_y == 0:
        return 0.0
    num = sum(W_S[i, j] * abs(y[i] - y[j])
              for i in range(n) for j in range(n))
    return float(num / (2 * n ** 2 * mean_y))


# ---------------------------------------------------------------------------
# 3.  Build temporal weight matrix M from statistics
# ---------------------------------------------------------------------------

_DECAY_REGISTRY: dict[str, callable] = {}


def _norm_row(M: np.ndarray) -> np.ndarray:
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return M / rs


def build_twm_from_stats(stats: Sequence[float],
                          method: Literal["moran", "geary", "getis", "gini",
                                          "raw"] = "moran",
                          row_standardize: bool = True,
                          bandwidth: float = 1.0) -> np.ndarray:
    """
    Build a T×T temporal weight matrix M from a sequence of T spatial
    statistics (one per period).

    The (t, s) entry of M is proportional to the *similarity* of period t
    and period s as measured by their statistics:

        M_raw[t, s] = exp(-|stat_t - stat_s| / b),  t != s   (paper, Section 2.2)

    Zero diagonal is enforced before row-normalisation.

    Parameters
    ----------
    stats          : length-T sequence of per-period statistics
    method         : label used for display only (all methods use the same
                     kernel — pass ``method='raw'`` if stats are pre-computed)
    row_standardize: row-normalise M (recommended)
    bandwidth      : b in the kernel

    Returns
    -------
    M : (T, T) temporal weight matrix
    """
    s = np.asarray(stats, float)
    T = len(s)
    diff = np.abs(s[:, None] - s[None, :])
    M = np.exp(-diff / bandwidth)
    np.fill_diagonal(M, 0.0)
    if row_standardize:
        M = _norm_row(M)
    return M


def build_twm_parametric(T: int,
                          rho: float = 0.6,
                          form: Literal["exponential", "power",
                                        "linear"] = "exponential",
                          row_standardize: bool = True) -> np.ndarray:
    """
    Parametric temporal weight matrix with M_{ts} ∝ decay(|t - s|).

    Parameters
    ----------
    T    : number of periods
    rho  : decay parameter
    form : 'exponential' → rho^|t-s|,
           'power'       → |t-s|^{-rho},
           'linear'      → max(0, 1 - rho*|t-s|)
    """
    M = np.zeros((T, T))
    for t in range(T):
        for s in range(T):
            if t == s:
                continue
            d = abs(t - s)
            if form == "exponential":
                M[t, s] = rho ** d
            elif form == "power":
                M[t, s] = d ** (-rho)
            elif form == "linear":
                M[t, s] = max(0.0, 1.0 - rho * d)
    if row_standardize:
        M = _norm_row(M)
    return M


# ---------------------------------------------------------------------------
# 4.  Same-period and cross-period blocks (paper, Section 2)
# ---------------------------------------------------------------------------

def kronecker_weights(M: np.ndarray, W_S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Kronecker weights of Section 2.2: W_0 = I_T kron W_S (same period),
    W_1 = M kron W_S (cross period), W = W_0 + W_1.

    M must have a zero diagonal. With W_S and M row-stochastic each row of W_0
    and each row of W_1 sums to one (Assumption 3(i)).
    """
    M, W_S = np.asarray(M, float), np.asarray(W_S, float)
    if np.any(np.diag(M) != 0):
        raise ValueError("M must have a zero diagonal; the same-period links are in W_0")
    return np.kron(np.eye(M.shape[0]), W_S), np.kron(M, W_S)


def block_row_normalise(W, n: int, T: int):
    """
    Row-normalise the same-period part and the cross-period part of W separately
    (Assumption 3(i)); returns (W_0, W_1). Normalising the whole row would shrink
    the same-period share as source periods accumulate.
    """
    from scipy import sparse
    W = sparse.csr_matrix(W, dtype=float)
    year = np.repeat(np.arange(T), n)
    C = W.tocoo()
    same = year[C.row] == year[C.col]
    out = []
    for mask in (same, ~same):
        B = sparse.csr_matrix((C.data[mask], (C.row[mask], C.col[mask])), shape=W.shape)
        rs = np.asarray(B.sum(1)).ravel()
        rs[rs == 0] = 1.0
        out.append(sparse.diags(1 / rs) @ B)
    return out[0].tocsr(), out[1].tocsr()


def parameter_space(W0, W1, n: int, T: int) -> dict:
    """
    Check Assumption 3 and Lemma 2 for given blocks and report the parameter space.

    Returns the maximum row sums of W_0 and W_1 (each must be <= 1), the column-sum
    bound c_W of W, the column-sum bound c_0 of the diagonal blocks, whether W is
    recursive (no block above the diagonal), and the implied bounds:
    |rho0| + |rho1| < 1/max(1, c_W) in general (Assumption 3(iii)); with recursive
    weights only |rho0| < 1/max(1, c_0) is needed and rho1 is unrestricted (Lemma 2).
    """
    from scipy import sparse
    W0, W1 = sparse.csr_matrix(W0), sparse.csr_matrix(W1)
    W = (W0 + W1).tocoo()
    year = np.repeat(np.arange(T), n)
    recursive = not bool(np.any(year[W.col][W.data != 0] > year[W.row][W.data != 0]))
    D = W0.tocoo()
    diag_only = bool(np.all(year[D.row] == year[D.col]))
    c_W = float(np.abs(W0 + W1).sum(0).max())
    c_0 = float(max(np.abs(W0[t * n:(t + 1) * n, t * n:(t + 1) * n]).sum(0).max() for t in range(T)))
    return {
        "row_sum_W0": float(np.abs(W0).sum(1).max()),
        "row_sum_W1": float(np.abs(W1).sum(1).max()),
        "zero_diagonal": bool(np.all((W0 + W1).diagonal() == 0)),
        "nonnegative": bool((W0.data >= 0).all() and (W1.data >= 0).all()),
        "W0_block_diagonal": diag_only,
        "recursive": recursive,
        "c_W": c_W,
        "c_0": c_0,
        "bound_rho_sum": 1 / max(1.0, c_W),
        "bound_rho0_recursive": 1 / max(1.0, c_0) if recursive else None,
    }


def stwm_summary(W: np.ndarray, n: int, T: int) -> dict:
    """
    Key properties of an assembled STWM.

    Returns
    -------
    dict with shape, sparsity, spectral_radius, row_sum stats,
    and eigenvalue decomposition info.
    """
    assert W.shape == (n * T, n * T), \
        f"Expected ({n * T}, {n * T}), got {W.shape}."
    nnz = np.count_nonzero(W)
    sparsity = 1.0 - nnz / W.size
    eigvals = np.linalg.eigvals(W).real
    row_sums = W.sum(axis=1)
    off = W[np.eye(n * T, dtype=bool) == False]
    off = off[off != 0]
    return {
        "shape": W.shape,
        "n_spatial": n,
        "T_temporal": T,
        "N": n * T,
        "sparsity": round(float(sparsity), 4),
        "spectral_radius": round(float(np.max(np.abs(eigvals))), 6),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "row_sum_mean": float(row_sums.mean()),
        "weight_min": float(off.min()) if len(off) else 0.0,
        "weight_max": float(off.max()) if len(off) else 0.0,
        "weight_mean": float(off.mean()) if len(off) else 0.0,
        "eigenvalue_range": (float(eigvals.min()), float(eigvals.max())),
    }


# ---------------------------------------------------------------------------
# 5.  Eigenvalue helpers (Kronecker log-determinant, eq. (2.5))
# ---------------------------------------------------------------------------

def eigvals_kronecker(M: np.ndarray, W_S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (complex) of M and W_S for the Kronecker log-determinant."""
    return linalg.eigvals(M), linalg.eigvals(W_S)


def logdet_kronecker(rho, eigvals_M: np.ndarray, eigvals_WS: np.ndarray) -> float:
    """
    ln|det(I - rho0 I_T kron W_S - rho1 M kron W_S)|
        = sum_j sum_i ln|1 - (rho0 + rho1 mu_j) lambda_i|        (paper, eq. (2.5))

    rho = (rho0, rho1). Returns -inf if the matrix is singular.
    """
    rho0, rho1 = rho
    z = 1 - np.outer(rho0 + rho1 * np.asarray(eigvals_M), np.asarray(eigvals_WS))
    a = np.abs(z)
    if not np.all(a > 1e-12):
        return -np.inf
    return float(np.sum(np.log(a)))
