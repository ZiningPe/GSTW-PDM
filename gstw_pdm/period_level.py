"""
period_level.py
---------------
Period-level formation variables, L = T (paper, Section 5).

With W_0 = I_T kron W_S, W_1 = M(h) kron W_S and A = I_T kron iota_n, the
control A eps_hat omits delta_c A P_Z eps, a vector of T period constants
that does not shrink with n, and the CF-QMLE (the naive estimator) converges
to a point that depends on the realised shocks (Proposition 3).

The constants are period effects and are removed exactly by projecting each
period on the complement of iota_n (Lemma 4). With F_n an n x (n-1) orthonormal
basis orthogonal to iota_n and W_S* = F_n' W_S F_n,

    Y* = (I_T kron F_n') Y,   X* = (I_T kron F_n') X,

follow the same model with W_S* in place of W_S and without the period
constants. within_period_qmle maximises its likelihood (eq. (5.2), Theorem 4).
The first stage is not used. anderson_rubin is the exact AR statistic of
Appendix I on the same transformed data.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg, stats
from typing import Optional, Sequence

from .multi_w import KroneckerNetwork
from .cf_qmle import _cols, _full_rank, _fit


def complement_basis(n: int) -> np.ndarray:
    """F_n: n x (n-1) orthonormal columns orthogonal to iota_n."""
    q = linalg.qr(np.column_stack([np.ones(n), np.eye(n)[:, :n - 1]]), mode="economic")[0]
    return q[:, 1:]


def _to_star(A: np.ndarray, Fn: np.ndarray, T: int) -> np.ndarray:
    """(I_T kron F_n') A for time-major rows."""
    n = Fn.shape[0]
    A = np.asarray(A, float)
    one = A.ndim == 1
    B = A.reshape(T, n, -1)
    out = np.einsum("ij,tik->tjk", Fn, B).reshape(T * (n - 1), -1)
    return out.ravel() if one else out


def _from_star(A: np.ndarray, Fn: np.ndarray, T: int) -> np.ndarray:
    """(I_T kron F_n) A."""
    m = Fn.shape[1]
    A = np.asarray(A, float)
    one = A.ndim == 1
    B = A.reshape(T, m, -1)
    out = np.einsum("ij,tjk->tik", Fn, B).reshape(T * Fn.shape[0], -1)
    return out.ravel() if one else out


def within_period_qmle(y, X, M: np.ndarray, W_S: np.ndarray, durbin: bool = True, extra=None,
                       x_names=None, extra_names=None, rho0=None, variance: bool = True,
                       names: Sequence[str] = ("rho0", "rho1")) -> dict:
    """
    QMLE on within-period deviations (Theorem 4), for any delta_c and any realisation of eps.

    Requires W_S iota_n = iota_n. Regressors constant within a period (intercept,
    period dummies, period-level variables) are absorbed and dropped. The variance is
    I*^{-1} V* I*^{-1}, with V* from Lemma 9 applied to xi* = (I_T kron F_n') xi.
    """
    M, W_S = np.asarray(M, float), np.asarray(W_S, float)
    n, T = W_S.shape[0], M.shape[0]
    if not np.allclose(W_S.sum(1), 1):
        raise ValueError("within_period_qmle needs a row-stochastic W_S (Lemma 4)")
    y = np.asarray(y, float).ravel()
    N = len(y)
    Fn = complement_basis(n)
    Ws_star = Fn.T @ W_S @ Fn
    net = KroneckerNetwork(M, Ws_star, names)
    full = KroneckerNetwork(M, W_S, names)
    X = _cols(X, N)
    E = _cols(extra, N)
    x_names = list(x_names) if x_names is not None else [f"x{i + 1}" for i in range(X.shape[1])]
    e_names = list(extra_names) if extra_names is not None else [f"e{i + 1}" for i in range(E.shape[1])]
    blocks, rn = [X], list(x_names)
    if durbin:
        for nm, W in zip(full.names, full.W):
            blocks.append(W @ X)
            rn += [f"{nm}*{x}" for x in x_names]
    blocks.append(E)
    rn += e_names
    R = _to_star(np.column_stack(blocks), Fn, T)
    keep = _full_rank(R)
    R, rn = R[:, keep], [rn[i] for i in keep]
    ys = _to_star(y, Fn, T)
    U = _to_star(np.column_stack([W @ y for W in full.W]), Fn, T)

    def basis_diag(A):
        m = n - 1
        d = np.zeros(T * n)
        for t in range(T):
            blk = A[t * m:(t + 1) * m, t * m:(t + 1) * m]
            d[t * n:(t + 1) * n] = np.sum((Fn @ blk) * Fn, axis=1)
        return d

    out = _fit(ys, U, R, rn, {}, net, None, rho0, variance,
               basis_diag=basis_diag, basis_vec=lambda c: _from_star(c, Fn, T),
               group=n, resid_orig=lambda e: _from_star(e, Fn, T), label="within-period QMLE")
    out["N_star"] = len(ys)
    return out


def anderson_rubin(y, X, M: np.ndarray, W_S: np.ndarray, r: Sequence[float], extra=None) -> dict:
    """
    Exact Anderson-Rubin statistic for H0: (rho0, rho1) = r (paper, Appendix I).

    X~ = within-period deviations of [X, W_0 X, W_1 X, E], G~ = within-period
    deviations of W_0^2 X, W_0^3 X, W_1^2 X, W_1^3 X residualised on X~. Under
    Gaussian xi, AR(r) ~ F(m, T(n-1) - k - m) exactly at the true r, whatever delta_c.
    """
    M, W_S = np.asarray(M, float), np.asarray(W_S, float)
    n, T = W_S.shape[0], M.shape[0]
    y = np.asarray(y, float).ravel()
    N = len(y)
    X = _cols(X, N)
    E = _cols(extra, N)
    Fn = complement_basis(n)
    W0, W1 = np.kron(np.eye(T), W_S), np.kron(M, W_S)
    Xt = _to_star(np.column_stack([X, W0 @ X, W1 @ X, E]), Fn, T)
    Xt = Xt[:, _full_rank(Xt)]
    G = _to_star(np.column_stack([W0 @ W0 @ X, W0 @ W0 @ W0 @ X, W1 @ W1 @ X, W1 @ W1 @ W1 @ X]), Fn, T)
    Qx = linalg.orth(Xt)
    G = G - Qx @ (Qx.T @ G)
    G = G[:, _full_rank(G)]
    Qg = linalg.orth(G)
    k, m = Qx.shape[1], Qg.shape[1]
    es = _to_star(y - r[0] * (W0 @ y) - r[1] * (W1 @ y), Fn, T)
    et = es - Qx @ (Qx.T @ es)
    num = float(np.sum((Qg.T @ et) ** 2))
    dof = T * (n - 1) - k - m
    stat = (num / m) / ((et @ et - num) / dof)
    return {"stat": stat, "df": (m, dof), "p": float(stats.f.sf(stat, m, dof))}


def ar_confidence_set(y, X, M, W_S, grid0: Sequence[float], grid1: Sequence[float], level: float = 0.95,
                      extra=None) -> dict:
    """AR confidence set {r : AR(r) <= F_level} on a grid of (rho0, rho1)."""
    P = np.array([[anderson_rubin(y, X, M, W_S, (a, b), extra)["p"] for b in grid1] for a in grid0])
    return {"grid0": np.asarray(grid0), "grid1": np.asarray(grid1), "p": P, "in_set": P > 1 - level}
