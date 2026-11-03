"""
inference.py
------------
Spillover effects and their inference (paper, eqs. (2.2)-(2.3) and Section 6).

The effect of regressor k is the N x N matrix

    S_k = S(rho)^{-1} (beta_k I + sum_j theta_jk W_j),

and its block averages are

    DE_t       = n^{-1} tr([S_k]_(t,t))                        direct effect
    IEw_t      = n^{-1} iota' [S_k]_(t,t) iota - DE_t            within-period indirect
    IE_(t<-s)  = n^{-1} iota' [S_k]_(t,s) iota,  t != s          cross-period (eq. (6.1))

Standard errors use the delta method with the analytic gradient (eq. (6.3)):
d/d rho_j = n^{-1} iota'[S^{-1} W_j S_k]_(t,s) iota, d/d beta_k = n^{-1} iota'[S^{-1}]_(t,s) iota,
d/d theta_jk = n^{-1} iota'[S^{-1} W_j]_(t,s) iota. The block sums need one solve with
T right-hand sides; the direct effects need the diagonal blocks of S^{-1}.

Multiple testing over the T(T-1)/2 (recursive) or T(T-1) cross-period effects uses the
step-up procedure of Benjamini-Hochberg, or Benjamini-Yekutieli, which does not need
the positive dependence the shared estimate would have to satisfy.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg, stats
from typing import Optional, Sequence


def _period_indicators(n: int, T: int) -> np.ndarray:
    return np.kron(np.eye(T), np.ones((n, 1)))          # (N, T), column s = indicator of period s


def _block_means(A: np.ndarray, n: int, T: int) -> np.ndarray:
    """(N, T) -> (T, T): mean over the units of each destination period."""
    return A.reshape(T, n, -1).mean(1)


def _diag_block_traces(A: np.ndarray, n: int, T: int) -> np.ndarray:
    d = np.diag(A).reshape(T, n)
    return d.mean(1)


def spillover_effects(net, rho, beta: float, theta: Sequence[float], direct: bool = True) -> dict:
    """
    Block-averaged effects of one regressor.

    theta holds theta_jk for each lag matrix of net (zeros if the model has no Durbin term).
    Returns IE (T x T; the diagonal holds the total same-period effect DE + IEw), DE, IEw.
    """
    n, T = net.n, net.T
    assert np.isfinite(net.factor(rho))
    D = _period_indicators(n, T)
    V = beta * D + sum(th * (W @ D) for th, W in zip(theta, net.W))
    IE = _block_means(net.solve(V), n, T)
    out = {"IE": IE}
    if direct:
        Sinv = net.inverse()
        Sk = beta * Sinv + sum(th * (Sinv @ W.toarray()) for th, W in zip(theta, net.W))
        out["DE"] = _diag_block_traces(Sk, n, T)
        out["IEw"] = np.diag(IE) - out["DE"]
    return out


def effects_inference(net, fit: dict, x: str, V: Optional[np.ndarray] = None, direct: bool = True,
                      alpha: float = 0.05, fdr: str = "by") -> dict:
    """
    Effects of regressor x with delta-method standard errors.

    fit is a cf_qmle / cf_2sls / within_period_qmle / aqs_fit result (names, theta, V);
    the Durbin coefficients are the entries named '<lag>*<x>'. V overrides fit['V']
    (for example a cluster-robust variance). The cross-period p-values are adjusted
    with fdr = 'bh' or 'by'.
    """
    names = list(fit["names"])
    theta_hat = np.asarray(fit["theta"], float)
    V = fit["V"] if V is None else V
    p = len(net.names)
    i_rho = [names.index(nm) for nm in net.names]
    i_beta = names.index(x)
    i_th = [names.index(f"{nm}*{x}") if f"{nm}*{x}" in names else None for nm in net.names]
    rho = theta_hat[i_rho]
    beta = theta_hat[i_beta]
    th = np.array([theta_hat[i] if i is not None else 0.0 for i in i_th])
    n, T = net.n, net.T
    assert np.isfinite(net.factor(rho))
    D = _period_indicators(n, T)
    WD = [W @ D for W in net.W]
    SD = net.solve(D)                                           # S^{-1} u_s
    SWD = [net.solve(w) for w in WD]                            # S^{-1} W_j u_s
    Z = beta * SD + sum(t * s for t, s in zip(th, SWD))         # S_k u_s
    IE = _block_means(Z, n, T)
    grads = {}                                                  # index -> (T, T) gradient
    for j in range(p):
        grads[i_rho[j]] = _block_means(net.solve(net.W[j] @ Z), n, T)
    grads[i_beta] = _block_means(SD, n, T)
    for j in range(p):
        if i_th[j] is not None:
            grads[i_th[j]] = _block_means(SWD[j], n, T)
    idx = sorted(grads)
    Gt = np.stack([grads[i] for i in idx], -1)                  # (T, T, q)
    Vs = V[np.ix_(idx, idx)]
    SE = np.sqrt(np.maximum(np.einsum("tsa,ab,tsb->ts", Gt, Vs, Gt), 0))
    zt = IE / np.where(SE > 0, SE, np.nan)
    pv = 2 * stats.norm.sf(np.abs(zt))
    out = {"IE": IE, "SE": SE, "z": zt, "p": pv}
    # recursive weights: effects above the diagonal are zero and not tested
    off = ~np.eye(T, dtype=bool)
    lower = np.tril(np.ones((T, T), bool), -1)
    test = lower if np.allclose(IE[np.triu_indices(T, 1)], 0) else off
    padj = np.full((T, T), np.nan)
    rej = np.zeros((T, T), bool)
    r_, a_ = fdr_correction(pv[test], alpha, fdr)
    padj[test], rej[test] = a_, r_
    out.update({"p_adjusted": padj, "reject": rej, "tested": test, "fdr": fdr})
    if direct:
        Sinv = net.inverse()
        SW = [Sinv @ W.toarray() for W in net.W]
        Sk = beta * Sinv + sum(t * s for t, s in zip(th, SW))
        dmean = lambda A: _diag_block_traces(A, n, T)
        DE = dmean(Sk)
        gD = {}
        for j in range(p):
            gD[i_rho[j]] = dmean(SW[j] @ Sk)
        gD[i_beta] = dmean(Sinv)
        for j in range(p):
            if i_th[j] is not None:
                gD[i_th[j]] = dmean(SW[j])
        Gd = np.stack([gD[i] for i in idx], -1)                 # (T, q)
        se_DE = np.sqrt(np.maximum(np.einsum("ta,ab,tb->t", Gd, Vs, Gd), 0))
        diagIE = np.diag(IE)
        Gw = np.stack([np.diagonal(grads[i]) - gD[i] for i in idx], -1)
        se_IEw = np.sqrt(np.maximum(np.einsum("ta,ab,tb->t", Gw, Vs, Gw), 0))
        out.update({"DE": DE, "SE_DE": se_DE, "IEw": diagIE - DE, "SE_IEw": se_IEw})
    return out


# ---------------------------------------------------------------------------
# multiple testing
# ---------------------------------------------------------------------------

def fdr_correction(p_values, alpha: float = 0.05, method: str = "bh"):
    """
    Step-up FDR procedure: 'bh' (Benjamini-Hochberg 1995) or 'by'
    (Benjamini-Yekutieli 2001, alpha divided by sum_{j<=R} 1/j).

    Returns (reject, adjusted p-values), both with the shape of p_values.
    """
    p_values = np.asarray(p_values, float)
    shape = p_values.shape
    pv = p_values.ravel()
    m = len(pv)
    if m == 0:
        return np.zeros(shape, bool), np.zeros(shape)
    c = np.sum(1.0 / np.arange(1, m + 1)) if method == "by" else 1.0
    order = np.argsort(pv)
    ps = pv[order]
    below = ps <= alpha * np.arange(1, m + 1) / (m * c)
    rs = np.zeros(m, bool)
    if below.any():
        rs[:np.where(below)[0].max() + 1] = True
    adj = np.minimum.accumulate((c * m / np.arange(m, 0, -1)) * ps[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    rej, pad = np.empty(m, bool), np.empty(m)
    rej[order], pad[order] = rs, adj
    return rej.reshape(shape), pad.reshape(shape)


def bh_correction(p_values, alpha: float = 0.05):
    """Benjamini-Hochberg step-up procedure; returns (reject, adjusted p-values)."""
    return fdr_correction(p_values, alpha, "bh")


def by_correction(p_values, alpha: float = 0.05):
    """Benjamini-Yekutieli step-up procedure; returns (reject, adjusted p-values)."""
    return fdr_correction(p_values, alpha, "by")


def print_ie_table(result: dict, period_labels: Optional[Sequence] = None) -> str:
    """Cross-period effects with SE; * marks rejection after the FDR adjustment."""
    IE, SE, rej = result["IE"], result["SE"], result.get("reject", np.zeros_like(result["IE"], bool))
    T = IE.shape[0]
    labs = list(period_labels) if period_labels is not None else list(range(1, T + 1))
    lines = [f"Cross-period effects IE(t<-s), SE in parentheses, * = rejected at FDR ({result.get('fdr', 'bh')})"]
    lines.append("t\\s   " + "".join(f"{str(l):>17}" for l in labs))
    for t in range(T):
        row = f"{str(labs[t]):<6}"
        for s in range(T):
            star = "*" if rej[t, s] else " "
            row += f"  {IE[t, s]:>7.3f}({SE[t, s]:.3f}){star}"
        lines.append(row)
    return "\n".join(lines)
