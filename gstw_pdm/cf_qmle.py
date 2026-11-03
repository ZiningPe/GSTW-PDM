"""
cf_qmle.py
----------
Control-function QMLE (paper, Section 3 eq. (3.5), Theorems 1-2).

Model, time-major rows (period by period, units within period):

    Y = sum_j rho_j W_j Y + X beta + sum_j W_j X theta_j + E gamma_E + eb_hat delta_c + xi

The paper's case has two lag matrices, the same-period blocks W_0 and the
cross-period blocks W_1 of one weight matrix; any number of lag matrices is
allowed (for example two networks, each split by split_same_cross). E holds
regressors without a spatial lag (intercept, period dummies). eb_hat = A eps_hat
is the first-stage residual of the formation equation h = Z_W pi + eps, with
A = I_N when L = N and A = I_T kron iota_n when L = T.

Given rho, the other mean parameters are OLS coefficients of S(rho) Y on
R = [X, W_j X, E, eb_hat] and sigma2 is the residual mean square, so the
likelihood is maximised over rho only. ln|det S(rho)| comes from the network:
KroneckerNetwork (eigenvalues of M and W_S), RecursiveNetwork (same-period
blocks only, Lemma 2) or DenseNetwork (N x N factorisation).

Variance (Theorem 2, L = N):

    Var = I^{-1} (V + D Sigma_pi D') I^{-1},

with I the information, V the variance of the score from Lemma 9 (linear-quadratic
forms, mu3 and mu4 of the residuals), and D Sigma_pi D' the generated-regressor
term: D has rows delta_c / sigma2 * Ubar' Z_W on the (rho, beta, theta, E)
coordinates and zero rows on delta_c and sigma2, Sigma_pi = sigma_eps^2 (Z_W'Z_W)^{-1}.
At L = T the expansion is not valid (Proposition 3): the estimator converges to a
point that depends on the realised period shocks. Use within_period_qmle there.
"""

from __future__ import annotations
import warnings
import numpy as np
from scipy import linalg, optimize, stats
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------

def _full_rank(A: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if A.shape[1] == 0:
        return np.zeros(0, int)
    q, r, piv = linalg.qr(A, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    return np.sort(piv[d > tol * d.max()])


def _cols(a, N):
    if a is None:
        return np.zeros((N, 0))
    a = np.asarray(a, float)
    return a[:, None] if a.ndim == 1 else a


def control_from_first_stage(fs: dict, n: int, T: int) -> np.ndarray:
    """eb_hat = A eps_hat: A = I_N if the first stage has N rows, I_T kron iota_n if it has T."""
    e = np.asarray(fs["eps_hat"], float)
    if len(e) == n * T:
        return e.copy()
    if len(e) == T:
        return np.repeat(e, n)
    raise ValueError(f"first stage has {len(e)} rows; expected N = {n * T} or T = {T}")


def build_design(y, X, net, durbin: bool = True, extra=None, first_stages: Optional[dict] = None,
                 controls: Optional[dict] = None, x_names=None, extra_names=None):
    """
    Lag columns U = [W_j y] and regressors R = [X, W_j X, E, controls].

    Returns (U, R, R_names, ctrl), ctrl mapping a first-stage name to its column in R.
    Columns of R that are collinear with earlier ones (for example W_j iota when X
    has an intercept and W_j is row-normalised) are dropped.
    """
    y = np.asarray(y, float).ravel()
    N = len(y)
    X = _cols(X, N)
    E = _cols(extra, N)
    x_names = list(x_names) if x_names is not None else [f"x{i + 1}" for i in range(X.shape[1])]
    e_names = list(extra_names) if extra_names is not None else [f"e{i + 1}" for i in range(E.shape[1])]
    U = np.column_stack([W @ y for W in net.W])
    blocks, names = [X], list(x_names)
    if durbin:
        for nm, W in zip(net.names, net.W):
            blocks.append(W @ X)
            names += [f"{nm}*{x}" for x in x_names]
    blocks.append(E)
    names += e_names
    cols, cnames = [], []
    for m, fs in (first_stages or {}).items():
        cols.append(control_from_first_stage(fs, net.n, net.T))
        cnames.append("delta_" + m)
    for m, c in (controls or {}).items():
        cols.append(np.asarray(c, float).ravel())
        cnames.append("delta_" + m)
    R = np.column_stack(blocks + ([np.column_stack(cols)] if cols else []))
    names += cnames
    keep = _full_rank(R)
    R = R[:, keep]
    names = [names[i] for i in keep]
    ctrl = {m: names.index("delta_" + m) for m in (first_stages or {}) if "delta_" + m in names}
    return U, R, names, ctrl


# ---------------------------------------------------------------------------
# profile likelihood
# ---------------------------------------------------------------------------

def profile_loglik(rho, net, yt: np.ndarray, Ut: np.ndarray, N: int) -> float:
    """ln|det S(rho)| - N/2 ln sigma2(rho), with yt, Ut already residualised on R."""
    ld = net.factor(rho)
    if not np.isfinite(ld):
        return -np.inf
    e = yt - Ut @ np.asarray(rho, float)
    return ld - 0.5 * N * np.log(e @ e / N)


def _maximise(net, yt, Ut, N, rho0):
    def f(r):
        if not np.all(np.isfinite(r)):
            return 1e12
        v = -profile_loglik(r, net, yt, Ut, N)
        return min(v, 1e12) if np.isfinite(v) else 1e12
    r = optimize.minimize(f, np.asarray(rho0, float), method="Nelder-Mead",
                          options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 6000})
    r2 = optimize.minimize(f, r.x, method="BFGS", options={"gtol": 1e-8})
    best = r2 if r2.fun <= r.fun else r
    return best.x, -best.fun


# ---------------------------------------------------------------------------
# variance
# ---------------------------------------------------------------------------

def _moments(e: np.ndarray, s2: float, g: Optional[int]):
    """mu3, mu4 of xi from residuals; with g, the residuals are deviations from means over groups of size g."""
    if g is None or g <= 1:
        return float(np.mean(e ** 3)), float(np.mean(e ** 4))
    mu3 = float(np.mean(e ** 3) * g ** 2 / ((g - 1) * (g - 2))) if g > 2 else 0.0
    k4 = (g - 1) * (g ** 2 - 3 * g + 3) / g ** 3
    c4 = 3 * (g - 1) * (2 * g - 3) / g ** 3
    return mu3, float((np.mean(e ** 4) - c4 * s2 ** 2) / k4)


def qmle_information_and_score_variance(net, R: np.ndarray, rho, gamma, s2: float, mu3: float, mu4: float,
                                        basis_diag=None, basis_vec=None):
    """
    Information I and score variance V of zeta = (rho, gamma, sigma2) at the estimate.

    Score at the truth, with G_j = W_j S^{-1} and m = R gamma:
        rho_j : (G_j m)' xi / s2 + xi' G_j xi / s2 - tr(G_j)
        gamma : R' xi / s2
        sigma2: xi' xi / (2 s2^2) - N / (2 s2)
    V applies the linear-quadratic form lemma entry by entry. When the model is
    estimated on transformed data xi* = P' xi (within-period deviations),
    basis_diag(A) = diag(P A P') and basis_vec(c) = P c give the mu3/mu4 terms.
    """
    assert np.isfinite(net.factor(rho))
    N = R.shape[0]
    p, k = len(rho), R.shape[1]
    P = p + k + 1
    dg = basis_diag or (lambda A: np.diag(A).copy())
    bv = basis_vec or (lambda c: c)
    Sinv = net.inverse()
    G = [np.asarray(W @ Sinv) for W in net.W]
    del Sinv
    m = R @ gamma
    Gm = [Gj @ m for Gj in G]
    trG = np.array([np.trace(Gj) for Gj in G])
    trGG = np.array([[np.sum(G[j] * G[l].T) for l in range(p)] for j in range(p)])
    trGtG = np.array([[np.sum(G[j] * G[l]) for l in range(p)] for j in range(p)])
    dG = [dg(Gj) for Gj in G]
    dI = dg(np.eye(N))
    pGm = [bv(v) for v in Gm]
    pR = bv(R)
    del G

    Info = np.zeros((P, P))
    for j in range(p):
        for l in range(p):
            Info[j, l] = Gm[j] @ Gm[l] / s2 + trGtG[j, l] + trGG[j, l]
        Info[j, p:p + k] = Info[p:p + k, j] = Gm[j] @ R / s2
        Info[j, -1] = Info[-1, j] = trG[j] / s2
    Info[p:p + k, p:p + k] = R.T @ R / s2
    Info[-1, -1] = N / (2 * s2 ** 2)

    # non-Gaussian terms of the linear-quadratic form lemma
    k4 = mu4 - 3 * s2 ** 2
    V = Info.copy()
    for j in range(p):
        for l in range(p):
            V[j, l] += (mu3 * (pGm[j] @ dG[l] + pGm[l] @ dG[j]) + k4 * dG[j] @ dG[l]) / s2 ** 2
        V[j, p:p + k] += mu3 * (dG[j] @ pR) / s2 ** 2
        V[p:p + k, j] = V[j, p:p + k]
        V[j, -1] += (mu3 * pGm[j] @ dI + k4 * dG[j] @ dI) / (2 * s2 ** 3)
        V[-1, j] = V[j, -1]
    V[p:p + k, -1] += mu3 * (dI @ pR) / (2 * s2 ** 3)
    V[-1, p:p + k] = V[p:p + k, -1]
    V[-1, -1] += k4 * (dI @ dI) / (4 * s2 ** 4)
    return Info, V, trG


def _Z(fs):
    return np.asarray(fs["Z_within"] if "Z_within" in fs else fs["Z"], float)


def joint_pi_covariance_pooled(first_stages: Sequence[dict]) -> np.ndarray:
    """Cov(pi_hat) across first stages: sigma_ab (Z_a'Z_a)^{-1} Z_a'Z_b (Z_b'Z_b)^{-1}."""
    e = [np.asarray(fs["eps_hat"], float) for fs in first_stages]
    Zs = [_Z(fs) for fs in first_stages]
    inv = [fs["ZtZ_inv"] if "ZtZ_inv" in fs else linalg.inv(Z.T @ Z) for fs, Z in zip(first_stages, Zs)]
    dof = [fs.get("dof", len(ea) - Z.shape[1]) for fs, ea, Z in zip(first_stages, e, Zs)]
    m = len(e)
    blocks = [[(e[a] @ e[b] / min(dof[a], dof[b])) * inv[a] @ (Zs[a].T @ Zs[b]) @ inv[b]
               for b in range(m)] for a in range(m)]
    return np.block(blocks)


def generated_regressor_term(Ubar: np.ndarray, gamma, ctrl: dict, first_stages: dict, s2: float,
                             R: np.ndarray, p: int, k: int):
    """
    D and Sigma_pi of Theorem 2 for one or several first stages at L = N.

    Rows of D: delta_c,m * scale_m / s2 * Ubar' Z_m on the coordinates in Ubar
    (rho and every column of R except the controls); zero on the controls and sigma2.
    The control column may be any multiple of eps_hat; the multiple is recovered.
    """
    modes = [m for m in first_stages if m in ctrl]
    Sig = joint_pi_covariance_pooled([first_stages[m] for m in modes])
    D = np.zeros((p + k + 1, Sig.shape[0]))
    rows = list(range(p)) + [p + c for c in range(k) if c not in ctrl.values()]
    off = 0
    for m in modes:
        Z = _Z(first_stages[m])
        kz = Z.shape[1]
        ci = ctrl[m]
        eh = np.asarray(first_stages[m]["eps_hat"], float)
        scale = float(R[:, ci] @ eh / (eh @ eh))
        D[rows, off:off + kz] = gamma[ci] * scale * (Ubar.T @ Z) / s2
        off += kz
    return D, Sig


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------

def _fit(y, U, R, names, ctrl, net, first_stages, rho0, variance, basis_diag=None, basis_vec=None,
         group=None, resid_orig=None, label="CF-QMLE"):
    N, p, k = len(y), U.shape[1], R.shape[1]
    Q = linalg.orth(R)
    yt = y - Q @ (Q.T @ y)
    Ut = U - Q @ (Q.T @ U)
    rho0 = np.zeros(p) if rho0 is None else np.asarray(rho0, float)
    rho, ll = _maximise(net, yt, Ut, N, rho0)
    gamma = linalg.lstsq(R, y - U @ rho)[0]
    e = y - U @ rho - R @ gamma
    s2 = float(e @ e / N)
    out = {"estimator": label, "rho": rho, "gamma": gamma, "sigma2": s2,
           "theta": np.r_[rho, gamma, s2], "names": list(net.names) + list(names) + ["sigma2"],
           "loglik": float(ll - 0.5 * N * (1 + np.log(2 * np.pi))), "resid": e, "N": N}
    if not variance:
        return out
    er = e if resid_orig is None else resid_orig(e)
    mu3, mu4 = _moments(er, s2, group)
    Info, Vs, trG = qmle_information_and_score_variance(net, R, rho, gamma, s2, mu3, mu4, basis_diag, basis_vec)
    Ii = linalg.inv(Info)
    V_rob = Ii @ Vs @ Ii
    V = V_rob
    gen = {m: fs for m, fs in (first_stages or {}).items() if m in ctrl}
    if gen:
        if any(len(fs["eps_hat"]) != N for fs in gen.values()):
            warnings.warn("first stage with L = T: Theorem 2 does not apply and the generated-regressor "
                          "term is not added. The CF-QMLE is the naive estimator of Proposition 3 and "
                          "is inconsistent unless delta_c = 0; use within_period_qmle.")
        else:
            nonctrl = [c for c in range(k) if c not in ctrl.values()]
            Ubar = np.column_stack([U, R[:, nonctrl]])
            D, Sig = generated_regressor_term(Ubar, gamma, ctrl, gen, s2, R, p, k)
            V = Ii @ (Vs + D @ Sig @ D.T) @ Ii
            out["D"], out["Sigma_pi"] = D, Sig
    out.update({"V": V, "V_no_generated": V_rob, "V_information": Ii, "Information": Info,
                "score_variance": Vs, "mu3": mu3, "mu4": mu4,
                "se": np.sqrt(np.maximum(np.diag(V), 0)),
                "se_information": np.sqrt(np.maximum(np.diag(Ii), 0))})
    return out


def cf_qmle(y, X, net, first_stages: Optional[dict] = None, controls: Optional[dict] = None,
            durbin: bool = True, extra=None, x_names=None, extra_names=None,
            rho0=None, variance: bool = True) -> dict:
    """
    Control-function QMLE with the two-step variance of Theorem 2.

    Parameters
    ----------
    y, X        : (N,) outcome and (N, K) regressors, time-major.
    net         : KroneckerNetwork, RecursiveNetwork, DenseNetwork or TiedNetwork;
                  its lag matrices W_j define rho_j (paper: W_0 same period, W_1 cross period).
    first_stages: {name: first_stage_fit(...) result}. The control A eps_hat enters R as
                  'delta_<name>' and its first-stage uncertainty enters the variance.
    controls    : {name: (N,) column} extra control functions without first-stage correction.
    durbin      : include W_j X for every lag (theta_j).
    extra       : (N, q) regressors without spatial lags (intercept, period dummies).
    rho0        : starting value for rho.

    Returns
    -------
    dict with theta = (rho, gamma, sigma2), names, se (Theorem 2), se_information
    (information only), V, V_no_generated, loglik, resid, mu3, mu4.
    """
    U, R, names, ctrl = build_design(y, X, net, durbin, extra, first_stages, controls, x_names, extra_names)
    y = np.asarray(y, float).ravel()
    return _fit(y, U, R, names, ctrl, net, first_stages, rho0, variance)


def qmle(y, X, net, durbin: bool = True, extra=None, x_names=None, extra_names=None,
         rho0=None, variance: bool = True) -> dict:
    """QMLE without a control function (exogenous weights, or the static model when net has W_0 only)."""
    return cf_qmle(y, X, net, None, None, durbin, extra, x_names, extra_names, rho0, variance)


def wald_delta_c(fit: dict, names: Optional[Sequence[str]] = None) -> dict:
    """
    Wald test of delta_c = 0 with the information variance (paper, Appendix F.2).

    On the naive estimator at L = T the test is valid when no other period effects are
    present; at L = N it is the usual exogeneity test of the weights.
    """
    names = names or [nm for nm in fit["names"] if nm.startswith("delta_")]
    idx = [fit["names"].index(nm) for nm in names]
    b = fit["theta"][idx]
    V = fit["V_information"][np.ix_(idx, idx)]
    w = float(b @ linalg.solve(V, b))
    return {"stat": w, "df": len(idx), "p": float(stats.chi2.sf(w, len(idx))), "names": list(names)}
