"""
fixed_effects.py
----------------
Unit fixed effects with cross-period weights at fixed T.

Estimators on the equation

    Y = U rho + R gamma + (iota_T kron alpha) + xi,      U = [W_j Y]_j,

with city effects alpha removed by F = C_T kron I_n:

  * dummy_qmle   : concentrated QMLE (city dummies), biased O(1/T) for
                   cross-period coefficients (paper, Proposition 2);
  * aqs          : adjusted quasi-score, g_rho_j = (W_j Y)' F e / s2 - tr(F G_j)
                   (paper, Theorem 3); with adjust_regressors=True the trace is
                   tr((F - P_R) G_j), P_R the projection on the demeaned
                   regressors, which also removes the O(k) mean of the score
                   left by concentrating gamma out (it matters when R holds
                   many year or region-year dummies); with the sandwich variance
                   Gamma^{-1} (Upsilon + D Sigma_pi D') Gamma^{-T}, where
                   Upsilon uses the linear-quadratic-form variance with mu3,
                   mu4, and D Sigma_pi D' is the generated-regressor term of
                   the control functions;
  * cf_2sls_fe   : 2SLS after demeaning, instruments supplied by the caller;
  * first_stage_fe : within-city first stage for the formation variable.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg, stats, optimize
from typing import Optional, Sequence
from .multi_w import RecursiveNetwork, city_demean

np.seterr(all="ignore")


# ---------------------------------------------------------------------------
# first stage
# ---------------------------------------------------------------------------

def _full_rank(A: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    q, r, piv = linalg.qr(A, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    return np.sort(piv[d > tol * d.max()])


def first_stage_fe(h: np.ndarray, Z: np.ndarray, n: int, T: int,
                   n_instruments: Optional[int] = None) -> dict:
    """
    Within-city OLS  F h = F Z pi + F eps.

    Z holds the excluded instruments first, then year dummies and controls.
    Returns the residual eps_hat (already within), the demeaned regressors
    kept after a rank check, pi_hat, the iid and city-cluster covariances,
    and the cluster F statistic of the first `n_instruments` columns.
    """
    hf = city_demean(h, n, T)
    Zf = city_demean(Z, n, T)
    keep = _full_rank(Zf)
    Zf = Zf[:, keep]
    pi = linalg.lstsq(Zf, hf)[0]
    e = hf - Zf @ pi
    N, k = Zf.shape
    inv = linalg.inv(Zf.T @ Zf)
    s2 = e @ e / (N - n - k)
    sc = (Zf * e[:, None]).reshape(T, n, -1).sum(0)
    V_cl = inv @ (sc.T @ sc) @ inv * n / (n - 1)
    out = {"eps_hat": e, "Z_within": Zf, "keep": keep, "pi": pi, "sigma2": s2, "dof": N - n - k,
           "V_iid": s2 * inv, "V_cluster": V_cl, "ZtZ_inv": inv,
           "resid_sd": float(np.std(e)), "R2_within": 1 - float(e @ e) / float(hf @ hf)}
    if n_instruments:
        kz = int(np.sum(keep < n_instruments))
        b = pi[:kz]
        V = V_cl[:kz, :kz]
        F = float(b @ linalg.solve(V, b) / kz)
        out.update({"cluster_F": F, "cluster_p": float(stats.chi2.sf(F * kz, kz)), "n_instruments": kz})
    return out


def joint_pi_covariance(first_stages: Sequence[dict], n: int, T: int) -> np.ndarray:
    """Joint covariance of the first-stage coefficients across modes (iid errors, correlated across modes)."""
    m = len(first_stages)
    e = [fs["eps_hat"] for fs in first_stages]
    N = len(e[0])
    kmax = max(fs["Z_within"].shape[1] for fs in first_stages)
    dof = min(fs.get("dof", N - n - kmax) for fs in first_stages)
    S = np.array([[e[a] @ e[b] / dof for b in range(m)] for a in range(m)])
    blocks = [[S[a, b] * first_stages[a]["ZtZ_inv"] @ (first_stages[a]["Z_within"].T @ first_stages[b]["Z_within"])
               @ first_stages[b]["ZtZ_inv"] for b in range(m)] for a in range(m)]
    return np.block(blocks)


# ---------------------------------------------------------------------------
# within design
# ---------------------------------------------------------------------------

class WithinDesign:
    """Demeaned y, U, R and the versions of y and U with F R partialled out."""

    def __init__(self, y, U, R, n: int, T: int):
        self.n, self.T = n, T
        self.y_raw, self.U_raw, self.R_raw = np.asarray(y, float), np.asarray(U, float), np.asarray(R, float)
        self.yF = city_demean(self.y_raw, n, T)
        self.UF = city_demean(self.U_raw, n, T)
        RF = city_demean(self.R_raw, n, T)
        self.keep = _full_rank(RF)
        self.RF = RF[:, self.keep]
        self.Q = linalg.orth(self.RF)
        self.yt = self.yF - self.Q @ (self.Q.T @ self.yF)
        self.Ut = self.UF - self.Q @ (self.Q.T @ self.UF)
        self.N = len(self.yF)
        self.k = self.RF.shape[1]

    def gamma(self, rho):
        return linalg.lstsq(self.RF, self.yF - self.UF @ rho)[0]

    def resid(self, rho):
        return self.yF - self.UF @ rho - self.RF @ self.gamma(rho)


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def dummy_drift(net: RecursiveNetwork, rho) -> dict:
    """
    Drift of the dummy-variable QMLE (paper, eq. (4.5), Proposition 2).

    The concentrated rho_j score at the truth has conditional mean
    -(T - 1)^{-1} sum_{t != s} tr([G_j]_ts), a sum over the cross-period blocks of
    G_j = W_j S^{-1}. Returns that drift and its value per observation, d_j / (T - 1).
    It is zero for block-diagonal weights; when it is not, the dummy-variable
    estimator of rho is off by O(1/T) whatever n, and the AQS is needed.
    """
    assert np.isfinite(net.factor(rho))
    n, T = net.n, net.T
    tFG, SB = net.tr_FG()
    tr_all = np.array([np.trace((W @ SB).reshape(T, n, n).sum(0)) for W in net.W])    # sum_{t,s} tr([G_j]_ts)
    tr_G = tFG + tr_all / T                                                           # tr(G_j)
    cross = tr_all - tr_G
    drift = -cross / (T - 1)
    return {"names": list(net.names), "drift": drift, "per_obs": drift / net.N, "cross_trace": cross}


def dummy_qmle(net: RecursiveNetwork, wd: WithinDesign, rho0: Sequence[float]) -> dict:
    """Concentrated QMLE with city dummies: maximise -N/2 ln s2(rho) + ln|det S(rho)|."""
    N = wd.N

    def negll(rho):
        ld = net.factor(rho)
        if not np.isfinite(ld):
            return 1e10
        e = wd.yt - wd.Ut @ rho
        return 0.5 * N * np.log(e @ e / N) - ld

    r = optimize.minimize(negll, np.asarray(rho0, float), method="Nelder-Mead",
                          options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
    r = optimize.minimize(negll, r.x, method="BFGS", options={"gtol": 1e-7})
    rho = r.x
    return {"rho": rho, "gamma": wd.gamma(rho), "loglik": -r.fun,
            "sigma2": float(wd.resid(rho) @ wd.resid(rho) / N)}


def aqs_equations(net: RecursiveNetwork, wd: WithinDesign, rho, dof: int, adjust_regressors: bool = False):
    ld = net.factor(rho)
    if not np.isfinite(ld):
        return np.full(len(rho), 1e6), np.nan, None
    e = wd.yt - wd.Ut @ rho
    s2 = e @ e / dof
    tFG, SB = net.tr_FG()
    if adjust_regressors:
        tFG = tFG - net.tr_PG(wd.Q)[0]
    return wd.Ut.T @ e / s2 - tFG, s2, SB


def aqs_solve(net: RecursiveNetwork, wd: WithinDesign, rho0: Sequence[float],
              dof: Optional[int] = None, adjust_regressors: bool = False) -> np.ndarray:
    """Root of the adjusted quasi-score in rho, starting from rho0 (usually the dummy QMLE)."""
    if dof is None:
        dof = wd.N - wd.n - wd.k
    f = lambda rho: aqs_equations(net, wd, rho, dof, adjust_regressors)[0]
    scale = max(1.0, float(np.abs(wd.Ut.T @ wd.yt).max()))
    rho0 = np.asarray(rho0, float)
    best = None
    for method, start in [("hybr", rho0), ("lm", rho0), ("hybr", 0.5 * rho0), ("lm", np.zeros_like(rho0))]:
        opts = {"xtol": 1e-12} if method == "hybr" else {"xtol": 1e-12, "ftol": 1e-12}
        sol = optimize.root(f, start, method=method, options=opts)
        err = np.max(np.abs(f(sol.x))) / scale
        if best is None or err < best[1]:
            best = (sol.x, err)
        if err < 1e-9:
            break
    if best[1] > 1e-7:
        raise RuntimeError(f"AQS did not converge (relative residual {best[1]:.2e})")
    return best[0]


def aqs_variance(net: RecursiveNetwork, wd: WithinDesign, rho, R_names: Sequence[str],
                 first_stages: Optional[dict] = None, cf_prefix: str = "delta_",
                 cluster: Optional[dict] = None, adjust_regressors: bool = False) -> dict:
    """
    Sandwich variance of the AQS estimator for theta = (rho, gamma, sigma2).

    Gamma is the expected Jacobian and Upsilon the variance of the estimating
    equations (paper, Appendix E, with the mu3/mu4 terms of Lemma 9).
    first_stages maps a mode name to its first_stage_fe() result; the control
    function of mode m must be the column named cf_prefix + m in R_names (it
    may be any multiple of the first-stage residual; the scale is recovered).

    wd.U_raw may contain weights on years before the estimation sample (a
    predetermined history); that part enters the mean of each lag.

    cluster maps a label to an array of n cluster ids (one per unit). For each
    label the result holds V_cluster[label]: Gamma^{-1} Ups_c Gamma^{-T} with
    Ups_c the outer product of cluster sums of the score contributions (with
    the first-stage influence functions added) plus the cross-cluster part of
    tr(F G_j F G_l), which the outer product cannot see. It allows arbitrary
    correlation of the errors within a cluster; Theorem 3 does not cover it.
    """
    assert np.isfinite(net.factor(rho))
    n, T, N = wd.n, wd.T, wd.N
    p = len(rho)
    RF, k = wd.RF, wd.k
    gam = wd.gamma(rho)
    e = wd.resid(rho)
    dof = N - n - k
    s2 = e @ e / dof
    tFG, SB = net.tr_FG()
    tFGG = net.tr_FGG(SB)
    if adjust_regressors:
        tPG, SQ = net.tr_PG(wd.Q)
        tFG = tFG - tPG
        tFGG = tFGG - net.tr_PGG(wd.Q, SQ)
    P = p + k + 1
    names = list(R_names[j] for j in wd.keep)

    Sinv = net.inverse()
    Q = []
    dPG = np.zeros((p, N))                        # diag(P_R G_j)
    for j in range(p):
        Gj = np.asarray(net.W[j] @ Sinv)
        Q.append(city_demean(Gj, n, T))          # F G_j
        if adjust_regressors:
            dPG[j] = np.sum(wd.Q * (Gj.T @ wd.Q), axis=1)
        del Gj
    del Sinv
    e_raw = wd.y_raw - wd.U_raw @ rho - wd.R_raw[:, wd.keep] @ gam
    alpha = e_raw.reshape(T, n).mean(0)
    # lags = W_j Y (sample years) + hist_j (weights on earlier years, predetermined)
    hist = wd.U_raw - np.column_stack([net.W[j] @ wd.y_raw for j in range(p)])
    histF = city_demean(hist, n, T)
    mu = wd.R_raw[:, wd.keep] @ gam + np.tile(alpha, T) + hist @ rho
    # third and fourth moments of xi from the within residuals r = xi - mean_t(xi):
    # E r^3 = (T-1)(T-2)/T^2 mu3,  E r^4 = (T-1)(T^2-3T+3)/T^3 mu4 + 3(T-1)(2T-3)/T^3 s2^2
    mu3 = float(np.mean(e ** 3) * T ** 2 / ((T - 1) * (T - 2))) if T > 2 else 0.0
    k4 = (T - 1) * (T ** 2 - 3 * T + 3) / T ** 3
    c4 = 3 * (T - 1) * (2 * T - 3) / T ** 3
    mu4 = float((np.mean(e ** 4) - c4 * s2 ** 2) / k4)

    Gmu = [Qj @ mu + histF[:, j] for j, Qj in enumerate(Q)]   # F E[lag_j]
    a = Gmu                                   # c_j = F G_j mu, the linear part of the score
    dq = [np.diag(Qj).copy() for Qj in Q]
    trQQ = np.zeros((p, p))
    trQQt = np.zeros((p, p))
    Ups = np.zeros((P, P))
    Gam = np.zeros((P, P))
    Fii = 1 - 1 / T
    for j in range(p):
        for l in range(j, p):
            trQQ[j, l] = trQQ[l, j] = np.sum(Q[j] * Q[l])
            trQQt[j, l] = trQQt[l, j] = np.sum(Q[j] * Q[l].T)
            v = trQQ[j, l] + trQQt[j, l] + a[j] @ a[l] / s2 \
                + (mu4 - 3 * s2 ** 2) * (dq[j] @ dq[l]) / s2 ** 2 \
                + mu3 * (dq[j] @ a[l] + dq[l] @ a[j]) / s2 ** 2
            Ups[j, l] = Ups[l, j] = v
        Ups[j, p:p + k] = a[j] @ RF / s2 + mu3 * (dq[j] @ RF) / s2 ** 2
        Ups[p:p + k, j] = Ups[j, p:p + k]
        Ups[j, -1] = tFG[j] / s2 + (mu4 - 3 * s2 ** 2) * dq[j].sum() * Fii / (2 * s2 ** 3) \
            + mu3 * Fii * a[j].sum() / (2 * s2 ** 3)
        Ups[-1, j] = Ups[j, -1]
    Ups[p:p + k, p:p + k] = RF.T @ RF / s2
    Ups[p:p + k, -1] = mu3 * Fii * RF.sum(0) / (2 * s2 ** 3)
    Ups[-1, p:p + k] = Ups[p:p + k, -1]
    Ups[-1, -1] = (N - n) / (2 * s2 ** 2) + (mu4 - 3 * s2 ** 2) * N * Fii ** 2 / (4 * s2 ** 4)
    for j in range(p):
        for l in range(p):
            Gam[j, l] = -(Gmu[j] @ Gmu[l] / s2 + trQQ[j, l] + tFGG[j, l])
        Gam[j, p:p + k] = -Gmu[j] @ RF / s2
        Gam[p:p + k, j] = Gam[j, p:p + k]
        Gam[j, -1] = -tFG[j] / s2
        Gam[-1, j] = Gam[j, -1]
    Gam[p:p + k, p:p + k] = -RF.T @ RF / s2
    Gam[-1, -1] = -(N - n) / (2 * s2 ** 2)
    obs_unit = np.tile(np.arange(n), T)
    cross_tr = {}
    for lab, ids in (cluster or {}).items():
        co = np.asarray(ids)[obs_unit]
        diff = co[:, None] != co[None, :]
        cross_tr[lab] = np.array([[np.sum(Q[j].T * Q[l] * diff) for l in range(p)] for j in range(p)])
        del diff
    del Q

    # generated-regressor term
    Dm = None
    Sig_pi = None
    if first_stages:
        modes = list(first_stages)
        Sig_pi = joint_pi_covariance([first_stages[m] for m in modes], n, T)
        UR = np.column_stack([wd.UF, RF])
        Dm = np.zeros((P, Sig_pi.shape[0]))
        off = 0
        for m in modes:
            Zt = first_stages[m]["Z_within"]
            kz = Zt.shape[1]
            col = cf_prefix + m
            if col in names:
                ci = names.index(col)
                eh = first_stages[m]["eps_hat"]
                scale = float(RF[:, ci] @ eh / (eh @ eh))      # column = scale * eps_hat
                Dm[:p + k, off:off + kz] = gam[ci] * scale * UR.T @ Zt / s2
            off += kz
    Gi = linalg.inv(Gam)
    V_ups = Gi @ Ups @ Gi.T
    V_full = V_ups if Dm is None else Gi @ (Ups + Dm @ Sig_pi @ Dm.T) @ Gi.T
    # cluster-robust versions
    V_cl = {}
    if cluster:
        sc = np.zeros((N, P))
        sc[:, :p] = wd.UF * e[:, None] / s2 - np.column_stack(dq) + dPG.T
        sc[:, p:p + k] = RF * e[:, None] / s2
        sc[:, -1] = (e ** 2 - s2 * dof / N) / (2 * s2 ** 2)
        if Dm is not None:
            psi = np.column_stack([(first_stages[m]["Z_within"] * first_stages[m]["eps_hat"][:, None])
                                   @ first_stages[m]["ZtZ_inv"] for m in modes])
            sc = sc + psi @ Dm.T
        for lab, ids in cluster.items():
            ids = np.unique(np.asarray(ids), return_inverse=True)[1]
            Gc = ids.max() + 1
            Sc = np.zeros((Gc, P))
            np.add.at(Sc, ids[obs_unit], sc)
            Uc = Sc.T @ Sc * Gc / (Gc - 1)
            Uc[:p, :p] += cross_tr[lab]
            V_cl[lab] = Gi @ Uc @ Gi.T
    return {"theta": np.r_[rho, gam, s2], "names": list(net.names) + names + ["sigma2"],
            "V": V_full, "V_no_generated": V_ups, "V_information": -Gi, "V_cluster": V_cl,
            "Gamma": Gam, "Upsilon": Ups, "sigma2": s2, "mu3": mu3, "mu4": mu4,
            "alpha": alpha, "tr_FG": tFG, "resid_within": e}


def aqs_fit(net: RecursiveNetwork, wd: WithinDesign, R_names: Sequence[str],
            rho0: Optional[Sequence[float]] = None, first_stages: Optional[dict] = None,
            variance: bool = True, cluster: Optional[dict] = None, adjust_regressors: bool = False) -> dict:
    """Dummy QMLE start, AQS root, sandwich variance. Returns coefficient vector, SE, names."""
    if rho0 is None:
        rho0 = np.zeros(net.p)
    dq = dummy_qmle(net, wd, rho0)
    rho = aqs_solve(net, wd, dq["rho"], adjust_regressors=adjust_regressors)
    out = {"rho": rho, "rho_dummy": dq["rho"], "loglik_dummy": dq["loglik"], "gamma": wd.gamma(rho)}
    if variance:
        va = aqs_variance(net, wd, rho, R_names, first_stages, cluster=cluster, adjust_regressors=adjust_regressors)
        out.update(va)
        out["se"] = np.sqrt(np.maximum(np.diag(va["V"]), 0))
        out["se_information"] = np.sqrt(np.maximum(np.diag(va["V_information"]), 0))
        out["se_cluster"] = {lab: np.sqrt(np.maximum(np.diag(Vc), 0)) for lab, Vc in va["V_cluster"].items()}
    return out


def cf_2sls_fe(wd: WithinDesign, Z_excl: np.ndarray, R_names: Sequence[str], net_names: Sequence[str],
               cluster: Optional[np.ndarray] = None) -> dict:
    """2SLS after city demeaning; Z_excl are the excluded instruments on the raw rows.
    Standard errors are clustered by unit, or by the n cluster ids in `cluster`."""
    n, T = wd.n, wd.T
    ZF = city_demean(Z_excl, n, T)
    Zt = ZF - wd.Q @ (wd.Q.T @ ZF)
    Qz = linalg.orth(Zt)
    Uh = Qz @ (Qz.T @ wd.Ut)
    rho = linalg.lstsq(Uh, wd.yt)[0]
    gam = wd.gamma(rho)
    e = wd.resid(rho)
    Ufull = np.column_stack([wd.UF, wd.RF])
    Qall = linalg.orth(np.column_stack([wd.RF, ZF]))
    Uhat = Qall @ (Qall.T @ Ufull)
    M = linalg.pinv(Uhat)
    b = M @ wd.yF
    sc = (M.T * e[:, None]).reshape(T, n, -1).sum(0)
    g = n
    if cluster is not None:
        ids = np.unique(np.asarray(cluster), return_inverse=True)[1]
        g = ids.max() + 1
        scg = np.zeros((g, sc.shape[1]))
        np.add.at(scg, ids, sc)
        sc = scg
    fac = g / (g - 1) * (len(e) - 1) / (len(e) - Ufull.shape[1] - n)
    V = fac * sc.T @ sc
    F = []
    for j in range(wd.Ut.shape[1]):
        bz = linalg.lstsq(Zt, wd.Ut[:, j])[0]
        r = wd.Ut[:, j] - Zt @ bz
        F.append(float((bz @ Zt.T @ Zt @ bz / Zt.shape[1]) / (r @ r / (len(r) - Zt.shape[1] - n - wd.k))))
    return {"theta": b, "V": V, "se": np.sqrt(np.diag(V)), "names": list(net_names) + [R_names[j] for j in wd.keep],
            "partial_F": np.array(F), "n_instruments": int(Zt.shape[1]), "resid_within": e}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def wald_test(theta: np.ndarray, V: np.ndarray, R: np.ndarray, r: Optional[np.ndarray] = None) -> dict:
    """Wald test of R theta = r."""
    R = np.atleast_2d(np.asarray(R, float))
    r = np.zeros(R.shape[0]) if r is None else np.asarray(r, float)
    d = R @ theta - r
    VV = R @ V @ R.T
    w = float(d @ linalg.solve(VV, d))
    return {"stat": w, "df": int(R.shape[0]), "p": float(stats.chi2.sf(w, R.shape[0]))}


def restriction_matrix(names: Sequence[str], restrictions: Sequence) -> np.ndarray:
    """
    Build R from restrictions given as names or (name_a, name_b) pairs.
    A single name tests coefficient = 0; a pair tests coef_a - coef_b = 0.
    """
    R = np.zeros((len(restrictions), len(names)))
    for i, rs in enumerate(restrictions):
        if isinstance(rs, str):
            R[i, names.index(rs)] = 1
        else:
            R[i, names.index(rs[0])] = 1
            R[i, names.index(rs[1])] = -1
    return R
