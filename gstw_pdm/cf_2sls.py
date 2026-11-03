"""
cf_2sls.py
----------
Control-function 2SLS (paper, eq. (3.6), Appendix H).

Regress Y on

    Ubar = [W_j Y, X, W_j X, E, eb_hat]

with instruments

    Qbar = [X, W_j X, W_j W_l X (all ordered pairs), E, eb_hat],

so that for the paper's two lag matrices W^2 X stands for W_0^2 X, W_0 W_1 X,
W_1 W_0 X and W_1^2 X. The control removes the endogeneity of W and the
instruments handle the simultaneity of W Y; only linear moments are used.

Variance (Theorem H.2): with e = xi + delta_c delta, delta = A Z_W (pi_hat - pi),

    Var = B Ubar'P_Q Sigma_e P_Q Ubar B,   B = (Ubar' P_Q Ubar)^{-1},
    Sigma_e = sigma2 I + sum_{a,b} dc_a dc_b A Z_a Cov(pi_a, pi_b) Z_b' A'.

At L = N this is consistent. At L = T sigma_eps^2 comes from T residuals and the
plug-in is consistent only when the local imbalance Omega_A is zero (Appendix H);
a warning is issued.
"""

from __future__ import annotations
import warnings
import numpy as np
from scipy import linalg, stats
from typing import Optional

from .cf_qmle import _cols, _full_rank, _Z, control_from_first_stage, joint_pi_covariance_pooled


def cf_2sls_design(y, X, net, durbin: bool = True, extra=None, first_stages: Optional[dict] = None,
                   controls: Optional[dict] = None, x_names=None, extra_names=None, power: int = 2):
    """Ubar, Qbar and the names of Ubar. power = 2 adds W_j W_l X; power = 3 adds W_j W_l W_m X too."""
    y = np.asarray(y, float).ravel()
    N = len(y)
    X = _cols(X, N)
    E = _cols(extra, N)
    x_names = list(x_names) if x_names is not None else [f"x{i + 1}" for i in range(X.shape[1])]
    e_names = list(extra_names) if extra_names is not None else [f"e{i + 1}" for i in range(E.shape[1])]
    WX = [W @ X for W in net.W]
    cols, cnames, ctrl_names = [], [], []
    for m, fs in (first_stages or {}).items():
        cols.append(control_from_first_stage(fs, net.n, net.T))
        cnames.append("delta_" + m)
        ctrl_names.append(m)
    for m, c in (controls or {}).items():
        cols.append(np.asarray(c, float).ravel())
        cnames.append("delta_" + m)
    C = np.column_stack(cols) if cols else np.zeros((N, 0))
    # regressors
    U = np.column_stack([W @ y for W in net.W])
    Rb = [X] + (WX if durbin else []) + [E, C]
    Rn = list(x_names) + ([f"{nm}*{x}" for nm in net.names for x in x_names] if durbin else []) + e_names + cnames
    R = np.column_stack(Rb)
    keep = _full_rank(R)
    R, Rn = R[:, keep], [Rn[i] for i in keep]
    Ub = np.column_stack([U, R])
    names = list(net.names) + Rn
    # instruments
    inst = [X] + WX
    level = WX
    for _ in range(power - 1):
        level = [W @ A for W in net.W for A in level]
        inst += level
    inst += [E, C]
    Q = np.column_stack(inst)
    Q = Q[:, _full_rank(Q)]
    return y, Ub, Q, names


def cf_2sls(y, X, net, first_stages: Optional[dict] = None, controls: Optional[dict] = None,
            durbin: bool = True, extra=None, x_names=None, extra_names=None, power: int = 2,
            robust: bool = False) -> dict:
    """
    CF-2SLS estimate and variance.

    Parameters as in cf_qmle; power sets the highest power of W in the instruments.
    robust=True replaces sigma2 I in Sigma_e by diag(e_i^2) (heteroskedasticity).

    Returns dict with theta, names, se, V, V_no_generated, sigma2, resid,
    first-stage partial F of each spatial lag on the excluded instruments.
    """
    y, Ub, Q, names = cf_2sls_design(y, X, net, durbin, extra, first_stages, controls, x_names, extra_names, power)
    N, P = Ub.shape
    Qo = linalg.orth(Q)
    PU = Qo @ (Qo.T @ Ub)
    B = linalg.inv(PU.T @ Ub)
    kap = B @ (PU.T @ y)
    e = y - Ub @ kap
    s2 = float(e @ e / (N - P))
    meat = (PU.T * (e ** 2 if robust else s2)) @ PU
    V0 = B @ meat @ B
    V = V0
    gen = {m: fs for m, fs in (first_stages or {}).items() if "delta_" + m in names}
    if gen:
        L = {len(fs["eps_hat"]) for fs in gen.values()}
        if L != {N}:
            warnings.warn("first stage with L = T: the generated-regressor variance uses sigma_eps^2 from T "
                          "residuals and is consistent only when Omega_A = 0 (paper, Appendix H).")
        modes = list(gen)
        Sig = joint_pi_covariance_pooled([gen[m] for m in modes])
        H = []
        for m in modes:
            ci = names.index("delta_" + m)
            ctl = Ub[:, ci]
            eh = np.asarray(gen[m]["eps_hat"], float)
            Z = _Z(gen[m])
            AZ = Z if len(eh) == N else np.repeat(Z, net.n, axis=0)
            AE = ctl if len(eh) == N else ctl[::net.n]
            scale = float(AE @ eh / (eh @ eh))
            H.append(kap[ci] * scale * (PU.T @ AZ))
        H = np.column_stack(H)
        V = B @ (meat + H @ Sig @ H.T) @ B
    # partial F of each spatial lag on the excluded instruments
    Rcols = Ub[:, len(net.names):]
    Ro = linalg.orth(Rcols)
    Qt = Q - Ro @ (Ro.T @ Q)
    Qt = Qt[:, _full_rank(Qt)]
    F = []
    for j in range(len(net.names)):
        u = Ub[:, j] - Ro @ (Ro.T @ Ub[:, j])
        b = linalg.lstsq(Qt, u)[0]
        r = u - Qt @ b
        F.append(float(((u @ u - r @ r) / Qt.shape[1]) / (r @ r / (N - Qt.shape[1] - Ro.shape[1]))))
    se = np.sqrt(np.maximum(np.diag(V), 0))
    z = kap / np.where(se > 0, se, np.nan)
    return {"estimator": "CF-2SLS", "theta": kap, "names": names, "V": V, "V_no_generated": V0, "se": se,
            "z": z, "p": 2 * stats.norm.sf(np.abs(z)), "sigma2": s2, "resid": e, "N": N,
            "partial_F": np.array(F), "n_instruments": int(Q.shape[1])}


def cf_2sls_summary(result: dict) -> str:
    """Stata-style table of a cf_2sls or cf_qmle result."""
    from .report import regression_table
    k = len(result["se"])
    return regression_table(result["names"][:k], result["theta"][:k], result["se"], title=result.get("estimator", ""))
