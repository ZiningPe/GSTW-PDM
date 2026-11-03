"""
first_stage.py
--------------
OLS first-stage estimation of the entry equation

    h = Z_W π + ε

where h is the variable that forms the weights: one value per unit and
period (L = N, e.g. a city's accessibility in year t; the main case of the
paper) or one value per period (L = T, e.g. Moran's I in period t). Z_W are
the instruments and ε is the formation shock (paper, eq. (3.1)).

The residuals ε̂ are subsequently passed to control_function.py to form
the control function ε̄̂ = A ε̂.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg
from typing import Optional


def first_stage(h: np.ndarray,
                Z_W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    OLS first stage:  h = Z_W π + ε.

    Parameters
    ----------
    h   : (L,) vector of entry-equation outcomes (one per time period if L=T,
          or one per observation if L=N)
    Z_W : (L, k_Z) matrix of excluded instruments

    Returns
    -------
    eps_hat : (L,) OLS residuals  ε̂ = M_{Z_W} h  (where M_{Z_W} = I − P_{Z_W})
    pi_hat  : (k_Z,) OLS coefficient estimates
    """
    h   = np.asarray(h, float).ravel()
    Z_W = np.asarray(Z_W, float)
    if Z_W.ndim == 1:
        Z_W = Z_W[:, None]

    pi_hat  = linalg.lstsq(Z_W, h, cond=None)[0]
    eps_hat = h - Z_W @ pi_hat
    return eps_hat, pi_hat


def first_stage_stats(h: np.ndarray,
                      Z_W: np.ndarray) -> dict:
    """
    Extended first-stage output with F-statistic and R².

    Returns
    -------
    dict with keys:
        eps_hat, pi_hat, fitted, sigma2,
        R2, F_stat, F_pval, n_obs, n_params
    """
    from scipy import stats as scipy_stats

    h   = np.asarray(h, float).ravel()
    Z_W = np.asarray(Z_W, float)
    if Z_W.ndim == 1:
        Z_W = Z_W[:, None]

    L, k = Z_W.shape

    # R², F, and σ² require a model with an intercept.
    # If Z_W has no constant column, augment it so the formulas are valid.
    _col_const = np.all(np.abs(np.diff(Z_W, axis=0)) < 1e-10, axis=0)
    if not _col_const.any():
        Z_W_aug = np.column_stack([np.ones(L), Z_W])
    else:
        Z_W_aug = Z_W

    eps_hat, pi_hat = first_stage(h, Z_W_aug)
    fitted  = h - eps_hat
    k_aug   = Z_W_aug.shape[1]
    sigma2  = float(eps_hat @ eps_hat) / max(L - k_aug, 1)

    # R²  (valid because the model now contains an intercept)
    h_mean = h.mean()
    SS_tot = float((h - h_mean) @ (h - h_mean))
    SS_res = float(eps_hat @ eps_hat)
    R2 = max(0.0, 1.0 - SS_res / SS_tot) if SS_tot > 1e-15 else 0.0

    # F-statistic  (H₀: all slope coefficients = 0, excl. intercept)
    df_reg = k_aug - 1    # number of slope coefficients
    df_res = L - k_aug
    if df_res > 0 and df_reg > 0:
        SS_reg = SS_tot - SS_res
        F_stat = (SS_reg / df_reg) / (SS_res / df_res)
        F_pval = float(scipy_stats.f.sf(F_stat, df_reg, df_res))
    else:
        F_stat = np.nan
        F_pval = np.nan

    # Return eps_hat from the original Z_W (without added intercept)
    # for downstream control-function use, if intercept was added here
    if not _col_const.any():
        eps_hat_cf, pi_hat_cf = first_stage(h, Z_W)
    else:
        eps_hat_cf, pi_hat_cf = eps_hat, pi_hat

    return {
        "eps_hat" : eps_hat_cf,   # residuals from original Z_W (for CF use)
        "pi_hat"  : pi_hat_cf,
        "fitted"  : fitted,
        "sigma2"  : sigma2,
        "R2"      : round(R2, 6),
        "F_stat"  : round(float(F_stat), 4) if not np.isnan(F_stat) else np.nan,
        "F_pval"  : round(F_pval, 6) if not np.isnan(F_pval) else np.nan,
        "n_obs"   : L,
        "n_params": k,
    }


def projection_matrix(Z_W: np.ndarray) -> np.ndarray:
    """
    Orthogonal projection matrix P_{Z_W} = Z_W (Z_W'Z_W)^{-1} Z_W'.

    Used to form the projected residuals P_{Z_W} ε̂ needed in some
    variance-correction formulas.

    Parameters
    ----------
    Z_W : (L, k) instrument matrix

    Returns
    -------
    P : (L, L) projection matrix
    """
    Z_W = np.asarray(Z_W, float)
    if Z_W.ndim == 1:
        Z_W = Z_W[:, None]
    ZtZ_inv = linalg.pinv(Z_W.T @ Z_W)
    return Z_W @ ZtZ_inv @ Z_W.T


def first_stage_fit(h: np.ndarray, Z_W: np.ndarray, n_instruments: Optional[int] = None) -> dict:
    """
    First stage h = Z_W pi + eps (paper, eq. (3.1)) in the form the estimators use.

    Z_W holds the excluded instruments first and then any included regressors
    (intercept, X, period dummies). Returns eps_hat, Z (rank-checked), pi, sigma2
    (sigma_eps^2 with L - k_z degrees of freedom), ZtZ_inv, V_iid, dof, R2, and, with
    n_instruments, the F statistic of the first n_instruments columns.
    Pass the result in first_stages={'name': ...} to cf_qmle or cf_2sls.
    """
    from scipy import stats as scipy_stats
    h = np.asarray(h, float).ravel()
    Z = np.asarray(Z_W, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    q, r, piv = linalg.qr(Z, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    keep = np.sort(piv[d > 1e-9 * d.max()])
    Z = Z[:, keep]
    L, k = Z.shape
    inv = linalg.inv(Z.T @ Z)
    pi = inv @ (Z.T @ h)
    e = h - Z @ pi
    s2 = float(e @ e / (L - k))
    hc = h - h.mean()
    out = {"eps_hat": e, "Z": Z, "keep": keep, "pi": pi, "sigma2": s2, "ZtZ_inv": inv,
           "V_iid": s2 * inv, "dof": L - k, "L": L, "R2": 1 - float(e @ e) / float(hc @ hc)}
    if n_instruments:
        kz = int(np.sum(keep < n_instruments))
        b = pi[:kz]
        F = float(b @ linalg.solve(out["V_iid"][:kz, :kz], b) / kz)
        out.update({"F": F, "F_p": float(scipy_stats.f.sf(F, kz, L - k)), "n_instruments": kz})
    return out
