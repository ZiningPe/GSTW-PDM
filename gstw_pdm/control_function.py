"""
control_function.py
-------------------
Construct the aggregated control function  ε̄̂ = A ε̂  from the first-stage
residuals ε̂.

The aggregation matrix A depends on the dimension of the entry equation:

  L = T  (one entry observation per time period):
      A = I_T ⊗ ι_n  ∈ ℝ^{N×T},   ε̄̂_p = ε̂_{t(p)}
      Each structural observation in period t inherits ε̂_t.

  L = N  (one entry observation per structural observation):
      A = I_N  ∈ ℝ^{N×N},   ε̄̂_p = ε̂_p
      Each structural observation gets its own first-stage residual directly.

The control function ε̄̂ enters the CF-QMLE and CF-2SLS as an additional regressor
(paper, Section 3). At L = T it omits a period constant that does not shrink
with n (Proposition 3); see period_level.within_period_qmle.
"""

from __future__ import annotations
import numpy as np
from typing import Literal


def aggregate_eps_hat(eps_hat: np.ndarray,
                      n: int,
                      T: int,
                      mode: Literal["L_eq_T", "L_eq_N"] = "L_eq_T") -> np.ndarray:
    """
    Form the N-vector ε̄̂ = A ε̂.

    Parameters
    ----------
    eps_hat : first-stage OLS residuals, shape (L,)
    n       : number of spatial units
    T       : number of time periods
    mode    : ``'L_eq_T'``  →  A = I_T ⊗ ι_n  (period-level entry equation)
              ``'L_eq_N'``  →  A = I_N          (unit-level entry equation)

    Returns
    -------
    bar_eps_hat : (N,) control-function vector,  N = n*T
    """
    eps_hat = np.asarray(eps_hat, float).ravel()
    N = n * T

    if mode == "L_eq_T":
        if len(eps_hat) != T:
            raise ValueError(
                f"mode='L_eq_T' expects len(eps_hat)={T} (=T), got {len(eps_hat)}."
            )
        # A = I_T ⊗ ι_n: repeat each ε̂_t for n units  (time-major stacking)
        return np.repeat(eps_hat, n)          # shape (N,)

    elif mode == "L_eq_N":
        if len(eps_hat) != N:
            raise ValueError(
                f"mode='L_eq_N' expects len(eps_hat)={N} (=N=n*T), got {len(eps_hat)}."
            )
        return eps_hat.copy()                  # A = I_N

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'L_eq_T' or 'L_eq_N'.")


def aggregation_matrix(n: int, T: int,
                       mode: Literal["L_eq_T", "L_eq_N"] = "L_eq_T") -> np.ndarray:
    """
    Return the explicit N×L aggregation matrix A.

    Caution: for large N this is an N×N dense matrix.  Prefer
    ``aggregate_eps_hat`` (which never materialises A) in production code.

    Parameters
    ----------
    n, T : spatial units and time periods
    mode : same as in ``aggregate_eps_hat``

    Returns
    -------
    A : (N, L) dense matrix
    """
    N = n * T
    if mode == "L_eq_T":
        # I_T ⊗ ι_n  →  block-diagonal with T blocks, each an n-vector of 1s
        iota_n = np.ones((n, 1))
        return np.kron(np.eye(T), iota_n)     # (N, T)
    elif mode == "L_eq_N":
        return np.eye(N)                       # (N, N)
    else:
        raise ValueError(f"Unknown mode '{mode}'.")


def detect_mode(eps_hat: np.ndarray, n: int, T: int) -> str:
    """
    Auto-detect whether L = T or L = N from the length of eps_hat.

    Returns 'L_eq_T' or 'L_eq_N'.  Raises ValueError if ambiguous.
    """
    L = len(eps_hat)
    if L == T:
        return "L_eq_T"
    elif L == n * T:
        return "L_eq_N"
    else:
        raise ValueError(
            f"Cannot determine mode: len(eps_hat)={L} is neither T={T} nor N=n*T={n * T}."
        )
