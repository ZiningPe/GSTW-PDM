"""
period_level_example.py
-----------------------
Period-level formation variable (L = T, Section 5 of the paper).

W_0 = I_T kron W_S, W_1 = M(h) kron W_S with M_ts proportional to exp(-|h_t - h_s|),
and every unit of period t loads on the period shock eps_t. The first stage has
T observations, so the control function never converges: the naive CF-QMLE
lands on a point that depends on the realised shocks (Proposition 3). Projecting
each period on the complement of iota_n removes the shocks exactly (Lemma 4),
and within_period_qmle is consistent for every delta_c (Theorem 4). The
Anderson-Rubin set of Appendix I is exact under Gaussian errors.
"""

import warnings
import numpy as np
from scipy import linalg
import gstw_pdm as g

rng = np.random.default_rng(7)
n_side, T = 12, 6
n, N = n_side ** 2, n_side ** 2 * T
W_S = g.rook_weights(n_side)

Zt = np.column_stack([np.ones(T), rng.normal(size=T)])
eps = rng.normal(size=T)
h = Zt @ np.array([0.0, 1.0]) + eps
M = g.build_twm_from_stats(h, bandwidth=1.0)
net = g.KroneckerNetwork(M, W_S)
W0, W1 = (W.toarray() for W in net.W)

X = rng.normal(size=(N, 1))
S = np.eye(N) - 0.3 * W0 - 0.2 * W1
y = linalg.solve(S, X[:, 0] + 0.5 * W0 @ X[:, 0] + 0.3 * W1 @ X[:, 0] + 1.0 * np.repeat(eps, n)
                 + 0.7 * rng.normal(size=N))

fs = g.first_stage_fit(h, Zt)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    naive = g.cf_qmle(y, X, net, first_stages={"h": fs}, extra=np.ones(N), extra_names=["const"])
within = g.within_period_qmle(y, X, M, W_S)
print(g.comparison_table(
    [{"title": "naive CF-QMLE", "names": naive["names"], "coef": naive["theta"], "se": naive["se"]},
     {"title": "within-period", "names": within["names"], "coef": within["theta"], "se": within["se"]}],
    rows=["rho0", "rho1", "x1", "rho0*x1", "rho1*x1"]))

grid0, grid1 = np.linspace(0.0, 0.6, 31), np.linspace(-0.3, 0.7, 41)
ar = g.ar_confidence_set(y, X, M, W_S, grid0, grid1)
ii = np.argwhere(ar["in_set"])
if len(ii):
    print(f"95% AR set: rho0 in [{grid0[ii[:, 0].min()]:.2f}, {grid0[ii[:, 0].max()]:.2f}], "
          f"rho1 in [{grid1[ii[:, 1].min()]:.2f}, {grid1[ii[:, 1].max()]:.2f}]")
