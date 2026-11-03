"""
recursive_network_example.py
----------------------------
Main case of the paper (L = N): a recursive network formed from a unit-period
variable h_it that is correlated with the outcome disturbance.

    W_ts[i, j] = rook_ij * exp(-|h_it - h_js|)   for s <= t,   0 for s > t
    W_0 = same-period blocks, W_1 = earlier-period blocks, each block row-normalised
    h = Z pi + eps,   V = delta_c eps + xi

Without the control the QMLE is biased; the CF-QMLE is not, and its Theorem 2
variance adds the first-stage term. The cross-period effects are reported with
Benjamini-Yekutieli control.
"""

import numpy as np
from scipy import sparse, linalg
import gstw_pdm as g

rng = np.random.default_rng(2026)
n_side, T = 10, 5
n, N = n_side ** 2, n_side ** 2 * T

# formation variable and instruments, one value per unit and period
Z = np.column_stack([np.ones(N), rng.normal(size=(N, 2))])
eps = rng.normal(size=N)
h = Z @ np.array([0.0, 0.8, 0.8]) + eps

# recursive network from h, same- and cross-period blocks normalised separately
rook = g.rook_weights(n_side) > 0
H = h.reshape(T, n)
K = np.zeros((N, N))
for t in range(T):
    for s in range(t + 1):
        K[t * n:(t + 1) * n, s * n:(s + 1) * n] = rook * np.exp(-np.abs(H[t][:, None] - H[s][None, :]))
W0, W1 = g.block_row_normalise(sparse.csr_matrix(K), n, T)
print(g.parameter_space(W0, W1, n, T))

net = g.RecursiveNetwork([("rho0", W0, False), ("rho1", W1, True)], n, T)

# outcome
X = rng.normal(size=(N, 1))
xi = 0.7 * rng.normal(size=N)
S = np.eye(N) - 0.3 * W0.toarray() - 0.3 * W1.toarray()
y = linalg.solve(S, 1.0 * X[:, 0] + 0.5 * (W0 @ X)[:, 0] + 0.2 * (W1 @ X)[:, 0] + 0.6 * eps + xi)

fs = g.first_stage_fit(h, np.column_stack([Z, X]), n_instruments=3)
print(f"first stage F = {fs['F']:.1f}")

naive = g.qmle(y, X, net, extra=np.ones(N), extra_names=["const"])
cf = g.cf_qmle(y, X, net, first_stages={"h": fs}, extra=np.ones(N), extra_names=["const"])
tsls = g.cf_2sls(y, X, net, first_stages={"h": fs}, extra=np.ones(N), extra_names=["const"])

print(g.comparison_table(
    [{"title": "QMLE (no CF)", "names": naive["names"], "coef": naive["theta"], "se": naive["se"]},
     {"title": "CF-QMLE", "names": cf["names"], "coef": cf["theta"], "se": cf["se"]},
     {"title": "CF-2SLS", "names": tsls["names"], "coef": tsls["theta"], "se": tsls["se"]}],
    rows=["rho0", "rho1", "x1", "rho0*x1", "rho1*x1", "delta_h"]))
print("Wald delta_c = 0:", g.wald_delta_c(cf))

ef = g.effects_inference(net, cf, "x1", fdr="by")
print(g.print_ie_table(ef))
print("direct effects by period:", np.round(ef["DE"], 3))
