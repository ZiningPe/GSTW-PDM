"""
fixed_effects_example.py
------------------------
Unit fixed effects at fixed T (paper, Section 4.3). With a recursive network the
dummy-variable QMLE drifts by O(1/T) (Proposition 2); the adjusted quasi-score
(Theorem 3) removes the drift.

    python examples/fixed_effects_example.py
"""

import numpy as np
from scipy import sparse, linalg
import gstw_pdm as g

rng = np.random.default_rng(1)
n_side, T = 10, 5
n, N = n_side ** 2, n_side ** 2 * T
rook = g.rook_weights(n_side) > 0

# recursive network formed by h, with unit effects in the formation equation as well
a_h = rng.normal(size=n)
Z = rng.normal(size=(N, 2))
eps = rng.normal(size=N)
h = np.tile(a_h, T) + Z @ np.array([0.8, 0.8]) + eps
H = h.reshape(T, n)
K = np.zeros((N, N))
for t in range(T):
    for s in range(t + 1):
        K[t * n:(t + 1) * n, s * n:(s + 1) * n] = rook * np.exp(-np.abs(H[t][:, None] - H[s][None, :]))
W0, W1 = g.block_row_normalise(sparse.csr_matrix(K), n, T)
net = g.RecursiveNetwork([("rho0", W0, False), ("rho1", W1, True)], n, T)

# outcome with unit effects alpha
alpha = rng.normal(size=n)
X = rng.normal(size=(N, 1)) + 0.5 * np.tile(alpha, T)[:, None]
S = np.eye(N) - 0.3 * W0.toarray() - 0.3 * W1.toarray()
y = linalg.solve(S, X[:, 0] + 0.5 * (W0 @ X)[:, 0] + np.tile(alpha, T) + 0.6 * eps + 0.7 * rng.normal(size=N))

# 1. within-unit first stage (instruments first, then year dummies)
D = np.kron(np.eye(T), np.ones((n, 1)))[:, 1:]                 # year dummies, first year dropped
fs = g.first_stage_fe(h, np.column_stack([Z, D]), n, T, n_instruments=2)
print(f"first-stage cluster F = {fs['cluster_F']:.1f}")

# 2. design: spatial lags U, regressors R (the control must be named 'delta_<name>')
U = np.column_stack([W @ y for W in net.W])
R = np.column_stack([X, W0 @ X, W1 @ X, fs["eps_hat"], D])
R_names = ["x", "rho0*x", "rho1*x", "delta_h"] + [f"year{t + 1}" for t in range(1, T)]
wd = g.WithinDesign(y, U, R, n, T)

# 3. dummy-variable QMLE and its drift
dq = g.dummy_qmle(net, wd, [0.0, 0.0])
print("dummy-variable QMLE rho:", np.round(dq["rho"], 3))
print("drift of the rho scores (Prop. 2):", np.round(g.dummy_drift(net, dq["rho"])["drift"], 2))

# 4. adjusted quasi-score with the generated-regressor term and unit-clustered SEs
res = g.aqs_fit(net, wd, R_names, first_stages={"h": fs}, adjust_regressors=True,
                cluster={"unit": np.arange(n)})
k = len(res["se"])
print(g.regression_table(res["names"][:k], res["theta"][:k], res["se"], title="AQS (Theorem 3 SE)",
                         skip=["year"]))
print(g.regression_table(res["names"][:k], res["theta"][:k], res["se_cluster"]["unit"],
                         title="AQS (unit-clustered SE)", skip=["year"]))
