"""
quickstart.py
-------------
Step-by-step walk-through of the main case (L = N) on simulated data.
Every step mirrors the README tutorial; replace the simulated arrays with your own data.

    python examples/quickstart.py
"""

import numpy as np
from scipy import sparse, linalg
import gstw_pdm as g

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Step 0. Data in long format: one row per (unit, period)
# ---------------------------------------------------------------------------
n, T = 100, 5                      # units and periods
N = n * T
coords = rng.uniform(0, 10, size=(n, 2))          # unit locations
unit = np.tile(np.arange(n), T)                    # rows already stacked period by period
year = np.repeat(np.arange(T), n)
z1, z2 = rng.normal(size=N), rng.normal(size=N)    # instruments for the formation variable
eps = rng.normal(size=N)                           # formation shock (unobserved)
h = 0.8 * z1 + 0.8 * z2 + eps                      # formation variable, e.g. accessibility
x = rng.normal(size=N)                             # a regressor

# ---------------------------------------------------------------------------
# Step 1. Time-major order: sort by period, then by unit (row p = t*n + i)
# ---------------------------------------------------------------------------
order = np.lexsort((unit, year))
assert np.all(order == np.arange(N)), "sort your data by (period, unit) first"

# ---------------------------------------------------------------------------
# Step 2. Build the weight matrix: unit i in period t is linked to unit j in period s <= t
# ---------------------------------------------------------------------------
dist = np.linalg.norm(coords[:, None] - coords[None], axis=2)
near = (dist < 1.5) & ~np.eye(n, dtype=bool)        # spatial neighbours
H = h.reshape(T, n)
K = np.zeros((N, N))
for t in range(T):
    for s in range(t + 1):                          # only the past and the present transmit
        K[t * n:(t + 1) * n, s * n:(s + 1) * n] = near * np.exp(-np.abs(H[t][:, None] - H[s][None, :]))
W0, W1 = g.block_row_normalise(sparse.csr_matrix(K), n, T)   # same-period / cross-period blocks
print(g.parameter_space(W0, W1, n, T))

# ---------------------------------------------------------------------------
# Step 3. The network object
# ---------------------------------------------------------------------------
net = g.RecursiveNetwork([("rho0", W0, False), ("rho1", W1, True)], n, T)

# simulate the outcome (skip this with real data)
X = x[:, None]
S = np.eye(N) - 0.3 * W0.toarray() - 0.3 * W1.toarray()
y = linalg.solve(S, X[:, 0] + 0.5 * (W0 @ X)[:, 0] + 0.2 * (W1 @ X)[:, 0] + 0.6 * eps + 0.7 * rng.normal(size=N))

# ---------------------------------------------------------------------------
# Step 4. First stage of the formation variable
# ---------------------------------------------------------------------------
Zw = np.column_stack([z1, z2, np.ones(N), x])       # excluded instruments first, then included regressors
fs = g.first_stage_fit(h, Zw, n_instruments=2)
print(f"first-stage F = {fs['F']:.1f}")

# ---------------------------------------------------------------------------
# Step 5. Estimate
# ---------------------------------------------------------------------------
fit = g.cf_qmle(y, X, net, first_stages={"h": fs}, extra=np.ones(N),
                x_names=["x"], extra_names=["const"])
print(g.regression_table(fit["names"], fit["theta"], fit["se"], title="CF-QMLE", depvar="y"))

# ---------------------------------------------------------------------------
# Step 6. Tests
# ---------------------------------------------------------------------------
tests = {
    "delta_h = 0 (weights exogenous)": g.wald_delta_c(fit),
    "rho1 = rho1*x = 0 (no cross-period)":
        g.wald_test(fit["theta"], fit["V"], g.restriction_matrix(fit["names"], ["rho1", "rho1*x"])),
    "rho0 = rho1 (same = cross)":
        g.wald_test(fit["theta"], fit["V"], g.restriction_matrix(fit["names"], [("rho0", "rho1")])),
}
print(g.tests_block(tests))

# ---------------------------------------------------------------------------
# Step 7. Spillover effects of x, with Benjamini-Yekutieli control
# ---------------------------------------------------------------------------
eff = g.effects_inference(net, fit, "x", fdr="by")
print(g.print_ie_table(eff, period_labels=[2018, 2019, 2020, 2021, 2022]))
print("direct effect by period:   ", np.round(eff["DE"], 3))
print("within-period indirect:    ", np.round(eff["IEw"], 3))

# ---------------------------------------------------------------------------
# Step 8. Robustness: CF-2SLS and the QMLE without the control
# ---------------------------------------------------------------------------
tsls = g.cf_2sls(y, X, net, first_stages={"h": fs}, extra=np.ones(N), x_names=["x"], extra_names=["const"])
naive = g.qmle(y, X, net, extra=np.ones(N), x_names=["x"], extra_names=["const"])
cols = [{"title": t, "names": r["names"], "coef": r["theta"], "se": r["se"]}
        for t, r in [("QMLE, no CF", naive), ("CF-QMLE", fit), ("CF-2SLS", tsls)]]
print(g.comparison_table(cols, rows=["rho0", "rho1", "x", "rho0*x", "rho1*x", "delta_h"]))
