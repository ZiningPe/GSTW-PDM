# GSTW-PDM

**Generalized Spatio-Temporally Weighted Panel Data Model**

GSTW-PDM is a Python package for **spatial panel models in which units affect
each other both within a period and across periods**, and in which the links
themselves may be **endogenous**. For example, a city's patenting may respond to
neighbouring cities' patenting this year and last year, through a travel-time
network that is itself shaped by local economic shocks.

It implements the estimators and inference of Wang, *Generalized
Spatio-Temporally Weighted Panel Data Model* (working paper):

- **CF-QMLE** and **CF-2SLS**, which correct endogenous weights with a control function
- **unit fixed effects** at fixed T through an adjusted quasi-score
- a **period-level** estimator with exact Anderson–Rubin confidence sets
- **direct, within-period and cross-period spillover effects** with standard
  errors and false-discovery-rate control

You don't need to read the paper to use the package. This page walks through
everything step by step.

---

## Contents

1. [Installation](#1-installation)
2. [The model in one minute](#2-the-model-in-one-minute)
3. [Preparing your data](#3-preparing-your-data)
4. [Which estimator should I use?](#4-which-estimator-should-i-use)
5. [Tutorial A: endogenous network, one formation value per unit and period (L = N)](#5-tutorial-a-endogenous-network-l--n)
6. [Tutorial B: unit fixed effects](#6-tutorial-b-unit-fixed-effects)
7. [Tutorial C: one formation value per period (L = T)](#7-tutorial-c-period-level-formation-variable-l--t)
8. [Reading the results](#8-reading-the-results)
9. [Spillover effects](#9-spillover-effects)
10. [Function reference](#10-function-reference)
11. [FAQ and troubleshooting](#11-faq-and-troubleshooting)
12. [Where each result is in the paper](#12-where-each-result-is-in-the-paper)
13. [Citation](#13-citation)

---

## 1. Installation

```bash
pip install git+https://github.com/ZiningPe/GSTW-PDM.git
```

To work on the code itself:

```bash
git clone https://github.com/ZiningPe/GSTW-PDM.git
cd GSTW-PDM
pip install -e ".[dev]"
pytest tests/          # 29 tests, a few seconds
```

Requirements: Python ≥ 3.9, NumPy ≥ 1.24, SciPy ≥ 1.10. Nothing else.

Check the installation:

```python
import gstw_pdm as g
print(g.__version__)      # 0.3.0
```

Three complete scripts run out of the box and are a good starting point:

| Script | What it shows |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | Tutorial A from start to finish (start here) |
| [`examples/fixed_effects_example.py`](examples/fixed_effects_example.py) | Tutorial B: unit fixed effects |
| [`examples/period_level_example.py`](examples/period_level_example.py) | Tutorial C: period-level formation variable |

```bash
python examples/quickstart.py
```

---

## 2. The model in one minute

```
Y = ρ0·W0·Y + ρ1·W1·Y + X·β + W0·X·θ0 + W1·X·θ1 + V
```

| Symbol | Meaning |
|---|---|
| `n`, `T`, `N = n·T` | number of units, number of periods, number of observations |
| `W0` | **same-period** links: unit i in year t ← unit j in year t |
| `W1` | **cross-period** links: unit i in year t ← unit j in an earlier year s |
| `ρ0`, `ρ1` | how strongly outcomes respond to linked outcomes in the same / other periods |
| `θ0`, `θ1` | the same for linked regressors (spatial Durbin terms) |
| `ρ1 = θ1 = 0` | the standard static spatial Durbin panel |

**Endogenous weights.** The weights are built from a *formation variable* `h`,
such as a city's rail accessibility, through a first stage

```
h = Z_W·π + ε,        V = δc·A·ε + ξ
```

If the shock `ε` that moves the links also moves the outcome (`δc ≠ 0`), the
weights are endogenous and ordinary estimators are biased. The package adds
the first-stage residual as a **control function** and corrects the standard
errors for the fact that it is estimated.

`L` is the number of first-stage observations:

- **L = N**: one value `h_it` per unit and period, so `A = I`. This is the
  main case and works for any `n`.
- **L = T**: one value `h_t` per period, so every unit in period t shares
  `ε_t`. The control function never becomes precise as `n` grows, so a
  different estimator is needed (Tutorial C).

---

## 3. Preparing your data

### 3.1 Row order: period by period ("time-major")

All vectors and matrices stack the panel **period by period, units within a
period**. Row `p = t·n + i` (0-based) is unit `i` in period `t`:

```
y = [y(unit 1, year 1), ..., y(unit n, year 1), y(unit 1, year 2), ..., y(unit n, year T)]
```

With a pandas DataFrame in long format (pandas is not a dependency):

```python
df = df.sort_values(["year", "city"])       # period first, then unit
n, T = df["city"].nunique(), df["year"].nunique()
assert len(df) == n * T                      # balanced panel required
y = df["patents"].to_numpy()
X = df[["gdp", "pop"]].to_numpy()
```

If your data are stacked unit by unit, `g.to_time_major(v, n, T)` converts them.

### 3.2 The weight matrix

You need an `N × N` matrix whose block `(t, s)` gives the links from units in
period `s` to units in period `t`. Then split it into the same-period and
cross-period parts, and row-normalise each part **separately** (the paper's
Assumption 3: normalising the whole row would shrink the same-period share as
past periods accumulate):

```python
from scipy import sparse
W0, W1 = g.block_row_normalise(sparse.csr_matrix(K), n, T)
print(g.parameter_space(W0, W1, n, T))       # checks the assumptions
```

`parameter_space` reports whether the matrix is **recursive** (only the past
and present transmit, nothing from later periods), the row and column sums,
and the bounds on `ρ`. With recursive weights `ρ1` is unrestricted and only
`|ρ0| < bound_rho0_recursive` is needed (Lemma 2).

If you have one spatial matrix `W_S` (n × n) and one temporal matrix `M`
(T × T, zero diagonal), the Kronecker form is ready-made:

```python
W_S = g.rook_weights(10)                          # or knn_weights, inverse_distance_weights, ...
M = g.build_twm_parametric(T, rho=0.5)            # M_ts ∝ 0.5^|t-s|
# or from per-period statistics:  M = g.build_twm_from_stats(h_t, bandwidth=1.0)
W0, W1 = g.kronecker_weights(M, W_S)              # I_T ⊗ W_S and M ⊗ W_S
```

### 3.3 The network object

Every estimator takes a *network object*. It holds the lag matrices and
computes the log-determinant efficiently:

| Your weights | Network object | Speed |
|---|---|---|
| Recursive (no links from later periods) | `g.RecursiveNetwork([("rho0", W0, False), ("rho1", W1, True)], n, T)` | fast: only n × n blocks |
| Kronecker `I⊗W_S`, `M⊗W_S` | `g.KroneckerNetwork(M, W_S)` | fast: eigenvalues |
| Anything else | `g.DenseNetwork([("rho0", W0), ("rho1", W1)], n, T)` | one N × N factorisation per step |
| Not sure | `g.make_network([("rho0", W0), ("rho1", W1)], n, T)` | picks recursive when it applies |

The third element `True` in `RecursiveNetwork` marks a cross-period matrix.
The names you give (`"rho0"`, `"rho1"`) become the coefficient names. You can
pass more than two lag matrices, for example rail and road networks, each split
into same- and cross-period parts:

```python
net = g.RecursiveNetwork([("rail0", R0, False), ("rail1", R1, True),
                          ("road0", D0, False), ("road1", D1, True)], n, T)
```

---

## 4. Which estimator should I use?

```
Is the formation variable observed once per unit and period (h_it)?
│
├─ yes (L = N) ── Do you need unit (city) fixed effects?
│                 ├─ no  → cf_qmle            (Tutorial A)  + cf_2sls as a check
│                 └─ yes → aqs_fit            (Tutorial B)  + cf_2sls_fe as a check
│
└─ no, once per period (h_t, L = T) → within_period_qmle  (Tutorial C)
                                        + anderson_rubin for robust confidence sets
```

| Function | Use when | What it does |
|---|---|---|
| `cf_qmle` | L = N, no unit effects | QMLE with the control function; Theorem 2 standard errors |
| `cf_2sls` | robustness check | instrumental variables, only linear moments |
| `qmle` | weights exogenous | QMLE without a control function |
| `aqs_fit` | L = N with unit fixed effects | removes the O(1/T) bias of the dummy-variable QMLE |
| `within_period_qmle` | L = T | removes the period shocks exactly; does not need the first stage |
| `anderson_rubin` | L = T, weak identification | exact test of (ρ0, ρ1) under normal errors |

**Do not** use `cf_qmle` with a period-level first stage. It is then
inconsistent whenever the weights are endogenous (Proposition 3), and the
package warns you.

---

## 5. Tutorial A: endogenous network (L = N)

The full script is [`examples/quickstart.py`](examples/quickstart.py). Here are the key steps.

**Step 1: build the network.** Here unit `i` in year `t` is linked to its
spatial neighbours `j` in every year `s ≤ t`, with strength
`exp(-|h_it - h_js|)`:

```python
import numpy as np
from scipy import sparse
import gstw_pdm as g

H = h.reshape(T, n)
K = np.zeros((N, N))
for t in range(T):
    for s in range(t + 1):
        K[t*n:(t+1)*n, s*n:(s+1)*n] = near * np.exp(-np.abs(H[t][:, None] - H[s][None, :]))
W0, W1 = g.block_row_normalise(sparse.csr_matrix(K), n, T)
net = g.RecursiveNetwork([("rho0", W0, False), ("rho1", W1, True)], n, T)
```

**Step 2: first stage.** Put the excluded instruments first, then the
included regressors, and say how many are excluded:

```python
Zw = np.column_stack([z1, z2, np.ones(N), x])
fs = g.first_stage_fit(h, Zw, n_instruments=2)
print(fs["F"])                 # first-stage F of the excluded instruments
```

**Step 3: estimate.**

```python
X = x[:, None]                                   # (N, K) regressors
fit = g.cf_qmle(y, X, net, first_stages={"h": fs},
                extra=np.ones(N),                # regressors without spatial lags
                x_names=["x"], extra_names=["const"])
print(g.regression_table(fit["names"], fit["theta"], fit["se"], title="CF-QMLE"))
```

```
CF-QMLE
----------------------------------------------------------------------------------
             y |      Coef.   Std. Err.       z    P>|z|     [95% Conf. Interval]
---------------+----------------------------------------------------------------------
          rho0 |     0.2287     0.0388     5.90    0.000      0.1527      0.3047  ***
          rho1 |     0.3176     0.0524     6.06    0.000      0.2149      0.4202  ***
             x |     1.0153     0.0437    23.21    0.000      0.9296      1.1010  ***
        rho0*x |     0.5978     0.0750     7.97    0.000      0.4508      0.7449  ***
        rho1*x |     0.1500     0.1054     1.42    0.155     -0.0565      0.3566
         const |    -0.0302     0.0443    -0.68    0.495     -0.1170      0.0566
       delta_h |     0.6317     0.0318    19.85    0.000      0.5694      0.6941  ***
        sigma2 |     0.4795     0.0282    16.98    0.000      0.4242      0.5349  ***
----------------------------------------------------------------------------------
```

- `rho0`, `rho1`: same-period and cross-period transmission
- `x`: own effect β. `rho0*x` and `rho1*x` are θ0 and θ1, the effects of
  neighbours' `x` in the same and earlier periods
- `delta_h` is δc. If it is significant, the weights are endogenous and the
  correction matters.

**Step 4: tests.** `restriction_matrix` takes coefficient names (tests
`= 0`) or pairs of names (tests equality):

```python
tests = {
    "delta_h = 0 (weights exogenous)": g.wald_delta_c(fit),
    "rho1 = rho1*x = 0 (no cross-period)":
        g.wald_test(fit["theta"], fit["V"], g.restriction_matrix(fit["names"], ["rho1", "rho1*x"])),
    "rho0 = rho1 (same = cross)":
        g.wald_test(fit["theta"], fit["V"], g.restriction_matrix(fit["names"], [("rho0", "rho1")])),
}
print(g.tests_block(tests))
```

**Step 5: compare estimators side by side.**

```python
tsls  = g.cf_2sls(y, X, net, first_stages={"h": fs}, extra=np.ones(N), x_names=["x"], extra_names=["const"])
naive = g.qmle(y, X, net, extra=np.ones(N), x_names=["x"], extra_names=["const"])
cols = [{"title": t, "names": r["names"], "coef": r["theta"], "se": r["se"]}
        for t, r in [("QMLE, no CF", naive), ("CF-QMLE", fit), ("CF-2SLS", tsls)]]
print(g.comparison_table(cols, rows=["rho0", "rho1", "x", "rho0*x", "rho1*x", "delta_h"]))
```

```
                               QMLE, no CF          CF-QMLE          CF-2SLS
rho0                             0.3477***        0.2287***        0.3559***
rho1                             0.5623***        0.3176***         0.2331**
rho1*x                          -0.4041***           0.1500           0.2286
```

The QMLE without the control function overstates cross-period transmission
(the true `ρ1` is 0.3 in this simulation).

**Several formation variables.** Pass one first stage per network, for
example `first_stages={"rail": fs_rail, "road": fs_road}`. Each adds a column
`delta_rail`, `delta_road`, and the variance accounts for the correlation
between the first stages.

---

## 6. Tutorial B: unit fixed effects

Full script: [`examples/fixed_effects_example.py`](examples/fixed_effects_example.py).

With unit effects and a recursive network, subtracting each unit's time mean
does **not** remove the effects (Lemma 3). City dummies leave a bias of order
1/T in `ρ` that does not shrink with `n` (Proposition 2). `aqs_fit` corrects
it.

```python
# 1. first stage within units: excluded instruments first, then year dummies
D  = np.kron(np.eye(T), np.ones((n, 1)))[:, 1:]
fs = g.first_stage_fe(h, np.column_stack([Z, D]), n, T, n_instruments=2)

# 2. spatial lags U and regressors R; the control column must be named "delta_<name>"
U = np.column_stack([W @ y for W in net.W])
R = np.column_stack([X, W0 @ X, W1 @ X, fs["eps_hat"], D])
R_names = ["x", "rho0*x", "rho1*x", "delta_h"] + [f"year{t+1}" for t in range(1, T)]
wd = g.WithinDesign(y, U, R, n, T)

# 3. how large is the dummy-variable bias?
dq = g.dummy_qmle(net, wd, [0.0, 0.0])
print(g.dummy_drift(net, dq["rho"])["drift"])    # zero only for block-diagonal weights

# 4. adjusted quasi-score
res = g.aqs_fit(net, wd, R_names, first_stages={"h": fs},
                adjust_regressors=True,               # recommended when R has many dummies
                cluster={"unit": np.arange(n)})       # optional cluster-robust SE
k = len(res["se"])
print(g.regression_table(res["names"][:k], res["theta"][:k], res["se"], skip=["year"]))
print(g.regression_table(res["names"][:k], res["theta"][:k], res["se_cluster"]["unit"], skip=["year"]))
```

- `res["se"]`: the Theorem 3 standard errors (independent errors)
- `res["se_cluster"]["unit"]`: allows any correlation of a unit's errors over
  time. Use it when residuals are serially correlated, which is common in
  annual city data. Theorem 3 itself assumes independent errors.
- If your lags use weights on years before the estimation sample (a
  predetermined history), compute `U` on the full history and pass only the
  estimation rows. The variance accounts for it.
- `g.cf_2sls_fe(wd, Z_excl, R_names, net.names)` is the instrumental-variables
  comparison estimator, with unit-clustered standard errors.

---

## 7. Tutorial C: period-level formation variable (L = T)

Full script: [`examples/period_level_example.py`](examples/period_level_example.py).

Here the temporal weights come from one number per period (for example
`M_ts ∝ exp(-|h_t - h_s|)`), and every unit in period `t` shares the shock `ε_t`.

```python
M   = g.build_twm_from_stats(h_t, bandwidth=1.0)
fit = g.within_period_qmle(y, X, M, W_S)                 # needs a row-stochastic W_S
print(g.regression_table(fit["names"], fit["theta"], fit["se"], title="within-period QMLE"))

# exact confidence set for (rho0, rho1), valid even under weak identification
grid0, grid1 = np.linspace(0.0, 0.6, 31), np.linspace(-0.3, 0.7, 41)
ar = g.ar_confidence_set(y, X, M, W_S, grid0, grid1)     # ar["in_set"] marks the 95% set
print(g.anderson_rubin(y, X, M, W_S, r=(0.3, 0.2)))      # test one value
```

Each period is projected on the part orthogonal to the period mean, which
removes the shared shocks exactly. As a consequence:

- you do **not** pass a first stage; the estimator is the same whatever `δc` is
- the intercept and anything constant within a period (period dummies,
  national variables) are absorbed and drop out of the table
- `g.wald_delta_c(fit)` on a `cf_qmle` fit still tests `δc = 0`, valid when no
  other period effects are present (Appendix F.2)

---

## 8. Reading the results

Every estimator returns a dictionary:

| Key | Content |
|---|---|
| `names` | coefficient names: `rho...`, regressors, Durbin terms `"<lag>*<x>"`, extra regressors, `delta_<name>`, `sigma2` |
| `theta` | estimates, in the order of `names` |
| `se`, `V` | standard errors and covariance matrix used for inference |
| `se_information`, `V_information` | (`cf_qmle`) information-only version, without the first-stage correction |
| `V_no_generated` | (`cf_qmle`, `cf_2sls`) robust to non-normal errors but without the first-stage term |
| `se_cluster`, `V_cluster` | (`aqs_fit`) cluster-robust versions, one per entry of `cluster=` |
| `loglik`, `sigma2`, `resid`, `mu3`, `mu4` | log-likelihood, error variance, residuals, residual third and fourth moments |
| `partial_F`, `n_instruments` | (`cf_2sls`) instrument strength for each spatial lag |

Why `se` and `se_information` differ: the control function is estimated. When
the first-stage instruments are correlated with the regressors (for example,
when `X` enters the first stage), information-only intervals are too narrow.
In a simulation with `X` in the first stage they covered about 82% instead of
95%. Always report `se`.

Tables: `regression_table` (one model, Stata layout), `comparison_table` (several
models, esttab layout), `tests_block` (Wald tests), `horizon_table` (impulse responses).

---

## 9. Spillover effects

The coefficients are not the full effect of `x`, because effects travel
through the network. `effects_inference` turns them into interpretable
quantities, with delta-method standard errors:

```python
eff = g.effects_inference(net, fit, "x", fdr="by")
print(g.print_ie_table(eff, period_labels=[2018, 2019, 2020, 2021, 2022]))
eff["DE"], eff["SE_DE"]       # direct effect of a unit's own x, by period
eff["IEw"], eff["SE_IEw"]     # within-period indirect effect (on the other units)
eff["IE"], eff["SE"]          # T × T: IE[t, s] = average effect on period t of x in period s
eff["p_adjusted"], eff["reject"]
```

```
Cross-period effects IE(t<-s), SE in parentheses, * = rejected at FDR (by)
t\s                2018             2019             2020             2021             2022
2018      2.091(0.088)     0.000(0.000)     0.000(0.000)     0.000(0.000)     0.000(0.000)
2019      1.056(0.101)*    2.091(0.088)     0.000(0.000)     0.000(0.000)     0.000(0.000)
2020      0.743(0.089)*    0.532(0.051)*    2.091(0.088)     0.000(0.000)     0.000(0.000)
```

- Off-diagonal cell `(t, s)`: raising `x` by one in every unit in period `s`
  changes the period-`t` outcome by this much on average. With recursive
  weights, effects of later periods on earlier ones are zero and are not tested.
- Diagonal cell: the total same-period effect, `DE + IEw`.
- There are many cross-period effects, so the p-values are adjusted for
  multiple testing. `fdr="by"` (Benjamini–Yekutieli, the default) is valid
  under any dependence. `fdr="bh"` (Benjamini–Hochberg) is less conservative.
- To use cluster-robust standard errors: `g.effects_inference(net, res, "x", V=res["V_cluster"]["unit"])`.

`fdr_correction(p, alpha, "bh" | "by")` works on any array of p-values.

---

## 10. Function reference

Call any function as `g.<name>` after `import gstw_pdm as g`; each has a
docstring (`help(g.cf_qmle)`).

**Weights**
| Function | Purpose |
|---|---|
| `rook_weights`, `queen_weights`, `knn_weights`, `inverse_distance_weights` | spatial matrices W_S |
| `compute_morans_i`, `compute_gearys_c`, `compute_getis_ord_g`, `compute_spatial_gini` | per-period spatial statistics |
| `build_twm_from_stats(stats, bandwidth)` | temporal matrix M from per-period statistics |
| `build_twm_parametric(T, rho, form)` | M with exponential / power / linear decay |
| `kronecker_weights(M, W_S)` | `(W0, W1) = (I_T⊗W_S, M⊗W_S)` |
| `block_row_normalise(W, n, T)` | split and normalise same- and cross-period blocks |
| `split_same_cross(W, n, T)` | split without normalising |
| `parameter_space(W0, W1, n, T)` | check the assumptions, bounds on ρ |
| `stwm_summary`, `eigvals_kronecker`, `logdet_kronecker`, `check_admissibility` | diagnostics |

**Networks**: `RecursiveNetwork`, `KroneckerNetwork`, `DenseNetwork`, `TiedNetwork` (equal coefficients across lags), `make_network`

**First stage**: `first_stage_fit` (pooled), `first_stage_fe` (within units), `aggregate_eps_hat`, `first_stage`, `first_stage_stats`

**Estimators**
| Function | Section |
|---|---|
| `cf_qmle`, `qmle` | Tutorial A |
| `cf_2sls` | Tutorial A |
| `WithinDesign`, `dummy_qmle`, `dummy_drift`, `aqs_fit`, `aqs_solve`, `aqs_variance`, `cf_2sls_fe` | Tutorial B |
| `within_period_qmle`, `anderson_rubin`, `ar_confidence_set` | Tutorial C |

**Inference**: `wald_test`, `restriction_matrix`, `wald_delta_c`, `effects_inference`, `spillover_effects`, `fdr_correction`, `bh_correction`, `by_correction`

**Output**: `regression_table`, `comparison_table`, `tests_block`, `horizon_table`, `print_ie_table`, `stars`

**Utilities**: `to_time_major`, `to_unit_major`, `panel_demean`, `city_demean`, `ned_test`, `matrix_condition`, `check_instrument_strength`

---

## 11. FAQ and troubleshooting

**How large can my panel be?** The estimates themselves scale well with
`RecursiveNetwork` and `KroneckerNetwork`. The variance needs dense
`N × N` matrices (`G_j = W_j S⁻¹`), about `8·N²` bytes each, one per lag
matrix: N = 5,000 needs about 200 MB per lag. Panels of a few thousand
observations run in seconds to minutes. Pass `variance=False` to
`cf_qmle` for quick point estimates.

**My panel is unbalanced.** The package assumes a balanced panel (`N = n·T`).
Restrict the sample to units observed in every period.

**The rows of my weight matrix sum to 0 for the first year.** That is
expected. In a recursive network the first period has no earlier periods, so
the first block row of `W1` is zero.

**`aqs_fit` raises "AQS did not converge".** The adjusted score can have no
root near the data, typically when the estimate is near the boundary of the
parameter space (`multiplier_row_sums()` of the network becomes very large).
Check `parameter_space`, try another starting value (`rho0=`), or use a
less dense network. Don't run a non-converging specification inside a loop.

**Warning "first stage with L = T".** You passed a first stage with `T` rows to
`cf_qmle`. See Tutorial C.

**A column disappeared from the table.** Columns of `R` that are collinear with
earlier ones are dropped automatically, for example `W0 @ const` when `W0` is
row-normalised (it equals the intercept). In `within_period_qmle` everything
constant within a period drops.

**`RuntimeWarning: divide by zero encountered in matmul` on macOS.** This is a
known spurious warning of NumPy with Apple's Accelerate library. The results are
unaffected.

**My residuals are serially correlated.** The theorems assume independent
errors. Serial correlation also biases the AQS adjustment itself, not only the
standard errors. Cluster-robust standard errors (`cluster=`) fix the inference
part. Adding the lagged outcome as a regressor is often the better
specification.

**Which version of the paper does this match?** Package 0.3.0 follows the
paper with separate same- and cross-period coefficients. The 0.1.0 code (one
`δ`, CF-GMM) is at commit `61eba57`.

---

## 12. Where each result is in the paper

| Paper | Package |
|---|---|
| Model (2.1), effects (2.2)–(2.3) | `cf_qmle`, `spillover_effects` |
| Kronecker log-determinant (2.5) | `KroneckerNetwork`, `logdet_kronecker` |
| Lemma 2, recursive log-determinant (2.6) | `RecursiveNetwork`, `parameter_space` |
| Assumption 3 (block normalisation) | `block_row_normalise`, `parameter_space` |
| First stage (3.1), CF-QMLE (3.5), CF-2SLS (3.6) | `first_stage_fit`, `cf_qmle`, `cf_2sls` |
| Theorems 1–2 (two-step variance), Lemma 9 (μ3, μ4) | `cf_qmle` → `se` |
| Lemma 3, Proposition 2 (dummy-variable drift) | `dummy_qmle`, `dummy_drift` |
| Theorem 3 (adjusted quasi-score) | `aqs_fit` |
| Proposition 3 (naive estimator at L = T) | warning in `cf_qmle` |
| Lemma 4, Theorem 4 (within-period QMLE) | `within_period_qmle` |
| Section 6, (6.1)–(6.3), BH / BY | `effects_inference`, `fdr_correction` |
| Appendix F.2 (test of δc = 0) | `wald_delta_c` |
| Appendix H (CF-2SLS variance) | `cf_2sls` |
| Appendix I (exact Anderson–Rubin) | `anderson_rubin`, `ar_confidence_set` |

---

## 13. Citation

```bibtex
@software{GSTW-PDM2025,
  author  = {Wang, Zining},
  title   = {{GSTW-PDM}: Generalized Spatio-Temporally Weighted Panel Data Model},
  year    = {2025},
  url     = {https://github.com/ZiningPe/GSTW-PDM}
}
```

Also cite the underlying STWM package:

```bibtex
@software{stwm2025,
  author = {Zining, Pe},
  title  = {{STWM}: Spatial-Temporal Weight Matrix for Panel Econometrics},
  year   = {2025},
  url    = {https://github.com/ZiningPe/STWM}
}
```

License: MIT.
