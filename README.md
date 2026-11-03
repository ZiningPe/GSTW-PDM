# GSTW-PDM

**Generalized Spatio-Temporally Weighted Panel Data Model**

Estimators and inference for spatial Durbin panels whose weight matrix links
units both within and across periods, and whose weights may be endogenous.
The package implements the model and results of Wang, *Generalized
Spatio-Temporally Weighted Panel Data Model* (working paper).

```bash
pip install git+https://github.com/ZiningPe/GSTW-PDM.git
```

Requirements: Python ≥ 3.9, NumPy ≥ 1.24, SciPy ≥ 1.10.

---

## Model

Observations are stacked period by period (`p = (t-1) n + i`, time-major).

```
Y = ρ0 W0 Y + ρ1 W1 Y + X β + W0 X θ0 + W1 X θ1 + V
V = δc · A ε + ξ,        h = Z_W π + ε
```

* `W0` holds the same-period blocks of the weight matrix and `W1 = W − W0`
  the cross-period blocks. Each is row-normalised on its own
  (`block_row_normalise`); `ρ1 = θ1 = 0` is the static spatial Durbin panel.
* `h` is the variable that forms the weights (for example a city's
  accessibility in year t). Its first-stage residual is the control function;
  `δc = 0` if and only if the weights are exogenous.
* `L` = number of first-stage observations. `L = N` (one `h_it` per unit and
  period, `A = I_N`) is the main case; `L = T` (one `h_t` per period,
  `A = I_T ⊗ ι_n`) is the period-level case.

Any number of lag matrices is allowed (for example two networks, each split
into same- and cross-period parts); the coefficients are named after the
network's lags, and the Durbin terms `"<lag>*<x>"`.

## What is implemented

| Paper | Function |
|---|---|
| Kronecker weights `W0 = I_T ⊗ W_S`, `W1 = M ⊗ W_S`; log-det (2.5) | `kronecker_weights`, `KroneckerNetwork` |
| Recursive weights (only the past transmits), log-det (2.6), Lemma 2 | `RecursiveNetwork`, `split_same_cross`, `make_network` |
| General `W`, N × N log-det | `DenseNetwork` |
| Assumption 3 / Lemma 2 parameter space | `parameter_space` |
| First stage (3.1) | `first_stage_fit`, `first_stage_fe` (within-unit) |
| CF-QMLE (3.5), Theorem 1; two-step variance of Theorem 2 with μ3, μ4 (Lemma 9) and the generated-regressor term | `cf_qmle`, `qmle` |
| CF-2SLS (3.6), Appendix H variance | `cf_2sls` |
| Unit fixed effects: dummy-variable QMLE and its drift (4.5), Proposition 2 | `dummy_qmle`, `dummy_drift` |
| Adjusted quasi-score (4.6), Theorem 3; profiled `tr(M_R G_j)` form; cluster-robust option | `aqs_fit` (`adjust_regressors=True`), `aqs_variance` |
| Demeaned CF-2SLS (comparison estimator for the AQS) | `cf_2sls_fe` |
| L = T: QMLE on within-period deviations (5.2), Theorem 4 | `within_period_qmle` |
| L = T: exact Anderson–Rubin statistic and set, Appendix I | `anderson_rubin`, `ar_confidence_set` |
| Wald test of `δc = 0`, Appendix F.2 | `wald_delta_c` |
| Direct, within-period and cross-period effects (2.3), (6.1); delta method with the analytic gradient (6.3) | `spillover_effects`, `effects_inference` |
| Benjamini–Hochberg and Benjamini–Yekutieli control | `fdr_correction`, `bh_correction`, `by_correction` |

At `L = T` the CF-QMLE is the *naive* estimator of Proposition 3: it converges
to a point that depends on the realised period shocks, and `cf_qmle` warns.
Use `within_period_qmle` there; it does not use the first stage and is
consistent for every `δc`.

---

## Quick start: L = N with a recursive network

```python
import numpy as np
import gstw_pdm as g

# W: (nT x nT) time-major weights with no block above the diagonal
W0, W1 = g.block_row_normalise(W, n, T)              # same- and cross-period blocks
print(g.parameter_space(W0, W1, n, T))               # Assumption 3 / Lemma 2
net = g.RecursiveNetwork([("rho0", W0, False), ("rho1", W1, True)], n, T)

fs = g.first_stage_fit(h, np.column_stack([Z_excl, np.ones(N), X]), n_instruments=Z_excl.shape[1])
fit = g.cf_qmle(y, X, net, first_stages={"h": fs}, extra=np.ones(N), extra_names=["const"])
print(g.regression_table(fit["names"], fit["theta"], fit["se"], title="CF-QMLE"))
print(g.wald_delta_c(fit))                           # exogeneity of the weights

eff = g.effects_inference(net, fit, "x1", fdr="by")  # IE(t<-s), DE_t, IEw_t with SEs
print(g.print_ie_table(eff))

tsls = g.cf_2sls(y, X, net, first_stages={"h": fs}, extra=np.ones(N))
```

`fit["se"]` is the Theorem 2 standard error, `fit["se_information"]` the
information-only one (it under-covers when the first-stage instruments are
correlated with the regressors, for example when `X` enters the first stage).

## Period-level formation variable (L = T)

```python
M = g.build_twm_from_stats(h_t, bandwidth=b)         # M_ts ∝ exp(-|h_t - h_s| / b)
fit = g.within_period_qmle(y, X, M, W_S)             # Theorem 4
ar = g.ar_confidence_set(y, X, M, W_S, grid0, grid1) # exact under Gaussian ξ
```

## Unit fixed effects at fixed T

Time demeaning removes unit effects only if `Σ_s W_ts` is the same for every
`t` (Lemma 3), which fails for accumulating networks. The dummy-variable QMLE
then has an `O(1/T)` drift (`dummy_drift`); the adjusted quasi-score removes it.

```python
U = np.column_stack([W @ y for W in net.W])           # lags (may include a predetermined history)
R = np.column_stack([X, W0 @ X, W1 @ X, fs['eps_hat'], year_dummies])
wd = g.WithinDesign(y, U, R, n, T)
fs = g.first_stage_fe(h, Z, n, T, n_instruments=k)
res = g.aqs_fit(net, wd, R_names, first_stages={"h": fs}, adjust_regressors=True,
                cluster={"unit": np.arange(n)})
```

The control-function column must be named `delta_<name>` in `R_names` for
the generated-regressor term to be added. `cluster=` gives a cluster-robust
variance that allows within-unit serial correlation of the errors; Theorem 3
itself assumes independent errors.

---

## Package structure

```
gstw_pdm/
├── weight_construction.py  spatial W, temporal M, Kronecker blocks, block normalisation, parameter space
├── multi_w.py              Recursive / Dense / Kronecker / Tied networks, same/cross split
├── first_stage.py          first stage h = Z_W π + ε
├── control_function.py     A ε̂ for L = N or L = T
├── cf_qmle.py              CF-QMLE, Theorem 2 variance, Wald test of δc
├── cf_2sls.py              CF-2SLS
├── period_level.py         within-period QMLE and Anderson–Rubin (L = T)
├── fixed_effects.py        dummy QMLE, drift, AQS, demeaned CF-2SLS, Wald tests
├── inference.py            spillover effects, delta method, BH / BY
├── report.py               Stata-style tables
└── utils.py                NED diagnostic, reshaping, checks
examples/
├── recursive_network_example.py   L = N, endogenous recursive network
└── period_level_example.py        L = T, naive vs within-period QMLE, AR set
```

### Changes in 0.3.0

The estimators follow the paper's model with separate same- and cross-period
coefficients. `cf_qmle` and `cf_2sls` take a network object and estimate
`(ρ0, ρ1, β, θ0, θ1, δc)`; the old single-`δ` interface, CF-GMM and the
`Ω_A` module were removed (the paper no longer uses them). New:
`DenseNetwork`, `KroneckerNetwork`, `make_network`, `first_stage_fit`,
`within_period_qmle`, `anderson_rubin`, `effects_inference`, `dummy_drift`,
`parameter_space`, `block_row_normalise`, `by_correction`. The 0.1.0
package and its Monte Carlo scripts for the earlier single-`δ` model remain
available at commit `61eba57`.

---

## Citation

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
