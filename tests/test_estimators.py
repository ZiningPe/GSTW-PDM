"""
test_estimators.py
------------------
Checks of the estimators against the paper (Sections 2-6) on small synthetic panels.

Run with:  pytest tests/ -v
"""

import warnings
import numpy as np
import pytest
from scipy import linalg, sparse

import gstw_pdm as g


@pytest.fixture(scope="module")
def kron():
    rng = np.random.default_rng(42)
    n_side, T = 6, 4
    n = n_side ** 2
    N = n * T
    Ws = g.rook_weights(n_side)
    M = g.build_twm_parametric(T, rho=0.5)
    net = g.KroneckerNetwork(M, Ws)
    W0, W1 = (W.toarray() for W in net.W)
    S = np.eye(N) - 0.3 * W0 - 0.2 * W1
    X = rng.normal(size=(N, 1))
    Z = np.column_stack([np.ones(N), X[:, 0], rng.normal(size=(N, 2))])
    eps = rng.normal(size=N)
    h = Z @ np.array([0.0, 0.8, 1.0, 1.0]) + eps
    y = linalg.solve(S, X[:, 0] + 0.5 * W0 @ X[:, 0] + 0.3 * W1 @ X[:, 0] + 0.6 * eps + 0.7 * rng.normal(size=N))
    return dict(n=n, T=T, N=N, Ws=Ws, M=M, net=net, W0=W0, W1=W1, X=X, Z=Z, h=h, y=y, rng=rng)


@pytest.fixture(scope="module")
def big():
    rng = np.random.default_rng(11)
    n_side, T = 10, 5
    n = n_side ** 2
    N = n * T
    Ws = g.rook_weights(n_side)
    M = g.build_twm_parametric(T, rho=0.5)
    net = g.KroneckerNetwork(M, Ws)
    W0, W1 = (W.toarray() for W in net.W)
    S = np.eye(N) - 0.3 * W0 - 0.2 * W1
    X = rng.normal(size=(N, 1))
    Z = np.column_stack([np.ones(N), X[:, 0], rng.normal(size=(N, 2))])
    eps = rng.normal(size=N)
    h = Z @ np.array([0.0, 0.8, 1.0, 1.0]) + eps
    y = linalg.solve(S, X[:, 0] + 0.5 * W0 @ X[:, 0] + 0.3 * W1 @ X[:, 0] + 0.6 * eps + 0.7 * rng.normal(size=N))
    yT = linalg.solve(S, X[:, 0] + 0.5 * W0 @ X[:, 0] + 0.3 * W1 @ X[:, 0]
                      + np.repeat(2 * rng.normal(size=T), n) + 0.7 * rng.normal(size=N))
    truth = {"rho0": 0.3, "rho1": 0.2, "x1": 1.0, "rho0*x1": 0.5, "rho1*x1": 0.3}
    return dict(n=n, T=T, N=N, Ws=Ws, M=M, net=net, X=X, Z=Z, h=h, y=y, yT=yT, truth=truth)


class TestWeights:
    def test_rook_row_sum(self):
        assert np.allclose(g.rook_weights(4).sum(1), 1)

    def test_kronecker_blocks(self, kron):
        W0, W1 = g.kronecker_weights(kron["M"], kron["Ws"])
        assert np.allclose(W0, kron["W0"]) and np.allclose(W1, kron["W1"])
        assert np.allclose(W0.sum(1), 1) and np.allclose(W1.sum(1), 1)

    def test_logdet_kronecker(self, kron):
        eM, eW = g.eigvals_kronecker(kron["M"], kron["Ws"])
        for r in [(0.3, 0.2), (-0.4, 0.5)]:
            S = np.eye(kron["N"]) - r[0] * kron["W0"] - r[1] * kron["W1"]
            assert abs(g.logdet_kronecker(r, eM, eW) - np.linalg.slogdet(S)[1]) < 1e-8

    def test_block_row_normalise(self, kron):
        n, T = kron["n"], kron["T"]
        W0, W1 = g.block_row_normalise(sparse.csr_matrix(3 * kron["W0"] + kron["W1"]), n, T)
        assert np.allclose(W0.toarray(), kron["W0"]) and np.allclose(W1.toarray(), kron["W1"])

    def test_parameter_space(self, kron):
        ps = g.parameter_space(kron["W0"], kron["W1"], kron["n"], kron["T"])
        assert ps["row_sum_W0"] <= 1 + 1e-12 and ps["row_sum_W1"] <= 1 + 1e-12
        assert ps["zero_diagonal"] and not ps["recursive"]


class TestNetworks:
    def test_dense_kron_recursive_logdet(self, kron):
        n, T = kron["n"], kron["T"]
        W1r = np.tril(kron["W1"])
        dn = g.DenseNetwork([("a", kron["W0"]), ("b", W1r)], n, T)
        rn = g.make_network([("a", kron["W0"]), ("b", W1r)], n, T)
        assert isinstance(rn, g.RecursiveNetwork)
        assert abs(dn.factor([0.3, 0.7]) - rn.factor([0.3, 0.7])) < 1e-9
        B = kron["rng"].normal(size=(kron["N"], 2))
        assert np.abs(dn.solve(B) - rn.solve(B)).max() < 1e-10

    def test_make_network_dense(self, kron):
        net = g.make_network([("W", kron["W0"] + kron["W1"])], kron["n"], kron["T"])
        assert isinstance(net, g.DenseNetwork)


class TestFirstStage:
    def test_residual_orthogonal(self, kron):
        fs = g.first_stage_fit(kron["h"], kron["Z"], n_instruments=4)
        assert np.abs(kron["Z"].T @ fs["eps_hat"]).max() < 1e-8
        assert fs["dof"] == kron["N"] - 4 and fs["F"] > 10

    def test_aggregate_L_eq_T(self):
        assert np.allclose(g.aggregate_eps_hat(np.array([1.0, 2.0]), 3, 2), [1, 1, 1, 2, 2, 2])


class TestCFQMLE:
    def test_kron_equals_dense(self, kron):
        fs = g.first_stage_fit(kron["h"], kron["Z"])
        dn = g.DenseNetwork([("rho0", kron["W0"]), ("rho1", kron["W1"])], kron["n"], kron["T"])
        a = g.cf_qmle(kron["y"], kron["X"], kron["net"], {"h": fs}, extra=np.ones(kron["N"]))
        b = g.cf_qmle(kron["y"], kron["X"], dn, {"h": fs}, extra=np.ones(kron["N"]))
        assert np.abs(a["theta"] - b["theta"]).max() < 1e-6
        assert np.abs(a["se"] - b["se"]).max() < 1e-6

    def test_names_and_generated_term(self, kron):
        fs = g.first_stage_fit(kron["h"], kron["Z"])
        f = g.cf_qmle(kron["y"], kron["X"], kron["net"], {"h": fs}, extra=np.ones(kron["N"]), extra_names=["const"])
        assert f["names"][:5] == ["rho0", "rho1", "x1", "rho0*x1", "rho1*x1"]
        i = f["names"].index("x1")
        # X is in the first stage, so the beta variance gains the generated-regressor term
        assert f["se"][i] > f["se_information"][i] * 1.05
        # the delta_c and sigma2 rows of D are zero (Theorem 2)
        assert np.allclose(f["D"][f["names"].index("delta_h")], 0) and np.allclose(f["D"][-1], 0)

    def test_gaussian_sandwich_close_to_information(self, kron):
        f = g.qmle(kron["y"], kron["X"], kron["net"], extra=np.ones(kron["N"]))
        # mean-parameter block; the sigma2 entry depends on the sample kurtosis
        assert np.allclose(f["V"][:-1, :-1], f["V_information"][:-1, :-1], rtol=0.05, atol=1e-5)

    def test_lq_variance_by_simulation(self):
        """Score variance with skewed errors: Lemma 9 against simulation."""
        rng = np.random.default_rng(3)
        n_side, T = 5, 3
        n = n_side ** 2
        N = n * T
        net = g.KroneckerNetwork(g.build_twm_parametric(T, 0.5), g.rook_weights(n_side))
        W0, W1 = (W.toarray() for W in net.W)
        rho = np.array([0.3, 0.2])
        S = np.eye(N) - rho[0] * W0 - rho[1] * W1
        R = np.column_stack([np.ones(N), rng.normal(size=N)])
        gam = np.array([0.5, 1.0])
        s2 = 2.0
        from gstw_pdm.cf_qmle import qmle_information_and_score_variance
        draws = (rng.chisquare(2, size=(4000, N)) - 2) / 2 * np.sqrt(s2)
        m3, m4 = np.mean(draws ** 3), np.mean(draws ** 4)
        Info, V, _ = qmle_information_and_score_variance(net, R, rho, gam, s2, m3, m4)
        Sinv = linalg.inv(S)
        G = [W0 @ Sinv, W1 @ Sinv]
        sc = []
        for xi in draws:
            Y = Sinv @ (R @ gam + xi)
            s = [(W @ Y) @ xi / s2 - np.trace(Gj) for W, Gj in zip([W0, W1], G)]
            s += list(R.T @ xi / s2) + [xi @ xi / (2 * s2 ** 2) - N / (2 * s2)]
            sc.append(s)
        Vs = np.cov(np.array(sc).T)
        d = np.sqrt(np.diag(V))
        assert np.abs(Vs / np.outer(d, d) - V / np.outer(d, d)).max() < 0.08

    def test_wald_delta_c(self, kron):
        fs = g.first_stage_fit(kron["h"], kron["Z"])
        f = g.cf_qmle(kron["y"], kron["X"], kron["net"], {"h": fs}, extra=np.ones(kron["N"]))
        w = g.wald_delta_c(f)
        assert w["df"] == 1 and w["p"] < 0.01


class TestCF2SLS:
    def test_close_to_truth(self, big):
        fs = g.first_stage_fit(big["h"], big["Z"])
        a = g.cf_2sls(big["y"], big["X"], big["net"], {"h": fs}, extra=np.ones(big["N"]))
        b = g.cf_qmle(big["y"], big["X"], big["net"], {"h": fs}, extra=np.ones(big["N"]))
        assert a["names"] == b["names"][:-1]
        for k, v in big["truth"].items():
            i = a["names"].index(k)
            assert abs(a["theta"][i] - v) < 4 * a["se"][i]
        assert np.all(a["se"] > 0) and np.all(a["partial_F"] > 10)
        assert np.all(a["se"] >= np.sqrt(np.diag(a["V_no_generated"])) - 1e-12)


class TestPeriodLevel:
    def test_within_period_removes_period_constants(self, kron):
        n, T, N = kron["n"], kron["T"], kron["N"]
        rng = np.random.default_rng(7)
        S = np.eye(N) - 0.3 * kron["W0"] - 0.2 * kron["W1"]
        X = rng.normal(size=(N, 1))
        y = linalg.solve(S, X[:, 0] + np.repeat(5 * rng.normal(size=T), n) + 0.5 * rng.normal(size=N))
        w = g.within_period_qmle(y, X, kron["M"], kron["Ws"], durbin=False)
        # adding any period constants to the outcome equation leaves the estimate unchanged
        y2 = linalg.solve(S, S @ y + np.repeat(rng.normal(size=T), n))
        w2 = g.within_period_qmle(y2, X, kron["M"], kron["Ws"], durbin=False)
        assert np.abs(w["theta"] - w2["theta"]).max() < 1e-6
        assert w["N_star"] == T * (n - 1)

    def test_anderson_rubin(self, big):
        ar = g.anderson_rubin(big["yT"], big["X"], big["M"], big["Ws"], (0.3, 0.2))
        far = g.anderson_rubin(big["yT"], big["X"], big["M"], big["Ws"], (-0.3, 0.8))
        assert ar["p"] > 0.01 and far["p"] < 0.01
        assert ar["df"][1] == big["T"] * (big["n"] - 1) - 3 - ar["df"][0]

    def test_within_period_close_to_truth(self, big):
        w = g.within_period_qmle(big["yT"], big["X"], big["M"], big["Ws"])
        for k, v in big["truth"].items():
            i = w["names"].index(k)
            assert abs(w["theta"][i] - v) < 4 * w["se"][i]

    def test_naive_warns_at_L_eq_T(self, kron):
        n, T, N = kron["n"], kron["T"], kron["N"]
        fs = g.first_stage_fit(np.arange(T, dtype=float) + np.random.default_rng(1).normal(size=T),
                               np.column_stack([np.ones(T), np.arange(T)]))
        with pytest.warns(UserWarning):
            g.cf_qmle(kron["y"], kron["X"], kron["net"], {"h": fs}, extra=np.ones(N))


class TestEffects:
    def test_effects_match_dense(self, kron):
        n, T, N = kron["n"], kron["T"], kron["N"]
        f = g.qmle(kron["y"], kron["X"], kron["net"], extra=np.ones(N))
        ef = g.effects_inference(kron["net"], f, "x1")
        th = dict(zip(f["names"], f["theta"]))
        Sk = linalg.inv(np.eye(N) - th["rho0"] * kron["W0"] - th["rho1"] * kron["W1"]) @ (
            th["x1"] * np.eye(N) + th["rho0*x1"] * kron["W0"] + th["rho1*x1"] * kron["W1"])
        IE = np.array([[Sk[t * n:(t + 1) * n, s * n:(s + 1) * n].sum() / n for s in range(T)] for t in range(T)])
        assert np.abs(IE - ef["IE"]).max() < 1e-10
        assert np.allclose(ef["DE"] + ef["IEw"], np.diag(IE))
        assert np.all(ef["SE"] > 0)

    def test_block_diagonal_has_no_cross_effects(self, kron):
        n, T, N = kron["n"], kron["T"], kron["N"]
        net = g.DenseNetwork([("rho0", kron["W0"])], n, T)
        f = g.qmle(kron["y"], kron["X"], net, extra=np.ones(N))
        ef = g.effects_inference(net, f, "x1", direct=False)
        assert np.abs(ef["IE"][~np.eye(T, dtype=bool)]).max() < 1e-12

    def test_fdr(self):
        p = np.array([0.001, 0.01, 0.02, 0.04, 0.5])
        rb, ab = g.bh_correction(p, 0.05)
        ry, ay = g.by_correction(p, 0.05)
        assert rb.sum() >= ry.sum() and np.all(ay >= ab)
        assert rb[:4].all() and not rb[4]


class TestFixedEffects:
    def test_dummy_drift_zero_for_block_diagonal(self, kron):
        n, T = kron["n"], kron["T"]
        net = g.RecursiveNetwork([("rho0", sparse.csr_matrix(kron["W0"]), False)], n, T)
        assert abs(g.dummy_drift(net, [0.3])["drift"][0]) < 1e-9

    def test_dummy_drift_matches_dense(self, kron):
        n, T, N = kron["n"], kron["T"], kron["N"]
        W1 = np.tril(kron["W1"])
        net = g.RecursiveNetwork([("a", sparse.csr_matrix(kron["W0"]), False), ("b", sparse.csr_matrix(W1), True)], n, T)
        d = g.dummy_drift(net, [0.3, 0.4])
        Sinv = linalg.inv(np.eye(N) - 0.3 * kron["W0"] - 0.4 * W1)
        for j, Wj in enumerate([kron["W0"], W1]):
            G = Wj @ Sinv
            cr = sum(np.trace(G[t * n:(t + 1) * n, s * n:(s + 1) * n]) for t in range(T) for s in range(T) if t != s)
            assert abs(d["drift"][j] + cr / (T - 1)) < 1e-9


class TestUtils:
    def test_check_admissibility(self, kron):
        eM, eW = g.eigvals_kronecker(kron["M"], kron["Ws"])
        assert g.check_admissibility((0.3, 0.2), eM, eW)

    def test_ned_keys(self, kron):
        out = g.ned_test(kron["y"], kron["n"], kron["T"], kron["Ws"])
        assert {"morans_by_period", "temporal_autocorr"} <= set(out)
