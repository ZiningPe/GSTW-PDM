"""Checks for the multi-W / fixed-effects modules on a small synthetic recursive panel."""
import numpy as np
from scipy import sparse, linalg
from gstw_pdm import (split_same_cross, RecursiveNetwork, TiedNetwork, WithinDesign, city_demean,
                      first_stage_fe, aqs_variance, dummy_qmle, aqs_solve)


def _network(rng, n, Tf):
    pos = rng.uniform(size=(n, 2))
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    W = {}
    for m in ['a', 'b']:
        K = np.zeros((n * Tf, n * Tf))
        for s in range(Tf):
            Ks = np.exp(-(4 if m == 'a' else 8) * dist * rng.uniform(.5, 1.5, size=(n, n))) * (1 - np.eye(n))
            for t in range(s, Tf):
                K[t * n:(t + 1) * n, s * n:(s + 1) * n] = Ks
        W[m] = sparse.csr_matrix(K / K.sum(1, keepdims=True))
    return W


def test_split_and_traces():
    rng = np.random.default_rng(0)
    n, Tf = 12, 4
    T = Tf - 1
    W = _network(rng, n, Tf)
    rows = np.arange(n, n * Tf)
    lags = []
    for m in ['a', 'b']:
        W0, W1 = split_same_cross(W[m], n, Tf)
        assert abs((W0 + W1) - W[m]).max() < 1e-12
        lags += [(m + '0', W0[rows][:, rows], False), (m + '1', W1[rows][:, rows], True)]
    net = RecursiveNetwork(lags, n, T)
    rho = np.array([.3, .2, .2, .3])
    ld = net.factor(rho)
    N = n * T
    S = np.eye(N) - sum(r * l[1].toarray() for r, l in zip(rho, lags))
    assert abs(ld - np.linalg.slogdet(S)[1]) < 1e-10
    B = rng.normal(size=(N, 3))
    assert np.abs(S @ net.solve(B) - B).max() < 1e-10
    F = np.kron(np.eye(T) - np.ones((T, T)) / T, np.eye(n))
    Sinv = linalg.inv(S)
    G = [l[1].toarray() @ Sinv for l in lags]
    tFG, SB = net.tr_FG()
    for j in range(4):
        assert abs(tFG[j] - np.trace(F @ G[j])) < 1e-9
    tFGG = net.tr_FGG(SB)
    for j in range(4):
        for l in range(4):
            assert abs(tFGG[j, l] - np.trace(F @ G[j] @ G[l])) < 1e-9


def test_aqs_variance_matches_dense():
    rng = np.random.default_rng(1)
    n, Tf = 25, 5
    T = Tf - 1
    N = n * T
    W = _network(rng, n, Tf)
    rows = np.arange(n, n * Tf)
    lags = []
    Wfull = []
    for m in ['a', 'b']:
        W0, W1 = split_same_cross(W[m], n, Tf)
        lags += [(m + '0', W0[rows][:, rows], False), (m + '1', W1[rows][:, rows], True)]
        Wfull += [W0, W1]
    rho = np.array([.3, .15, .2, .25])
    X = rng.normal(size=(n * Tf, 2))
    Z = rng.normal(size=(n * Tf, 3))
    eps = rng.normal(size=n * Tf)
    h = Z @ np.array([1., .5, -.5]) + eps
    alpha = rng.normal(size=n)
    xi = rng.normal(size=n * Tf) * .5
    mu = X @ np.array([1., -.5]) + np.tile(alpha, Tf) + .4 * eps + xi
    Sf = sparse.identity(n * Tf) - sum(r * Wm for r, Wm in zip(rho, Wfull))
    y = sparse.linalg.spsolve(Sf.tocsc(), mu)
    mask = np.repeat(np.arange(Tf), n) >= 1
    yd = np.eye(Tf)[np.repeat(np.arange(Tf), n)][:, 1:]
    fs = first_stage_fe(h[mask], np.column_stack([Z[mask], yd[mask][:, 1:]]), n, T, n_instruments=3)
    U = np.column_stack([Wm @ y for Wm in Wfull])[mask]
    R = np.column_stack([X[mask], fs['eps_hat'], yd[mask][:, 1:]])
    names = ['b1', 'b2', 'delta_a', 'y2', 'y3', 'y4']
    wd = WithinDesign(y[mask], U, R, n, T)
    net = RecursiveNetwork(lags, n, T)
    va = aqs_variance(net, wd, rho, names, first_stages={'a': fs})
    # independent dense implementation at the same rho
    S = np.eye(N) - sum(r * l[1].toarray() for r, l in zip(rho, lags))
    Sinv = linalg.inv(S)
    G = [l[1].toarray() @ Sinv for l in lags]
    F = np.kron(np.eye(T) - np.ones((T, T)) / T, np.eye(n))
    RF, gam, s2 = wd.RF, wd.gamma(rho), va['sigma2']
    e = wd.resid(rho)
    hist = U - np.column_stack([l[1] @ y[mask] for l in lags])      # weights on the dropped first year
    muv = wd.R_raw[:, wd.keep] @ gam + np.tile(va['alpha'], T) + hist @ rho
    p, k = 4, RF.shape[1]
    P = p + k + 1
    mu3 = np.mean(e ** 3) * T ** 2 / ((T - 1) * (T - 2))
    mu4 = (np.mean(e ** 4) - 3 * (T - 1) * (2 * T - 3) / T ** 3 * s2 ** 2) / ((T - 1) * (T ** 2 - 3 * T + 3) / T ** 3)
    A = [Gj.T @ F / s2 for Gj in G]
    c = [F @ (Gj @ muv + hist[:, j]) for j, Gj in enumerate(G)]
    a = [cj / s2 for cj in c]
    allA = A + [np.zeros((N, N))] * k + [F / (2 * s2 ** 2)]
    alla = a + [RF[:, c] / s2 for c in range(k)] + [np.zeros(N)]
    Ups = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            A1s = (allA[i] + allA[i].T) / 2
            A2s = (allA[j] + allA[j].T) / 2
            Ups[i, j] = 2 * s2 ** 2 * np.trace(A1s @ A2s) + s2 * alla[i] @ alla[j] \
                + (mu4 - 3 * s2 ** 2) * np.diag(allA[i]) @ np.diag(allA[j]) \
                + mu3 * (np.diag(allA[i]) @ alla[j] + np.diag(allA[j]) @ alla[i])
    Gam = np.zeros((P, P))
    for j in range(p):
        for l in range(p):
            Gam[j, l] = -(c[j] @ c[l] / s2 + np.trace(G[j].T @ F @ G[l]) + np.trace(F @ G[j] @ G[l]))
        Gam[j, p:p + k] = -c[j] @ RF / s2
        Gam[p:p + k, j] = Gam[j, p:p + k]
        Gam[j, -1] = -np.trace(F @ G[j]) / s2
        Gam[-1, j] = Gam[j, -1]
    Gam[p:p + k, p:p + k] = -RF.T @ RF / s2
    Gam[-1, -1] = -(N - n) / (2 * s2 ** 2)
    rel = lambda A_, B_: np.abs(A_ - B_).max() / np.abs(B_).max()
    assert rel(va['Gamma'], Gam) < 1e-10
    assert rel(va['Upsilon'], Ups) < 1e-10
    # rescaling the control function leaves the variance of the other coefficients unchanged
    R2 = R.copy()
    R2[:, 2] /= 7.0
    wd2 = WithinDesign(y[mask], U, R2, n, T)
    vb = aqs_variance(net, wd2, rho, names, first_stages={'a': fs}, cluster={'unit': np.arange(n)})
    keep = [i for i in range(P) if i != p + 2]
    assert rel(vb['V'][np.ix_(keep, keep)], va['V'][np.ix_(keep, keep)]) < 1e-8
    assert np.all(np.diag(vb['V_cluster']['unit']) > 0)
    # tied coefficients: A' Gamma A
    A_ = np.array([[1., 0], [1, 0], [0, 1], [0, 1]])
    tn = TiedNetwork(net, A_, ['a', 'b'])
    rt = np.array([.2, .25])
    wdt = WithinDesign(y[mask], U @ A_, R, n, T)
    vt = aqs_variance(tn, wdt, rt, names, first_stages={'a': fs})
    vf = aqs_variance(net, wd, A_ @ rt, names, first_stages={'a': fs})
    B_ = linalg.block_diag(A_, np.eye(k + 1))
    assert rel(vt['Gamma'], B_.T @ vf['Gamma'] @ B_) < 1e-10
    assert rel(vt['Upsilon'], B_.T @ vf['Upsilon'] @ B_) < 1e-10


def test_score_variance_by_simulation():
    """Variance of the adjusted score for rho at the truth equals c'c/s2 + tr(QQ) + tr(QQ'), c = F G mu."""
    rng = np.random.default_rng(0)
    n, T = 40, 6
    N = n * T
    yr = np.repeat(np.arange(T), n)
    K = rng.uniform(size=(N, N)) * (rng.uniform(size=(N, N)) < .1)
    K[yr[:, None] < yr[None, :]] = 0
    np.fill_diagonal(K, 0)
    K /= K.sum(1, keepdims=True)
    W0 = K * (yr[:, None] == yr[None, :])
    W1 = K - W0
    Sinv = linalg.inv(np.eye(N) - .3 * W0 - .4 * W1)
    F = np.kron(np.eye(T) - np.ones((T, T)) / T, np.eye(n))
    mu = 2 + .5 * yr + np.tile(rng.normal(size=n), T) + rng.normal(size=N)
    s2 = .25
    xi = rng.normal(size=(20000, N)) * np.sqrt(s2)
    for Wj in (W0, W1):
        Q = F @ Wj @ Sinv
        g = np.einsum('ri,ri->r', (mu + xi) @ (Wj @ Sinv).T, xi @ F) / s2 - np.trace(Q)
        c = Q @ mu
        v = c @ c / s2 + np.sum(Q * Q) + np.sum(Q * Q.T)
        assert abs(g.mean()) < 4 * np.sqrt(v / len(g))
        assert abs(g.var() / v - 1) < .05


if __name__ == '__main__':
    test_split_and_traces()
    test_aqs_variance_matches_dense()
    test_score_variance_by_simulation()
    print('ok')
