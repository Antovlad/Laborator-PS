import numpy as np


def make_time_series(N=1000, seed=0):
    np.random.seed(seed)
    t = np.arange(N)
    trend = 0.0005 * t**2 + 0.05 * t + 5
    season = 3*np.sin(2*np.pi*t/50) + 2*np.sin(2*np.pi*t/120)
    noise = np.random.normal(0, 1, N)
    x = trend + season + noise
    return x


def ar_design_matrix(x, p):
    x = np.asarray(x)
    N = len(x)
    y = x[p:]
    X = np.zeros((N - p, p))
    for k in range(1, p + 1):
        X[:, k - 1] = x[p - k:N - k]
    return X, y

def add_intercept(X):
    return np.column_stack([np.ones((X.shape[0], 1)), X])

def fit_ar_ols(x, p):
    Xlags, y = ar_design_matrix(x, p)
    X = add_intercept(Xlags)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c = beta[0]
    phi = beta[1:]
    return c, phi


def fit_ar_greedy_forward(x, p, k_max):
    Xlags, y = ar_design_matrix(x, p)

    selected = []
    remaining = list(range(p))

    X0 = np.ones((len(y), 1))
    beta0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    best_rss = np.sum((y - X0 @ beta0)**2)

    for _ in range(k_max):
        best_j = None
        for j in remaining:
            cols = selected + [j]
            X = add_intercept(Xlags[:, cols])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            rss = np.sum((y - X @ beta)**2)
            if rss < best_rss:
                best_rss = rss
                best_j = j

        if best_j is None:
            break

        selected.append(best_j)
        remaining.remove(best_j)

    phi = np.zeros(p)
    if selected:
        X = add_intercept(Xlags[:, selected])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        c = beta[0]
        for i, j in enumerate(selected):
            phi[j] = beta[i + 1]
    else:
        c = beta0[0]

    return c, phi


def roots_via_companion(coeffs):
    c = np.asarray(coeffs, dtype=float)
    c = c / c[0]  
    n = len(c) - 1

    C = np.zeros((n, n))
    C[1:, :-1] = np.eye(n - 1)
    C[:, -1] = -c[1:][::-1]

    return np.linalg.eigvals(C)


def is_stationary_ar(phi):
    coeffs = np.r_[1, -phi]
    roots = roots_via_companion(coeffs)
    return np.all(np.abs(roots) > 1), roots


def fit_ar_l1_cvxopt(x, p, lam):
    from cvxopt import matrix, solvers
    solvers.options["show_progress"] = False

    X, y = ar_design_matrix(x, p)

    y_mean = y.mean()
    y0 = y - y_mean
    X_mean = X.mean(axis=0)
    X0 = X - X_mean

    X_std = X0.std(axis=0)
    X_std[X_std == 0] = 1
    Xs = X0 / X_std

    XtX = Xs.T @ Xs
    Xty = Xs.T @ y0

    Q = np.block([
        [XtX, np.zeros((p, p))],
        [np.zeros((p, p)), np.zeros((p, p))]
    ])
    q = np.concatenate([-Xty, lam * np.ones(p)])

    G = np.block([
        [ np.eye(p), -np.eye(p)],
        [-np.eye(p), -np.eye(p)],
        [np.zeros((p, p)), -np.eye(p)]
    ])
    h = np.zeros(3 * p)

    sol = solvers.qp(matrix(Q), matrix(q), matrix(G), matrix(h))
    z = np.array(sol['x']).ravel()
    b = z[:p]

    phi = b / X_std
    c = y_mean - X_mean @ phi
    return c, phi


if __name__ == "__main__":
    x = make_time_series()

    p = 200
    K = 10
    lam = 50.0

    c_ols, phi_ols = fit_ar_ols(x, p)
    stat_ols, _ = is_stationary_ar(phi_ols)

    c_g, phi_g = fit_ar_greedy_forward(x, p, K)
    stat_g, _ = is_stationary_ar(phi_g)

    c_l1, phi_l1 = fit_ar_l1_cvxopt(x, p, lam)
    stat_l1, _ = is_stationary_ar(phi_l1)

    print("AR OLS staționar:", stat_ols)
    print("AR Greedy staționar:", stat_g)
    print("AR L1 staționar:", stat_l1)
