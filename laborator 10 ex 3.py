import numpy as np
import matplotlib.pyplot as plt


def ar_design_matrix(x, p):
    
    x = np.asarray(x).ravel()
    N = len(x)
    y = x[p:]
    X = np.zeros((N - p, p))
    for k in range(1, p + 1):
        X[:, k - 1] = x[p - k: N - k]
    return X, y

def ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    return beta, resid, rss

def add_intercept(X):
    return np.column_stack([np.ones((X.shape[0], 1)), X])


def fit_ar_greedy_forward(x, p, k_max, tol=1e-8):
    
    Xlags, y = ar_design_matrix(x, p)

    X0 = np.ones((len(y), 1))
    beta0, _, rss0 = ols_fit(X0, y)

    selected = []
    remaining = list(range(p))  
    rss_history = [rss0]
    current_rss = rss0

    for _ in range(k_max):
        best_rss = np.inf
        best_j = None
        best_beta = None

        for j in remaining:
            cols = selected + [j]
            X = add_intercept(Xlags[:, cols])
            beta, _, rss = ols_fit(X, y)
            if rss < best_rss:
                best_rss = rss
                best_j = j
                best_beta = beta

        if best_j is None:
            break

        improvement = current_rss - best_rss
        if improvement <= tol:
            break

        selected.append(best_j)
        remaining.remove(best_j)
        current_rss = best_rss
        rss_history.append(current_rss)

    phi = np.zeros(p)
    if len(selected) > 0:
        X_final = add_intercept(Xlags[:, selected])
        beta, _, _ = ols_fit(X_final, y)
        c = beta[0]
        phi_sel = beta[1:]
        for idx, j in enumerate(selected):
            phi[j] = phi_sel[idx]
    else:
        c = float(beta0[0])

    lags_1_indexed = [j + 1 for j in selected]  
    return c, phi, lags_1_indexed, rss_history


def fit_ar_l1_cvxopt(x, p, lam, standardize=True):
    
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False

    X, y = ar_design_matrix(x, p)
    y = y.astype(float)

    y_mean = y.mean()
    y0 = y - y_mean

    X0 = X.copy()
    X_mean = X0.mean(axis=0)
    X0 = X0 - X_mean

    if standardize:
        X_std = X0.std(axis=0, ddof=0)
        X_std[X_std == 0] = 1.0
        Xs = X0 / X_std
    else:
        X_std = np.ones(p)
        Xs = X0

   
    XtX = Xs.T @ Xs
    Xty = Xs.T @ y0

    Q = np.block([
        [XtX,               np.zeros((p, p))],
        [np.zeros((p, p)),  np.zeros((p, p))]
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
    b_hat = z[:p]

    phi = b_hat / X_std

    
    c = y_mean - X_mean @ phi

    info = {
        "status": sol["status"],
        "objective": float(sol["primal objective"]),
        "nnz": int(np.sum(np.abs(phi) > 1e-8))
    }
    return c, phi, info


if __name__ == "__main__":
    np.random.seed(0)
    N = 1000
    t = np.arange(N)
    trend = 0.0005 * t**2 + 0.05 * t + 5
    season = 3*np.sin(2*np.pi*t/50) + 2*np.sin(2*np.pi*t/120)
    noise = np.random.normal(0, 1, N)
    x = trend + season + noise

    p = 200  

    K = 10
    c_g, phi_g, lags_g, rss_hist = fit_ar_greedy_forward(x, p, k_max=K)
    print("Greedy: lag-uri selectate =", lags_g)
    print("Greedy: nnz =", np.sum(np.abs(phi_g) > 1e-8))

    lam = 50.0
    c_l1, phi_l1, info = fit_ar_l1_cvxopt(x, p, lam=lam, standardize=True)
    print("L1/CVXOPT:", info)

    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.stem(phi_g, basefmt=" ")
    plt.title(f"Greedy Forward AR({p}) cu K={K} (coef. pe lag-uri 1..p)")

    plt.subplot(2,1,2)
    plt.stem(phi_l1, basefmt=" ")
    plt.title(f"L1 (Lasso) AR({p}) cu lambda={lam} (coef. pe lag-uri 1..p)")
    plt.tight_layout()
    plt.show()