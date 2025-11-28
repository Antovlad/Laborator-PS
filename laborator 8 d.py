import numpy as np



N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t + 5

season = 3 * np.sin(2 * np.pi * t / 50) + 2 * np.sin(2 * np.pi * t / 120)

noise = np.random.normal(0, 1, N)

x = trend + season + noise



def fit_ar_least_squares(x, p):
    
    x = np.asarray(x)
    N = len(x)

    y = x[p:]

    X_lags = []
    for k in range(1, p+1):
        X_lags.append(x[p-k:N-k])
    X = np.column_stack(X_lags) 

    X = np.column_stack([np.ones(len(y)), X]) 

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    c = beta[0]
    a = beta[1:]
    return c, a



def one_step_mse(x, p, m):
   
    x = np.asarray(x)
    N = len(x)
    assert m > p, "Trebuie să existe cel puțin p observații pentru estimare."

    c, a = fit_ar_least_squares(x[:m], p)

    preds = np.zeros(N - m)

    for idx, t_idx in enumerate(range(m, N)):
        past_vals = x[t_idx - p:t_idx][::-1]       
        preds[idx] = c + np.dot(a, past_vals)

    true_vals = x[m:]
    mse = np.mean((true_vals - preds)**2)
    return mse



p_values = range(1, 21)          
m_values = range(200, 801, 100) 
best_err = np.inf
best_p = None
best_m = None

for p in p_values:
    for m in m_values:
        if m <= p:
            continue  

        mse_val = one_step_mse(x, p, m)

        print(f"p = {p:2d}, m = {m:4d}, MSE = {mse_val:.4f}")

        if mse_val < best_err:
            best_err = mse_val
            best_p = p
            best_m = m

print("\n=========================")
print("Cei mai buni hiperparametri:")
print(f"p optim = {best_p}")
print(f"m optim = {best_m}")
print(f"MSE minim = {best_err:.4f}")
print("=========================")
