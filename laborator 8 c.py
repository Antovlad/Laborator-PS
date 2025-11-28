import numpy as np
import matplotlib.pyplot as plt



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

p = 10

c, a = fit_ar_least_squares(x, p)
print("Intercept c =", c)
print("Coeficienți a =", a)

y_pred = np.empty_like(x)
y_pred[:] = np.nan  

for t_idx in range(p, N):
    
    past_vals = x[t_idx-p:t_idx][::-1]
    y_pred[t_idx] = c + np.dot(a, past_vals)



plt.figure(figsize=(12,5))
plt.plot(x, label="Seria originală")
plt.plot(y_pred, label=f"Predicție AR({p})", linewidth=2)
plt.title(f"Model AR({p}) pe seria de timp")
plt.xlabel("t")
plt.ylabel("valoare")
plt.legend()
plt.grid(True)
plt.show()
