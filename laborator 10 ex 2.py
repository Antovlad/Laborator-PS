

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)

N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t + 5
season = 3*np.sin(2*np.pi*t/50) + 2*np.sin(2*np.pi*t/120)
noise = np.random.normal(0, 1, N)

x = trend + season + noise


def fit_ar_ols(x, p):
   
    x = np.asarray(x)
    N = len(x)

    y = x[p:]
    X = np.ones((N - p, p + 1))  

    for k in range(1, p + 1):
        X[:, k] = x[p - k:N - k]

    
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    c = beta[0]
    phi = beta[1:]

    resid = y - X @ beta
    sigma2 = np.var(resid)

    return c, phi, resid, sigma2


def predict_ar(x, c, phi, steps=1):
    p = len(phi)
    x_hist = list(x.copy())
    preds = []

    for _ in range(steps):
        past = np.array(x_hist[-p:][::-1])
        yhat = c + phi @ past
        preds.append(yhat)
        x_hist.append(yhat)

    return np.array(preds)


p = 20
c, phi, resid, sigma2 = fit_ar_ols(x, p)

print("Model AR(", p, ")", sep="")
print("Intercept c =", c)
print("Primii 5 coeficienti phi =", phi[:5])
print("Var reziduuri =", sigma2)


x_next = predict_ar(x, c, phi, steps=1)[0]
print("Predictie x[N] =", x_next)


plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(x)
plt.title("Seria de timp")

plt.subplot(3, 1, 2)
plt.plot(resid)
plt.title("Reziduuri AR(p)")

plt.subplot(3, 1, 3)
plt.stem(phi)
plt.title("Coeficienti AR(p)")

plt.tight_layout()
plt.show()
