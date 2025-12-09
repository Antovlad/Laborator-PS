import numpy as np
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t + 5
season = 3*np.sin(2*np.pi*t/50) + 2*np.sin(2*np.pi*t/120)
noise  = np.random.normal(0, 1, N)

x = trend + season + noise

def fit_arma(x, p, q):
    
    model = ARIMA(x, order=(p, 0, q))
    res = model.fit()
    x_hat = res.fittedvalues
    eps = x - x_hat
    return res, x_hat, eps  


p_max = 20
q_max = 20

best_aic = np.inf
best_order = None
best_result = None
best_x_hat = None
best_eps = None

for p in range(0, p_max + 1):
    for q in range(0, q_max + 1):
        if p == 0 and q == 0:
            continue
        try:
            res, x_hat, eps = fit_arma(x, p, q)
            aic = res.aic
            print(f"Model ARMA({p},{q}) -> AIC = {aic:.2f}")
            if aic < best_aic:
                best_aic = aic
                best_order = (p, q)
                best_result = res
                best_x_hat = x_hat
                best_eps = eps
        except Exception as e:
            print(f"Model ARMA({p},{q}) a esuat: {e}")
            continue

print("\n======================================")
print(f"Cel mai bun model (dupa AIC): ARMA{best_order} cu AIC = {best_aic:.2f}")
print(best_result.summary())


eps = best_eps
mse_eps = np.mean(eps**2)
print(f"\nMSE al erorilor epsilon pentru ARMA{best_order} = {mse_eps:.4f}")


plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(x, label="x (original)")
plt.title("Seria de timp originala")
plt.legend()

plt.subplot(3,1,2)
plt.plot(x,      label="x (original)", alpha=0.4)
plt.plot(best_x_hat, label=f"ARMA{best_order} - valori estimate")
plt.title("Model ARMA - serie estimata")
plt.legend()

plt.subplot(3,1,3)
plt.plot(eps, label=r"epsilon[i] = x[i] - x_hat[i]")
plt.axhline(0, color="k", linewidth=1)
plt.title("Termenii de eroare ε[i]")
plt.legend()

plt.tight_layout()
plt.show()
