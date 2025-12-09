import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t + 5
season = 3*np.sin(2*np.pi*t/50) + 2*np.sin(2*np.pi*t/120)
noise = np.random.normal(0, 1, N)

x = trend + season + noise

plt.figure(figsize=(12,7))
plt.subplot(4,1,1); plt.plot(x);      plt.title("Seria de timp x")
plt.subplot(4,1,2); plt.plot(trend);  plt.title("Trend (grad 2)")
plt.subplot(4,1,3); plt.plot(season); plt.title("Sezon (2 frecvente)")
plt.subplot(4,1,4); plt.plot(noise);  plt.title("Zgomot alb gaussian")
plt.tight_layout()
plt.show()

def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred)**2)

alpha_grid = np.linspace(0.01, 0.99, 50)


print("\n=== MEDIERE EXPONENTIALA SIMPLA (SES) ===")

alpha_fixed = 0.3  
ses_fixed = SimpleExpSmoothing(x).fit(smoothing_level=alpha_fixed,
                                      optimized=False)
x_ses_fixed = ses_fixed.fittedvalues
mse_ses_fixed = mse(x, x_ses_fixed)
print(f"SES cu alpha fix = {alpha_fixed:.2f}, MSE = {mse_ses_fixed:.4f}")

best_alpha_ses = None
best_mse_ses = np.inf
for a in alpha_grid:
    m = SimpleExpSmoothing(x).fit(smoothing_level=a, optimized=False)
    x_hat = m.fittedvalues
    err = mse(x, x_hat)
    if err < best_mse_ses:
        best_mse_ses = err
        best_alpha_ses = a

ses_opt = SimpleExpSmoothing(x).fit(smoothing_level=best_alpha_ses,
                                    optimized=False)
x_ses_opt = ses_opt.fittedvalues
print(f"SES alpha optim = {best_alpha_ses:.4f}, MSE = {best_mse_ses:.4f}")


print("\n=== MEDIERE EXPONENTIALA DUBLA (HOLT) ===")

beta_fixed = 0.2  

holt_fixed = Holt(x).fit(smoothing_level=alpha_fixed,
                         smoothing_trend=beta_fixed,
                         optimized=False)
x_holt_fixed = holt_fixed.fittedvalues
mse_holt_fixed = mse(x, x_holt_fixed)
print(f"Holt cu alpha fix = {alpha_fixed:.2f}, beta fix = {beta_fixed:.2f}, "
      f"MSE = {mse_holt_fixed:.4f}")

best_alpha_holt = None
best_mse_holt = np.inf
for a in alpha_grid:
    m = Holt(x).fit(smoothing_level=a,
                    smoothing_trend=beta_fixed,
                    optimized=False)
    x_hat = m.fittedvalues
    err = mse(x, x_hat)
    if err < best_mse_holt:
        best_mse_holt = err
        best_alpha_holt = a

holt_opt = Holt(x).fit(smoothing_level=best_alpha_holt,
                       smoothing_trend=beta_fixed,
                       optimized=False)
x_holt_opt = holt_opt.fittedvalues
print(f"Holt alpha optim = {best_alpha_holt:.4f} (beta = {beta_fixed:.2f}), "
      f"MSE = {best_mse_holt:.4f}")


print("\n=== MEDIERE EXPONENTIALA TRIPLA (HOLT–WINTERS) ===")

seasonal_periods = 50   
beta_hw_fixed = 0.2
gamma_hw_fixed = 0.2

hw_fixed = ExponentialSmoothing(
    x,
    trend="add",
    seasonal="add",
    seasonal_periods=seasonal_periods
).fit(smoothing_level=alpha_fixed,
      smoothing_trend=beta_hw_fixed,
      smoothing_seasonal=gamma_hw_fixed,
      optimized=False)

x_hw_fixed = hw_fixed.fittedvalues
mse_hw_fixed = mse(x, x_hw_fixed)
print("HW cu alpha fix = {:.2f}, beta = {:.2f}, gamma = {:.2f}, MSE = {:.4f}"
      .format(alpha_fixed, beta_hw_fixed, gamma_hw_fixed, mse_hw_fixed))

best_alpha_hw = None
best_mse_hw = np.inf
for a in alpha_grid:
    m = ExponentialSmoothing(
        x,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods
    ).fit(smoothing_level=a,
          smoothing_trend=beta_hw_fixed,
          smoothing_seasonal=gamma_hw_fixed,
          optimized=False)
    x_hat = m.fittedvalues
    err = mse(x, x_hat)
    if err < best_mse_hw:
        best_mse_hw = err
        best_alpha_hw = a

hw_opt = ExponentialSmoothing(
    x,
    trend="add",
    seasonal="add",
    seasonal_periods=seasonal_periods
).fit(smoothing_level=best_alpha_hw,
      smoothing_trend=beta_hw_fixed,
      smoothing_seasonal=gamma_hw_fixed,
      optimized=False)

x_hw_opt = hw_opt.fittedvalues
print("HW alpha optim = {:.4f} (beta = {:.2f}, gamma = {:.2f}), MSE = {:.4f}"
      .format(best_alpha_hw, beta_hw_fixed, gamma_hw_fixed, best_mse_hw))


plt.figure(figsize=(12,10))

# SES
plt.subplot(3,1,1)
plt.plot(x, label="Original", alpha=0.4)
plt.plot(x_ses_fixed, label=f"SES alpha fix = {alpha_fixed:.2f}")
plt.plot(x_ses_opt,  label=f"SES alpha optim = {best_alpha_ses:.2f}", linestyle="--")
plt.title("Mediere exponentiala simpla (SES)")
plt.legend()

# Holt
plt.subplot(3,1,2)
plt.plot(x, label="Original", alpha=0.4)
plt.plot(x_holt_fixed, label=f"Holt alpha fix = {alpha_fixed:.2f}")
plt.plot(x_holt_opt,  label=f"Holt alpha optim = {best_alpha_holt:.2f}", linestyle="--")
plt.title("Mediere exponentiala dubla (Holt)")
plt.legend()

# Holt–Winters
plt.subplot(3,1,3)
plt.plot(x, label="Original", alpha=0.4)
plt.plot(x_hw_fixed, label=f"HW alpha fix = {alpha_fixed:.2f}")
plt.plot(x_hw_opt,  label=f"HW alpha optim = {best_alpha_hw:.2f}", linestyle="--")
plt.title("Mediere exponentiala tripla (Holt–Winters)")
plt.legend()

plt.tight_layout()
plt.show()
