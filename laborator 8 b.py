import numpy as np
import matplotlib.pyplot as plt



N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t + 5

season = 3 * np.sin(2 * np.pi * t / 50) + 2 * np.sin(2 * np.pi * t / 120)

noise = np.random.normal(0, 1, N)

x = trend + season + noise



def autocorr_numpy(x, max_lag):
    x = np.asarray(x)
    x_centered = x - np.mean(x)

    corr_full = np.correlate(x_centered, x_centered, mode='full')

    mid = corr_full.size // 2
    corr = corr_full[mid:mid + max_lag + 1]

    corr = corr / corr[0]
    return corr

max_lag = 50
r = autocorr_numpy(x, max_lag)

plt.figure(figsize=(10,4))
plt.stem(range(max_lag + 1), r)
plt.title("Autocorelația seriei de timp (NumPy)")
plt.xlabel("Lag")
plt.ylabel("r(k)")
plt.grid(True)
plt.show()
