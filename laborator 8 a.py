import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t + 5

season = 3*np.sin(2*np.pi*t/50) + 2*np.sin(2*np.pi*t/120)

noise = np.random.normal(0, 1, N)

x = trend + season + noise

plt.figure(figsize=(12,7))
plt.subplot(4,1,1); plt.plot(x); plt.title("Seria de timp")
plt.subplot(4,1,2); plt.plot(trend); plt.title("Trend (grad 2)")
plt.subplot(4,1,3); plt.plot(season); plt.title("Sezon (2 frecvențe)")
plt.subplot(4,1,4); plt.plot(noise); plt.title("Zgomot alb gaussian")
plt.tight_layout()
plt.show()
