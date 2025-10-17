import numpy as np
import matplotlib.pyplot as plt


Fs = 200       
Ts = 1 / Fs   
t = np.arange(0, 0.03 + Ts, Ts) 


x = np.cos(520 * np.pi * t + np.pi / 3)
y = np.cos(280 * np.pi * t - np.pi / 3)
z = np.cos(120 * np.pi * t + np.pi / 3)


plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.stem(t, x, basefmt=" ")
plt.title('x[n] = cos(520πnTs + π/3)')
plt.xlabel('t [s]')
plt.ylabel('x[n]')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(t, y, basefmt=" ")
plt.title('y[n] = cos(280πnTs - π/3)')
plt.xlabel('t [s]')
plt.ylabel('y[n]')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(t, z, basefmt=" ")
plt.title('z[n] = cos(120πnTs + π/3)')
plt.xlabel('t [s]')
plt.ylabel('z[n]')
plt.grid(True)

plt.tight_layout()
plt.show()
