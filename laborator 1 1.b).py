import numpy as np
import matplotlib.pyplot as plt


t = np.arange(0, 0.03 + 0.0005, 0.0005)


x = np.cos(520 * np.pi * t + np.pi / 3)
y = np.cos(280 * np.pi * t - np.pi / 3)
z = np.cos(120 * np.pi * t + np.pi / 3)


plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, x, 'b')
plt.title('x(t) = cos(520πt + π/3)')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, y, 'r')
plt.title('y(t) = cos(280πt - π/3)')
plt.xlabel('t [s]')
plt.ylabel('y(t)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, z, 'g')
plt.title('z(t) = cos(120πt + π/3)')
plt.xlabel('t [s]')
plt.ylabel('z(t)')
plt.grid(True)

plt.tight_layout()
plt.show()
