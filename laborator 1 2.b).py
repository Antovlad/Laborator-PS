import numpy as np
import matplotlib.pyplot as plt

# Semnal sinusoidal de 800 Hz, durată 3 secunde
Fs = 8000
Ts = 1 / Fs
durata = 3.0
t = np.arange(0, durata, Ts)
x = np.sin(2 * np.pi * 800 * t)

plt.figure()
plt.plot(t, x)
plt.title('b) Sinus 800 Hz, durată 3 s')
plt.xlabel('t [s]')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.show()
