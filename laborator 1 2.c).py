import numpy as np
import matplotlib.pyplot as plt


Fs = 8000
Ts = 1 / Fs
f = 240
T = 1 / f
durata = 0.05
t = np.arange(0, durata, Ts)

# formula sawtooth în [-1, 1]
x = 2 * ((t / T) - np.floor(0.5 + (t / T)))

plt.figure()
plt.plot(t, x)
plt.title('c) Sawtooth 240 Hz')
plt.xlabel('t [s]')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.show()
