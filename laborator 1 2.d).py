import numpy as np
import matplotlib.pyplot as plt


Fs = 8000
Ts = 1 / Fs
f = 300
durata = 0.05
t = np.arange(0, durata, Ts)

x = np.sign(np.sin(2 * np.pi * f * t))
x[x == 0] = 1  

plt.figure()
plt.plot(t, x)
plt.title('d) Square 300 Hz')
plt.xlabel('t [s]')
plt.ylabel('Amplitudine')
plt.ylim(-1.2, 1.2)
plt.grid(True)
plt.show()
