import numpy as np
import matplotlib.pyplot as plt


Fs = 8000            
Ts = 1 / Fs
N = 1600
t = np.arange(N) * Ts
x = np.sin(2 * np.pi * 400 * t)

plt.figure()
plt.plot(t, x)
plt.title('a) Sinus 400 Hz, 1600 eșantioane')
plt.xlabel('t [s]')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.show()
