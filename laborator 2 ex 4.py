import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth  

fs = 1000          
T = 1.0            
t = np.arange(0, T, 1/fs)


A1 = 1.0
f1 = 5.0           
phi1 = 0.0         
x1 = A1 * np.sin(2 * np.pi * f1 * t + phi1)


A2 = 1.0
f2 = 3.0           
phi2 = np.pi / 6   
x2 = A2 * sawtooth(2 * np.pi * f2 * t + phi2, width=1.0)



x_sum = x1 + x2


plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
plt.plot(t, x1, 'b')
plt.title('Semnal 1: Sinusoidal (A=1, f=5 Hz)')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, x2, 'orange')
plt.title('Semnal 2: Sawtooth (A=1, f=3 Hz)')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, x_sum, 'r')
plt.title('Suma celor două semnale')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.tight_layout()
plt.show()
