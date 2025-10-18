import numpy as np
import matplotlib.pyplot as plt


A = 2.0        
f = 5.0        
phi = 0.7      


fs = 1000                  
T = 1.0                    
t = np.arange(0, T, 1/fs)  


sine = A * np.sin(2 * np.pi * f * t + phi)


cosine = A * np.cos(2 * np.pi * f * t + (phi - np.pi/2)) 


plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, sine, 'b')
plt.title('Semnal sinusoidal')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, cosine, 'r')
plt.title('Semnal cosinus ajustat (identic cu sinusul)')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.tight_layout()
plt.show()



