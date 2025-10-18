import numpy as np
import matplotlib.pyplot as plt


fs = 1000      
T = 1.0       
t = np.arange(0, T, 1/fs)


f1 = fs / 2    
f2 = fs / 4
f3 = 0.0       


x1 = np.sin(2 * np.pi * f1 * t)  
x2 = np.sin(2 * np.pi * f2 * t)  
x3 = np.sin(2 * np.pi * f3 * t) 


plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
plt.plot(t, x1, 'r')
plt.title(f'(a) f = fs/2 = {f1:.1f} Hz')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, x2, 'g')
plt.title(f'(b) f = fs/4 = {f2:.1f} Hz')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, x3, 'b')
plt.title(f'(c) f = 0 Hz')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.tight_layout()
plt.show()



