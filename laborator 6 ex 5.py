import numpy as np
import matplotlib.pyplot as plt



def rect_window(N):
    """Fereastră dreptunghiulară"""
    return np.ones(N)

def hanning_window(N):
    """Fereastră Hanning (Hann)"""
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))




Nw = 200       
fs = 10_000    
f = 100        
A = 1          
phi = 0        

t = np.arange(Nw) / fs      
x = A * np.sin(2 * np.pi * f * t + phi)  




w_rect = rect_window(Nw)
w_hann = hanning_window(Nw)

x_rect = x * w_rect
x_hann = x * w_hann




plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(x_rect, label="Sinusoidă cu fereastră dreptunghiulară")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x_hann, label="Sinusoidă cu fereastră Hanning")
plt.grid(True)
plt.legend()

plt.suptitle("Aplicarea ferestrelor pe o sinusoidă (f = 100 Hz, Nw = 200)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
