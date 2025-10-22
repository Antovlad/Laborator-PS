import numpy as np
import matplotlib.pyplot as plt


N = 256                 
f0 = 9                  
t = np.arange(N)
x = np.sin(2*np.pi*f0*t/N)  


def color_by_radius(z):
    r = np.abs(z)
    rng = np.ptp(r)  
    r = (r - np.min(r)) / (rng if rng != 0 else 1.0)
    return r


y = x * np.exp(-1j * 2*np.pi * t / N)  
c = color_by_radius(y)

fig1 = plt.figure(figsize=(9,4))

ax1 = fig1.add_subplot(1,2,1)
ax1.plot(t, x, linewidth=1.5)
ax1.set_xlabel("Timp (eşantioane)")
ax1.set_ylabel("Amplitudine")
ax1.grid(True, alpha=0.3)


ax2 = fig1.add_subplot(1,2,2)

ax2.plot(y.real, y.imag, alpha=0.25)

sc = ax2.scatter(y.real, y.imag, c=c, s=10)
ax2.set_xlabel("Real")
ax2.set_ylabel("Imaginar")
ax2.set_aspect('equal', 'box')
ax2.grid(True, alpha=0.3)
fig1.suptitle("Figura 1: Reprezentarea unui semnal în planul complex", y=0.02)
plt.tight_layout()


omegas = [1, 3, f0, 7] 
fig2, axes = plt.subplots(2, 2, figsize=(8,7))
axes = axes.ravel()

for ax, w in zip(axes, omegas):
    z = x * np.exp(-1j * 2*np.pi * w * t / N)  
    c = color_by_radius(z)
    ax.plot(z.real, z.imag, alpha=0.25)        
    ax.scatter(z.real, z.imag, c=c, s=8)       
    
    z_sum = np.sum(z) / N
    ax.plot([0, z_sum.real], [0, z_sum.imag], linewidth=2)
    ax.scatter([z_sum.real], [z_sum.imag], s=30)
    ax.set_title(r"$\omega = {}$".format(w))
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginar")
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)

fig2.suptitle("Figura 2: Reprezentarea transformatei Fourier în planul complex", y=0.02)
plt.tight_layout()
plt.show()
