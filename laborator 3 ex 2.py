import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


N  = 500               
fs = 1000              
f0 = 7                
t  = np.arange(N) / fs
x  = np.sin(2*np.pi*f0*t)

def color_by_radius(z):
    
    r = np.abs(z)
    rng = np.ptp(r)                     
    return (r - np.min(r)) / (rng if rng else 1.0)


y = x * np.exp(-2j*np.pi*t)
c = color_by_radius(y)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))


ax1.plot(t, x, linewidth=1.5)
ax1.set_xlabel("Timp (eșantioane)")
ax1.set_ylabel("Amplitudine")
ax1.set_title("Semnal sinusoidal în timp")
ax1.grid(True, alpha=0.3)


ax2.plot(y.real, y.imag, alpha=0.25)
sc1 = ax2.scatter(y.real, y.imag, c=c, s=10)
ax2.set_xlabel("Real")
ax2.set_ylabel("Imaginar")
ax2.set_aspect('equal', 'box')
ax2.set_title("Reprezentarea în planul complex")
ax2.grid(True, alpha=0.3)

fig1.suptitle("Figura 1: Reprezentarea unui semnal în planul complex (statică)", y=0.02)
plt.tight_layout()

omegas = [1, 2, f0, 5]
fig2, axes = plt.subplots(2, 2, figsize=(8, 7))
axes = axes.ravel()

for ax, w in zip(axes, omegas):
    z = x * np.exp(-2j*np.pi*w*t)
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

fig2.suptitle("Figura 2: Reprezentarea transformatei Fourier în planul complex (statică)", y=0.02)
plt.tight_layout()

fig_anim1, (ax1a, ax2a) = plt.subplots(1, 2, figsize=(10, 4))


ax1a.plot(t, x, color='green')
dot_time, = ax1a.plot([], [], 'ro', markersize=8)
ax1a.set_xlabel('Timp (eșantioane)')
ax1a.set_ylabel('Amplitudine')
ax1a.set_xlim(t[0], t[-1])
ax1a.set_ylim(-1.1, 1.1)
ax1a.set_title('Semnal sinusoidal în timp')
ax1a.grid(True, alpha=0.3)


ax2a.plot(np.real(y), np.imag(y), color='lightblue')
dot_complex, = ax2a.plot([], [], 'ro', markersize=8)
ax2a.set_xlabel('Real')
ax2a.set_ylabel('Imaginar')
ax2a.set_xlim(-1.1, 1.1)
ax2a.set_ylim(-1.1, 1.1)
ax2a.set_aspect('equal')
ax2a.set_title('Reprezentare în planul complex')
ax2a.grid(True, alpha=0.3)


fig_anim2, axes2a = plt.subplots(2, 2, figsize=(8, 7))
axes2a = axes2a.ravel()
z_list, dots = [], []

for i, w in enumerate(omegas):
    z = x * np.exp(-2j*np.pi*w*t)
    z_list.append(z)
    axes2a[i].plot(np.real(z), np.imag(z), color='violet', alpha=0.5)
    d, = axes2a[i].plot([], [], 'ko', markersize=6)
    dots.append(d)
    axes2a[i].set_title(r"$\omega = {}$".format(w))
    axes2a[i].set_xlabel("Real")
    axes2a[i].set_ylabel("Imaginar")
    axes2a[i].set_xlim(-1.1, 1.1)
    axes2a[i].set_ylim(-1.1, 1.1)
    axes2a[i].set_aspect('equal')
    axes2a[i].grid(True, alpha=0.3)


def init_all():
    dot_time.set_data([], [])
    dot_complex.set_data([], [])
    for d in dots:
        d.set_data([], [])
    return [dot_time, dot_complex, *dots]

def update_all(frame):
    
    dot_time.set_data([t[frame]], [x[frame]])
    dot_complex.set_data([np.real(y[frame])], [np.imag(y[frame])])
    
    for d, z in zip(dots, z_list):
        d.set_data([np.real(z[frame])], [np.imag(z[frame])])
    return [dot_time, dot_complex, *dots]

ani_all1 = FuncAnimation(fig_anim1, update_all, frames=len(t),
                         init_func=init_all, interval=20, blit=True)
ani_all2 = FuncAnimation(fig_anim2, update_all, frames=len(t),
                         init_func=init_all, interval=20, blit=True)

plt.tight_layout()
plt.show()
