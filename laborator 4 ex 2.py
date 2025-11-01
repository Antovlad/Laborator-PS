import numpy as np
import matplotlib.pyplot as plt


fs = 200.0          
f0 = 850.0          
A  = 1.0            
phi = 0.0           
T_view = 0.03       


def alias_frequency(f, fs):
   
    f_wrapped = ((f + fs/2) % fs) - fs/2
    return abs(f_wrapped)

f_alias = alias_frequency(f0, fs)   
f_alt1  = f0 - fs                   
f_alt2  = -f0 + 5*fs               


t = np.linspace(0, T_view, 3000, endpoint=True) 
n = np.arange(0, int(T_view*fs) + 1)            
tn = n/fs                                        

def x_ct(f, t, A=1.0, phi=0.0):
    return A * np.cos(2*np.pi*f*t + phi)


x0_c = x_ct(f0,     t, A, phi)
x1_c = x_ct(f_alt1, t, A, phi)
x2_c = x_ct(f_alt2, t, A, phi)
xa_c = x_ct(f_alias,t, A, phi)      


samples = x_ct(f0, tn, A, phi)


fig, axs = plt.subplots(4, 1, figsize=(7, 6), sharex=True)


axs[0].plot(t, x0_c, linewidth=2, alpha=0.6)
axs[0].set_ylim(-1.1, 1.1)


axs[1].plot(t, x0_c, linewidth=2, alpha=0.6)
axs[1].plot(tn, samples, 'o', markersize=8,
            markerfacecolor='gold', markeredgecolor='gold')


axs[3-1].plot(t, x1_c, linewidth=2)
axs[3-1].plot(tn, samples, 'o', markersize=8,
              markerfacecolor='gold', markeredgecolor='gold')


axs[3].plot(t, xa_c, linewidth=2)
axs[3].plot(tn, samples, 'o', markersize=8,
            markerfacecolor='gold', markeredgecolor='gold')

for ax in axs:
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_yticks([-1, 0, 1])

axs[-1].set_xlabel('timp [s]')
axs[0].set_title('Fenomenul de aliere — trei frecvențe diferite, aceleași mostre')


axs[0].text(T_view*0.995, 0.75, f'f0 = {f0:g} Hz', ha='right')
axs[1].text(T_view*0.995, 0.75, f'fs = {fs:g} Hz', ha='right')
axs[2].text(T_view*0.995, 0.75, f'alias #1: {f_alt1:g} Hz', ha='right')
axs[3].text(T_view*0.995, 0.75, f'aparent: {f_alias:g} Hz', ha='right')

plt.tight_layout()
plt.show()
