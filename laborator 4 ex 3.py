import numpy as np
import matplotlib.pyplot as plt


fs_sub = 200.0      
f0 = 850.0          
A, phi = 1.0, 0.0

f_alt1 = f0 - fs_sub          
f_alt2 = -f0 + 5*fs_sub       


fs_hi = 3000.0  
T_view = 0.03   

t  = np.linspace(0, T_view, 4000, endpoint=True)    
n  = np.arange(0, int(T_view * fs_hi) + 1)          
tn = n / fs_hi

def x_ct(f, t, A=1.0, phi=0.0):
    return A * np.cos(2*np.pi*f*t + phi)

x0_c    = x_ct(f0,     t, A, phi)
xalt1_c = x_ct(f_alt1, t, A, phi)
xalt2_c = x_ct(f_alt2, t, A, phi)

s0    = x_ct(f0,     tn, A, phi)
salt1 = x_ct(f_alt1, tn, A, phi)
salt2 = x_ct(f_alt2, tn, A, phi)


fig, axs = plt.subplots(4, 1, figsize=(7, 6), sharex=True)

axs[0].plot(t, x0_c, linewidth=2, alpha=0.7)
axs[0].plot(tn, s0, 'o', markersize=7, markerfacecolor='gold', markeredgecolor='gold')
axs[0].set_ylim(-1.1, 1.1)
axs[0].grid(True, linestyle=':')
axs[0].set_title('Eșantionare peste Nyquist – nu apare aliasing')
axs[0].text(T_view*0.995, 0.75, f'f0 = {f0:g} Hz', ha='right')

axs[1].plot(t, xalt1_c, linewidth=2, alpha=0.7)
axs[1].plot(tn, salt1, 'o', markersize=7, markerfacecolor='gold', markeredgecolor='gold')
axs[1].set_ylim(-1.1, 1.1)
axs[1].grid(True, linestyle=':')
axs[1].text(T_view*0.995, 0.75, f'f_alt1 = {f_alt1:g} Hz', ha='right')

axs[2].plot(t, xalt2_c, linewidth=2, alpha=0.7)
axs[2].plot(tn, salt2, 'o', markersize=7, markerfacecolor='gold', markeredgecolor='gold')
axs[2].set_ylim(-1.1, 1.1)
axs[2].grid(True, linestyle=':')
axs[2].text(T_view*0.995, 0.75, f'f_alt2 = {f_alt2:g} Hz', ha='right')

axs[3].stem(tn, s0,    basefmt=" ", linefmt='C0-', markerfmt='C0o')
axs[3].stem(tn, salt1, basefmt=" ", linefmt='C1-', markerfmt='C1o')
axs[3].stem(tn, salt2, basefmt=" ", linefmt='C2-', markerfmt='C2o')
axs[3].set_ylim(-1.1, 1.1)
axs[3].grid(True, linestyle=':')
axs[3].set_xlabel('timp [s]')
axs[3].text(T_view*0.995, 0.75, f'fs = {fs_hi:g} Hz (> 2·max(f))', ha='right')

plt.tight_layout()
plt.show()
