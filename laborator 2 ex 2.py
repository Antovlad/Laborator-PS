import numpy as np
import matplotlib.pyplot as plt


A = 1.0            
f = 5.0            
fs = 1000          
T = 1.0            
t = np.arange(0, T, 1/fs)


phases = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]


signals = [A * np.sin(2*np.pi*f*t + phi) for phi in phases]

plt.figure(figsize=(10, 5))
for phi, x in zip(phases, signals):
    plt.plot(t, x, label=fr'$\varphi={phi:.2f}$ rad')
plt.title('Semnale sinusoidale (A=1, f=5Hz, 4 faze diferite)')
plt.xlabel('t [s]')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()  


rng = np.random.default_rng(42)
x_ref = signals[0]  

target_snrs = [0.1, 1.0, 10.0, 100.0]  
noisy_signals = []
measured_snrs = []

z = rng.normal(loc=0.0, scale=1.0, size=len(t))
x_norm2 = np.linalg.norm(x_ref)**2
z_norm2 = np.linalg.norm(z)**2

for snr in target_snrs:
    gamma = np.sqrt(x_norm2 / (snr * z_norm2))
    x_noisy = x_ref + gamma * z
    noisy_signals.append((snr, x_noisy))
    snr_meas = (np.linalg.norm(x_ref)**2) / (np.linalg.norm(gamma*z)**2)
    measured_snrs.append(snr_meas)


print("\nRezultate SNR:")
for snr_t, snr_m in zip(target_snrs, measured_snrs):
    print(f" SNR țintă = {snr_t:>6} | SNR măsurat = {snr_m:>10.4f}")


plt.figure(figsize=(10, 6))
plt.plot(t, x_ref, linewidth=2, label='Semnal curat')
for snr, x_n in noisy_signals:
    plt.plot(t, x_n, alpha=0.8, label=f'cu zgomot (SNR={snr:g})')
plt.title('Semnal cu zgomot gaussian pentru SNR-uri impuse')
plt.xlabel('t [s]')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()  
