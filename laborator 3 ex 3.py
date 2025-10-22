import numpy as np
import matplotlib.pyplot as plt


fs = 1000             
T  = 1.0              
N  = int(fs*T)        


freqs_sig = np.array([7, 23,  eighty := 80]) 
amps_sig  = np.array([1.5, 3.0,  1.0])        
phases    = np.array([0.0, 0.0,  0.0])        

t = np.arange(N) / fs


x = np.zeros(N, dtype=float)
for A, f, phi in zip(amps_sig, freqs_sig, phases):
    x += A * np.sin(2*np.pi*f*t + phi)


X_loop = np.zeros(N, dtype=complex)
for k in range(N):  
    s = 0.0 + 0.0j
    for n in range(N):
        s += x[n] * np.exp(-2j*np.pi*k*n/N)
    X_loop[k] = s


k = np.arange(N)
n = k.reshape(N, 1)
F = np.exp(-2j*np.pi*k*n/N)          
X_mat = F @ x                         


assert np.allclose(X_loop, X_mat, atol=1e-8)


freq_axis = np.arange(N) * fs / N
mag = np.abs(X_loop)


bins_of_components = np.round(freqs_sig * N / fs).astype(int)


plt.figure(figsize=(11,4.5))


plt.subplot(1,2,1)
plt.plot(t, x, linewidth=1.2)
plt.xlabel("Timp (s)")
plt.ylabel("x(t)")
plt.title("Semnal compus (≥ 3 componente)")
plt.grid(True, alpha=0.3)


plt.subplot(1,2,2)

half = N//2
try:
    markerline, stemlines, baseline = plt.stem(freq_axis[:half], mag[:half], basefmt=" ", use_line_collection=True)
except TypeError:
    markerline, stemlines, baseline = plt.stem(freq_axis[:half], mag[:half], basefmt=" ")
plt.xlabel("Frecvența (Hz)")
plt.ylabel(r"$|X(\omega)|$")
plt.title("Modulul DFT (relația (1))")
plt.grid(True, alpha=0.3)


for f, kbin in zip(freqs_sig, bins_of_components):
    if kbin < half:
        plt.axvline(freq_axis[kbin], linestyle="--", linewidth=1)
        plt.text(freq_axis[kbin]+0.5, mag[:half].max()*0.9, f"{f:.0f} Hz", rotation=90, va='top')

plt.tight_layout()
plt.show()
