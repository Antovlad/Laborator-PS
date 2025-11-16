import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


CSV_PATH = r"C:\Users\Antoniu\Desktop\Train.csv"

Fs = 1.0          
Ts = 1.0 / Fs



df = pd.read_csv(CSV_PATH)

if df.select_dtypes(include='number').shape[1] > 0:
    s = df.select_dtypes(include='number').iloc[:, 0]
else:
    s = pd.to_numeric(df.iloc[:, 0], errors='coerce')

data = s.dropna().to_numpy(dtype=float)

N_total = len(data)
print("Număr total de mostre în fișier:", N_total)


ore_3_zile = 3 * 24   
start = 0             



x = data[start:start + ore_3_zile]           
x = np.asarray(x, dtype=float)               
t = np.arange(len(x)) * Ts                   

plt.figure(figsize=(10, 4))
plt.plot(t, x)
plt.xlabel("t [ore]")
plt.ylabel("Număr vehicule")
plt.title("Semnal trafic - fereastră de 3 zile")
plt.grid(True)
plt.tight_layout()
plt.show()



def moving_average(x, w):
    window = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, window, mode='valid')

wins = [5, 9, 13, 17]

plt.figure(figsize=(10, 8))

plt.subplot(len(wins) + 1, 1, 1)
plt.plot(x, label="brut")
plt.ylabel("x[n]")
plt.legend()
plt.grid(True)

for i, w in enumerate(wins, start=2):
    y_ma = moving_average(x, w)
    plt.subplot(len(wins) + 1, 1, i)
    plt.plot(y_ma, label=f"medie alunecătoare, w={w}")
    plt.ylabel("y[n]")
    plt.legend()
    plt.grid(True)

plt.xlabel("n")
plt.tight_layout()
plt.show()


"""
Fs = 1 [eșantion/oră]  =>  f_Nyquist = Fs/2 = 0.5 [cicluri/oră]

Traficul are o componentă zilnică: perioada ~24h => f ≈ 1/24 ≈ 0.042 [cicluri/oră].
Alegem o frecvență de tăiere f_c ~ 1/6 ≈ 0.167 [cicluri/oră]
(păstrăm variațiile cu perioade > ~6 ore, filtrăm fluctuațiile foarte rapide).
"""
f_c = 1.0 / 6.0        
f_nyq = Fs / 2.0
Wn = f_c / f_nyq       

print("\n=== Parametri filtrare ===")
print("f_c       =", f_c, "cicluri/oră")
print("f_Nyquist =", f_nyq, "cicluri/oră")
print("Wn        =", Wn, "(frecvență normalizată)")



N = 5
rp = 5  

b_butt, a_butt = signal.butter(N, Wn, btype='low')
b_cheb, a_cheb = signal.cheby1(N, rp, Wn, btype='low')

w_butt, h_butt = signal.freqz(b_butt, a_butt)
w_cheb, h_cheb = signal.freqz(b_cheb, a_cheb)

freq = w_butt / np.pi * f_nyq

plt.figure(figsize=(10, 5))
plt.plot(freq, 20 * np.log10(np.abs(h_butt) + 1e-12), label="Butterworth")
plt.plot(freq, 20 * np.log10(np.abs(h_cheb) + 1e-12), label="Chebyshev (rp=5 dB)")
plt.axvline(f_c, color='k', linestyle='--', label="f_c")
plt.xlabel("Frecvență [cicluri/oră]")
plt.ylabel("Amplitudine [dB]")
plt.title("Răspuns în frecvență - filtre trece-jos")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

y_butt = signal.filtfilt(b_butt, a_butt, x)
y_cheb = signal.filtfilt(b_cheb, a_cheb, x)

plt.figure(figsize=(10, 5))
plt.plot(t, x, label="semnal brut", alpha=0.5)
plt.plot(t, y_butt, label="Butterworth", linewidth=2)
plt.plot(t, y_cheb, label="Chebyshev", linewidth=2)
plt.xlabel("t [ore]")
plt.ylabel("Număr vehicule")
plt.title("Filtrarea datelor de trafic")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""
Dintre cele două filtre, am ales filtrul Butterworth deoarece oferă un răspuns fără ondulații în banda de trecere și o netezire mai naturală a datelor de trafic
"""


ordere = [3, 5, 7]
for N_test in ordere:
    bB, aB = signal.butter(N_test, Wn, btype='low')
    yB = signal.filtfilt(bB, aB, x)

    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="brut", alpha=0.5)
    plt.plot(t, yB, label=f"Butterworth, N={N_test}", linewidth=2)
    plt.xlabel("t [ore]")
    plt.ylabel("Vehicule")
    plt.title(f"Butterworth - ordin {N_test}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

rp_vals = [1, 3, 5, 10]
for rp_test in rp_vals:
    bC, aC = signal.cheby1(5, rp_test, Wn, btype='low')  
    yC = signal.filtfilt(bC, aC, x)

    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="brut", alpha=0.5)
    plt.plot(t, yC, label=f"Chebyshev, N=5, rp={rp_test} dB", linewidth=2)
    plt.xlabel("t [ore]")
    plt.ylabel("Vehicule")
    plt.title(f"Chebyshev - rp={rp_test} dB")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
"""
În cadrul experimentelor de la punctul (f), am testat mai multe ordine ale filtrului Butterworth (N=3,5,7) și diferite valori ale ripplului pentru filtrul Chebyshev (rp = 1, 3, 5, 10 dB).
Am observat că Butterworth cu ordin mediu (N=5) oferă cea mai naturală netezire, fără ondulații. Chebyshev filtrează mai agresiv, dar introduce ondulații în banda de trecere atunci când rp este mare.
În consecință, filtrul optim pentru semnalul de trafic este Butterworth de ordin 5.
"""