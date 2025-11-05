

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\Antoniu\Desktop\Train.csv"  
df = pd.read_csv(CSV_PATH)

assert {"Datetime", "Count"}.issubset(df.columns), "Fișierul trebuie să conțină coloanele 'Datetime' și 'Count'"

df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d-%m-%Y %H:%M")
x = df["Count"].astype(float).to_numpy()
N = len(x)

deltas = df["Datetime"].diff().dropna()
dt_seconds = deltas.dt.total_seconds().median() 
Fs_hz = 1.0 / dt_seconds              
Fs_per_hour = Fs_hz * 3600.0          

print("(a) Frecvența de eșantionare:")
print(f"   Fs = {Fs_per_hour:.6g} / oră  = {Fs_hz:.10f} Hz")
print()

start = df["Datetime"].min()
end = df["Datetime"].max()
span = end - start
print("(b) Intervalul de timp acoperit:")
print(f"   start = {start}, end = {end}, durată = {span}")
print()

fmax_hz = Fs_hz / 2.0
fmax_cpd = fmax_hz * 86400.0  # cicluri/zi
print("(c) Frecvența maximă (Nyquist):")
print(f"   f_max = {fmax_hz:.10f} Hz = {fmax_cpd:.6g} cicluri/zi")
print()

mean_count = np.mean(x)
std_count = np.std(x)
print("(e) Componenta continuă (DC):")
print(f"   media = {mean_count:.4f}, abatere standard = {std_count:.4f}")
print("   DC prezentă? ", "DA" if abs(mean_count) > 1e-9 else "NU")
print("   O eliminăm prin centrare (x - mean).")
x_detrended = x - mean_count
print()

Fs = Fs_hz  
X = np.fft.fft(x_detrended)
freqs = np.fft.fftfreq(N, d=1.0 / Fs) 
half = N // 2
freqs_pos = freqs[:half]
mag = np.abs(X[:half]) / N

plt.figure()
plt.plot(freqs_pos, mag)
plt.xlabel("Frecvență (Hz)")
plt.ylabel("|X(f)| / N")
plt.title("Modulul transformatei Fourier (fără componenta DC)")
plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

cycles_per_day = freqs_pos * 86400.0
spec = pd.DataFrame({
    "freq_hz": freqs_pos,
    "cycles_per_day": cycles_per_day,
    "magnitude": mag
})


mask_meaningful = spec["cycles_per_day"] > 0.1
peaks = spec[mask_meaningful].nlargest(4, "magnitude").copy()
peaks["period_hours"] = (1.0 / peaks["freq_hz"]) / 3600.0
print("(f) Primele 4 vârfuri (după magnitudine), excluzând < 0.1 cicluri/zi:")
print(peaks[["freq_hz", "cycles_per_day", "period_hours", "magnitude"]].to_string(index=False))
print("""
   Interpretare uzuală:
     ~1 ciclu/zi  -> periodicitate zilnică (program lucru/casă).
     ~2 cicluri/zi -> vârfuri dimineață/seară.
     ~1/7 cicluri/zi (~0.143) -> periodicitate săptămânală (weekend vs. zile lucrătoare).
""")

monday_idxs = df.index[df["Datetime"].dt.dayofweek == 0]  
if len(monday_idxs) == 0:
    raise RuntimeError("Nu am găsit nicio zi de luni în setul de date.")
start_idx = int(monday_idxs[0])
end_idx = min(start_idx + 1000, N)
seg = df.iloc[start_idx:end_idx]

plt.figure()
plt.plot(seg["Datetime"], seg["Count"])
plt.xlabel("Timp")
plt.ylabel("Nr. mașini")
plt.title("O lună de trafic (~1000 eșantioane) începând de luni")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("(g) Fereastra afișată pentru ~o lună (1000 eșantioane):")
print(f"   {seg['Datetime'].iloc[0]}  →  {seg['Datetime'].iloc[-1]} (N={len(seg)})")
print()

method_text = """
(h) Metodă propusă (doar din semnal):
  1) Estimează perioada săptămânală din autocorelație (vârf în jur de 7*24 h).
  2) Construiește un "șablon săptămânal": media pe ore pe mai multe săptămâni
     (profil oră-din-săptămână, 0..167).
  3) Găsește faza optimă prin cross-corelare între șablon și seria brută
     (deplasarea care maximizează corelația). Ora 0 a zilei cu nivel minim
     (de regulă duminică noaptea) dă reperul; următoarea zi este Luni 00:00.
  4) Etichetezi toate eșantioanele în consecință.
  Limitări: presupune stabilitate a tiparului săptămânal; sărbătorile/vacanțele
  pot deplasa faza; rezoluția maximă este de o oră.
"""
print(method_text)


window_hours = 24
kernel = np.ones(window_hours) / window_hours
x_filt = np.convolve(x, kernel, mode="same")

two_weeks = 24 * 14
idx2 = start_idx + two_weeks
seg2 = df.iloc[start_idx:idx2]

plt.figure()
plt.plot(seg2["Datetime"], seg2["Count"], label="original")
plt.plot(seg2["Datetime"], x_filt[start_idx:idx2], label=f"filtrat (media mobilă {window_hours}h)")
plt.xlabel("Timp")
plt.ylabel("Nr. mașini")
plt.title("Semnal original vs. filtrat (frecvențe înalte eliminate)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


