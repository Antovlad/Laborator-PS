import numpy as np


np.random.seed(0)


n = 20
k = np.arange(n)


x = np.sin(2 * np.pi * 3 * k / n) + 0.1 * np.random.randn(n)

d = np.random.randint(0, n)   
y = np.roll(x, d)

print("Vectorul x:", np.round(x, 3))
print("Vectorul y (x deplasat cu d pozitii):", np.round(y, 3))
print("Deplasarea reala d =", d)
print("-" * 60)


X = np.fft.fft(x)
Y = np.fft.fft(y)

R_xy = np.fft.ifft(np.conj(X) * Y)

lags = np.arange(n)
d_est1 = np.argmax(np.abs(R_xy))

print("Metoda 1: IFFT(conj(FFT(x)) * FFT(y))  - corelare")
print("Magnitudinea corelatiei:", np.round(np.abs(R_xy), 3))
print("Deplasare estimata (pozitia maximului) =", d_est1)
print("-" * 60)


eps = 1e-10
ratio = Y / (X + eps)
delta = np.fft.ifft(ratio)

d_est2 = np.argmax(np.abs(delta))

print("Metoda 2: IFFT(FFT(y) / FFT(x))  - \"deconvolutie\"")
print("Magnitudinea rezultatului:", np.round(np.abs(delta), 3))
print("Deplasare estimata (pozitia maximului) =", d_est2)
print("-" * 60)

print("Comparatie:")
print(f"  d real   = {d}")
print(f"  d_est1   = {d_est1}  (din corelare)")
print(f"  d_est2   = {d_est2}  (din impartire)")

#Metoda 1 (cu conj(FFT(x)) * FFT(y)) îți dă un vector de corelație circulară, care are un vârf (maxim) la lag-ul d
#Metoda 2 (cu FFT(y) / FFT(x)) îți dă în mod ideal ceva foarte apropiat de un impuls (un singur maxim pronunțat) la poziția d



