import numpy as np


N = 5              
coef_min = -5      
coef_max = 5


p = np.random.randint(coef_min, coef_max + 1, size=N + 1)
q = np.random.randint(coef_min, coef_max + 1, size=N + 1)

print("Coeficienți p(x):", p)
print("Coeficienți q(x):", q)


r_direct = np.convolve(p, q)
print("\nProdus direct (np.convolve):")
print(r_direct)


L = len(p) + len(q) - 1

P = np.fft.fft(p, n=L)
Q = np.fft.fft(q, n=L)

R_fft = P * Q                       
r_fft = np.fft.ifft(R_fft)          

r_fft_real = np.round(np.real(r_fft)).astype(int)

print("\nProdus prin FFT:")
print(r_fft_real)


print("\nCoincid cele două rezultate? ", np.array_equal(r_direct, r_fft_real))
