import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from numpy.fft import fft2, ifft2, fftshift, ifftshift


image_path = r"C:\Users\Antoniu\Desktop\raccoon.png"
img = imread(image_path)

if img.ndim == 3:
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])

X = img.astype(float)

if X.max() <= 1.0:
    X = X * 255.0

rows, cols = X.shape

pixel_noise = 200
noise = np.random.randint(-pixel_noise, pixel_noise + 1, size=X.shape)
X_noisy = X + noise
X_noisy = np.clip(X_noisy, 0, 255)

def snr_db(original, test):
    signal_energy = np.sum(original ** 2)
    noise_energy  = np.sum((original - test) ** 2) + 1e-12
    return 10 * np.log10(signal_energy / noise_energy)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap='gray')
plt.title("Raton original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(X_noisy, cmap='gray')
plt.title("Raton cu zgomot (pt. Ex. 3)")
plt.axis('off')

plt.tight_layout()
plt.show()



Y = fft2(X)
Y_shift = fftshift(Y)

mag = np.abs(Y_shift).ravel()

idx_sorted = np.argsort(mag)[::-1]
mag_sorted = mag[idx_sorted]

SNR_target = 20  

total_energy_freq = np.sum(mag**2)
energy_kept = 0.0
K = len(mag_sorted)  

for k, v in enumerate(mag_sorted):
    energy_kept += v**2
    noise_energy_freq = total_energy_freq - energy_kept
    SNR_freq = 10 * np.log10((energy_kept + 1e-12) /
                             (noise_energy_freq + 1e-12))
    if SNR_freq >= SNR_target:
        K = k + 1
        break

print("=== EXERCIȚIUL 2 ===")
print(f"SNR țintă în frecvență : {SNR_target:.2f} dB")
print(f"Coeficienți păstrați   : {K} / {mag.size}")
print(f"Compresie              : {100 * (1 - K/mag.size):.2f} %")

mask_flat = np.zeros_like(mag, dtype=bool)
mask_flat[idx_sorted[:K]] = True
mask = mask_flat.reshape(Y_shift.shape)

Yc_shift = Y_shift * mask
Yc = ifftshift(Yc_shift)
X_compressed = np.real(ifft2(Yc))

snr_ex2 = snr_db(X, X_compressed)
print(f"SNR (imagine originală vs. comprimată): {snr_ex2:.2f} dB")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap='gray')
plt.title("Raton original (pt. compresie)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(X_compressed, cmap='gray')
plt.title(f"Comprimată FFT\nSNR = {snr_ex2:.2f} dB")
plt.axis('off')

plt.tight_layout()
plt.show()



print("\n=== EXERCIȚIUL 3 ===")

snr_before = snr_db(X, X_noisy)

Y_n = fft2(X_noisy)
Y_n_shift = fftshift(Y_n)

u = np.arange(-cols // 2, cols // 2)
v = np.arange(-rows // 2, rows // 2)
U, V = np.meshgrid(u, v)
D = np.sqrt(U**2 + V**2)

D0 = 30  
H = (D <= D0).astype(float)

Yf_shift = Y_n_shift * H
Yf = ifftshift(Yf_shift)
X_denoised = np.real(ifft2(Yf))

snr_after = snr_db(X, X_denoised)

print(f"SNR înainte filtrare (X vs X_noisy)   : {snr_before:.2f} dB")
print(f"SNR după filtrare  (X vs denoised)   : {snr_after:.2f} dB")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(X, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(X_noisy, cmap='gray')
plt.title(f"Noisy (din setup)\nSNR = {snr_before:.2f} dB")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(X_denoised, cmap='gray')
plt.title(f"Denoised (LPF)\nSNR = {snr_after:.2f} dB")
plt.axis('off')

plt.tight_layout()
plt.show()
