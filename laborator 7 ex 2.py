import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from numpy.fft import fft2, ifft2, fftshift, ifftshift


img = imread(r"C:\Users\Antoniu\Desktop\raccoon.png")

if img.ndim == 3:
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])

X = img.astype(float)
signal_energy = np.sum(X**2)


Y = fft2(X)
Y_shift = fftshift(Y)

mag = np.abs(Y_shift).ravel()
idx_sorted = np.argsort(mag)[::-1]


SNR_target = 20  

energy_kept = 0
K = None

for k in range(len(idx_sorted)):
    i = idx_sorted[k]
    energy_kept += mag[i]**2

    noise_energy = (np.sum(mag**2) - energy_kept)
    SNR = 10 * np.log10((energy_kept + 1e-12) / (noise_energy + 1e-12))

    if SNR >= SNR_target:
        K = k + 1
        break

if K is None:
    K = len(idx_sorted)
    print("Atenție: pragul SNR nu a putut fi atins → păstrăm toți coeficienții.")

print("Coeficienți păstrați:", K, "/", mag.size)
print("Compresie:", 100 * (1 - K/mag.size), "%")


mask = np.zeros_like(mag, dtype=bool)
mask[idx_sorted[:K]] = True
mask = mask.reshape(Y_shift.shape)

Yc = Y_shift * mask


Xc = np.real(ifft2(ifftshift(Yc)))


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(X, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(Xc, cmap='gray')
plt.title(f"Compressed (SNR={SNR_target} dB)")
plt.axis('off')

plt.show()
