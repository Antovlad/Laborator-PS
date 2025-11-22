
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio   




X_rgb = imageio.imread(r"C:\Users\Antoniu\Desktop\raccoon.png")

print("Dimensiunea inițială a imaginii:", X_rgb.shape)

if X_rgb.ndim == 3 and X_rgb.shape[2] == 4:
    X_rgb = X_rgb[..., :3]   # păstrăm doar R,G,B


if X_rgb.ndim == 3:
    X = (0.299 * X_rgb[..., 0] +
         0.587 * X_rgb[..., 1] +
         0.114 * X_rgb[..., 2])
else:
    X = X_rgb.astype(np.float64)

X = X.astype(np.float64)

print("Dimensiunea imaginii gri:", X.shape) 

plt.imshow(X, cmap="gray")
plt.title("Original (grayscale)")
plt.axis("on")
plt.show()




def compute_snr(ref, test):
    ref  = ref.astype(np.float64)
    test = test.astype(np.float64)
    signal_power = np.mean(ref**2)
    noise_power  = np.mean((ref - test)**2) + 1e-12
    return 10 * np.log10(signal_power / noise_power)


def low_pass_mask(shape, R):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist2 = (y - crow)**2 + (x - ccol)**2
    return dist2 <= R**2




Y = np.fft.fft2(X)
Y_shift = np.fft.fftshift(Y)

R_comp = 80     

mask_comp = low_pass_mask(X.shape, R_comp)

Y_comp_shift = Y_shift * mask_comp
Y_comp = np.fft.ifftshift(Y_comp_shift)

X_comp = np.fft.ifft2(Y_comp).real


plt.imshow(X_comp, cmap="gray")
plt.title(f"Imagine comprimată (LPF R={R_comp})")
plt.axis("on")
plt.show()

snr_comp = compute_snr(X, X_comp)
print("=== Exercițiul 2 ===")
print(f"SNR după compresie: {snr_comp:.2f} dB\n")




Yc = np.fft.fft2(X_comp)
Yc_shift = np.fft.fftshift(Yc)

R_denoise = 40  

mask_denoise = low_pass_mask(X.shape, R_denoise)

Y_denoised_shift = Yc_shift * mask_denoise
Y_denoised = np.fft.ifftshift(Y_denoised_shift)

X_denoised = np.fft.ifft2(Y_denoised).real


plt.imshow(X_denoised, cmap="gray")
plt.title(f"Denoised (LPF R={R_denoise})")
plt.axis("on")
plt.show()

snr_before = compute_snr(X, X_comp)
snr_after  = compute_snr(X, X_denoised)

print("=== Exercițiul 3 ===")
print(f"SNR înainte de filtrare (compresie): {snr_before:.2f} dB")
print(f"SNR după filtrare (denoising):       {snr_after:.2f} dB\n")



fig, ax = plt.subplots(1, 3, figsize=(15,5))

ax[0].imshow(X, cmap="gray")
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(X_comp, cmap="gray")
ax[1].set_title(f"Compressed\nSNR={snr_before:.2f} dB")
ax[1].axis("off")

ax[2].imshow(X_denoised, cmap="gray")
ax[2].set_title(f"Denoised\nSNR={snr_after:.2f} dB")
ax[2].axis("off")

plt.show()
