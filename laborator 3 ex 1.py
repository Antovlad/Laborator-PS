import numpy as np
import matplotlib.pyplot as plt

N = 8
n = np.arange(N)
k = n.reshape((N, 1))


F = (1/np.sqrt(N)) * np.exp(-2j * np.pi * k * n / N)


unitarity_check = np.allclose(F.conj().T @ F, np.eye(N))
print("E F unitara?", unitarity_check)


fig, axes = plt.subplots(2, 1, figsize=(10, 6))
for row in F:
    axes[0].plot(np.real(row), marker='o')
    axes[1].plot(np.imag(row), marker='o')

axes[0].set_title("Partea reală a liniilor matricii Fourier")
axes[1].set_title("Partea imaginară a liniilor matricii Fourier")

plt.tight_layout()
plt.show()
