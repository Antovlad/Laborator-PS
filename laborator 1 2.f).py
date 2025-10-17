import numpy as np
import matplotlib.pyplot as plt


rows, cols = 128, 128
ii, jj = np.indices((rows, cols))
checker = (ii + jj) % 2
checker = checker.astype(float)

plt.figure()
plt.imshow(checker, cmap='gray', interpolation='nearest')
plt.title('f) Semnal 2D personalizat (checkerboard 128×128)')
plt.axis('off')
plt.show()
