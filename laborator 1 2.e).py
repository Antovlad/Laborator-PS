import numpy as np
import matplotlib.pyplot as plt

# Semnal 2D aleator (128x128)
I = np.random.rand(128, 128)

plt.figure()
plt.imshow(I, cmap='gray', interpolation='nearest')
plt.title('e) Semnal 2D aleator (128×128)')
plt.axis('off')
plt.colorbar()
plt.show()
