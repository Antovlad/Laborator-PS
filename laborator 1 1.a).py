import numpy as np

start, stop, step = 0.0, 0.03, 0.0005
n = int(round((stop - start) / step))  # 60
t = np.linspace(start, stop, n + 1)    # 61 puncte, inclusiv 0.03

print(t)
