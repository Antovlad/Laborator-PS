import numpy as np
import matplotlib.pyplot as plt



def plot_iterations(x0, title):
    """
    x0 – semnalul inițial
    title – titlul figurii
    """
    x_list = [x0]
    x = x0.copy()

    for _ in range(3):
        x = np.convolve(x, x)
        x_list.append(x)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(title, fontsize=14)

    for i, (ax, sig) in enumerate(zip(axes, x_list)):
        ax.plot(sig, linewidth=1.5)
        ax.set_ylabel(f"Iterația {i}")
        ax.grid(True)

    axes[-1].set_xlabel("n")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




N = 100
x_random = np.random.randn(N)   

plot_iterations(x_random, "Cerința 2: Semnal aleator – convoluții iterative")




x_rect = np.zeros(N)
x_rect[45:55] = 1.0   

plot_iterations(x_rect, "Cerința 2: Bloc dreptunghiular – convoluții iterative")
