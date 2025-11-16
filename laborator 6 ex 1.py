import numpy as np
import matplotlib.pyplot as plt


B = 1.0                      
t_min, t_max = -3.0, 3.0     
Fs_list = [1.0, 1.5, 2.0, 4.0]   
N_t = 4000                   

def x_continuous(t, B):
    return np.sinc(B * t) ** 2

def reconstruct_from_samples(t, t_samp, x_samp, Ts):
    
    x_hat = np.zeros_like(t)
    for tk, xk in zip(t_samp, x_samp):
        x_hat += xk * np.sinc((t - tk) / Ts)
    return x_hat

t = np.linspace(t_min, t_max, N_t)
x_t = x_continuous(t, B)

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
axes = axes.ravel()

for ax, Fs in zip(axes, Fs_list):
    Ts = 1.0 / Fs

    n_min = int(np.ceil(t_min / Ts))
    n_max = int(np.floor(t_max / Ts))
    n = np.arange(n_min, n_max + 1)

    t_samp = n * Ts
    x_samp = x_continuous(t_samp, B)

    x_hat = reconstruct_from_samples(t, t_samp, x_samp, Ts)

    ax.plot(t, x_t, linewidth=1.5, label=r"$x(t)$ (original)")

    ax.plot(t, x_hat, linestyle="--", linewidth=1.5,
            label=r"$\hat{x}(t)$ (reconstruită)")

    ax.stem(t_samp, x_samp, linefmt='C2-', markerfmt='C2o', basefmt='k-',
            label=r"$x[n]$")

    ax.set_title(rf"$F_s = {Fs:.2f}\ \mathrm{{Hz}}$")
    ax.set_xlabel(r"$t\ [s]$")
    ax.set_ylabel("Amplitudă")
    ax.grid(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)

fig.suptitle(rf"Funcția $\mathrm{{sinc}}^2(B t)$, B = {B:.2f}", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.90])

plt.show()
