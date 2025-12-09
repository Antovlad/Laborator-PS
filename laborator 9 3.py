import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. Seria de timp utilizata anterior (3 componente)
# ---------------------------------------------------------
N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t + 5
season = 3*np.sin(2*np.pi*t/50) + 2*np.sin(2*np.pi*t/120)
noise = np.random.normal(0, 1, N)

x = trend + season + noise

# ---------------------------------------------------------
# 1. Functie pentru model MA cu orizont q
#    y_ma[i] = media(x[i-q+1 : i])           (media mobila)
#    e[i]    = x[i] - y_ma[i]               (termen de eroare)
# ---------------------------------------------------------
def moving_average_model(x, q):
    """
    x  - seria de timp (vector 1D)
    q  - orizontul mediei mobile (fereastra)
    returneaza:
        ma   - seria mediei mobile (aceeasi lungime ca x, cu NaN la inceput)
        eps  - erorile e[i] = x[i] - ma[i]
    """
    x = np.asarray(x)
    # media mobila doar acolo unde fereastra e completa
    ma_valid = np.convolve(x, np.ones(q) / q, mode="valid")
    
    # aliniem la lungimea initiala (primele q-1 pozitii nu au media definita)
    ma = np.full_like(x, fill_value=np.nan, dtype=float)
    ma[q-1:] = ma_valid
    
    eps = x - ma   # epsilon[i] = deviatia fata de media pe fereastra
    return ma, eps

# ---------------------------------------------------------
# 2. Aplicam modelul MA pentru mai multe orizonturi q
# ---------------------------------------------------------
q_list = [5, 20, 50]   # poti schimba sau adauga alte valori

results = {}

for q in q_list:
    ma_q, eps_q = moving_average_model(x, q)
    results[q] = (ma_q, eps_q)
    # ignoram NaN-urile de la inceput cand calculam statistici
    valid = ~np.isnan(eps_q)
    mse_q = np.mean(eps_q[valid]**2)
    print(f"q = {q:2d}  ->  MSE al erorilor epsilon = {mse_q:.4f}")

# ---------------------------------------------------------
# 3. Plot pentru un q ales (de exemplu q = 20)
# ---------------------------------------------------------
q_plot = 20
ma_q, eps_q = results[q_plot]

plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(x, label="x (original)")
plt.title("Seria de timp originala")
plt.legend()

plt.subplot(3,1,2)
plt.plot(x, label="x (original)", alpha=0.4)
plt.plot(ma_q, label=f"MA cu orizont q = {q_plot}", linewidth=2)
plt.title("Model MA (media mobila)")
plt.legend()

plt.subplot(3,1,3)
plt.plot(eps_q, label=r"epsilon[i] = x[i] - MA_q[i]")
plt.title("Termenii de eroare ε[i]")
plt.legend()

plt.tight_layout()
plt.show()
