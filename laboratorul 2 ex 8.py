import numpy as np
import matplotlib.pyplot as plt


alpha = np.linspace(-np.pi/2, np.pi/2, 2001)


s = np.sin(alpha)


s_lin  = alpha                                  
s_pade = (alpha - (7*alpha**3)/60) / (1 + alpha**2/20)  


err_lin  = s - s_lin
err_pade = s - s_pade
abs_err_lin, abs_err_pade = np.abs(err_lin), np.abs(err_pade)


plt.figure(figsize=(10,5))
plt.plot(alpha, s, label='sin(α)', linewidth=2)
plt.plot(alpha, s_lin, '--', label='Aproximare liniară: α')
plt.plot(alpha, s_pade, '-.', label='Aproximare Padé')
plt.title('sin(α) vs. aproximații pe [-π/2, π/2]')
plt.xlabel('α [rad]')
plt.ylabel('Valoare')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,5))
plt.plot(alpha, err_lin, label='Eroare: sin(α) − α')
plt.plot(alpha, err_pade, label='Eroare: sin(α) − Padé')
plt.title('Erori (semnate) ale aproximațiilor')
plt.xlabel('α [rad]')
plt.ylabel('Eroare')
plt.axhline(0, color='k', linewidth=0.8)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


eps = 1e-16
plt.figure(figsize=(10,5))
plt.semilogy(alpha, np.maximum(abs_err_lin, eps), label='|sin(α) − α|')
plt.semilogy(alpha, np.maximum(abs_err_pade, eps), label='|sin(α) − Padé|')
plt.title('Erori absolute pe axă Oy logaritmică')
plt.xlabel('α [rad]')
plt.ylabel('|eroare| (log)')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()
