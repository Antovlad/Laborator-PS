import numpy as np
import matplotlib.pyplot as plt


fs = 1000           
f = 50              
T = 0.1             
t = np.arange(0, T, 1/fs)


x = np.sin(2 * np.pi * f * t)

M = 4               
x_dec1 = x[::M]     
t_dec1 = t[::M]


plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, x, 'b', label='Semnal original (fs=1000 Hz)')
plt.stem(t_dec1, x_dec1, linefmt='r-', markerfmt='ro', basefmt=' ')
plt.title('(a) Decimare la 1/4 (păstrăm fiecare al 4-lea eșantion)')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.legend()
plt.grid(True)


x_dec2 = x[1::M]     
t_dec2 = t[1::M]

plt.subplot(2, 1, 2)
plt.plot(t, x, 'b', label='Semnal original (fs=1000 Hz)')
plt.stem(t_dec2, x_dec2, linefmt='g-', markerfmt='go', basefmt=' ')
plt.title('(b) Decimare la 1/4 (pornind de la al doilea eșantion)')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("Observații:")
print("• În (a), semnalul decimat păstrează forma sinusului, dar cu o frecvență de eșantionare de 250 Hz (1/4 din fs).")
print("• În (b), decimarea pornind de la al doilea eșantion schimbă faza semnalului rezultat — semnalul decimat este deplasat în fază.")
print("• Decimarea fără filtru trece-jos prealabil poate duce la aliasing dacă frecvența semnalului este prea mare față de noul fs.")
