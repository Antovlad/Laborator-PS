import numpy as np
import matplotlib.pyplot as plt


n = 20
d = 5     


x = np.zeros(n)
x[5:15] = np.linspace(0, 1, 10)

print("x =", x)


y = np.roll(x, d)
print("y =", y)


X = np.fft.fft(x)
Y = np.fft.fft(y)

R1 = np.fft.ifft(X * Y)          
r1 = np.real(R1)


R2 = np.fft.ifft(Y * X)
r2 = np.real(R2)



print("\nRezultatul IFFT(FFT(x) * FFT(y)):")
print(np.round(r1, 3))

print("\nRezultatul IFFT(FFT(y) * FFT(x)):")
print(np.round(r2, 3))


d_rec_1 = np.argmax(r1)
d_rec_2 = np.argmax(r2)

print(f"\nDeplasarea reală      d = {d}")
print(f"Deplasarea recuperată din formula 1 = {d_rec_1}")
print(f"Deplasarea recuperată din formula 2 = {d_rec_2}")


plt.figure(figsize=(10,5))
plt.stem(r1, label="IFFT(FFT(x) * FFT(y))")
plt.stem(r2, linefmt="C1-", markerfmt="C1o", basefmt="k-", label="IFFT(FFT(y) * FFT(x))")
plt.legend()
plt.title("Compararea rezultatelor celor două expresii FFT")
plt.grid(True)
plt.show()
#Rezultatul e acelasi deoarece inmultirea e comutativa