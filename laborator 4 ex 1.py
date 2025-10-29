import numpy as np
import time
import math
import matplotlib.pyplot as plt
import pandas as pd



def dft(x: np.ndarray) -> np.ndarray:
    
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)   
    return W @ x

def fft_radix2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    if N <= 1:
        return x
    if N % 2 != 0:
        
        return dft(x)

    X_even = fft_radix2(x[::2])
    X_odd  = fft_radix2(x[1::2])

    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate((
        X_even + factor[:N // 2] * X_odd,
        X_even - factor[:N // 2] * X_odd
    ))



def main():
    Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
    rng = np.random.default_rng(42)

    times_dft = []
    times_fft = []
    times_np  = []

    repeats_small = 5
    repeats_large = 3

    for N in Ns:
        x = rng.normal(size=N) + 1j * rng.normal(size=N)

       
        if N <= 2048:
            reps = repeats_small if N <= 1024 else repeats_large
            t0 = time.perf_counter()
            for _ in range(reps):
                _ = dft(x)
            t_dft = (time.perf_counter() - t0) / reps
        else:
            t_dft = np.nan
        times_dft.append(t_dft)

       
        reps = repeats_small if N <= 4096 else repeats_large
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = fft_radix2(x)
        t_fft = (time.perf_counter() - t0) / reps
        times_fft.append(t_fft)

        
        reps = repeats_small if N <= 4096 else repeats_large
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = np.fft.fft(x)
        t_np = (time.perf_counter() - t0) / reps
        times_np.append(t_np)

    
    plt.figure(figsize=(8, 5))
    plt.plot(Ns, times_dft, marker='o', label='DFT (naiv)')
    plt.plot(Ns, times_fft, marker='o', label='FFT (implementare proprie)')
    plt.plot(Ns, times_np,  marker='o', label='NumPy FFT')
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xlabel('Dimensiunea vectorului N')
    plt.ylabel('Timp execuție (secunde, scară log)')
    plt.title('Compararea timpilor DFT / FFT vs numpy.fft')
    plt.legend()
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig('fft_timing.png', dpi=160)

    
    df = pd.DataFrame({
        'N': Ns,
        'DFT_naiv_s': times_dft,
        'FFT_propriu_s': times_fft,
        'NumPy_FFT_s': times_np,
    })
    df.to_csv('fft_timing.csv', index=False)

    
    print(df)
    print('Grafic salvat ca fft_timing.png, datele în fft_timing.csv')

if __name__ == "__main__":
    main()
