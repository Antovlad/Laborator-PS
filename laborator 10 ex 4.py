import numpy as np

def roots_via_companion(coeffs):
    
    c = np.asarray(coeffs, dtype=float).ravel()
    if c.size < 2:
        raise ValueError("Polinomul trebuie să aibă grad >= 1 (minim 2 coeficienți).")
    if c[0] == 0:
        raise ValueError("Coeficientul lider (a_n) trebuie să fie nenul.")

    b = c / c[0]
    n = b.size - 1 

    C = np.zeros((n, n), dtype=float)
    C[1:, :-1] = np.eye(n - 1)
    C[:, -1] = -b[1:][::-1]  

    return np.linalg.eigvals(C)

if __name__ == "__main__":
    coeffs = [1, -6, 11, -6]
    r = roots_via_companion(coeffs)
    print("Rădăcini:", np.sort(r))
