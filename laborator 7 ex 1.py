import numpy as np
import matplotlib.pyplot as plt


N = 64
n1 = np.arange(N)
n2 = np.arange(N)
N1, N2 = np.meshgrid(n1, n2, indexing='ij')

def show_image_and_spectrum(x, title):
   
    Y = np.fft.fft2(x)
    Y_shift = np.fft.fftshift(Y)      
    mag = np.log1p(np.abs(Y_shift))   

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(x, cmap='gray')
    axs[0].set_title(f'{title} - x(n1,n2)')
    axs[0].axis('off')

    axs[1].imshow(mag, cmap='gray')
    axs[1].set_title(f'{title} - |Y(m1,m2)|')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 1a) x_{n1,n2} = sin(2π n1 + 3π n2)
#    => sinusoidă 2D cu o singură frecvență; spectrul va avea două vârfuri conjugate.
# ------------------------------------------------------------------
x1 = np.sin(2 * np.pi * N1 + 3 * np.pi * N2)
show_image_and_spectrum(x1, 'x1 = sin(2π n1 + 3π n2)')


# ------------------------------------------------------------------
# 1b) x_{n1,n2} = sin(4π n1) + cos(6π n2)
#    => sumă de două sinusoide independente pe n1 și n2;
#       spectrul va avea patru vârfuri (câte două pentru fiecare sinusoidă).
# ------------------------------------------------------------------
x2 = np.sin(4 * np.pi * N1) + np.cos(6 * np.pi * N2)
show_image_and_spectrum(x2, 'x2 = sin(4π n1) + cos(6π n2)')



def show_from_spectrum(Y, title):
    """
    Primește un spectru Y(m1,m2), reconstruiește x(n1,n2) prin IFFT
    și afișează atât x cât și |Y|.
    """
    x = np.fft.ifft2(Y)
    x = np.real(x)  
    Y_shift = np.fft.fftshift(Y)
    mag = np.log1p(np.abs(Y_shift))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(x, cmap='gray')
    axs[0].set_title(f'{title} - x(n1,n2)')
    axs[0].axis('off')

    axs[1].imshow(mag, cmap='gray')
    axs[1].set_title(f'{title} - |Y(m1,m2)|')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


# (i) Y_{0,5} = Y_{0,N-5} = 1, altfel 0
#     => sinusoidă pură de-a lungul direcției n2 (două muchii de frecvență ±).
Y1 = np.zeros((N, N), dtype=complex)
Y1[0, 5]     = 1
Y1[0, N - 5] = 1
show_from_spectrum(Y1, 'Y(0,5) = Y(0,N-5) = 1')


# (ii) Y_{5,0} = Y_{N-5,0} = 1, altfel 0
#      => sinusoidă pură de-a lungul direcției n1.
Y2 = np.zeros((N, N), dtype=complex)
Y2[5, 0]     = 1
Y2[N - 5, 0] = 1
show_from_spectrum(Y2, 'Y(5,0) = Y(N-5,0) = 1')


# (iii) Y_{5,5} = Y_{N-5,N-5} = 1, altfel 0
#       => sinusoidă oblică (frecvență nenulă pe ambele axe).
Y3 = np.zeros((N, N), dtype=complex)
Y3[5, 5]           = 1
Y3[N - 5, N - 5]   = 1
show_from_spectrum(Y3, 'Y(5,5) = Y(N-5,N-5) = 1')
