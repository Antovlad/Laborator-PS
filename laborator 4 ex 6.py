import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


PATH_WAV = r"C:\Users\Antoniu\Desktop\aeiou.wav"     
fs, x = wavfile.read(PATH_WAV)

x = x.astype(np.float64)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= np.max(np.abs(x)) + 1e-12

N = len(x)


win_len = max(2, int(0.01 * N))      
win_len += win_len % 2              
hop = win_len // 2                  
n_frames = 1 + max(0, (N - win_len) // hop)

window = np.hanning(win_len)

nfft = 1 << int(np.ceil(np.log2(win_len)))  


spec_cols = []
for m in range(n_frames):
    start = m * hop
    frame = x[start:start + win_len]
    if len(frame) < win_len:
        frame = np.pad(frame, (0, win_len - len(frame)))
    frame_win = frame * window
    X = np.fft.rfft(frame_win, n=nfft)
    spec_cols.append(np.abs(X))


S = np.stack(spec_cols, axis=1)

S_db = 20 * np.log10(S + 1e-12)

freqs = np.fft.rfftfreq(nfft, d=1.0/fs)      
times = (np.arange(n_frames) * hop + win_len/2) / fs  


plt.figure(figsize=(9, 4.5))
extent = [times[0], times[-1], freqs[0]/1000, freqs[-1]/1000]  
im = plt.imshow(S_db,
                origin='lower',
                aspect='auto',
                extent=extent,
                cmap='plasma',
                vmin=-100, vmax=0)   
plt.colorbar(im, label='dBFS')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [kHz]')
plt.title('Spectrogramă (ferestre 1% din N, overlap 50%)')
plt.tight_layout()
plt.show()