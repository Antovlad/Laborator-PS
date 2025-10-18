import numpy as np
import sounddevice as sd
from scipy.io import wavfile

Fs = 8000  

def play(sig, fs=Fs, title=""):
    print(f"Redare: {title}  (durata ~ {len(sig)/fs:.3f} s)")
    sd.play(sig, fs)
    sd.wait()  

def to_int16(x):
    """Convertire float [-1,1] -> int16 pentru WAV."""
    x = np.asarray(x, dtype=np.float32)
    mx = np.max(np.abs(x)) or 1.0
    y = x / mx  
    return (y * 32767).astype(np.int16)


f_a = 400
N_a = 1600
t_a = np.arange(N_a) / Fs
x_a = np.sin(2*np.pi*f_a*t_a)
play(x_a, Fs, "a) Sinus 400 Hz, 1600 eșantioane")


f_b = 800
dur_b = 3.0
t_b = np.arange(int(dur_b*Fs)) / Fs
x_b = np.sin(2*np.pi*f_b*t_b)
play(x_b, Fs, "b) Sinus 800 Hz, 3 s")


f_c = 240
dur_c = 2.0
t_c = np.arange(int(dur_c*Fs)) / Fs
T_c = 1/f_c
x_c = 2*((t_c/T_c) - np.floor(0.5 + t_c/T_c))  # în [-1,1]
play(x_c, Fs, "c) Sawtooth 240 Hz")

f_d = 300
dur_d = 2.0
t_d = np.arange(int(dur_d*Fs)) / Fs
x_d = np.sign(np.sin(2*np.pi*f_d*t_d)).astype(float)
x_d[x_d == 0] = 1
play(x_d, Fs, "d) Square 300 Hz")


wav_path = "sinus_800Hz_3s.wav"
wavfile.write(wav_path, Fs, to_int16(x_b))
print(f"Salvat: {wav_path}")


Fs_read, x_read = wavfile.read(wav_path)
print(f"Încărcat din fișier: Fs={Fs_read} Hz, dtype={x_read.dtype}, shape={x_read.shape}")


play(x_read.astype(np.float32)/32768.0, Fs_read, "Redare din fișier .wav")
