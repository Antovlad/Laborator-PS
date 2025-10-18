import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


fs = 44100        
T = 1.0           
f1, f2 = 440.0, 880.0  

t = np.linspace(0, T, int(fs*T), endpoint=False)


x1 = np.sin(2*np.pi*f1*t)
x2 = np.sin(2*np.pi*f2*t)


x = np.concatenate([x1, x2]).astype(np.float32)


amp = 0.3  
print("Redau două tonuri: 440 Hz apoi 880 Hz...")
sd.play(amp * x, fs)
sd.wait()  
print("Gata 🎧")
