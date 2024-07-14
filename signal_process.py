import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy.fft import rfft, rfftfreq  

with open('signal.txt', 'r') as f:
    g = [float(line) for line in f]

with open('heartbeats.txt', 'r') as f:
    heartbeats = [float(line) for line in f]

heartbeats = [int(i) for i in heartbeats]
beatLines = []
for i in range(len(g)):
    if(i in heartbeats):
        beatLines.append(max(g))
    else:
        beatLines.append(min(g))


# 5-tap moving average filter
window_size = 5
avgG = []
for i in range(len(g) - window_size + 1):
    avgG.append(sum(g[i:i+window_size]) / window_size)
avgG = avgG[window_size-1:]  # remove the first few elements


# Bandpass filter parameters
lowcut = 0.1
highcut = 3.25
fs = 30

# Apply bandpass filter
def butter_bandpass_filter(data, low, high, fs, order=5):
    b, a = butter(order, [low, high], btype='band', fs=fs)
    y = lfilter(b, a, data)
    return y

# Apply bandpass filter to signal 'g'
h = butter_bandpass_filter(avgG, lowcut, highcut, fs)

ph = rfft(h)
freqs = rfftfreq(len(h), 1/fs)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.plot(g)
ax1.plot(beatLines,'r')
ax2.plot(h)
ax3.plot(freqs, np.abs(ph))

plt.tight_layout()
plt.show(block=True)
