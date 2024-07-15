import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy.fft import rfft, rfftfreq

def shift_signal(data, shift):
    return [data[0]] * shift + data
def load_data(filename):
    with open(filename, 'r') as f:
        return [float(line) for line in f]

def moving_average(data, window_size):
    avg = []
    for i in range(len(data) - window_size + 1):
        avg.append(sum(data[i:i+window_size]) / window_size)
    return avg

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return lfilter(b, a, data)

def main():
    g = load_data('signal.txt')
    markers = load_data('markers.txt')
    markers = [int(i) for i in markers]

    _, (ax1, ax2, ax3) = plt.subplots(3, 1)
    for i in markers:
        ax1.axvline(x=i, color='g')
        ax2.axvline(x=i, color='g')

    avg_signal = moving_average(g, 5)
    h = butter_bandpass_filter(avg_signal, 0.75, 3.25, 30)
    freqs = rfftfreq(len(h), 1/30)
    ph = rfft(h)

    ax1.plot(g)
    ax1.plot(shift_signal(avg_signal,2), 'r')
    ax2.plot(h)
    ax3.plot(freqs, np.abs(ph))

    plt.tight_layout()
    plt.show(block=True)

if __name__ == '__main__':
    main()

