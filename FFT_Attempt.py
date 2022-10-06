from scipy.fft import fftfreq, fft
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt


def butter_bandpass(w_L, w_H, f_s, order=5):
    nyq = 0.5 * f_s
    low = w_L / nyq
    high = w_H / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(array, lowcut, highcut, f_s,  order=5):
    b, a = butter_bandpass(lowcut, highcut, f_s, order=order)
    filtered = signal.lfilter(b, a, array)
    return filtered


y = np.genfromtxt("C:/Users/amir9/AppData/Roaming/JetBrains/PyCharmEdu2022.2/scratches/Otot2_Red_Signal.csv")  # live_forehead_Red_HSV_Backups  /  Otot2_Red_Signal
y_new = 200 * (y - np.mean(y))  # amplification of the zero centered signal
N = y.size

# order = 5
f_s = 30
w_L = 0.11  # frequency for heartrate of 30BPM which is very low and helps us eliminate DC noise.
w_H = 1  # frequency for heartrate of 210 which is very high for any standard human being not currently playing against messi, ronaldinho, ronaldo and maradona all together
filter_order = 5

# for some reason the frequency response plot doesn't show cutoff frequencies of 0.5 and 3.5 as wanted. The websites demo works fine.

plt.figure(1)
plt.clf()
b, a = butter_bandpass(w_L, w_H, f_s, order=filter_order)
w, h = signal.freqz(b, a, fs=f_s, worN=8000)
plt.plot((f_s * 0.5 / np.pi) * w, abs(h), label="order = %d" % filter_order)
plt.plot([0, f_s], [np.sqrt(0.5), np.sqrt(0.5)],
         '--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')
# plt.show()

T = N / f_s
n = int(N)
t = np.linspace(0, T, n, endpoint=False)
y_filtered = butter_bandpass_filter(y_new, w_L, w_H, f_s, order=filter_order)
plt.figure(2)
plt.clf()
plt.plot(t, y_new, label='Noisy signal')

plt.plot(t, y_filtered, label='filtered signal')
plt.xlabel('time (seconds)')
# plt.hlines([-a, a], 0, T, linestyles='--')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='lower left')
# plt.show()
# cutoff = 7.5398

# b, a = butter_lowpass(cutoff, fs, order)
# w, h = signal.freqz(b, a, fs=fs, worN=8000)
# plt.subplot(2, 1, 1)
# plt.plot(w, np.abs(h), 'b')
# plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
# plt.axvline(cutoff, color='k')
# plt.xlim(0, 0.5*fs)
# plt.title("Lowpass Filter Frequency Response")
# plt.xlabel('Frequency [Hz]')
# plt.grid()
# plt.show


# n = int(N)

# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
# data = y[20:1200]

# filtered_y = butter_lowpass_filter(data, cutoff, fs, order)
# plt.subplot(2, 1, 2)
# plt.plot(t, data, 'b-', label='data')
# plt.plot(t, filtered_y, 'g-', linewidth=2, label='filtered data')
# plt.xlabel('Time [sec]')
# plt.grid()
# plt.legend()
# plt.subplots_adjust(hspace=0.35)
#
# plt.show

# x = np.linspace(0.0, y.size, y.size, endpoint=True)
# x_ax = x / 30
# b, a = signal.butter(4, 4, 'low', analog=False)
# sos = signal.butter(4, 4 * 2 * np.pi, 'low', fs=1000, output='sos')


# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(x_ax, y)
# ax1.set_title('regular signal')
# ax1.axis([0, 1, -2, 2])
# ax2.plot(x_ax, filtered_y)
# ax2.set_title('filtered signal')
# # ax2.axis([0, 1, -2, 2])
# ax2.set_xlabel('Time [sec]')
# plt.tight_layout()
# plt.show()

yf = fft(y_filtered)
xf = fftfreq(N, T)[:N // 2]

# plt.plot(x_ax, y_new, ".", color="red")
# plt.grid()
# plt.ylim(-500, np.max(y_new) + 50)
# plt.show()

# plt.figure(3)
# plt.clf()
# plt.psd(y_filtered, 1024, 1 / T)
# plt.show()
plt.figure(3)
plt.clf()
plt.plot(xf * N, 2.0 / N * np.abs(yf[0:N // 2]), color="aquamarine")
plt.grid(True)
plt.axis('tight')
plt.show()
