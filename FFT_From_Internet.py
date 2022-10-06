from scipy.fft import fftfreq, fft
import numpy as np
from matplotlib import pyplot as plt

N = 600
print(N)
T = 1 / 800
print(T)
x = np.linspace(0.0, N * T, N, endpoint=False)
print(x.size)
y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x) + 0.25 * np.sin(120.0 * 2.0 * np.pi * x)
print(y.size)
yf = fft(y)
print(yf.size)
xf = fftfreq(N, T)[:N // 2]
print(xf.size)
plt.figure(1)
plt.clf()
plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
plt.grid()

plt.figure(2)
plt.clf()
plt.plot(x, y, color="gold")
plt.show()
