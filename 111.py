import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt


# a = np.random.normal(0, 0.5, size=(8192,))
# b = a.copy()
# b[1024:] = 0
# c = np.abs(fft(a))[:4096]/4096
# d = np.abs(fft(b))[:4096]/4096
# e = np.abs(fft(b[:1024]))[:512]/512
# plt.subplot(311)
# plt.plot(c)
# plt.subplot(312)
# plt.plot(d)
# plt.subplot(313)
# plt.plot(e)
# #plt.ylim(0, 1.5)
# plt.show()


a = np.zeros((1024,))
a[:99] = 1
c = np.abs(fft(a))[:512]/512
plt.plot(c, c="k", lw=1.5)
plt.xlim(0, 256)
plt.ylim(0, 0.05)
plt.xticks([])
plt.yticks([])
plt.show()