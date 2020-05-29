import matplotlib.pyplot as plt
import numpy as np
from EMD_hu.inst_freq import InstantaneousFreq
from scipy.signal import hilbert


# hyper-parameter
DIM = 1024
START = 0
END = 1
DTYPE = np.float64
CHOICE = 0

# signal generate
t = np.linspace(START, END, DIM, dtype=DTYPE)
# for hilbert transform and arccos func comparison
phase1 = 50 * np.pi * (t ** 2)
phase2 = 40 * np.pi * t + 1 * np.pi * np.cos(8 * np.pi * t) + 1.3 * np.pi
kkk = np.diff(phase2) * DIM / 2 / np.pi
phases = np.array([phase1, phase2])
signals = np.cos(phases)
hilbert_s = hilbert(signals).imag


inst = InstantaneousFreq()
hilbert_freqs, _ = inst.hilbert_method(signals, sampling_rate=DIM)
arccos_freqs = inst.arccos_inst_freq(signals, sampling_rate=DIM)
actual_freqs = np.diff(phases) * DIM / 2 / np.pi
actual_freqs = np.concatenate((actual_freqs[:, :1], actual_freqs), axis=-1)

plt.figure()
plt.plot(t, signals[CHOICE], c="k", lw=1.5)
plt.plot(t, hilbert_s[CHOICE], c="r", lw=1, linestyle="--")
plt.xlim(0, 1)


plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(t, signals[CHOICE], c="k")
plt.ylim(-1.1, 1.1)
plt.xlim(0, 1)

plt.subplot(122)
plt.plot(t, actual_freqs[CHOICE], c="k")
plt.ylim(0, 100)
plt.xlim(0, 1)

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(t, hilbert_freqs[CHOICE], c="k")
plt.ylim(0, 100)
plt.xlim(0, 1)

plt.subplot(122)
plt.plot(t, arccos_freqs[CHOICE], c="k")
plt.ylim(0, 100)
plt.xlim(0, 1)

plt.show()

