from LMD.lmd import LocalMeanDecomp
from EMD_hu.emd import EmpiricalModeDecomp
from EMD_hu.inst_freq import InstantaneousFreq
import matplotlib.pyplot as plt
import numpy as np


def imfs_show(coord, signal, imfs, res, same_lim=True, name="IMF"):

    time_max, time_min = np.max(signal), np.min(signal)

    num = len(imfs) + 2
    plt.figure()
    plt.subplot(num, 1, 1)
    plt.plot(coord, signal, "b", lw=1.5)
    plt.xlim((coord[0], coord[-1]))
    plt.xticks([])
    if same_lim:
        plt.ylim(time_min, time_max)
    plt.ylabel("Signal")

    for i in range(len(imfs)):
        plt.subplot(num, 1, (i + 1) + 1)
        plt.plot(coord, imfs[i], 'r', lw=1.5)
        plt.xlim((coord[0], coord[-1]))
        plt.xticks([])
        if same_lim:
            plt.ylim(time_min, time_max)
        plt.ylabel(name + "_" + str(i + 1))

    plt.subplot(num, 1, num)
    plt.plot(coord, res, "g", lw=1.5)
    plt.xlim((coord[0], coord[-1]))
    if same_lim:
        plt.ylim(time_min, time_max)
    plt.ylabel("Res")


# hyper-parameter
DIM = 1024
START = 0
END = 1
DTYPE = np.float64
CHOICE = 1
MAX_FREQ = 300

# signal generate
t = np.linspace(START, END, DIM, dtype=DTYPE)

signal1 = 5 * (1 + np.cos(40 * np.pi * t)) * np.cos(200 * np.pi * t) + 15 * (1 + np.cos(40 * np.pi * t)) * \
          np.cos(600 * np.pi * t)
signal2 = 2 * (1.3 + np.cos(18 * np.pi * t)) * np.cos(400 * np.pi * t + 2 * np.cos(20 * np.pi * t)) + \
          4 * np.cos(5 * np.pi * t) * np.cos(120 * np.pi * t) + \
          2 * np.cos(40 * np.pi * t) + 0.8 * np.sin(np.pi * t) * np.sin(10 * np.pi * t)
signals = np.array([signal1, signal2])

# EMD decomposition
emd = EmpiricalModeDecomp()
imfs, res = emd(signals[CHOICE])

# time frequency representation
inst = InstantaneousFreq()
inst_freqs, amps = inst.hilbert_method(imfs, sampling_rate=DIM)

plt.figure(figsize=(9, 3))
plt.plot(t, inst_freqs[0], c="k", lw=1)
plt.xlim(START, END)
plt.ylim(0, 300)

for i in range(len(inst_freqs)):    # window=int(np.mean(inst_freqs[i]) / (DIM * 32))
    win = int(32 - DIM / np.mean(inst_freqs[i]))
    inst.inst_freq_ma_smooth(inst_freqs[i], window=win)
inst.time_freq_representation(inst_freqs, amps, freq_dim=MAX_FREQ, max_freq=MAX_FREQ, sampling_rate=DIM)
# inst.inst_freq_representation(t, signals[CHOICE], imfs, inst_freqs, amps, res, same_lim=False)

plt.figure(figsize=(9, 3))
plt.plot(t, inst_freqs[0], c="k", lw=1)
plt.xlim(START, END)
plt.ylim(0, 300)


plt.figure(figsize=(9, 3))
plt.plot(t, signals[CHOICE], c="k", lw=1)
plt.xlim(START, END)


plt.show()

