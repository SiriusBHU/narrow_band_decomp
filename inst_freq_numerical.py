from EMD_hu.emd import EmpiricalModeDecomp
from EMD_hu.inst_freq import InstantaneousFreq
from LMD.lmd import LocalMeanDecomp
import logging
import matplotlib.pyplot as plt
import numpy as np


def imfs_show(coord, signal, imfs, res, same_lim=True):

    time_max, time_min = np.max(signal), np.min(signal)

    num = len(imfs) + 2
    plt.figure()
    plt.subplot(num, 1, 1)
    plt.plot(coord, signal, "b", lw=1.5)
    plt.xlim((coord[0], coord[-1]))
    if same_lim:
        plt.ylim(time_min, time_max)
    plt.ylabel("Signal")

    for i in range(len(imfs)):
        plt.subplot(num, 1, (i + 1) + 1)
        plt.plot(coord, imfs[i], 'r', lw=1.5)
        plt.xlim((coord[0], coord[-1]))
        if same_lim:
            plt.ylim(time_min, time_max)
        plt.ylabel("IMF_ " + str(i + 1))

    plt.subplot(num, 1, num)
    plt.plot(coord, res, "g", lw=1.5)
    plt.xlim((coord[0], coord[-1]))
    if same_lim:
        plt.ylim(time_min, time_max)
    plt.ylabel("Res")
    plt.show()


if __name__ == "__main__":

    # param setting
    DIM, START, END, DTYPE = 1024, 0, 1, np.float64

    # signal generate
    coord = np.linspace(START, END, DIM, dtype=DTYPE)
    signal = 18 * np.sin(5 * np.pi * coord) * np.cos(40 * np.pi * coord) + 30 * np.cos(9 * np.pi * coord)
    # signal = (2 + np.cos(18 * np.pi * coord)) * np.cos(400 * np.pi * coord + 2 * np.cos(20 * np.pi * coord)) + \
    #          4 * np.cos(5 * np.pi * coord) * np.cos(120 * np.pi * coord) + \
    #          2 * np.cos(40 * np.pi * coord) + 0.8 * np.sin(np.pi * coord) * np.sin(10 * np.pi * coord)

    # EMD decomposition
    emd = EmpiricalModeDecomp()
    lmd = LocalMeanDecomp()
    imfs, res = emd(signal)

    # extract pue freq-modulated signal and its amplitude envelope
    ifs = InstantaneousFreq()
    inst_freqs, amps = ifs.hilbert_method(imfs, sampling_rate=DIM / (END - START))
    # inst_freqs, amps = ifs.arccos_method(imfs, sampling_rate=DIM / (END - START))
    # ifs.inst_freq_ma_smooth(inst_freqs[0], window=5)
    ifs.time_freq_representation(inst_freqs, amps, freq_dim=400, max_freq=80, sampling_rate=DIM / (END - START))
    ifs.inst_freq_representation(coord, signal, imfs, inst_freqs, amps, res, same_lim=False)
    plt.show()
