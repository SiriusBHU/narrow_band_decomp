from LMD.lmd import LocalMeanDecomp
from EMD_hu.emd import EmpiricalModeDecomp
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


def mode_mix_show(t, signal, x1, x2):

    d1 = np.diff(signal)
    d1 = np.append(d1[:1], d1)
    fig = plt.figure()
    plt.subplot(411)
    plt.plot(t, signal)
    plt.xticks([])
    plt.xlim(0, 1)
    plt.subplot(412)
    plt.plot(t, x1)
    plt.xticks([])
    plt.xlim(0, 1)
    plt.subplot(413)
    plt.plot(t, x2)
    plt.xticks([])
    plt.xlim(0, 1)
    plt.subplot(414)
    plt.plot(t, d1)
    plt.xlim(0, 1)


def emd_sifting_show(t, signal):

    emd = EmpiricalModeDecomp()
    # emd.calculate_imf(t, signal)
    ma1_ind, mi1_ind = emd.find_maxima_minima(signal)
    ma1, mi1 = emd.mirror_extension(t, signal, ma1_ind, mi1_ind)
    _, ma_s1 = emd.splines(t, ma1)
    _, mi_s1 = emd.splines(t, mi1)
    mean1 = (ma_s1 + mi_s1) / 2
    res1 = signal - mean1
    ma2_ind, mi2_ind = emd.find_maxima_minima(res1)
    ma2, mi2 = emd.mirror_extension(t, res1, ma2_ind, mi2_ind)
    _, ma_s2 = emd.splines(t, ma2)
    _, mi_s2 = emd.splines(t, mi2)
    mean2 = (ma_s2 + mi_s2) / 2
    res2 = res1 - mean2

    plt.figure(figsize=(10, 3))
    plt.plot(t, signal, c="k", lw=1.5)
    plt.plot(t, ma_s1, c="r", lw=0.8, linestyle="--")
    plt.plot(t, mi_s1, c="orange", lw=0.8, linestyle="--")
    plt.plot(t, mean1, c="blue", lw=1.2)
    plt.scatter(t.take(ma1_ind), signal.take(ma1_ind), c="darkred", s=20, alpha=0.8)
    plt.scatter(t.take(mi1_ind), signal.take(mi1_ind), c="green", s=20, alpha=0.8)
    plt.xlim(0, 1)
    plt.legend(["x(t)", "max_spline", "min_spline", "spline_mean", "maxima", "minima"])

    plt.figure(figsize=(10, 3))
    plt.plot(t, res1, c="k", lw=1.2)
    plt.plot(t, ma_s2, c="r", lw=0.5, linestyle="--")
    plt.plot(t, mi_s2, c="orange", lw=0.5, linestyle="--")
    plt.plot(t, mean2, c="blue", lw=1.)
    plt.scatter(t.take(ma2_ind), res1.take(ma2_ind), c="darkred", s=20, alpha=0.8)
    plt.scatter(t.take(mi2_ind), res1.take(mi2_ind), c="green", s=20, alpha=0.8)
    plt.xlim(0, 1)

    plt.figure(figsize=(10, 3))
    plt.plot(t, res2, c="k", lw=1.2)
    plt.xlim(0, 1)
    plt.show()


def lmd_sifting_show(t, signal):

    lmd = LocalMeanDecomp()
    e_ind1, _m, _a, window = lmd.get_mean_envelope_window(signal)
    ma_m1, ma_a1 = lmd.multiple_moving_average(_m, window), lmd.multiple_moving_average(_a, window)
    hij = signal - ma_m1
    s1 = hij / ma_a1

    e_ind2, _m, _a, window = lmd.get_mean_envelope_window(s1)
    ma_m2, ma_a2 = lmd.multiple_moving_average(_m, window), lmd.multiple_moving_average(_a, window)
    hij = s1 - ma_m2
    s2 = hij / ma_a2

    plt.figure(figsize=(10, 2))
    plt.plot(t, signal, c="k", lw=1.5)
    plt.plot(t, ma_m1, c="green", lw=1.2)
    plt.scatter(t.take(e_ind1), signal.take(e_ind1), c="darkred", s=20, alpha=0.8)
    plt.legend(["x(t)", "ma_mean", "extrema"])
    plt.xlim(0, 1)
    plt.xticks([])

    plt.figure(figsize=(10, 2))
    plt.plot(t, signal - ma_m1, c="blue", lw=1.5)
    plt.plot(t, ma_a1, c="r", lw=0.8, linestyle="--")
    plt.xlim(0, 1)
    plt.legend(["x(t)-mean(t)", "ma_amp"])

    plt.figure(figsize=(10, 2))
    plt.plot(t, s1, c="k", lw=1.5)
    plt.plot(t, ma_m2, c="green", lw=1.2)
    plt.scatter(t.take(e_ind2), s1.take(e_ind2), c="darkred", s=20, alpha=0.8)
    plt.xlim(0, 1)
    plt.xticks([])
    plt.figure(figsize=(10, 2))
    plt.plot(t, s1 - ma_m2, c="blue", lw=1.5)
    plt.plot(t, ma_a2, c="r", lw=0.8, linestyle="--")
    plt.xlim(0, 1)

    plt.figure(figsize=(10, 3))
    plt.plot(t, s2, c="k", lw=1.2)
    plt.xlim(0, 1)
    plt.show()


def inst_freq_representation(coord, signal, imfs, inst_freqs, amps, res,
                             freq_max=None, same_lim=True):

    if not freq_max:
        freq_max = max(np.mean(inst_freqs[0]) + 3 * np.std(inst_freqs[0]), 1.1 * np.max(inst_freqs))

    num_imf = len(imfs)
    num_total = num_imf + 2
    time_max, time_min = np.max(signal), np.min(signal)
    freq_max, freq_min = freq_max, 0

    plt.figure(figsize=(14, 8))
    plt.subplot(num_total, 2, 1)
    plt.plot(coord, signal, "k", lw=1.5)
    plt.xlim((coord[0], coord[-1]))
    if same_lim:
        plt.ylim(time_min, time_max)
    plt.ylabel("Signal")

    ylims = [(-10, 50), (-2, 10), (-1, 5)]

    for num in range(num_imf):
        plt.subplot(num_total, 2, 2 * (num + 1) + 1)
        plt.plot(coord, imfs[num], 'k', lw=1.5)
        plt.plot(coord, amps[num], 'r', linestyle="--", lw=1.)
        plt.plot(coord, -amps[num], 'r', linestyle="--", lw=1.)
        plt.xlim((coord[0], coord[-1]))
        if same_lim:
            plt.ylim(time_min, time_max)
        plt.ylabel("IMF_ " + str(num + 1))

        plt.subplot(num_total, 2, 2 * (num + 1) + 2)
        plt.plot(coord, inst_freqs[num], 'k', lw=1.)
        plt.xlim((coord[0], coord[-1]))
        plt.ylim(ylims[num])
        if same_lim:
            plt.ylim(freq_min, freq_max)
        plt.ylabel("IF_ " + str(num + 1))

    plt.subplot(num_total, 2, num_total * 2 - 1)
    plt.plot(coord, res, "k", lw=0.8)
    plt.xlim((coord[0], coord[-1]))
    if same_lim:
        plt.ylim(time_min, time_max)
    plt.ylabel("Res")


def inst_freq_representation1(coord, signal, imfs, inst_freqs, amps, res,
                             freq_max=None, same_lim=True):

    if not freq_max:
        freq_max = max(np.mean(inst_freqs[0]) + 3 * np.std(inst_freqs[0]), 1.1 * np.max(inst_freqs))

    num_imf = len(imfs)
    time_max, time_min = np.max(signal), np.min(signal)
    freq_max, freq_min = freq_max, 0

    plt.figure(figsize=(20, 8))

    ylims = [(-10, 50), (-2, 10), (-1, 5)]
    for num in range(num_imf):
        plt.subplot(num_imf, 1, (num + 1))
        plt.plot(coord, inst_freqs[num], 'k', lw=1.)
        plt.xlim((coord[0], coord[-1]))
        plt.ylim(ylims[num])
        if same_lim:
            plt.ylim(freq_min, freq_max)
        plt.ylabel("IF_ " + str(num + 1))

# hyper-parameter
DIM = 1024
START = 0
END = 1
DTYPE = np.float64
CHOICE = 2
fig_size = (10, 2.5)
cc = 0


# signal generate
t = np.linspace(START, END, DIM, dtype=DTYPE)
signal1 = np.cos(200 * np.pi * t + 2 * np.cos(10 * np.pi * t)) * (1 + 0.5 * np.cos(9 * np.pi * t)) + \
          3 * np.cos(20 * np.pi * t ** 2 + 6 * np.pi * t)
signal2 = 5 * (1 + np.cos(40 * np.pi * t)) * np.cos(200 * np.pi * t) + 15 * (1 + np.cos(40 * np.pi * t)) * \
          np.cos(600 * np.pi * t)
signal3 = 18 * np.sin(5 * np.pi * t) * np.cos(40 * np.pi * t) + 30 * np.cos(9 * np.pi * t)
signal4 = (2 + np.cos(18 * np.pi * t)) * np.cos(400 * np.pi * t + 2 * np.cos(20 * np.pi * t)) + \
          4 * np.cos(5 * np.pi * t) * np.cos(120 * np.pi * t) + \
          2 * np.cos(40 * np.pi * t) + 0.8 * np.sin(np.pi * t) * np.sin(10 * np.pi * t)
signals = np.array([signal1, signal2, signal3, signal4])


x1 = 2 * np.cos(2 * np.pi * t + 0.25 * np.pi) * np.cos(40 * np.pi * t)
x2 = 10 * np.cos(4 * np.pi * t)
signal = x1 + x2

# decomposition
# 1. EMD
emd = EmpiricalModeDecomp()
imfs, res_emd = emd(signals[CHOICE])
# 2. LMD
lmd = LocalMeanDecomp()
lmd.pf_max_iter = 2
s, envelopes, res_lmd = lmd(signal)
pfs = [s_ * e_ for s_, e_ in zip(s, envelopes)]

imfs_show(t, signal, imfs, res_emd, same_lim=False)
imfs_show(t, signal, pfs, res_lmd, same_lim=False, name="PF")
plt.show()


from EMD_hu.inst_freq import InstantaneousFreq
inst = InstantaneousFreq()
inst_freqs, amps = inst.arccos_method(imfs, sampling_rate=DIM)
hht_inst, hht_amps = inst.hilbert_method(imfs, sampling_rate=DIM)

# imfs_show(t, signals[CHOICE], imfs, res_emd, same_lim=False)
inst_freq_representation(t, signals[CHOICE], imfs, inst_freqs, amps, res_emd, same_lim=False)
plt.savefig("arccos_show.png")
inst_freq_representation(t, signals[CHOICE], imfs, hht_inst, hht_amps, res_emd, same_lim=False)
plt.savefig("hilbert_show.png")

# plt.savefig("IMF_show.png")
plt.show()

# ma_ind, mi_ind = emd.find_maxima_minima(imfs[cc])
# plt.figure(figsize=fig_size)
# plt.plot(t, imfs[cc], c="k", lw=1.5)
# plt.plot(t, amps[cc], c="r", lw=1., linestyle="--")
# plt.plot(t, -amps[cc], c="r", lw=1., linestyle="--")
# plt.scatter(t.take(ma_ind), imfs[cc].take(ma_ind), c="green", s=20, alpha=0.8)
# plt.scatter(t.take(mi_ind), imfs[cc].take(mi_ind), c="blue", s=20, alpha=0.8)
# # plt.legend(["IMF", "Amplitude", "Amplitude", "Maxima", "Minima"])
# plt.xlim(0, 1)
#
# puree = imfs[cc]/amps[cc]
# puree = inst._check_pure_freq_modulation(puree)
# plt.figure(figsize=fig_size)
# plt.plot(t, puree, c="k", lw=1.5)
# plt.xlim(0, 1)
# plt.ylim(-1.5, 1.5)
#
# plt.figure(figsize=fig_size)
# plt.plot(t, inst_freqs[cc], c="k", lw=1.5)
# plt.xlim(0, 1)
# plt.ylim(0, 100)
# plt.show()