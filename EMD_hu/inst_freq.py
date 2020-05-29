#!~D:\\Python_Projects\\narrow_band_decomp
# coding: UTF-8

"""
    Author: Sirius HU
    Created Date: 2019.11.28
"""

from __future__ import division, print_function
import logging
from scipy.signal import hilbert
import numpy as np
from EMD_hu.emd import EmpiricalModeDecomp
import matplotlib.pyplot as plt


class InstantaneousFreq(EmpiricalModeDecomp):

    """
        Get Instantaneous Frequency by mainly two methods:
            1. Hilbert Transform
            2. amplitude envelope and frequency-modulated signal sifting + arc-cosine func

        for the second method, two sifting mode are given:
            2.1. spline
            2.2. moving average
    """

    def __init__(self):

        super(EmpiricalModeDecomp, self).__init__()
        self.method = "hilbert"     # 'hilbert' or 'arccos'
        self.sift_mode = "spline"   # 'spline' or 'ma'
        self.spline_kind = "cubic"
        self.ma_mode = "simple"     # 'simple', 'triangle'
        self.max_ma_iter = 20
        self.mirror_pnums = 2

    @staticmethod
    def _check_input(imfs):

        if not isinstance(imfs, np.ndarray):
            imfs = np.array(imfs)

        if len(imfs.shape) == 1:
            imfs = imfs.reshape(1, -1)

        if len(imfs.shape) != 2:
            raise AttributeError("expected IMFs.shape=(num, length), but got {}" .format(imfs.shape))

        return imfs

    # hilbert method:
    @staticmethod
    def hilbert_inst_freq_amp(imf, sampling_rate=None):

        if not sampling_rate:
            sampling_rate = 1.

        analytic_signal = hilbert(imf)

        # calculate instantaneous frequency
        phases = np.unwrap(np.angle(analytic_signal))
        inst_freq = np.diff(phases, n=1) / (2 * np.pi) * sampling_rate
        # append to keep the same dim
        inst_freq = np.append(inst_freq[:1], inst_freq)

        # calculate instantaneous amplitude of the corresponding frequency
        amp = np.abs(analytic_signal)

        return inst_freq, amp

    def hilbert_method(self, imfs, sampling_rate=None):

        inst_freqs, amps = [], []
        for imf in imfs:
            _freq, _amp = self.hilbert_inst_freq_amp(imf, sampling_rate)
            inst_freqs.append(_freq)
            amps.append(_amp)
        inst_freqs, amps = np.array(inst_freqs), np.array(amps)

        return inst_freqs, amps

    # pure frequency demodulation + arc-cosine method
    @staticmethod
    def _check_alternate_extrema(phase, maxima_index, minima_index):

        # to guarantee the indexes are mono-increasing
        maxima_index = maxima_index[np.argsort(maxima_index)]
        minima_index = minima_index[np.argsort(minima_index)]

        # if np.any(phase.take(maxima_index) < np.pi / 2) or \
        #         np.any(phase.take(minima_index) > np.pi / 2):
        #     raise AttributeError("expected maxima > 0 and minima < 0, but got exception. \n"
        #                          "please check the phase!")
        #
        # if abs(len(maxima_index) - len(minima_index)) > 1:
        #     raise AttributeError("expected difference between number of maxima and number of minima "
        #                          "should be lower than 2, but got %d"
        #                          % abs(len(maxima_index) - len(minima_index)))

        if maxima_index[0] > minima_index[0]:
            for i in range(min(len(minima_index), len(maxima_index))):
                if maxima_index[i] < minima_index[i]:
                    return False
        elif maxima_index[0] < minima_index[0]:
            for i in range(min(len(minima_index), len(maxima_index))):
                if maxima_index[i] > minima_index[i]:
                    return False
        else:
            raise AttributeError("extrema index should be significant!")

        return True

    @staticmethod
    def _extrema_smoothing(phase, extrema_interval, correct_ind):

        # smoothing the extrema point
        for _ind in correct_ind:
            if _ind - 3 >= 0 and _ind + 3 < len(phase) and 3 <= extrema_interval:
                phase[_ind] = (phase[_ind - 3] + phase[_ind + 3]) / 2
                phase[_ind - 2] = phase[_ind - 3] * 2 / 3 + phase[_ind] * 1 / 3
                phase[_ind - 1] = phase[_ind - 3] * 1 / 3 + phase[_ind] * 2 / 3
                phase[_ind + 1] = phase[_ind] * 2 / 3 + phase[_ind + 3] * 1 / 3
                phase[_ind + 2] = phase[_ind] * 1 / 3 + phase[_ind + 3] * 2 / 3
            elif _ind - 2 >= 0 and _ind + 2 < len(phase) and 2 <= extrema_interval:
                phase[_ind] = (phase[_ind - 2] + phase[_ind + 2]) / 2
                phase[_ind - 1] = (phase[_ind - 2] + phase[_ind]) / 2
                phase[_ind + 1] = (phase[_ind] + phase[_ind + 2]) / 2
            else:
                phase[_ind] = (phase[_ind - 1] + phase[_ind + 1]) / 2

        return phase

    def _arccos_unwrap(self, phases):

        for phase in phases:
            maxima_index, minima_index = self.find_maxima_minima(phase)
            if not self._check_alternate_extrema(phase, maxima_index, minima_index):
                raise AttributeError("expected indexes of maxima and minima should be alternate, "
                                     "but got exception!")

            if minima_index[0] < maxima_index[0]:
                phase[:minima_index[0]] -= np.pi
                phase[:minima_index[0]] *= -1

                if len(minima_index) > len(maxima_index):
                    for i in range(len(maxima_index)):
                        phase[maxima_index[i]: minima_index[i + 1]] -= np.pi
                        phase[maxima_index[i]: minima_index[i + 1]] *= -1

                elif len(minima_index) == len(maxima_index):
                    for i in range(len(maxima_index) - 1):
                        phase[maxima_index[i]: minima_index[i + 1]] -= np.pi
                        phase[maxima_index[i]: minima_index[i + 1]] *= -1
                    phase[maxima_index[-1]:] -= np.pi
                    phase[maxima_index[-1]:] *= -1
                else:
                    raise AttributeError("got exception !")
            else:
                if len(minima_index) < len(maxima_index):
                    for i in range(len(maxima_index) - 1):
                        phase[maxima_index[i]: minima_index[i]] -= np.pi
                        phase[maxima_index[i]: minima_index[i]] *= -1
                    phase[maxima_index[-1]:] -= np.pi
                    phase[maxima_index[-1]:] *= -1
                elif len(minima_index) == len(maxima_index):
                    for i in range(len(maxima_index)):
                        phase[maxima_index[i]: minima_index[i]] -= np.pi
                        phase[maxima_index[i]: minima_index[i]] *= -1
                else:
                    raise AttributeError("got exception !")

            # unwrap the phase signal
            correct_ind = np.sort(np.append(maxima_index, minima_index))
            for _ind in correct_ind:
                phase[_ind:] += np.pi

            extrema_interval = np.min(np.diff(np.sort(np.append(maxima_index, minima_index))))
            self._extrema_smoothing(phase, extrema_interval, correct_ind)

            """fix me"""
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(phase, c="k")
            # plt.scatter(maxima_index, phase.take(maxima_index), c="r")
            # plt.scatter(minima_index, phase.take(minima_index), c="g")
            # plt.show()

        return phases

    def arccos_inst_freq(self, signals, sampling_rate=None):

        if not sampling_rate:
            sampling_rate = 1

        # calculate instantaneous frequency
        phases = np.arccos(signals)

        # get the monotonic increasing phases signal
        phases = self._arccos_unwrap(phases)
        # # calculate the instantaneous frequency
        # inst_freqs = np.diff(phases, n=1) / (2 * np.pi) * sampling_rate
        # # append to keep the same dim
        # inst_freqs = np.concatenate((inst_freqs[:, :1], inst_freqs), axis=-1)

        inst_freqs = np.gradient(phases, axis=-1) / (2 * np.pi) * sampling_rate

        return inst_freqs

    @staticmethod
    def _is_smoothed(curve):

        if np.any(np.diff(curve, n=1) == 0):
            return False

        return True

    @staticmethod
    def get_window(extrema_index, ma_mode):

        """
            1. get the max interval of extrema_index (time coordinates),
               set win_size as max / 3
            2. get the mean interval of extrema_index
               set win_size as mean
            3. other strategy

            here we use the first strategy


        """

        _win = np.max(np.diff(extrema_index, n=1))
        win_size = np.ceil(_win / 3)
        if win_size % 2 == 0:
            win_size += 1

        if win_size < 3:
            win_size = 3

        win_size = int(win_size)
        if ma_mode == 'simple':
            window = np.ones((win_size,)) / win_size

        elif ma_mode == 'triangle':
            window = np.arange(1, win_size // 2 + 2)
            window = np.concatenate((window, window[:-1][::-1]), axis=0)
        else:
            raise ValueError("expected ma mode should be within ['simple', 'triangle'], but got %s instead !"
                             % str(ma_mode))

        return window

    @staticmethod
    def get_raw_amp(coord, extrema, zero_index):

        # init amplitude curve and set its coordinate
        amp_curve = np.zeros((2, int(np.max(extrema[0]) - np.min(extrema[0]) + 1)))
        amp_curve[0, :] = np.arange(np.min(extrema[0]), np.max(extrema[0]) + 1, 1)

        # assign value according the zero-crossing points
        _ind = 0
        for i, (pre, post) in enumerate(zip(extrema[0, :-1], extrema[0, 1:])):
            pre, post = int(pre), int(post)

            # find the zero-crossing point between two adjacent extrema
            _zero_have = False
            while _ind < len(zero_index):
                if zero_index[_ind] < pre:
                    _ind += 1
                    continue
                elif zero_index[_ind] > post:
                    break
                else:
                    _zero_have = True
                    break

            # if there have a zero-crossing point, set the mid-point as the zero-crossing point,
            # if not, set the mid-point as the middle of the two adjacent extrema
            if _zero_have:
                mid_point = int(np.round(zero_index[_ind], 0))
            else:
                mid_point = (pre + post) // 2

            # calculate index
            _start = int(np.min(extrema[0]))
            pre_index, mid_index, post_index = pre - _start, mid_point - _start, post - _start

            # assign value
            amp_curve[1, pre_index: mid_index] = extrema[1, i]
            amp_curve[1, mid_index: post_index] = extrema[1, i + 1]
        # make sure the last point of the curve has a value
        amp_curve[1, -1] = amp_curve[1, -2]

        # according to the coordinate range, extract raw amplitude envelope from the curve
        is_extracted = (amp_curve[0] <= coord[-1]) & (amp_curve[0] >= coord[0])
        amp_envelope = amp_curve[1, is_extracted]

        return amp_envelope

    def moving_avg(self, coord, extrema, zero_points):

        # get the weight of the moving windows
        window = self.get_window(extrema[0], self.ma_mode)

        # get the raw amplitude envelope and window size for moving average
        raw_envelope = self.get_raw_amp(coord, extrema, zero_points)

        # implement moving average until we get smooth condition or reach max-iteration times
        pre_envelope, post_envelope = raw_envelope.copy(), np.zeros(raw_envelope.shape[-1])
        for j in range(self.max_ma_iter):

            #   1. according to the window size, padding both end of the pre-envelope
            win_size = window.shape[-1]
            _amp = np.concatenate((pre_envelope[:win_size//2][::-1],
                                   pre_envelope,
                                   pre_envelope[-win_size//2 + 1:][::-1]))
            length = _amp.shape[-1]

            #   2. convolve to get the next round of envelope (post-envelope)
            for i in range(window.shape[-1]):
                post_envelope += _amp[i:length - win_size + 1 + i] * window[i]

            #   3. exchange the value of the pre and post envelope
            pre_envelope[:] = post_envelope[:]
            post_envelope[:] = 0

            #   4. check if the envelope is smooth, if it is, break
            if self._is_smoothed(pre_envelope):
                break

        return pre_envelope

    @staticmethod
    def find_zero_crossing(signal):

        index = np.arange(signal.shape[-1], dtype=signal.dtype)

        # find zero points
        is_zero = signal == 0
        zero_point1 = index[is_zero]

        # find pairs of zero crossing points
        is_zero_cross = (signal[:-1] * signal[1:]) < 0
        _left = np.array(tuple(is_zero_cross) + (False, ))
        _right = np.array((False, ) + tuple(is_zero_cross))
        zero_point2 = signal[_left] / (signal[_left] - signal[_right]) + index[_left]

        return np.sort(np.append(zero_point1, zero_point2))

    def _check_pure_freq_modulation(self, signal):

        # divided by max value
        max_val = np.max(np.abs(signal))
        signal /= max_val

        # for each peak, normalization
        maxima_index, minima_index = self.find_maxima_minima(signal)
        extrema_index = np.sort(np.append(maxima_index, minima_index))
        zero_points = self.find_zero_crossing(signal)
        index = np.arange(signal.shape[-1])
        for pre, post in zip(zero_points[:-1], zero_points[1:]):
            _ind = (index > pre) & (index < post)
            max_val = np.max(np.abs(signal[_ind]))
            signal[_ind] /= max_val

        if extrema_index[0] < zero_points[0]:
            max_val = np.max(np.abs(signal[: int(zero_points[0])]))
            signal[: int(zero_points[0])] /= max_val

        if extrema_index[-1] > zero_points[-1]:
            max_val = np.max(np.abs(signal[int(zero_points[-1] + 1):]))
            signal[int(zero_points[-1] + 1):] /= max_val

        return signal

    def _demodulate(self, coord, imf, ee=1e-8):

        # get extrema of IMF
        maxima_index, minima_index = self.find_maxima_minima(imf)

        # mirror extent the maxima and minima
        maxima, minima = self.mirror_extension(coord, imf, maxima_index, minima_index)

        # concatenate extrema
        extrema = np.concatenate((maxima, minima), axis=-1)
        index_sorted = np.argsort(extrema[0])
        extrema = extrema[:, index_sorted]

        # we believe that the mean curve of a well-decomposed IMF is near to zero,
        # so we direct abs the extrema to get an amplitude curve
        extrema[1, :] = np.abs(extrema[1, :])

        if self.sift_mode == "spline":
            # spline amplitude curve
            _, amp = self.splines(coord, extrema)
        elif self.sift_mode == "ma":
            zero_points = self.find_zero_crossing(imf)
            amp = self.moving_avg(coord, extrema, zero_points)
        else:
            raise ValueError("expected sift mode is within ('spline', 'ma'), but got %s"
                             % str(self.sift_mode))

        # make sure there all components in the amplitude envelope is bigger than zero
        amp = np.abs(amp)
        amp[amp == 0] += ee

        # sift freq_module
        freq_modulated_sig = imf / (amp + ee)

        # check the demodulated freq signal to make sure it's extrema values are 1
        freq_modulated_sig = self._check_pure_freq_modulation(freq_modulated_sig)

        return freq_modulated_sig, amp

    def freq_amp_demodulate(self, imfs):

        coord = np.arange(imfs.shape[-1])
        coord, imfs = self._check_dtype(coord, imfs)

        modulated_freq_sigs, amps = [], []
        for imf in imfs:
            freq, amp = self._demodulate(coord, imf)
            modulated_freq_sigs.append(freq)
            amps.append(amp)
        modulated_freq_sigs, amps = np.array(modulated_freq_sigs), np.array(amps)

        return modulated_freq_sigs, amps

    def arccos_method(self, imfs, sampling_rate=None):

        imfs = self._check_input(imfs)
        modulated_freq_sigs, amps = self.freq_amp_demodulate(imfs)
        inst_freqs = self.arccos_inst_freq(modulated_freq_sigs, sampling_rate)

        return inst_freqs, amps

    @staticmethod
    def time_freq_representation(inst_freqs, amps,
                                 freq_dim=None, max_freq=None, sampling_rate=None):

        if not max_freq:
            if not sampling_rate:
                max_freq = max(np.mean(inst_freqs[0]) + 3 * np.std(inst_freqs[0]), 1.1 * np.max(inst_freqs))
            if sampling_rate:
                max_freq = sampling_rate / 2

        if not sampling_rate:
            sampling_rate = 1.

        time_dim = inst_freqs.shape[-1]
        time_resolution = 1. / sampling_rate
        if not freq_dim:
            freq_dim = time_dim
        freq_resolution = max_freq / freq_dim

        amps = np.log(amps + 1) + 0.1
        tfr = np.zeros((time_dim, freq_dim), dtype=np.float32)
        for IF, amp in zip(inst_freqs, amps):
            for t, (f, a) in enumerate(zip(IF, amp)):
                if f < 0 or f >= max_freq:
                    continue
                f_c = f / freq_resolution
                f_low, f_high = int(f_c), int(np.ceil(f_c))

                if f_high >= freq_dim:
                    print(1)
                    tfr[t, f_low] = a
                else:
                    tfr[t, f_low], tfr[t, f_high] = (f_c - f_low) * a, (f_high - f_c) * a

        xx, yy = np.meshgrid(np.arange(0, time_dim / sampling_rate, time_resolution),
                             np.arange(0, max_freq, freq_resolution))

        tfr = tfr.transpose(1, 0)
        plt.figure(figsize=(7, 6))
        plt.contourf(xx, yy, tfr, cmap=plt.cm.hot, levels=np.arange(0, np.max(amps), 0.1))
        plt.colorbar()
        # plt.show()

    @staticmethod
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
        plt.plot(coord, signal, "b", lw=1)
        plt.xlim((coord[0], coord[-1]))
        if same_lim:
            plt.ylim(time_min, time_max)
        plt.ylabel("Signal")

        for num in range(num_imf):
            plt.subplot(num_total, 2, 2 * (num + 1) + 1)
            plt.plot(coord, imfs[num], 'k', lw=0.8)
            plt.plot(coord, amps[num], 'r', linestyle="--", lw=0.5)
            plt.plot(coord, -amps[num], 'r', linestyle="--", lw=0.5)
            plt.xlim((coord[0], coord[-1]))
            if same_lim:
                plt.ylim(time_min, time_max)
            plt.ylabel("IMF_ " + str(num + 1))

            plt.subplot(num_total, 2, 2 * (num + 1) + 2)
            plt.plot(coord, inst_freqs[num], 'r', lw=0.8)
            plt.xlim((coord[0], coord[-1]))
            if same_lim:
                plt.ylim(freq_min, freq_max)
            plt.ylabel("IF_ " + str(num + 1))

        plt.subplot(num_total, 2, num_total * 2 - 1)
        plt.plot(coord, res, "g", lw=0.8)
        plt.xlim((coord[0], coord[-1]))
        if same_lim:
            plt.ylim(time_min, time_max)
        plt.ylabel("Res")

    @staticmethod
    def inst_freq_ma_smooth(inst_freq, window=7):

        minus_index = inst_freq < 0
        inst_freq[minus_index] = 0
        if window < 2:
            return

        if isinstance(window, int):
            window = int(window)

        if window % 2 == 0:
            window += 1

        kkk = np.concatenate((inst_freq[: window // 2][::-1],
                              inst_freq,
                              inst_freq[-window // 2 + 1:][::-1]), axis=-1)
        length = kkk.shape[-1]
        inst_new = np.zeros(inst_freq.shape)
        for i in range(window):
            inst_new += kkk[i: length - window + 1 + i] / window
        inst_freq[:] = inst_new


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # param setting
    DTYPE = np.float64
    N, tMin, tMax = 5000, 0, 1

    # signal generate
    coord = np.linspace(tMin, tMax, N, dtype=DTYPE)
    # signal = 18 * np.sin(5 * np.pi * T * (1 + 0.2 * T)) + T ** 2 + T * np.sin(13 * T)
    signal = 18 * np.sin(5 * np.pi * coord) * np.cos(40 * np.pi * coord) + 30 * np.cos(9 * np.pi * coord)
    # signal = (2 + np.cos(18 * np.pi * coord)) * np.cos(400 * np.pi * coord + 2 * np.cos(20 * np.pi * coord)) + \
    #          4 * np.cos(5 * np.pi * coord) * np.cos(120 * np.pi * coord) + \
    #          2 * np.cos(40 * np.pi * coord) + 0.8 * np.sin(np.pi * coord) * np.sin(10 * np.pi * coord)
    signal = signal.astype(DTYPE)

    # emd extract imfs
    emd = EmpiricalModeDecomp()
    imfs, res = emd(signal)
    imfsNo = len(imfs)

    # inst_freq
    ifs = InstantaneousFreq()

    # 1
    inst_freqs, amps = ifs.hilbert_method(imfs, sampling_rate=5000)
    ifs.time_freq_representation(inst_freqs, amps, freq_dim=400, sampling_rate=1000)
    ifs.inst_freq_representation(coord, signal, imfs, inst_freqs, amps, res, same_lim=False)

    # 2
    inst_freqs, amps = ifs.arccos_method(imfs, sampling_rate=5000)
    ifs.time_freq_representation(inst_freqs, amps, freq_dim=400, sampling_rate=1000)
    ifs.inst_freq_representation(coord, signal, imfs, inst_freqs, amps, res, same_lim=False)

    # 3
    imfs = ifs._check_input(imfs)
    modulated_freq_sigs, amps = ifs.freq_amp_demodulate(imfs)
    inst_freqs, amps1 = ifs.hilbert_method(modulated_freq_sigs, sampling_rate=5000)
    amps = amps * amps1

    ifs.time_freq_representation(inst_freqs, amps, freq_dim=400, sampling_rate=1000)
    ifs.inst_freq_representation(coord, signal, imfs, inst_freqs, amps, res, same_lim=False)
    plt.show()

    """
        EMD + spline 分解时，由于分解误差，造成极值点出，频率大范围波动
            采用修正方式 1 ： 对每两个相邻的过零点 所包围的 波 进行归一化，使其 arc-cosine 的值 一定经过 pi，
                            一定程度上消除了大范围的波动
    """
