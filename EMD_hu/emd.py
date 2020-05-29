#!~/opt/anaconda3/python.app/Contents/MacOS/python
# coding: UTF-8

"""
    Author: Sirius HU
    Email: sirius.hu@163.com
    Created Date: 2019.11.22

    Empirical Mode Decomposition

"""

# 对 python 2.X 版本，通过从 __future__ 导入 division, print_function, 可以
# 1、要求编程者必须按 python 3 的语法来，
# 2、python 3 的文档在 python 2.X 环境下也能运行
from __future__ import division, print_function
import logging
from scipy.interpolate import interp1d, Akima1DInterpolator
import numpy as np


class EmpiricalModeDecomp(object):

    """
        Empirical Mode Decomposition
            Method of decomposing signal into Intrinsic Mode Functions (IMFs)
            based on algorithm presented in Huang et al. [Huang1998]
    """

    def __init__(self):

        self.mirror_pnums = 2
        self.spline_kind = "cubic"

        # IMF end threshold
        self.scale_threshold = 0.001
        self.std_threshold = 0.2
        self.energy_threshold = 0.2

        # EMD end threshold
        self.power_threshold = 0.005
        self.range_threshold = 0.001
        self.max_imfs = 4

        self.logger = logging.getLogger(__name__)

    def __call__(self, signal, coord=None):
        return self.emd(coord, signal)

    @staticmethod
    def _check_dtype(coord, signal):

        data_type = signal.dtype
        coord = coord.astype(data_type)

        return coord, signal

    @staticmethod
    def _check_extrema(extrema):

        duplicate_idx = np.where(extrema[0, 1:] == extrema[0, :-1])
        return np.delete(extrema, duplicate_idx, axis=1)

    @staticmethod
    def find_maxima_minima(signal):

        # detect the extrema,
        # where the product of the pre_diff and post_diff of the extrema must be < 0
        diff_s = np.diff(signal, n=1)
        is_maxima = (diff_s[:-1] > 0) & ((diff_s[1:] * diff_s[:-1]) < 0)
        is_minima = (diff_s[:-1] < 0) & ((diff_s[1:] * diff_s[:-1]) < 0)

        # detect bad points, where there are continuous points with same value --> diff == 0
        diff_s_zero = diff_s == 0

        # calculate the mean point of continuous bad points, as the new extrema
        pre_index = 0
        pre_grad, post_grad = True, True
        while pre_index < len(diff_s_zero):

            # use 'pre_grad' as a tag,
            # to detect whether the bad points is maximum or is minimum
            if not diff_s_zero[pre_index]:
                pre_grad = diff_s[pre_index] > 0
                pre_index += 1
                continue

            # this is the index of diff_s_zero, not index in the original signal
            post_index = pre_index
            while post_index < len(diff_s_zero) and diff_s_zero[post_index]:
                post_index += 1
            mean_index = (pre_index + post_index - 1) // 2 - 1
            pre_index = post_index + 1

            if post_index == len(diff_s_zero):
                post_grad = ~pre_grad
            else:
                post_grad = diff_s[post_index] > 0

            if pre_grad and ~post_grad:
                is_maxima[mean_index] = True
            elif ~pre_grad and post_grad:
                is_minima[mean_index] = True
            else:
                # in this case, the gradient of the pre-bad-points is the same as the gradient of the post-bad-points
                # indicating that, those bad points are a period of interruption during a monotonous process
                continue

        is_maxima = np.array((False,) + tuple(is_maxima) + (False,))
        is_minima = np.array((False,) + tuple(is_minima) + (False,))

        index = np.arange(0, np.shape(signal)[0])
        maxima_index = index[is_maxima]
        minima_index = index[is_minima]

        return maxima_index, minima_index

    ''' this func is mainly according to the func in PyEMD written by @Dawid Laszuk '''
    def mirror_extension(self, coord, signal, maxima_index, minima_index):

        maxima_index, minima_index = maxima_index.astype(np.int), minima_index.astype(np.int)
        min_len, max_len = len(minima_index), len(maxima_index)

        # Local variables
        pnums = self.mirror_pnums

        # left bound - mirror (pnums) points to the left
        if maxima_index[0] < minima_index[0]:
            if signal[0] > signal[minima_index[0]]:
                # set the first maximum point as the sym-point
                left_ma_index = maxima_index[1: min(max_len, pnums + 1)][::-1]
                left_mi_index = minima_index[0: min(min_len, pnums + 0)][::-1]
                left_sym_index = maxima_index[0]
            else:
                left_ma_index = maxima_index[0: min(max_len, pnums)][::-1]
                left_mi_index = np.append(minima_index[0: min(min_len, pnums - 1)][::-1], 0)
                left_sym_index = 0
        else:
            if signal[0] < signal[maxima_index[0]]:
                left_ma_index = maxima_index[0: min(max_len, pnums + 0)][::-1]
                left_mi_index = minima_index[1: min(min_len, pnums + 1)][::-1]
                left_sym_index = minima_index[0]
            else:
                left_ma_index = np.append(maxima_index[0:min(max_len, pnums - 1)][::-1], 0)
                left_mi_index = minima_index[0:min(min_len, pnums)][::-1]
                left_sym_index = 0

        # right bound - mirror pnums points to the right
        if maxima_index[-1] < minima_index[-1]:
            if signal[-1] < signal[maxima_index[-1]]:
                right_ma_index = maxima_index[max(max_len - pnums, 0):][::-1]
                right_mi_index = minima_index[max(min_len - pnums - 1, 0):-1][::-1]
                right_sym_index = minima_index[-1]
            else:
                right_ma_index = np.append(maxima_index[max(max_len - pnums + 1, 0):], len(signal) - 1)[::-1]
                right_mi_index = minima_index[max(min_len - pnums, 0):][::-1]
                right_sym_index = len(signal) - 1
        else:
            if signal[-1] > signal[minima_index[-1]]:
                right_ma_index = maxima_index[max(max_len - pnums - 1, 0):-1][::-1]
                right_mi_index = minima_index[max(min_len - pnums, 0):][::-1]
                right_sym_index = maxima_index[-1]
            else:
                right_ma_index = maxima_index[max(max_len - pnums, 0):][::-1]
                right_mi_index = np.append(minima_index[max(min_len - pnums + 1, 0):], len(signal) - 1)[::-1]
                right_sym_index = len(signal) - 1

        # In case any array missing
        if not left_mi_index.size:
            left_mi_index = minima_index[::-1]
        if not right_mi_index.size:
            right_mi_index = minima_index[::-1]
        if not left_ma_index.size:
            left_ma_index = maxima_index[::-1]
        if not right_ma_index.size:
            right_ma_index = maxima_index[::-1]

        # the coordinates of left/right mirrored points
        coord_left_mi = 2 * coord[left_sym_index] - coord[left_mi_index]
        coord_left_ma = 2 * coord[left_sym_index] - coord[left_ma_index]
        coord_right_mi = 2 * coord[right_sym_index] - coord[right_mi_index]
        coord_right_ma = 2 * coord[right_sym_index] - coord[right_ma_index]

        # if mirrored points are not outside passed time range.
        # left case
        if coord_left_mi[0] > coord[0] or coord_left_ma[0] > coord[0]:
            if left_sym_index == maxima_index[0]:
                left_ma_index = maxima_index[0:min(max_len, pnums)][::-1]
            else:
                left_mi_index = minima_index[0:min(min_len, pnums)][::-1]
            if left_sym_index == 0:
                raise Exception('Left edge BUG')

            left_sym_index = 0
            coord_left_mi = 2 * coord[left_sym_index] - coord[left_mi_index]
            coord_left_ma = 2 * coord[left_sym_index] - coord[left_ma_index]
        # right case
        if coord_right_mi[-1] < coord[-1] or coord_right_ma[-1] < coord[-1]:
            if right_sym_index == maxima_index[-1]:
                right_ma_index = maxima_index[max(max_len - pnums, 0):][::-1]
            else:
                right_mi_index = minima_index[max(min_len - pnums, 0):][::-1]

            if right_sym_index == len(signal) - 1:
                raise Exception('Right edge BUG')

            right_sym_index = len(signal) - 1
            coord_right_mi = 2 * coord[right_sym_index] - coord[right_mi_index]
            coord_right_ma = 2 * coord[right_sym_index] - coord[right_ma_index]

        # the values of left/right mirrored points
        value_left_ma = signal[left_ma_index]
        value_left_mi = signal[left_mi_index]
        value_right_ma = signal[right_ma_index]
        value_right_mi = signal[right_mi_index]

        coord_mi = np.append(coord_left_mi, np.append(coord[minima_index], coord_right_mi))
        coord_ma = np.append(coord_left_ma, np.append(coord[maxima_index], coord_right_ma))
        value_mi = np.append(value_left_mi, np.append(signal[minima_index], value_right_mi))
        value_ma = np.append(value_left_ma, np.append(signal[maxima_index], value_right_ma))

        max_extrema = np.array([coord_ma, value_ma])
        min_extrema = np.array([coord_mi, value_mi])

        # Make double sure, that all extrema are significant
        max_extrema = self._check_extrema(max_extrema)
        min_extrema = self._check_extrema(min_extrema)

        return max_extrema, min_extrema

    def splines(self, coord, extrema):

        # make sure the time coordinate is between the maximum and minimum coordinates of the extrema
        t = coord[np.r_[coord >= extrema[0, 0]] & np.r_[coord <= extrema[0, -1]]]

        # calculate the values of the spline-curve over coordinates
        if self.spline_kind == "cubic":
            if extrema.shape[1] > 3:
                vals = interp1d(extrema[0], extrema[1], kind=self.spline_kind)(t)
            else:
                vals = interp1d(extrema[0], extrema[1], kind="quadratic")(t)

        elif self.spline_kind == "akima":
            vals = Akima1DInterpolator(extrema[0], extrema[1])(t)

        elif self.spline_kind in ('slinear', 'quadratic', 'linear'):
            vals = interp1d(extrema[0], extrema[1], kind=self.spline_kind)(t)

        else:
            raise ValueError("expected spline_kind is within ('cubic', 'akima', 'slinear', 'quadratic', 'linear'), "
                             "but got %s" % str(self.spline_kind))

        vals = vals.astype(np.float64)
        return t, vals

    def _is_imf_end(self, imf_diff, imf_old, maxima, minima, ee=1e-6):

        # criteria 1: maxima > 0 and minima < 0
        if np.any(maxima[1] < 0) or np.any(minima[1] > 0):
            return False

        # criteria 2: threshold for the difference of current and old IMF
        # threshold 1: variance of imf_diff / peak-to-peak value of imf_old
        if self.scale_threshold > np.sum(imf_diff ** 2) / (np.max(imf_old) - np.min(imf_old) + ee):
            return True

        # threshold 2: Huang's criteria - standard deviation
        if self.std_threshold > np.sum((imf_diff ** 2 / (imf_old ** 2 + ee))):
            return True

        # threshold 3: energy ratio - rms of imf_diff / rms of imf_old
        if self.energy_threshold > np.sum(imf_diff ** 2) / (np.sum(imf_old ** 2) + ee):
            return True

        return False

    def calculate_imf(self, coord, signal):

        imf_old, imf_cur = np.zeros(signal.shape), np.copy(signal)
        while True:
            maxima_index, minima_index = self.find_maxima_minima(imf_cur)

            ''' fix me '''
            if len(maxima_index) + len(minima_index) < 3:
                return None

            # mirror extent the maxima and minima
            maxima, minima = self.mirror_extension(coord, imf_cur, maxima_index, minima_index)

            # get spline-curves of extrema
            _, max_spline = self.splines(coord, maxima)
            _, min_spline = self.splines(coord, minima)
            mean_spline = (max_spline + min_spline) / 2

            # [:] means assigning the value, without it means assigning pointer
            imf_old[:], imf_cur[:] = imf_cur, imf_cur - mean_spline

            # " fix me "
            # import matplotlib.pyplot as plt
            # plt.subplot(211)
            # plt.plot(imf_old, c="k", lw=1.5)
            # plt.plot(max_spline, c="r", lw=0.8, linestyle="--")
            # plt.plot(min_spline, c="orange", lw=0.8, linestyle="--")
            # plt.plot(mean_spline, c="blue", lw=0.8)
            # plt.scatter(maxima_index, imf_old.take(maxima_index), c="darkred")
            # plt.scatter(minima_index, imf_old.take(minima_index), c="green")
            # plt.subplot(212)
            # plt.plot(imf_cur, c="k", lw=1.5)
            # plt.show()

            # check IMF end criteria, if satisfying, break
            if self._is_imf_end(-mean_spline, imf_old, maxima, minima):
                break

        return imf_cur

    def _is_emd_end(self, res):

        # criteria 1: extrema count threshold (=3)
        maxima_index, minima_index = self.find_maxima_minima(res)
        if len(maxima_index) + len(minima_index) < 3:
            return True

        # criteria 2: range threshold
        if np.max(res) - np.min(res) < self.range_threshold:
            return True

        # criteria 3: energy threshold
        if np.sum(np.abs(res)) < self.power_threshold:
            return True
        pass

    def emd(self, coord, signal):

        if coord is None:
            coord = np.arange(signal.shape[-1])
        coord, signal = self._check_dtype(coord, signal)

        res = np.copy(signal)
        imfs = []
        while True:
            if self._is_emd_end(res) or len(imfs) >= self.max_imfs:
                break

            imf = self.calculate_imf(coord, res)
            if imf is None:
                break

            imfs.append(imf)
            res[:] = res - imfs[-1]

        return imfs, res
