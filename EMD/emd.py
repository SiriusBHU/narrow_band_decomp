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
from EMD.splines import *
from scipy.interpolate import interp1d, Akima1DInterpolator
import numpy as np


class EMD(object):
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

    def __call__(self, signal, coord=None):
        return self.emd(coord, signal)

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

    @staticmethod
    def _check_extrema(extrema):

        duplicate_idx = np.where(extrema[0, 1:] == extrema[0, :-1])
        return np.delete(extrema, duplicate_idx, axis=1)

    ''' this func mainly according to the func written by @Dawid Laszuk '''
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

            " fix me "
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

    @staticmethod
    def _check_dtype(coord, signal):

        data_type = signal.dtype
        coord = coord.astype(data_type)

        return coord, signal

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

            if self._is_emd_end(res) or len(imfs) > self.max_imfs:
                break

            imf = self.calculate_imf(coord, res)
            if imf is None:
                break

            imfs.append(imf)
            res[:] = res - imfs[-1]

        return imfs, res


class EMDEnhanced(EMD):

    def __init__(self):
        super(EMD, self).__init__()
        # self.emd = EMD()
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

    def __call__(self, signal, coord=None):

        if coord is None:
            coord = np.arange(signal.shape[-1])
        coord, signal = self._check_dtype(coord, signal)
        imfs, res = self.emd(coord, signal)

        freqs, amps = [], []
        for imf in imfs:
            freq, amp = self.amp_freq_decomp(coord, imf)
            freqs.append(freq)
            amps.append(amp)

        return freqs, amps, res

    def amp_freq_decomp(self, coord, imf, ee=1e-6):

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

        # spline amplitude curve
        _, amp_module = self.splines(coord, extrema)
        amp_module = np.abs(amp_module)

        # sift freq_module
        freq_module = imf / (amp_module + ee)

        return freq_module, amp_module


class EMD_:

    """
    .. _EMD:

    **Empirical Mode Decomposition**

    Method of decomposing signal into Intrinsic Mode Functions (IMFs)
    based on algorithm presented in Huang et al. [Huang1998]_.

    Algorithm was validated with Rilling et al. [Rilling2003]_ Matlab's version from 3.2007.

    Threshold which control the goodness of the decomposition:
        * `std_thr` --- Test for the proto-IMF how variance changes between siftings.
        * `svar_thr` -- Test for the proto-IMF how energy changes between siftings.
        * `total_power_thr` --- Test for the whole decomp how much of energy is solved.
        * `range_thr` --- Test for the whole decomp whether the difference is tiny.

    Parameters
    ----------
    spline_kind : string, (default: 'cubic')
        Defines type of spline, which connects extrema.
        Possible: cubic, akima, slinear.
    nbsym : int, (default: 2)
        Number of extrema used in boundary mirroring.
    extrema_detection : string, (default: 'simple')
        How extrema are defined.

        * *simple* - Extremum is above/below neighbours.
        * *parabol* - Extremum is a peak of a parabola.

    References
    ----------
    .. [Huang1998] N. E. Huang et al., "The empirical mode decomposition and the
        Hilbert spectrum for non-linear and non stationary time series
        analysis", Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998
    .. [Rilling2003] G. Rilling, P. Flandrin and P. Goncalves, "On Empirical Mode
        Decomposition and its algorithms", IEEE-EURASIP Workshop on
        Nonlinear Signal and Image Processing NSIP-03, Grado (I), June 2003

    Examples
    --------
    # >>> import numpy as np
    # >>> T = np.linspace(0, 1, 100)
    # >>> S = np.sin(2*2*np.pi*T)
    # >>> EMD = EMD()
    # >>> EMD.extrema_detection = "parabol"
    # >>> IMFs = EMD.EMD(S)
    # >>> IMFs.shape
    (1, 100)
    """

    logger = logging.getLogger(__name__)

    def __init__(self, spline_kind='cubic', nbsym=2, **config):
        """Initiate *EMD* instance.

        Configuration, such as threshold values can be passed as config.

        # >>> config = {"std_thr": 0.01, "range_thr": 0.05}
        # >>> EMD = EMD(**config)
        """
        # Declare constants
        self.energy_ratio_thr = 0.2
        self.std_thr = 0.2
        self.svar_thr = 0.001
        self.total_power_thr = 0.005
        self.range_thr = 0.001

        self.nbsym = nbsym
        self.scale_factor = 1.

        self.spline_kind = spline_kind
        self.extrema_detection = 'simple'   # simple, parabol

        self.DTYPE = np.float64
        self.FIXE = 0
        self.FIXE_H = 0

        self.MAX_ITERATION = 1000

        # Instance global declaration
        self.imfs = None
        self.residue = None

        # Update based on options
        for key in config.keys():
            if key in self.__dict__.keys():
                self.__dict__[key] = config[key]

    def __call__(self, S, T=None, max_imf=None):
        return self.emd(S, T=T, max_imf=max_imf)

    @staticmethod
    def _not_duplicate(S):
        """
        Returns indices for not repeating values, where there is no extremum.

        Example
        -------
        # >>> S = [0, 1, 1, 1, 2, 3]
        # >>> idx = self._not_duplicate(S)
        [0, 1, 3, 4, 5]
        """
        dup = np.r_[S[1:-1]==S[0:-2]] & np.r_[S[1:-1]==S[2:]]
        not_dup_idx = np.arange(1, len(S)-1)[~dup]

        idx = np.empty(len(not_dup_idx)+2, dtype=np.int)
        idx[0] = 0
        idx[-1] = len(S)-1
        idx[1:-1] = not_dup_idx

        return idx

    def _find_extrema_parabol(self, T, S):
        """
        Performs parabol estimation of extremum, i.e. an extremum is a peak
        of parabol spanned on 3 consecutive points, where the mid point is
        the closest.

        See :meth:`EMD.find_extrema()`.
        """
        # Finds indexes of zero-crossings
        S1, S2 = S[:-1], S[1:]
        indzer = np.nonzero(S1*S2<0)[0]
        if np.any(S == 0):
            iz = np.nonzero(S==0)[0]
            if np.any(np.diff(iz)==1):
                zer = S == 0
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz == 1)[0]
                finz = np.nonzero(dz == -1)[0]-1
                indz = np.round((debz+finz)/2.)
            else:
                indz = iz

            indzer = np.sort(np.append(indzer, indz))

        dt = float(T[1]-T[0])
        scale = 2.*dt*dt

        idx = self._not_duplicate(S)
        T = T[idx]
        S = S[idx]

        # p - previous
        # 0 - current
        # n - next
        Tp, T0, Tn = T[:-2], T[1:-1], T[2:]
        Sp, S0, Sn = S[:-2], S[1:-1], S[2:]
        # a = Sn + Sp - 2*S0
        # b = 2*(Tn+Tp)*S0 - ((Tn+T0)*Sp+(T0+Tp)*Sn)
        # c = Sp*T0*Tn -2*Tp*S0*Tn + Tp*T0*Sn
        TnTp, T0Tn, TpT0 = Tn-Tp, T0-Tn, Tp-T0
        scale = Tp*Tn*Tn + Tp*Tp*T0 + T0*T0*Tn - Tp*Tp*Tn - Tp*T0*T0 - T0*Tn*Tn

        a = T0Tn*Sp + TnTp*S0 + TpT0*Sn
        b = (S0-Sn)*Tp**2 + (Sn-Sp)*T0**2 + (Sp-S0)*Tn**2
        c = T0*Tn*T0Tn*Sp + Tn*Tp*TnTp*S0 + Tp*T0*TpT0*Sn

        a = a/scale
        b = b/scale
        c = c/scale
        a[a==0] = 1e-14
        tVertex = -0.5*b/a
        idx = np.r_[tVertex<T0+0.5*(Tn-T0)] & np.r_[tVertex>=T0-0.5*(T0-Tp)]

        a, b, c = a[idx], b[idx], c[idx]
        tVertex = tVertex[idx]
        sVertex = a*tVertex*tVertex + b*tVertex + c

        local_max_pos, local_max_val = tVertex[a<0], sVertex[a<0]
        local_min_pos, local_min_val = tVertex[a>0], sVertex[a>0]

        return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer

    @classmethod
    def _find_extrema_simple(cls, T, S):
        """
        Performs extrema detection, where extremum is defined as a point,
        that is above/below its neighbours.

        See :meth:`EMD.find_extrema`.
        """

        # Finds indexes of zero-crossings
        S1, S2 = S[:-1], S[1:]
        indzer = np.nonzero(S1*S2<0)[0]
        if np.any(S==0):
            iz = np.nonzero(S==0)[0]
            if np.any(np.diff(iz)==1):
                zer = (S==0)
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz==1)[0]
                finz = np.nonzero(dz==-1)[0]-1
                indz = np.round((debz+finz)/2.)
            else:
                indz = iz

            indzer = np.sort(np.append(indzer, indz))

        # Finds local extrema
        d = np.diff(S)
        d1, d2 = d[:-1], d[1:]
        indmin = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 < 0])[0] + 1
        indmax = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 > 0])[0] + 1

        # When two or more points have the same value
        if np.any(d==0):

            imax, imin = [], []

            bad = (d==0)
            dd = np.diff(np.append(np.append(0, bad), 0))
            debs = np.nonzero(dd==1)[0]
            fins = np.nonzero(dd==-1)[0]

            # debs, fins, the start pos and end pos of the same value
            if debs[0] == 1:
                if len(debs)>1:
                    debs, fins = debs[1:], fins[1:]
                else:
                    debs, fins = [], []
            #

            if len(debs) > 0:
                if fins[-1] == len(S)-1:
                    if len(debs) > 1:
                        debs, fins = debs[:-1], fins[:-1]
                    else:
                        debs, fins = [], []

            lc = len(debs)
            if lc > 0:
                for k in range(lc):
                    if d[debs[k]-1] > 0:
                        if d[fins[k]] < 0:
                            imax.append(np.round((fins[k]+debs[k])/2.))
                    else:
                        if d[fins[k]] > 0:
                            imin.append(np.round((fins[k]+debs[k])/2.))

            if len(imax) > 0:
                indmax = indmax.tolist()
                for x in imax: indmax.append(int(x))
                indmax.sort()

            if len(imin) > 0:
                indmin = indmin.tolist()
                for x in imin: indmin.append(int(x))
                indmin.sort()

        local_max_pos = T[indmax]
        local_max_val = S[indmax]
        local_min_pos = T[indmin]
        local_min_val = S[indmin]
        # 没有将端点包含在内
        return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer

    def find_extrema(self, T, S):

        """
        Returns extrema (minima and maxima) for given signal S.
        Detection and definition of the extrema depends on
        ``extrema_detection`` variable, set on initiation of EMD.

        Parameters
        ----------
        T : numpy array
            Position or time array.
        S : numpy array
            Input data S(T).

        Returns
        -------
        local_max_pos : numpy array
            Position of local maxima.
        local_max_val : numpy array
            Values of local maxima.
        local_min_pos : numpy array
            Position of local minima.
        local_min_val : numpy array
            Values of local minima.
        """
        if self.extrema_detection == "parabol":
            return self._find_extrema_parabol(T, S)
        elif self.extrema_detection == "simple":
            return self._find_extrema_simple(T, S)
        else:
            raise ValueError("Incorrect extrema detection type. Please try: 'simple' or 'parabol'.")

    def _mirror_prepare_points_parabol(self, T, S, max_pos, max_val, min_pos, min_val):
        """
        Performs mirroring on signal which extrema do not necessarily
        belong on the position array.

        See :meth:`EMD.prepare_points`.
        """

        # Need at least two extrema to perform mirroring
        max_extrema = np.zeros((2,len(max_pos)), dtype=self.DTYPE)
        min_extrema = np.zeros((2,len(min_pos)), dtype=self.DTYPE)

        max_extrema[0], min_extrema[0] = max_pos, min_pos
        max_extrema[1], min_extrema[1] = max_val, min_val

        # Local variables
        nbsym = self.nbsym
        end_min, end_max = len(min_pos), len(max_pos)

        ####################################
        # Left bound
        d_pos = max_pos[0] - min_pos[0]
        left_ext_max_type = d_pos<0 # True -> max, else min

        # Left extremum is maximum
        if left_ext_max_type:
            if (S[0]>min_val[0]) and (np.abs(d_pos)>(max_pos[0]-T[0])):
                # mirror signal to first extrema
                expand_left_max_pos = 2*max_pos[0] - max_pos[1:nbsym+1]
                expand_left_min_pos = 2*max_pos[0] - min_pos[0:nbsym]
                expand_left_max_val = max_val[1:nbsym+1]
                expand_left_min_val = min_val[0:nbsym]
            else:
                # mirror signal to beginning
                expand_left_max_pos = 2*T[0] - max_pos[0:nbsym]
                expand_left_min_pos = 2*T[0] - np.append(T[0], min_pos[0:nbsym-1])
                expand_left_max_val = max_val[0:nbsym]
                expand_left_min_val = np.append(S[0], min_val[0:nbsym-1])

        # Left extremum is minimum
        else:
            if (S[0] < max_val[0]) and (np.abs(d_pos)>(min_pos[0]-T[0])):
                # mirror signal to first extrema
                expand_left_max_pos = 2*min_pos[0] - max_pos[0:nbsym]
                expand_left_min_pos = 2*min_pos[0] - min_pos[1:nbsym+1]
                expand_left_max_val = max_val[0:nbsym]
                expand_left_min_val = min_val[1:nbsym+1]
            else:
                # mirror signal to beginning
                expand_left_max_pos = 2*T[0] - np.append(T[0], max_pos[0:nbsym-1])
                expand_left_min_pos = 2*T[0] - min_pos[0:nbsym]
                expand_left_max_val = np.append(S[0], max_val[0:nbsym-1])
                expand_left_min_val = min_val[0:nbsym]

        if not expand_left_min_pos.shape:
            expand_left_min_pos, expand_left_min_val = min_pos, min_val
        if not expand_left_max_pos.shape:
            expand_left_max_pos, expand_left_max_val = max_pos, max_val

        expand_left_min = np.vstack((expand_left_min_pos[::-1], expand_left_min_val[::-1]))
        expand_left_max = np.vstack((expand_left_max_pos[::-1], expand_left_max_val[::-1]))

        ####################################
        # Right bound
        d_pos = max_pos[-1] - min_pos[-1]
        right_ext_max_type = d_pos > 0

        # Right extremum is maximum
        if not right_ext_max_type:
            if (S[-1] < max_val[-1]) and (np.abs(d_pos)>(T[-1]-min_pos[-1])):
                # mirror signal to last extrema
                idx_max = max(0, end_max-nbsym)
                idx_min = max(0, end_min-nbsym-1)
                expand_right_maxPos = 2*min_pos[-1] - max_pos[idx_max:]
                expand_right_min_pos = 2*min_pos[-1] - min_pos[idx_min:-1]
                expand_right_max_val = max_val[idx_max:]
                expand_right_min_val = min_val[idx_min:-1]
            else:
                # mirror signal to end
                idx_max = max(0, end_max-nbsym+1)
                idx_min = max(0, end_min-nbsym)
                expand_right_maxPos = 2*T[-1] - np.append(max_pos[idx_max:], T[-1])
                expand_right_min_pos = 2*T[-1] - min_pos[idx_min:]
                expand_right_max_val = np.append(max_val[idx_max:],S[-1])
                expand_right_min_val = min_val[idx_min:]

        # Right extremum is minimum
        else:
            if (S[-1] > min_val[-1]) and len(max_pos)>1 and (np.abs(d_pos)>(T[-1]-max_pos[-1])):
                # mirror signal to last extremum
                idx_max = max(0, end_max-nbsym-1)
                idx_min = max(0, end_min-nbsym)
                expand_right_maxPos = 2*max_pos[-1] - max_pos[idx_max:-1]
                expand_right_min_pos = 2*max_pos[-1] - min_pos[idx_min:]
                expand_right_max_val = max_val[idx_max:-1]
                expand_right_min_val = min_val[idx_min:]
            else:
                # mirror signal to end
                idx_max = max(0, end_max-nbsym)
                idx_min = max(0, end_min-nbsym+1)
                expand_right_maxPos = 2*T[-1] - max_pos[idx_max:]
                expand_right_min_pos = 2*T[-1] - np.append(min_pos[idx_min:], T[-1])
                expand_right_max_val = max_val[idx_max:]
                expand_right_min_val = np.append(min_val[idx_min:], S[-1])

        if not expand_right_min_pos.shape:
            expand_right_min_pos, expand_right_min_val = min_pos, min_val
        if not expand_right_maxPos.shape:
            expand_right_maxPos, expand_right_max_val = max_pos, max_val

        expand_right_min = np.vstack((expand_right_min_pos[::-1], expand_right_min_val[::-1]))
        expand_right_max = np.vstack((expand_right_maxPos[::-1], expand_right_max_val[::-1]))

        max_extrema = np.hstack((expand_left_max, max_extrema, expand_right_max))
        min_extrema = np.hstack((expand_left_min, min_extrema, expand_right_min))

        return max_extrema, min_extrema

    def _mirror_prepare_points_simple(self, T, S, max_pos, max_val, min_pos, min_val):
        """
        Performs mirroring on signal which extrema can be indexed on
        the position array.

        See :meth:`EMD.prepare_points`.
        """

        # Find indexes of pass
        ind_min = min_pos.astype(int)
        ind_max = max_pos.astype(int)

        # Local variables
        nbsym = self.nbsym
        end_min, end_max = len(min_pos), len(max_pos)

        ####################################
        # Left bound - mirror nbsym points to the left
        if ind_max[0] < ind_min[0]:
            if S[0] > S[ind_min[0]]:
                lmax = ind_max[1:min(end_max,nbsym+1)][::-1]
                lmin = ind_min[0:min(end_min,nbsym+0)][::-1]
                lsym = ind_max[0]
            else:
                lmax = ind_max[0:min(end_max,nbsym)][::-1]
                lmin = np.append(ind_min[0:min(end_min,nbsym-1)][::-1],0)
                lsym = 0
        else:
            if S[0] < S[ind_max[0]]:
                lmax = ind_max[0:min(end_max,nbsym+0)][::-1]
                lmin = ind_min[1:min(end_min,nbsym+1)][::-1]
                lsym = ind_min[0]
            else:
                lmax = np.append(ind_max[0:min(end_max,nbsym-1)][::-1],0)
                lmin = ind_min[0:min(end_min,nbsym)][::-1]
                lsym = 0

        ####################################
        # Right bound - mirror nbsym points to the right
        if ind_max[-1] < ind_min[-1]:
            if S[-1] < S[ind_max[-1]]:
                rmax = ind_max[max(end_max-nbsym,0):][::-1]
                rmin = ind_min[max(end_min-nbsym-1,0):-1][::-1]
                rsym = ind_min[-1]
            else:
                rmax = np.append(ind_max[max(end_max-nbsym+1,0):], len(S)-1)[::-1]
                rmin = ind_min[max(end_min-nbsym,0):][::-1]
                rsym = len(S)-1
        else:
            if S[-1] > S[ind_min[-1]]:
                rmax = ind_max[max(end_max-nbsym-1,0):-1][::-1]
                rmin = ind_min[max(end_min-nbsym,0):][::-1]
                rsym = ind_max[-1]
            else:
                rmax = ind_max[max(end_max-nbsym,0):][::-1]
                rmin = np.append(ind_min[max(end_min-nbsym+1,0):], len(S)-1)[::-1]
                rsym = len(S)-1

        # In case any array missing
        if not lmin.size: lmin = ind_min
        if not rmin.size: rmin = ind_min
        if not lmax.size: lmax = ind_max
        if not rmax.size: rmax = ind_max

        # Mirror points
        tlmin = 2*T[lsym]-T[lmin]
        tlmax = 2*T[lsym]-T[lmax]
        trmin = 2*T[rsym]-T[rmin]
        trmax = 2*T[rsym]-T[rmax]

        # If mirrored points are not outside passed time range.
        if tlmin[0] > T[0] or tlmax[0] > T[0]:
            if lsym == ind_max[0]:
                lmax = ind_max[0:min(end_max,nbsym)][::-1]
            else:
                lmin = ind_min[0:min(end_min,nbsym)][::-1]

            if lsym == 0:
                raise Exception('Left edge BUG')

            lsym = 0
            tlmin = 2*T[lsym]-T[lmin]
            tlmax = 2*T[lsym]-T[lmax]

        if trmin[-1] < T[-1] or trmax[-1] < T[-1]:
            if rsym == ind_max[-1]:
                rmax = ind_max[max(end_max-nbsym,0):][::-1]
            else:
                rmin = ind_min[max(end_min-nbsym,0):][::-1]

            if rsym == len(S)-1:
                raise Exception('Right edge BUG')

            rsym = len(S)-1
            trmin = 2*T[rsym]-T[rmin]
            trmax = 2*T[rsym]-T[rmax]

        zlmax = S[lmax]
        zlmin = S[lmin]
        zrmax = S[rmax]
        zrmin = S[rmin]

        tmin = np.append(tlmin, np.append(T[ind_min], trmin))
        tmax = np.append(tlmax, np.append(T[ind_max], trmax))
        zmin = np.append(zlmin, np.append(S[ind_min], zrmin))
        zmax = np.append(zlmax, np.append(S[ind_max], zrmax))

        max_extrema = np.array([tmax, zmax])
        min_extrema = np.array([tmin, zmin])

        # Make double sure, that each extremum is significant
        max_dup_idx = np.where(max_extrema[0,1:]==max_extrema[0,:-1])
        max_extrema = np.delete(max_extrema, max_dup_idx, axis=1)
        min_dup_idx = np.where(min_extrema[0,1:]==min_extrema[0,:-1])
        min_extrema = np.delete(min_extrema, min_dup_idx, axis=1)

        return max_extrema, min_extrema

    def mirror_extension_prepare_points(self, T, S, max_pos, max_val, min_pos, min_val):

        """
        Performs extrapolation on edges by adding extra extrema, also known
        as mirroring signal. The number of added points depends on *nbsym*
        variable.

        Parameters
        ----------
        S : numpy array
            Input signal.
        T : numpy array
            Position or time array.
        max_pos : iterable
            Sorted time positions of maxima.
        max_vali : iterable
            Signal values at max_pos positions.
        min_pos : iterable
            Sorted time positions of minima.
        min_val : iterable
            Signal values at min_pos positions.

        Returns
        -------
        min_extrema : numpy array (2 rows)
            Position (1st row) and values (2nd row) of minima.
        min_extrema : numpy array (2 rows)
            Position (1st row) and values (2nd row) of maxima.
        """
        if self.extrema_detection=="parabol":
            return self._mirror_prepare_points_parabol(T, S, max_pos, max_val, min_pos, min_val)
        elif self.extrema_detection=="simple":
            return self._mirror_prepare_points_simple(T, S, max_pos, max_val, min_pos, min_val)
        else:
            msg = "Incorrect extrema detection type. Please try: "
            msg+= "'simple' or 'parabol'."
            raise ValueError(msg)

    def spline_points(self, T, extrema):
        """
        Constructs spline over given points.

        Parameters
        ----------
        T : numpy array
            Position or time array.
        extrema : numpy array
            Position (1st row) and values (2nd row) of points.

        Returns
        -------
        T : numpy array
            Position array (same as input).
        spline : numpy array
            Spline array over given positions T.
        """

        kind = self.spline_kind.lower()
        t = T[np.r_[T>=extrema[0,0]] & np.r_[T<=extrema[0,-1]]]

        if kind == "akima":
            return t, akima(extrema[0], extrema[1], t)

        elif kind == 'cubic':
            if extrema.shape[1] > 3:
                return t, interp1d(extrema[0], extrema[1], kind=kind)(t)
            else:
                return cubic_spline_3pts(extrema[0], extrema[1], t)

        elif kind in ['slinear', 'quadratic', 'linear']:
            return T, interp1d(extrema[0], extrema[1], kind=kind)(t).astype(self.DTYPE)

        else:
            raise ValueError("No such interpolation method!")

    # import scipy.signal as signal
    # signal.argrelextrema()
    # np.take()

    def extract_max_min_spline(self, T, S):

        """
        Extracts top and bottom envelopes based on the signal,
        which are constructed based on maxima and minima, respectively.

        Parameters
        ----------
        T : numpy array
            Position or time array.
        S : numpy array
            Input data S(T).

        Returns
        -------
        max_spline : numpy array
            Spline spanned on S maxima.
        min_spline : numpy array
            Spline spanned on S minima.
        """

        # Get indexes of extrema
        ext_res = self.find_extrema(T, S)
        max_pos, max_val = ext_res[0], ext_res[1]
        min_pos, min_val = ext_res[2], ext_res[3]

        if len(max_pos) + len(min_pos) < 3: return [-1]*4

        #########################################
        # Extrapolation of signal (over boundaries)
        pp_res = self.mirror_extension_prepare_points(T, S, max_pos, max_val, min_pos, min_val)
        max_extrema, min_extrema = pp_res

        _, max_spline = self.spline_points(T, max_extrema)
        _, min_spline = self.spline_points(T, min_extrema)

        return max_spline, min_spline, max_extrema, min_extrema

    def end_condition(self, S, IMF):

        """Tests for end condition of whole EMD. The procedure will stop if:

        * Absolute amplitude (max - min) is below *range_thr* threshold, or
        * Metric L1 (mean absolute difference) is below *total_power_thr* threshold.

        Parameters
        ----------
        S : numpy array
            Original signal on which EMD was performed.
        IMF : numpy 2D array
            Set of IMFs where each row is IMF. Their order is not important.

        Returns
        -------
        end : bool
            Whether sifting is finished.
        """
        # When to stop EMD
        tmp = S - np.sum(IMF, axis=0)

        if np.max(tmp) - np.min(tmp) < self.range_thr:
            self.logger.info("FINISHED -- RANGE")
            return True

        if np.sum(np.abs(tmp)) < self.total_power_thr:
            self.logger.info("FINISHED -- SUM POWER")
            return True

        return False

    def check_imf(self, imf_new, imf_old, eMax, eMin):
        """
        Huang criteria for **IMF** (similar to Cauchy convergence test).
        Signal is an IMF if consecutive siftings do not affect signal
        in a significant manner.
        """
        # local max are > 0 and local min are < 0
        if np.any(eMax[1] < 0) or np.any(eMin[1] > 0):
            return False

        # Convergence
        if np.sum(imf_new ** 2) < 1e-10:
            return False

        # Precompute values
        imf_diff = imf_new - imf_old
        imf_diff_sqrd_sum = np.sum(imf_diff * imf_diff)

        # Scaled variance test
        svar = imf_diff_sqrd_sum / (max(imf_old) - min(imf_old))
        if svar < self.svar_thr:
            self.logger.info("Scaled variance -- PASSED")
            return True

        # Standard deviation test
        std = np.sum((imf_diff/imf_new)**2)
        if std < self.std_thr:
            self.logger.info("Standard deviation -- PASSED")
            return True

        energy_ratio = imf_diff_sqrd_sum/np.sum(imf_old*imf_old)
        if energy_ratio < self.energy_ratio_thr:
            self.logger.info("Energy ratio -- PASSED")
            return True

        # here std = sum ((imf_diff / imf_new) ** 2)
        #   energy = sum ((imf_diff / imf_old) ** 2)
        # so these two did not have much difference

        return False

    @staticmethod
    def _common_dtype(x, y):
        """Determines common numpy DTYPE for arrays."""
        dtype = np.find_common_type([x.dtype, y.dtype], [])
        if x.dtype != dtype: x = x.astype(dtype)
        if y.dtype != dtype: y = y.astype(dtype)

        return x, y

    def emd(self, S, T=None, max_imf=-1):

        """
        Performs Empirical Mode Decomposition on signal S.
        The decomposition is limited to *max_imf* imfs.
        Returns IMF functions in numpy array format.

        Parameters
        ----------
        S : numpy array,
            Input signal.
        T : numpy array, (default: None)
            Position or time array. If None passed or if self.extrema_detection == "simple",
            then numpy arange is created.
        max_imf : int, (default: -1)
            IMF number to which decomposition should be performed.
            Negative value means *all*.

        Returns
        -------
        IMF : numpy array
            Set of IMFs produced from input signal.
        """
        if T is not None and len(S) != len(T):
            raise ValueError("Time series have different sizes: len(S) -> {} != {} <- len(T)".format(len(S), len(T)))

        if T is None or self.extrema_detection == "simple":

            T = np.arange(len(S), dtype=S.dtype)

        # Make sure same types are dealt
        S, T = self._common_dtype(S, T)
        self.DTYPE = S.dtype
        N = len(S)

        residue = S.astype(self.DTYPE)
        imf = np.zeros(len(S), dtype=self.DTYPE)
        imf_old = np.nan

        if S.shape != T.shape:
            raise ValueError("Position or time array should be the same size as signal.")

        # Create arrays
        imfNo = 0
        IMF = np.empty((imfNo, N))  # Numpy container for IMF
        finished = False

        while not finished:
            self.logger.info('IMF -- '+str(imfNo))

            residue[:] = S - np.sum(IMF[:imfNo], axis=0)
            imf = residue.copy()
            mean = np.zeros(len(S), dtype=self.DTYPE)

            # Counters
            n = 0   # All iterations for current imf.
            n_h = 0 # counts when |#zero - #ext| <=1

            while(True):
                n += 1
                if n >= self.MAX_ITERATION:
                    self.logger.info("Max iterations reached for IMF. Continuing with another IMF.")
                    break

                ext_res = self.find_extrema(T, imf)
                max_pos, min_pos, indzer = ext_res[0], ext_res[2], ext_res[4]
                extNo = len(min_pos)+len(max_pos)
                nzm = len(indzer)

                if extNo > 2:

                    max_env, min_env, eMax, eMin = self.extract_max_min_spline(T, imf)
                    mean[:] = 0.5*(max_env+min_env)

                    imf_old = imf.copy()
                    imf[:] = imf - mean

                    # Fix number of iterations
                    if self.FIXE:
                        if n >= self.FIXE: break

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
                    elif self.FIXE_H:
                        tmp_residue = self.find_extrema(T, imf)
                        max_pos, min_pos, ind_zer = tmp_residue[0], tmp_residue[2], tmp_residue[4]
                        extNo = len(max_pos)+len(min_pos)
                        nzm = len(ind_zer)

                        if n == 1:
                            continue
                        
                        # If proto-IMF add one, or reset counter otherwise
                        n_h = n_h + 1 if abs(extNo-nzm) < 2 else 0

                        # STOP
                        if n_h >= self.FIXE_H: break

                    # Stops after default stopping criteria are met
                    else:
                        ext_res = self.find_extrema(T, imf)
                        max_pos, max_val, min_pos, min_val, ind_zer = ext_res
                        extNo = len(max_pos) + len(min_pos)
                        nzm = len(ind_zer)

                        if imf_old is np.nan: continue

                        f1 = self.check_imf(imf, imf_old, eMax, eMin)
                        f2 = abs(extNo - nzm) < 2

                        # STOP
                        if f1 and f2: break

                else:  # Less than 2 ext, i.e. trend
                    finished = True
                    break
            # END OF IMF SIFTING

            IMF = np.vstack((IMF, imf.copy()))
            imfNo += 1

            if self.end_condition(S, IMF) or imfNo == max_imf:
                finished = True
                break

        # Saving residuum
        self.residue = residue = S - np.sum(IMF, axis=0)
        self.imfs = IMF.copy()
        if not np.allclose(residue, 0):
            IMF = np.vstack((IMF, residue))

        return IMF

    def get_imfs_and_residue(self):
        """
        Provides access to separated imfs and residue from recently analysed signal.
        :return: (imfs, residue)
        """
        return self.imfs, self.residue

##################################################


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    # Logging options
    logging.basicConfig(level=logging.INFO)

    # EMD options
    max_imf = -1
    DTYPE = np.float64
    N, tMin, tMax = 1000, 0, 1

    # signal generate
    coord = np.linspace(tMin, tMax, N, dtype=DTYPE)
    # signal = 18 * np.sin(5 * np.pi * T * (1 + 0.2 * T)) + T ** 2 + T * np.sin(13 * T)
    signal = 18 * np.sin(5 * np.pi * coord) * np.cos(40 * np.pi * coord) + 30 * np.cos(9 * np.pi * coord)
    signal = signal.astype(DTYPE)
    print("Input S.dtype: " + str(signal.dtype))

    # # Prepare and run EMD
    # emd = EMD()
    # emd.FIXE_H = 5
    # emd.nbsym = 2
    # emd.spline_kind = 'cubic'
    # emd.DTYPE = DTYPE
    # imfs = emd.emd(signal, coord, max_imf)
    # imfNo = imfs.shape[0]

    # prepare for EMD_test

    emd_e = EMDEnhanced()
    freqs, amps, res = emd_e(signal)

    print(1)
    mmax = np.max(np.abs(freqs), axis=1).reshape(-1, 1)
    freqs = freqs / mmax
    phases = np.arccos(freqs)
    phases = phases - np.pi / 2
    phases = np.unwrap(phases, discont=np.pi/2)
    import matplotlib.pyplot as plt
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(phases)
    plt.show()

    from scipy.fftpack import hilbert
    emd = EMD()
    imfs, res = emd(signal)

    analytic_signals = hilbert(freqs[0] * amps[0])
    # calculate instantaneous frequency
    phases = np.angle(analytic_signals)
    # inst_freqs = np.diff(phases, n=1) / (2 * np.pi) * sampling_rate
    # # append to keep the same dim
    # inst_freqs = np.append(inst_freqs[:, :1], inst_freqs)
    # # calculate instantaneous amplitude of the corresponding frequency
    # amps = np.abs(analytic_signals)


    imfs.append(res)
    imfNo = len(imfs)

    # Plot results
    c = 1
    r = np.ceil((imfNo+1)/c)

    plt.ioff()
    plt.subplot(r, c, 1)
    plt.plot(coord, signal, 'r')
    plt.xlim((tMin, tMax))
    plt.title("Original signal")

    for num in range(imfNo):
        plt.subplot(r,c,num+2)
        plt.plot(coord, imfs[num], 'g')
        plt.xlim((tMin, tMax))
        plt.ylabel("Imf "+str(num+1))

    # plt.tight_layout()
    plt.show()

    print(1)

    # import matplotlib.pyplot as plt
    #
    # plt.subplot(311)
    # plt.plot(imf, c="k", lw=1.5)
    # plt.scatter(maxima_index, imf.take(maxima_index), c="orange")
    # plt.scatter(minima_index, imf.take(minima_index), c="orange")
    # plt.plot(amp_module, c="r", lw=0.8, linestyle="--")
    # plt.plot(-amp_module, c="r", lw=0.8, linestyle="--")
    # plt.subplot(312)
    # plt.plot(freq_module, c="b")
    # plt.subplot(313)
    # plt.plot(amp_module, c="b")
    # plt.show()