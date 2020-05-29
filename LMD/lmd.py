"""
    Author: Sirius HU
    Created Date: 2019.11.01

    Local Mean Decomposition signal processing algorithm

    References:
        [1] Jonathan S. Smith. The local mean decomposition and its application to EEG
        perception data. Journal of the Royal Society Interface, 2005, 2(5):443-454
"""
import logging
import matplotlib.pyplot as plt
from LMD.utils import *


def mean_envelop_extrema_plot(signal, _m, _a, extrema_index):

    plt.figure()
    plt.subplot(211)
    plt.plot(signal, c="k", lw=1.5)
    plt.plot(_m, c="r", lw=0.5, linestyle="--")
    plt.scatter(extrema_index, signal.take(extrema_index), s=20, c="r")
    plt.subplot(212)
    plt.plot(_a, c="b", lw=0.5, linestyle="--")

    return


def ma_mean_envelop_plot(_m, _a, ma_m, ma_a):

    plt.figure()
    plt.subplot(211)
    plt.plot(_m, c="k", lw=1.5)
    plt.plot(ma_m, c="r", lw=1, linestyle="--")
    plt.subplot(212)
    plt.plot(_a, c="k", lw=1.5)
    plt.plot(ma_a, c="r", lw=1, linestyle="--")

    return


def res_pf_s_a_plot(_s, envelope, res):

    plt.figure()
    plt.subplot(421)
    plt.plot(_s, c="k")
    plt.subplot(422)
    plt.plot(envelope, c="r")
    plt.subplot(412)
    plt.plot(_s * envelope, c="k")
    plt.plot(envelope, c="r", lw=0.5, linestyle="--")
    plt.subplot(413)
    plt.plot(res, c="k", lw=1.5)
    plt.plot(_s * envelope, c="r", lw=0.5, linestyle="--")
    plt.subplot(414)
    plt.plot(res - _s * envelope, c="k")

    return


class LocalMeanDecomp(object):

    def __init__(self,
                 ma_pad_mode='sym', ma_mode='triangle', ma_max_iter=20,
                 pf_max_iter=4, pf_en_threshold=1e-3):

        """
            initial the Local Mean Decomposition parameters

                :param include_endpoints: end-points as extrema, must set as True
                :param ma_pad_mode: the padding mode for moving average
                                     we can choose from ['sym', 'head', 'tail'], default is 'sym'
                :param ma_mode: the kernel of moving average
                                 we can choose from ['simple', 'triangle'], default is 'simple'
                :param ma_max_iter: the max iteration of moving average to get a smooth ma-curve
                :param pf_max_iter: the max iteration of PF calculation to get a ideal PF-curve
                :param pf_en_threshold: the threshold to stop PF calculation
        """

        # get decomposition logger
        self.logger = logging.getLogger(__name__)

        self.ma_pad_mode = ma_pad_mode
        self.ma_mode = ma_mode
        self.ma_max_iter = ma_max_iter
        self.pf_max_iter = pf_max_iter
        self.pf_en_threhold = pf_en_threshold

        self.logger.info("the parameters of LMD is set as: \n"
                         "include_endpoints = True\n"
                         "ma_pad_mode = %s\n"
                         "ma_mode = %s\n"
                         "ma_max_iter = %d\n"
                         "pf_max_iter = %d\n"
                         "pf_en_threshold = %f" % (self.ma_pad_mode, self.ma_mode, self.ma_max_iter,
                                                   self.pf_max_iter, self.pf_en_threhold))

    """ the first version of LMD didn't use mirror extension and cut """
    def mirror_extension(self, signal, mirror_rate):

        pass

    def mirror_cut(self, signal, mirrir_rate):

        pass

    def get_mean_envelope_window(self, signal):

        extrema_index = find_extrema(signal)
        mean_curve = get_mean_curve(signal, extrema_index)
        envelope_curve = get_envelope_curve(signal, extrema_index)
        window = self.get_window(extrema_index)

        return extrema_index, mean_curve, envelope_curve, window

    def get_window(self, extrema_index):

        """
            1. get the max interval of extrema_index (time coordinates),
               set window as max / 3
            2. get the mean interval of extrema_index
               set window as mean
            3. other strategy

            here we use the first strategy
        """

        _win = np.max(np.diff(extrema_index, n=1))
        window = np.ceil(_win / 3)
        if window % 2 == 0:
            window += 1

        return int(window)

    def multiple_moving_average(self, curve, window):

        weight = get_ma_weight(window, self.ma_mode, self.ma_pad_mode)

        for i in range(self.ma_max_iter):
            curve = ma_padding(curve, window, self.ma_pad_mode)
            curve = moving_average(curve, window, weight)

            if self._is_smoothed(curve):
                break

        return curve

    def _is_smoothed(self, curve):

        diff_curve_zero = np.diff(curve, n=1) == 0

        if True in diff_curve_zero:
            return False

        return True

    def pf_calculate(self, signal):

        _s = np.copy(signal)

        """ for debug, fix me """
        m, a, s = [], [], []

        envelop = 1
        for i in range(self.pf_max_iter):

            extrema_index, _m, _a, window = self.get_mean_envelope_window(_s)
            ma_m, ma_a = self.multiple_moving_average(_m, window), self.multiple_moving_average(_a, window)

            # """ for debug, fix me """
            # print(i)
            # mean_envelop_extrema_plot(_s, _m, _a, extrema_index)
            # mean_envelop_extrema_plot(_s, ma_m, ma_a, extrema_index)
            # ma_mean_envelop_plot(_m, _a, ma_m, ma_a)
            # plt.show()
            m.append(ma_m), a.append(ma_a), s.append(_s)

            hij = _s - ma_m
            _s = hij / ma_a
            envelop = envelop * ma_a
            if self._is_pf_envelop_approx_one(ma_a):
                break

        return _s, envelop, (m, a, s)

    def _is_pf_envelop_approx_one(self, curve):

        res = curve - 1
        if np.mean(np.abs(res)) <= self.pf_en_threhold:
            return True

        return False

    def __call__(self, signal):

        res = np.copy(signal)
        s, envelopes = [], []
        for i in range(self.pf_max_iter):

            _s, envelop, _ = self.pf_calculate(res)
            s.append(_s), envelopes.append(envelop)

            res -= _s * envelop
            if self._is_monotonous(res):
                break

        return s, envelopes, res

    @staticmethod
    def _is_monotonous(curve):

        diff_curve = np.diff(curve)
        diff_curve_mono_dec = diff_curve <= 0
        diff_curve_mono_inc = diff_curve >= 0

        if False in diff_curve_mono_dec and False in diff_curve_mono_inc:
            return False
        return True


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Logging options
    logging.basicConfig(level=logging.INFO)

    # EMD options
    DTYPE = np.float64

    # Signal options
    N, tMin, tMax = 800, 0, 1
    T = np.linspace(tMin, tMax, N, dtype=DTYPE)
    # S = 18 * np.sin(5 * np.pi * T * (1 + 0.2 * T)) + 20 * T ** 2 + T * np.sin(13 * T)
    # S = 18 * np.sin(5 * np.pi * T) * np.cos(40 * np.pi * T) + 30 * np.cos(9 * np.pi * T)
    S = (2 + np.cos(18 * np.pi * T)) * np.cos(400 * np.pi * T + 2 * np.cos(20 * np.pi * T)) + \
        4 * np.cos(5 * np.pi * T) * np.cos(120 * np.pi * T) + \
        2 * np.cos(40 * np.pi * T) + 0.8 * np.sin(np.pi * T) * np.sin(10 * np.pi * T)

    # S[-10:] = S[224]
    S = S.astype(DTYPE)
    logging.info("Input S.dtype: " + str(S.dtype))

    # LMD initial
    lmd = LocalMeanDecomp()
    s, envelopes, res = lmd(S)
    # extrema = LMD.find_extrema(S)
    # _m = LMD.get_mean_curve(S, extrema)
    # _a = LMD.get_envelope_curve(S, extrema)
    # win = LMD.get_window(extrema)
    #
    # __m = LMD.moving_average(_m, win)
    # __a = LMD.moving_average(_a, win)
    # _s, envelop, (m, a, s) = LMD.pf_calculate(S)

    # plt.figure(1)
    # plt.plot(S, c="k")
    # plt.scatter(extrema, S.take(extrema), c='r', s=20)
    # plt.plot(_m, c="blue")
    #
    # plt.figure(2)
    # plt.plot(_m, c="k")
    # plt.plot(__m, c="r")
    #
    # plt.figure(3)
    # plt.plot(_a, c="k")
    # plt.plot(__a, c="r")

    # plt.show()

    # import matplotlib.pyplot as plt
    plt.figure(2)
    plt.subplot(len(s) + 2, 2, 2)
    plt.plot(S)
    for i in range(1, len(s) + 1):
        plt.subplot(len(s) + 2, 2, i * 2 + 1)
        plt.plot(s[i - 1])
        plt.subplot(len(s) + 2, 2, i * 2 + 2)
        plt.plot(s[i - 1] * envelopes[i - 1])

    plt.subplot(len(s) + 2, 2, 2 * (len(s) + 2))
    plt.plot(res)

    # plt.subplot(4, 3, 10)
    # plt.plot(reses[2] - T ** 2)

    # plt.figure(3)
    # plt.plot(sum([s[i] * envelopes[i] for i in range(len(s))]) + res, c="k")
    # plt.plot(S, c="r")

    """
        效果不怎么样，

        s 的效果还可以
        但是 PF 分量的效果不好
        很有 所以关于 m 与 a 的估计可能 都有点问题

    """
    plt.show()
