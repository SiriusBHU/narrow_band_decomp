"""
    Author: Sirius HU
    Created Date: 2019.11.01

    Local Mean Decomposition signal processing algorithm

    References:
        [1] Jonathan S. Smith. The local mean decomposition and its application to EEG
        perception data. Journal of the Royal Society Interface, 2005, 2(5):443-454
"""
import logging
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


class robustLocalMeanDecomp(object):

    def __init__(self,
                 mirror_rate=0.1,
                 ma_pad_mode='sym', ma_mode='triangle', ma_max_iter=20,
                 pf_max_iter=20, pf_en_threshold=1e-3):

        """
            initial the Local Mean Decomposition parameters

                :param mirror_rate: the rate to calculate mirror len for mirror extension
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

        self.mirror_rate = mirror_rate
        self.ma_pad_mode = ma_pad_mode
        self.ma_mode = ma_mode
        self.ma_max_iter = ma_max_iter
        self.pf_max_iter = pf_max_iter
        self.pf_en_threhold = pf_en_threshold

        if self.mirror_rate < 0:
            raise ValueError("expected mirror rate should be bigger than zeros, but got %f < 0" % self.mirror_rate)

        self.logger.info("the parameters of LMD is set as: \n"
                         "ma_pad_mode = %s\n"
                         "ma_mode = %s\n"
                         "ma_max_iter = %d\n, "
                         "pf_max_iter = %d\n"
                         "pf_en_threshold = %f" % (self.ma_pad_mode, self.ma_mode, self.ma_max_iter,
                                                   self.pf_max_iter, self.pf_en_threhold))

        self.is_mirror = True

    def mirror_extension(self, signal, extrema_index):

        s = np.copy(signal)
        if len(extrema_index) <= 2:
            self.is_mirror = False
            return s

        l_end, l_e1, l_e2 = s[extrema_index[0]], s[extrema_index[1]], s[extrema_index[2]]
        r_end, r_e1, r_e2 = s[extrema_index[-1]], s[extrema_index[-2]], s[extrema_index[-3]]

        l_is_end = self._symmetry_point_judge(l_end, l_e1, l_e2)
        r_is_end = self._symmetry_point_judge(r_end, r_e1, r_e2)

        mirror_len = int(s.shape[-1] * self.mirror_rate)
        if not l_is_end:
            l_len = mirror_len + extrema_index[1] - extrema_index[0]
            l_start = extrema_index[1] + 1
        else:
            l_len = mirror_len
            l_start = extrema_index[0] + 1

        if not r_is_end:
            r_len = mirror_len + extrema_index[-1] - extrema_index[-2]
            r_start = extrema_index[-2]
        else:
            r_len = mirror_len
            r_start = extrema_index[-1]

        if l_start >= mirror_len or len(s) - r_start > mirror_len:
            self.is_mirror = False
            return s

        cat_tuple = (s[l_start: l_start + l_len][::-1],
                     s[l_start - 1: r_start + 1],
                     s[r_start - r_len: r_start][::-1])

        self.is_mirror = True
        return np.concatenate(cat_tuple, axis=-1)

    def _symmetry_point_judge(self, endpoint, e1, e2):

        if e1 > e2:
            if endpoint > e2:
                return False
            else:
                return True
        else:
            if endpoint < e2:
                return False
            else:
                return True

    def mirror_cut(self, signal):

        mirror_len = int(signal.shape[-1] / (1 + 2 * self.mirror_rate) * self.mirror_rate)
        if self.is_mirror:
            signal = signal[mirror_len: -mirror_len]

        return signal

    def get_mean_envelope_window(self, signal):

        # plt.subplot(211)
        # plt.plot(signal)
        # plt.xlim(-80, 880)
        _extrema = find_extrema(signal)
        s = self.mirror_extension(signal, _extrema)
        # plt.subplot(212)
        # plt.plot(signal)
        # plt.xlim(0, 960)
        #
        # plt.show()
        extrema_index = find_extrema(s)
        mean_curve = get_mean_curve(s, extrema_index)
        envelope_curve = get_envelope_curve(s, extrema_index)
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

        interval = np.diff(extrema_index, n=1)
        mean_win, std_win = np.mean(interval), np.std(interval)
        window = np.ceil(mean_win + 3 * std_win)
        if window % 2 == 0:
            window += 1

        return int(window)

    def multiple_moving_average(self, curve, window):

        weight = get_ma_weight(window, self.ma_mode, self.ma_pad_mode)

        for i in range(1):
            curve = ma_padding(curve, window, self.ma_pad_mode)
            curve = moving_average(curve, window, weight)

            # if self._is_smoothed(curve):
            #     break

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
        judge_a = []
        for i in range(self.pf_max_iter):

            extrema_index, _m, _a, window = self.get_mean_envelope_window(_s)
            ma_m, ma_a = self.multiple_moving_average(_m, window), self.multiple_moving_average(_a, window)
            _m, _a = self.mirror_cut(_m), self.mirror_cut(_a)
            kkk = (80 <= extrema_index) & (extrema_index < 880)
            extrema_index = extrema_index[kkk] - 80
            ma_m, ma_a = self.mirror_cut(ma_m), self.mirror_cut(ma_a)

            """ for debug, fix me """
            # print(i)
            # mean_envelop_extrema_plot(_s, _m, _a, extrema_index)
            # mean_envelop_extrema_plot(_s, ma_m, ma_a, extrema_index)
            # ma_mean_envelop_plot(_m, _a, ma_m, ma_a)
            # plt.show()


            """ for debug, fix me """
            m.append(ma_m), a.append(ma_a), s.append(_s)

            hij = _s - ma_m
            _s = hij / ma_a

            if len(judge_a) < 4:
                judge_a.append(ma_a)
            else:
                judge_a.pop(0), judge_a.append(ma_a)

            if len(judge_a) >= 4:
                if i == 4:
                    print("kkk", i)
                envelop = envelop * judge_a[0]

            if self._is_pf_envelop_approx_one(judge_a):
                break

        return _s, envelop, (m, a, s)

    def _is_pf_envelop_approx_one(self, curves):

        if len(curves) < 4:
            return False

        results = [self._sifting_ending_func(curve) for curve in curves]
        print(results[-1])
        if results[1] < results[2] < results[3]and results[0] <= self.pf_en_threhold:
            return True

        return False

    def _sifting_ending_func(self, curve):

        res = curve - 1
        rms = np.sqrt(np.mean(res ** 2))
        mean_res = np.mean(res)
        ek = np.mean((res - mean_res) ** 4) / (np.mean((res - mean_res) ** 2) ** 2) - 3

        return rms + ek

    def __call__(self, signal):

        res = np.copy(signal)

        """ for debug, fix me """
        reses, s, envelopes = [], [], []

        for i in range(self.pf_max_iter):
            print(i)
            if i == 9:
                print(1)
            _s, envelop, _ = self.pf_calculate(res)

            """ for debug, fix me """
            # res_pf_s_a_plot(_s, envelop, res)
            # plt.show()
            reses.append(np.copy(res)), s.append(_s), envelopes.append(envelop)

            res -= _s * envelop
            if self._is_monotonous(res):
                break

        return res, s, envelopes, reses

    def _is_monotonous(self, curve):

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
    N, tMin, tMax = 800, 0, 4
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
    lmd = robustLocalMeanDecomp()
    res, s, envelopes, reses = lmd(S)
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


a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b = a[5: 8]