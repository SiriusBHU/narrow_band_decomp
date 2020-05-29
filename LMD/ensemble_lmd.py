"""
    fix me
"""
import logging
import numpy as np


class LocalMeanDecomp(object):

    logger = logging.getLogger(__name__)

    # default Local Mean Decomposition parameters
    window = 5
    include_endpoints = True
    ma_pad_mode = 'sym'
    ma_mode = 'simple'#'triangle'#
    ma_max_iter = 20
    pf_max_iter = 20
    pf_en_threhold = 0.001

    def __init__(self,
                 include_endpoints=None,
                 ma_pad_mode=None, ma_mode=None, ma_max_iter=None,
                 pf_max_iter=None, pf_en_threhold=None,
                 window=None):

        """
            initial the Local Mean Decomposition parameters

                :param include_endpoints:
                :param ma_pad_mode:
                :param ma_mode:
                :param ma_max_iter:
                :param pf_max_iter:
                :param pf_en_threhold:
                :param window:
        """

        self.window = window
        self.include_endpoints = include_endpoints
        self.ma_pad_mode = ma_pad_mode
        self.ma_mode = ma_mode
        self.ma_max_iter = ma_max_iter
        self.pf_max_iter = pf_max_iter
        self.pf_en_threhold = pf_en_threhold

        if not self.window:
            self.window = 5
            self.include_endpoints = True
            self.ma_pad_mode = 'sym'
            self.ma_mode = 'simple'
            self.ma_max_iter = 20
            self.pf_max_iter = 20
            self.pf_en_threhold = 0.001

    def find_extrema(self, signal):

        """
            find the extrema of the signal,
            extrema are the set of maximum and/or minimum of the signal
            here we calculate the finite differences of the signal to find the extrema

            :param
            signal: the input signal
            :return the extrema's index of the signal
        """

        # detect the extrema,
        # where the product of the pre_diff and post_diff of the extrema must be < 0
        diff_s = np.diff(signal, n=1)
        is_extrema = (diff_s[1:] * diff_s[:-1]) < 0

        # detect bad points, where there are continuous points with same value --> diff == 0
        diff_s_zero = diff_s == 0

        # calculate the mean point of continuous bad points, as the new extrema
        pre_index = 0
        while pre_index < len(diff_s_zero):

            if not diff_s_zero[pre_index]:
                pre_index += 1
                continue

            # this is the index of diff_s_zero, not index in the original signal
            post_index = pre_index
            while post_index < len(diff_s_zero) and diff_s_zero[post_index]:
                post_index += 1
            mean_index = (pre_index + post_index - 1) // 2 - 1
            is_extrema[mean_index] = True
            pre_index = post_index + 1

        if self.include_endpoints:
            is_extrema = np.array((True, ) + tuple(is_extrema) + (True, ))

        index = np.arange(0, np.shape(signal)[0])
        extrema_index = index[is_extrema]

        return extrema_index

    def get_mean_curve(self, signal, extrema_index):

        extrema_index = np.sort(extrema_index)
        mean = np.zeros(signal.shape)
        for pre, post in zip(extrema_index[:-1], extrema_index[1:]):
            mean[pre: post] = (signal[pre] + signal[post]) / 2
        mean[-1] = mean[-2]

        # """ ****** fix me ****** """
        # # considering the end-points as the extrema to compute the mean curve
        # mean[:extrema_index[0]] = (signal[0] + signal[extrema_index[0]]) / 2
        # mean[extrema_index[-1]:] = (signal[-1] + signal[extrema_index[-1]]) / 2

        return mean

    def get_envelope_curve(self, signal, extrema_index):

        extrema_index = np.sort(extrema_index)
        envelope = np.zeros(signal.shape)
        for pre, post in zip(extrema_index[:-1], extrema_index[1:]):
            envelope[pre: post] = np.abs(signal[pre] - signal[post]) / 2
        envelope[-1] = envelope[-2]

        # """ ****** fix me ****** """
        # # considering the end-points as the extrema to compute the envelope curve
        # envelope[:extrema_index[0]] = (signal[0] + signal[extrema_index[0]]) / 2
        # envelope[extrema_index[-1]:] = (signal[-1] + signal[extrema_index[-1]]) / 2

        return envelope

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
        window = np.int(_win / 3)
        if window % 2 == 0:
            window += 1

        return int(window)

    """ adding other MA methods """
    def moving_average(self, curve, window):

        for i in range(self.ma_max_iter):

            curve = self._ma_padding(curve, window)
            if self.ma_mode == "simple":
                curve = self._simple_moving_average(curve, window)
            elif self.ma_mode == "triangle":
                curve = self._triangle_moving_average(curve, window)
            else:
                raise ValueError("expected ma_mode should be within ['simple'], but got '%s' instead !"
                                 % str(self.ma_mode))

            if self._is_smoothed(curve):
                break

        return curve

    def _ma_padding(self, curve, window):

        if self.ma_pad_mode == "sym":
            cat_tuple = (curve[1: int(window // 2 + 1)][::-1], curve, curve[- int(window // 2) - 1: -1][::-1])

        elif self.ma_pad_mode == "head":
            cat_tuple = (curve[1: int(window)][::-1], curve)

        elif self.ma_pad_mode == "tail":
            cat_tuple = (curve, curve[- int(window): -1][::-1])

        else:
            raise ValueError("expected padding mode shoule be within ['sym', 'head', 'tail'], but got %s instead !"
                             % str(self.ma_pad_mode))

        curve = np.concatenate(cat_tuple, axis=0)
        return curve

    def _is_smoothed(self, curve):

        diff_curve_zero = np.diff(curve, n=1) == 0

        if True in diff_curve_zero:
            return False

        return True

    def _simple_moving_average(self, curve, window):

        """
            the curve is the original signal + padding
            according to the mode of padding, different mode of moving average is chosen:

                head_mirror_padding: curve = x[1: window][::-1] + x
                                     x_smoothed[i] = (x[i-window+1] + x[i-window+2] + ... + x[i]) / window

                tail_mirror_padding: curve = x + x[-window: -1][::-1]
                                     x_smoothed[i] = (x[i] + x[i+1] + ... + x[i+window-1]) / window

                sym_mirror_padding:  curve = x[1: window//2+1][::-1] + x + x[-1-window//2: -1][::-1]
                                     x_smoothed[i] = (x[i-window//2 + 1] + ... + x[i] + ... + x[i+window//2]) / window

            :param curve:   padded curve of the original signal
            :param window:  the size of window
            :return:        smoothed curve

        """
        length = curve.shape[-1]
        smoothed_curve = np.zeros((length - window + 1))
        for i in range(window):
            smoothed_curve += curve[i: length - window + 1 + i]
        smoothed_curve /= window

        return smoothed_curve

    def _triangle_moving_average(self, curve, window):

        """
            the curve is the original signal + padding
            according to the mode of padding, different mode of moving average is chosen:

                head_mirror_padding: curve = x[1: window][::-1] + x
                                     weight is like [1, 2, 3, 4, 5] / 15

                tail_mirror_padding: curve = x + x[-window: -1][::-1]
                                     weight is like [5, 4, 3, 2, 1] / 15

                sym_mirror_padding:  curve = x[1: window//2+1][::-1] + x + x[-1-window//2: -1][::-1]
                                     weight is like [1, 2, 3, 2, 1] / 9

                smoothed_curve = x_padding ** weight,
                (** means convolution)

            :param curve:   padded curve of the original signal
            :param window:  the size of window
            :return:        smoothed curve

        """

        weight = None
        if self.ma_pad_mode == "head":
            weight = np.arange(1, window + 1)
            weight = weight / np.sum(weight)

        elif self.ma_pad_mode == "tail":
            weight = np.arange(1, window + 1)[::-1]
            weight = weight / np.sum(weight)

        elif self.ma_pad_mode == "sym":
            weight = np.arange(1, window // 2 + 2)
            weight = np.concatenate((weight, weight[:-1][::-1]), axis=0)
            weight = weight / np.sum(weight)

        else:
            raise ValueError("expected padding mode should be within ['sym', 'head', 'tail'], but got %s instead !"
                             % str(self.ma_pad_mode))

        length = curve.shape[-1]
        smoothed_curve = np.zeros((length - window + 1))
        for i in range(window):
            smoothed_curve += curve[i: length - window + 1 + i] * weight[i]
        # smoothed_curve /= window

        return smoothed_curve

        pass

    def pf_calculate(self, signal):

        _s = np.copy(signal)

        """ for debug, fix me """
        m, a, s = [], [], []

        envelop = 1
        for i in range(self.pf_max_iter):
            extrema_index = self.find_extrema(_s)
            _m, _a = self.get_mean_curve(_s, extrema_index), self.get_envelope_curve(_s, extrema_index)

            window = self.get_window(extrema_index)
            _m, _a = self.moving_average(_m, window), self.moving_average(_a, window)

            """ for debug, fix me """
            m.append(_m), a.append(_a), s.append(_s)

            hij = _s - _m
            _s = hij / _a
            envelop = envelop * _a
            if self._is_pf_envelop_approx_one(_a):
                break

        return _s, envelop, (m, a, s)

    def _is_pf_envelop_approx_one(self, curve):

        res = curve - 1
        if np.mean(np.abs(res)) <= self.pf_en_threhold:
            return True

        return False

    def __call__(self, signal):

        res = np.copy(signal)

        """ for debug, fix me """
        reses, s, envelopes = [], [], []

        for i in range(self.pf_max_iter):

            _s, envelop, _ = self.pf_calculate(res)

            """ for debug, fix me """
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
    N, tMin, tMax = 800, 0, 1
    T = np.linspace(tMin, tMax, N, dtype=DTYPE)
    # S = 18 * np.sin(5 * np.pi * T * (1 + 0.2 * T)) + T ** 2 + T * np.sin(13 * T)
    # S = 18 * np.sin(5 * np.pi * T) * np.cos(40 * np.pi * T) + 30 * np.cos(9 * np.pi * T)
    S = (2 + np.cos(18 * np.pi * T)) * np.cos(400 * np.pi * T + 2 * np.cos(20 * np.pi * T)) + \
        4 * np.cos(5 * np.pi * T) * np.cos(120 * np.pi * T) + \
        2 * np.cos(40 * np.pi * T) + 0.8 * np.sin(np.pi * T) * np.sin(10 * np.pi * T)

    # S[-10:] = S[224]
    S = S.astype(DTYPE)
    logging.info("Input S.dtype: " + str(S.dtype))

    # LMD initial
    lmd = LocalMeanDecomp()
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





# aaa = LMD.find_extrema(S)
# _m = LMD.get_mean_curve(S, aaa)
# _a = LMD.get_envelope_curve(S, aaa)
# win = LMD.get_window(aaa)
#
# __m = LMD.moving_average(_m, win)
# __a = LMD.moving_average(_a, win)
# _s, envelop, (m, a, s) = LMD.pf_calculate(S)

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.subplot(421)
# plt.plot(_s)
#
# plt.subplot(423)
# plt.plot(envelop)
#
# plt.subplot(425)
# plt.plot(_s * envelop)
#
# plt.subplot(424)
# plt.plot(S)
#
# plt.subplot(426)
# plt.plot(S - _s * envelop)
#
# plt.subplot(427)
# plt.plot(np.sin(20 * T * (1 + 0.2 * T)))
# plt.subplot(428)
# plt.plot(T ** 2 + T * np.sin(13 * T))
#
# plt.figure(2)
# n = 1
# for __m, __s, __a in zip(m, s, a):
#     plt.subplot(6, 2, n)
#     plt.plot(__s)
#     plt.plot(__m)
#
#     plt.subplot(6, 2, n+1)
#     plt.plot(__a)
#
#     n += 2



# """ fix me """
# def mean_interpolate(self, signal, extrema_index):
#
#     extrema_index = np.sort(extrema_index)
#     mean = np.zeros(signal.shape)
#     for pre, post in zip(extrema_index[:-1], extrema_index[1:]):
#         mean[pre: post] = (signal[pre] + signal[post]) / 2
#
#     """ ****** fix me ****** """
#     # considering the end-points as the extrema to compute the mean curve
#     mean[:extrema_index[0]] = (signal[0] + signal[extrema_index[0]]) / 2
#     mean[extrema_index[-1]:] = (signal[-1] + signal[extrema_index[-1]]) / 2
#
#     return mean
#
# """ fix me """
# def envelope_interpolate(self, signal, extrema_index):
#
#     extrema_index = np.sort(extrema_index)
#     envelope = np.zeros(signal.shape)
#     for pre, post in zip(extrema_index[:-1], extrema_index[1:]):
#         envelope[pre: post] = np.abs(signal[pre] - signal[post]) / 2
#
#     """ ****** fix me ****** """
#     # considering the end-points as the extrema to compute the envelope curve
#     envelope[:extrema_index[0]] = (signal[0] + signal[extrema_index[0]]) / 2
#     envelope[extrema_index[-1]:] = (signal[-1] + signal[extrema_index[-1]]) / 2
#
#     return envelope
# def is_monotonous(signal):
#     """判断信号是否是(非严格的)单调序列"""
#
#     def is_monotonous_increase(signal):
#         y0 = signal[0]
#         for y1 in signal:
#             if y1 < y0:
#                 return False
#             y0 = y1
#         return True
#
#     def is_monotonous_decrease(signal):
#         y0 = signal[0]
#         for y1 in signal:
#             if y1 > y0:
#                 return False
#             y0 = y1
#         return True
#
#     if len(signal) <= 0:
#         return True
#     else:
#         return is_monotonous_increase(signal) \
#                or is_monotonous_decrease(signal)
#
#
# def find_extrema(signal):
#     """找出信号的所有局部极值点位置."""
#
#     n = len(signal)
#
#     # 局部极值的位置列表
#     extrema = []
#
#     lookfor = None
#     for x in range(1, n):
#         y1 = signal[x - 1]
#         y2 = signal[x]
#         if lookfor == "min":
#             if y2 > y1:
#                 extrema.append(x - 1)
#                 lookfor = "max"
#         elif lookfor == "max":
#             if y2 < y1:
#                 extrema.append(x - 1)
#                 lookfor = "min"
#         else:
#             if y2 < y1:
#                 lookfor = "min"
#             elif y2 > y1:
#                 lookfor = "max"
#
#     # 优化(微调)极值点的位置:
#     # 当连续多个采样值相同时，取最中间位置作为极值点位置
#
#     def micro_adjust(x):
#         assert (0 <= x < n)
#         y = signal[x]
#         lo = hi = x
#         while (lo - 1 >= 0) and (signal[lo - 1] == y): lo -= 1
#         while (hi + 1 < n) and (signal[hi + 1] == y): hi += 1
#         if INCLUDE_ENDPOINTS:
#             if lo == 0: return 0
#             if hi == n - 1: return n - 1
#         return (lo + hi) // 2
#
#     extrema = [micro_adjust(x) for x in extrema]
#
#     if extrema and INCLUDE_ENDPOINTS:
#         if extrema[0] != 0:
#             extrema.insert(0, 0)
#         if extrema[-1] != n - 1:
#             extrema.append(n - 1)
#
#     return extrema
#
#
# def moving_average_smooth(signal, window):
#     """使用移动加权平均算法平滑方波信号."""
#
#     n = len(signal)
#
#     # at least one nearby sample is needed for average
#     if window < 3:
#         window = 3
#
#     # adjust length of sliding window to an odd number for symmetry
#     if (window % 2) == 0:
#         window += 1
#
#     half = window // 2
#     window = 5
#     weight = list(range(1, half + 2)) + list(range(half, 0, -1))
#     assert (len(weight) == window)
#
#     def sliding(signal, window):
#         for x in range(n):
#             x1 = x - half
#             x2 = x1 + window
#             w1 = 0
#             w2 = window
#             if x1 < 0: w1 = -x1; x1 = 0
#             if x2 > n: w2 = n - x2; x2 = n
#             yield signal[x1:x2], weight[w1:w2]
#
#     def weighted(signal, weight):
#         assert (len(signal) == len(weight))
#         return sum(y * w for y, w in zip(signal, weight)) / sum(weight)
#
#     def is_smooth(signal):
#         for x in range(1, n):
#             if signal[x] == signal[x - 1]:
#                 return False
#         return True
#
#     smoothed = signal
#     for i in range(MAX_SMOOTH_ITERATION):
#         smoothed = [weighted(s, w) for s, w in sliding(smoothed, window)]
#         if is_smooth(smoothed): break
#
#     return smoothed
#
#
# def local_mean_and_envelope(signal, extrema):
#     """根据极值点位置计算局部均值函数和局部包络函数."""
#
#     n = len(signal)
#     k = len(extrema)
#     assert (1 < k <= n)
#     # construct square signal
#     mean = []
#     enve = []
#     prev_mean = (signal[extrema[0]] + signal[extrema[1]]) / 2
#     prev_enve = abs(signal[extrema[0]] - signal[extrema[1]]) / 2
#     e = 1
#     for x in range(n):
#         if (x == extrema[e]) and (e + 1 < k):
#             next_mean = (signal[extrema[e]] + signal[extrema[e + 1]]) / 2
#             mean.append((prev_mean + next_mean) / 2)
#             prev_mean = next_mean
#             next_enve = abs(signal[extrema[e]] - signal[extrema[e + 1]]) / 2
#             enve.append((prev_enve + next_enve) / 2)
#             prev_enve = next_enve
#             e += 1
#         else:
#             mean.append(prev_mean)
#             enve.append(prev_enve)
#     # smooth square signal
#     window = max(extrema[i] - extrema[i - 1] for i in range(1, k)) // 3
#     return mean, moving_average_smooth(mean, window), \
#            enve, moving_average_smooth(enve, window)
#
#
# def extract_product_function(signal):
#
#     s = signal
#     n = len(signal)
#     envelopes = []  # 每次迭代得到的包络信号
#
#     def component():
#         c = []
#         for i in range(n):
#             y = s[i]
#             for e in envelopes:
#                 y = y * e[i]
#             c.append(y)
#         return c
#
#     for i in range(MAX_ENVELOPE_ITERATION):
#         extrema = find_extrema(s)
#         if len(extrema) <= 3: break
#         m0, m, a0, a = local_mean_and_envelope(s, extrema)
#         for y in a:
#             if y <= 0: raise ValueError("invalid envelope signal")
#         # 对原信号进行调制
#         h = [y1 - y2 for y1, y2 in zip(s, m)]
#         t = [y1 / y2 for y1, y2 in zip(h, a)]
#         if DBG_FIG:
#             DBG_FIG.plot(i, component(), s, m0, m, a0, a, h, t)
#         # 得到纯调频信号时终止迭代
#         err = sum(abs(1 - y) for y in a) / n
#         if err <= ENVELOPE_EPSILON: break
#         # 调制信号收敛时终止迭代
#         err = sum(abs(y1 - y2) for y1, y2 in zip(s, t)) / n
#         if err <= CONVERGENCE_EPSILON: break
#         envelopes.append(a);
#         s = t
#
#     return component()
#
#
# def LMD(signal):
#
#     pf = []
#     # 每次迭代得到一个PF分量, 直到残余函数接近为一个单调函数为止
#     residue = signal[:]
#     while (len(pf) < MAX_NUM_PF) and \
#             (not is_monotonous(residue)) and \
#             (len(find_extrema(residue)) >= 5):
#         component = extract_product_function(residue)
#         residue = [y1 - y2 for y1, y2 in zip(residue, component)]
#         pf.append((component, residue))
#
#     return pf
#
#
# if __name__ == "__main__":
#     #
#     # 算法控制参数
#     #
#     # 是否把信号的端点看作伪极值点
#     INCLUDE_ENDPOINTS = True
#
#     # 滑动平均算法: 最多迭代的次数
#     MAX_SMOOTH_ITERATION = 12
#
#     # 分离局部包络信号时最多迭代的次数
#     MAX_ENVELOPE_ITERATION = 200
#
#     ENVELOPE_EPSILON = 0.01
#     CONVERGENCE_EPSILON = 0.01
#
#     # 最多生成的积函数个数
#     MAX_NUM_PF = 8
#
#     # for debugging
#     DBG_FIG = None