"""
    Author: Sirius HU
    Created Date: 2019.11.03

    utils for local mean decomposition

"""


import numpy as np


def find_extrema(signal, end_as_extrema=True):

    """
        find the extrema of the signal,
        extrema are the set of maximum and/or minimum of the signal
        here we calculate the finite differences of the signal to find the extrema

        note that, by default, we set the end-points as extrema as well.

        :pa,ram
            signal: the input signal
        :return
            the extrema's index of the signal
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

    # end-points are also set as extrema
    if end_as_extrema:
        is_extrema = np.array((True, ) + tuple(is_extrema) + (True, ))
    else:
        is_extrema = np.array((False,) + tuple(is_extrema) + (False,))

    index = np.arange(0, np.shape(signal)[0])
    extrema_index = index[is_extrema]

    return extrema_index


def get_mean_curve(signal, extrema_index):

    extrema_index = np.sort(extrema_index)
    mean = np.zeros(signal.shape)
    for pre, post in zip(extrema_index[:-1], extrema_index[1:]):
        mean[pre: post] = (signal[pre] + signal[post]) / 2
    mean[-1] = mean[-2]

    return mean


def get_envelope_curve(signal, extrema_index):

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


def moving_average(curve, window, weight):

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
        smoothed_curve += weight[i] * curve[i: length - window + 1 + i]

    return smoothed_curve


def ma_padding(curve, window, ma_pad_mode):

    if ma_pad_mode == "sym":
        cat_tuple = (curve[1: int(window // 2 + 1)][::-1], curve, curve[- int(window // 2) - 1: -1][::-1])

    elif ma_pad_mode == "head":
        cat_tuple = (curve[1: int(window)][::-1], curve)

    elif ma_pad_mode == "tail":
        cat_tuple = (curve, curve[- int(window): -1][::-1])

    else:
        raise ValueError("expected padding mode shoule be within ['sym', 'head', 'tail'], but got %s instead !"
                         % str(ma_pad_mode))

    curve = np.concatenate(cat_tuple, axis=0)
    return curve


def get_ma_weight(window, ma_mode, ma_pad_mode):

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
    """

    if ma_mode == 'simple':
        weight = np.ones((window,)) / window

    elif ma_mode == 'triangle':

        if ma_pad_mode == "head":
            weight = np.arange(1, window + 1)
            weight = weight / np.sum(weight)

        elif ma_pad_mode == "tail":
            weight = np.arange(1, window + 1)[::-1]
            weight = weight / np.sum(weight)

        elif ma_pad_mode == "sym":
            weight = np.arange(1, window // 2 + 2)
            weight = np.concatenate((weight, weight[:-1][::-1]), axis=0)
            weight = weight / np.sum(weight)
        else:
            raise ValueError("expected padding mode should be within ['sym', 'head', 'tail'], but got %s instead !"
                             % str(ma_pad_mode))
    else:
        raise ValueError("expected ma mode should be within ['simple', 'triangle'], but got %s instead !"
                         % str(ma_mode))

    return weight