from EMD_hu.emd import EmpiricalModeDecomp
from EMD_hu.inst_freq import InstantaneousFreq
from dataflow.CWRU_data_load import *
from dataflow.MFPT_data_load import *
from dataflow.PU_data_load import *
from dataflow.SU_data_load import *
import logging
import matplotlib.pyplot as plt


def data_prepare(choice, path_project,
                 few_num,
                 is_generate=False, sample_len=None):
    if is_generate:
        sample_len_dict = {"CWRU": 2048, "MFPT": 4096, "PU": 4096, "SU": 2048}

        if sample_len is None:
            sample_len = sample_len_dict[choice]

        if not isinstance(sample_len, int):
            sample_len = int(sample_len)

        if sample_len <= 0:
            raise ValueError("expected sample length should be larger than zero, but got %d" % sample_len)

    if choice == "CWRU":
        cw_data = CaseWesternBearing(500, sample_len, path_project=path_project)
        if is_generate:
            cw_data.dataset_prepare_CWRU(500, sample_len)
            cw_data.normal_set_prepare_CWRU(sample_num=5000, sample_len=sample_len)
        data, labels = cw_data.working_condition_transferring(few_num)

    elif choice == "MFPT":
        mp_data = MFPTBearing(500, sample_len, path_project=path_project)
        if is_generate:
            mp_data.dataset_prepare_MFPT(500, sample_len)
            mp_data.normal_set_prepare_MFPT(sample_num=5000, sample_len=sample_len)
        data, labels = mp_data.working_condition_transferring(few_num)

    elif choice == "PU":
        pu_data = PaderbornBearing(300, sample_len, path_project=path_project)
        if is_generate:
            pu_data.dataset_prepare_PU(300, sample_len)
            pu_data.normal_set_prepare_PU(sample_num=1000, sample_len=sample_len)
        data, labels = pu_data.working_condition_transferring(few_num)

    elif choice == "SU":
        su_data = SoutheastBearingGear(500, sample_len, path_project=path_project)
        if is_generate:
            su_data.dataset_prepare_SU(500, sample_len)
            su_data.normal_set_prepare_SU(sample_num=5000, sample_len=sample_len)
        data, labels = su_data.bearing_working_condition_transferring(few_num=few_num, chs=2)

    else:
        raise ValueError("expected data-set choice should be within ['CWRU', 'SU', 'MFPT', 'PU'], but got %s"
                         % str(choice))

    return data, labels


def normalization(data):

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if len(data.shape) != 2:
        raise AttributeError("wrong dim != 2")

    mean, std = np.mean(data, axis=-1).reshape(-1, 1), np.std(data, axis=-1).reshape(-1, 1)
    data = (data - mean) / std
    data[np.isnan(data)] = 0

    return data


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
    # logger setting
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s")
    # hyper-param setting
    FEW_NUM = 1
    SAMPLING_RATE = 5000

    # prepare data
    path_project = "D:\\DataSet_Preparation\\"
    data, labels = data_prepare("SU", path_project=path_project,
                                few_num=FEW_NUM,
                                is_generate=False)
    labels = np.argmax(labels, axis=-1)

    wcs, nums, chs, length = data.shape
    data, labels = data.reshape(wcs * nums * chs, length), labels.reshape(wcs * nums)
    data = normalization(data)
    coord = np.arange(length) / SAMPLING_RATE

    ifff, ampp = [], []
    emd, inst = EmpiricalModeDecomp(), InstantaneousFreq()
    for i, signal in enumerate(data[:]):
        imfs, res = emd(signal)
        # imfs_show(coord, signal, imfs, res)
        inst_freqs, amps = inst.arccos_method(imfs, sampling_rate=36000)
        inst.inst_freq_ma_smooth(inst_freqs[0], window=51)
        inst.inst_freq_ma_smooth(inst_freqs[1], window=51)
        inst.inst_freq_ma_smooth(inst_freqs[2], window=51)
        inst.inst_freq_ma_smooth(inst_freqs[3], window=51)
        ifff.append(inst_freqs), ampp.append(amps)

        print(i, labels[i], i // 15, "[", np.sum(inst_freqs * amps / np.sum(amps, axis=-1).reshape(-1, 1), axis=-1), np.std(inst_freqs, axis=-1), "]")
        inst.time_freq_representation(inst_freqs, amps * 10, freq_dim=400, sampling_rate=36000)
        # inst.inst_freq_representation(coord[:], signal, imfs, inst_freqs, amps, res, same_lim=True)
    plt.show()
    ifff, ampp = np.array(ifff), np.array(ampp)
    if_mean, if_std = np.mean(ifff, axis=-1), np.std(ifff, axis=-1)
    am_mean, am_std = np.mean(ampp, axis=-1), np.std(ampp, axis=-1)

    data = [if_mean, if_std, am_mean, am_std]

    c1, c2 = 0, 2
    no_if = 0
    plt.scatter(data[c1][:500, no_if], data[c2][:500, no_if], c=labels[:500])
    plt.scatter(data[c1][200:300, no_if], data[c2][200:300, no_if], c="k")
    plt.show()

    print(1)
