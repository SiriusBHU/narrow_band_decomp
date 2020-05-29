from dataflow.CWRU_data_load import *
from dataflow.MFPT_data_load import *
from dataflow.PU_data_load import *
from dataflow.SU_data_load import *


def imbalance_data_prepare(choice, path_project,
                           normal_num, faulty_num,
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
        im_data, im_labels = cw_data.imbalance_case_wc(normal_num=normal_num, faulty_num=faulty_num)

    elif choice == "MFPT":
        mp_data = MFPTBearing(500, sample_len, path_project=path_project)
        if is_generate:
            mp_data.dataset_prepare_MFPT(500, sample_len)
            mp_data.normal_set_prepare_MFPT(sample_num=5000, sample_len=sample_len)
        im_data, im_labels = mp_data.imbalance_case(normal_num=normal_num, faulty_num=faulty_num)

    elif choice == "PU":
        pu_data = PaderbornBearing(300, sample_len, path_project=path_project)
        if is_generate:
            pu_data.dataset_prepare_PU(300, sample_len)
            pu_data.normal_set_prepare_PU(sample_num=1000, sample_len=sample_len)
        im_data, im_labels = pu_data.imbalance_case_ar(normal_num=normal_num, faulty_num=faulty_num)

    elif choice == "SU":
        su_data = SoutheastBearingGear(500, sample_len, path_project=path_project)
        if is_generate:
            su_data.dataset_prepare_SU(500, sample_len)
            su_data.normal_set_prepare_SU(sample_num=5000, sample_len=sample_len)
        im_data, im_labels = su_data.imbalance_case_wc(normal_num=normal_num, faulty_num=faulty_num, chs=4)

    else:
        raise ValueError("expected data-set choice should be within ['CWRU', 'SU', 'MFPT', 'PU'], but got %s"
                         % str(choice))

    return im_data, im_labels


def train_val_split(data, labels, val_rate=0.5):
    nums = data.shape[0]
    if nums != labels.shape[0]:
        raise AttributeError("dim mis-match")

    perm = np.arange(nums)
    np.random.shuffle(perm)

    train_indexs, val_indexes = perm[int(nums * val_rate):], perm[:int(nums * val_rate)]

    return data[train_indexs], labels[train_indexs], data[val_indexes], labels[val_indexes]


def normalization(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if len(data.shape) != 2:
        raise AttributeError("wrong dim != 2")

    mean, std = np.mean(data, axis=-1).reshape(-1, 1), np.std(data, axis=-1).reshape(-1, 1)
    data = (data - mean) / std
    data[np.isnan(data)] = 0

    return data


def standardization(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if len(data.shape) != 2:
        raise AttributeError("wrong dim != 2")

    mean, std = np.mean(data, axis=0).reshape(1, -1), np.std(data, axis=0).reshape(1, -1)
    data = (data - mean) / std
    data[np.isnan(data)] = 0

    return data




