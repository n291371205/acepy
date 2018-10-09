"""
Data Split
Split the original dataset into train/test label/unlabelset
Accept not only datamat, but also shape/list of instance name (for image datasets)
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division

import os

import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array

from utils.query_type import check_query_type
from utils.tools import check_matrix
from utils.tools import randperm


def split(X=None, y=None, instance_indexes=None, query_type=None, test_ratio=0.3, initial_label_rate=0.05,
          split_count=10, all_class=True, saving_path='.'):
    """Split given data.

    Parameters
    ----------
    X: array-like, optional
        data matrix with [n_samples, n_features]

    y: array-like, optional
        labels of given data [n_samples, n_labels] or [n_samples]

    instance_indexes: list, optional (default=None)
        list contains instances' names, used for image datasets,
        or provide index list instead of data matrix.
        Must provide one of [instance_names, X, y]

    query_type: str, optional (default='AllLabels')
        query type.

    test_ratio: float, optional (default=0.3)
        ratio of test set

    initial_label_rate: float, optional (default=0.05)
        ratio of initial label set
        e.g. initial_labelset*(1-test_ratio)*n_samples

    split_count: int, optional (default=10)
        random split data split_count times

    all_class: bool, optional (default=True)
        whether each split will contain at least one instance for each class.
        If False, a totally random split will be performed.

    saving_path: str, optional (default='.')

    Returns
    -------
    train_idx: array-like
        index of training set, shape like [n_split_count, n_training_samples]

    test_idx: array-like
        index of testing set, shape like [n_split_count, n_testing_samples]

    label_idx: array-like
        index of labeling set, shape like [n_split_count, n_labeling_samples]

    unlabel_idx: array-like
        index of unlabeling set, shape like [n_split_count, n_unlabeling_samples]
    """
    # check parameters
    if X is None and y is None and instance_indexes is None:
        raise Exception("Must provide one of X, y or instance_indexes.")
    len_of_parameters = [len(X) if X is not None else None, len(y) if y is not None else None,
                         len(instance_indexes) if instance_indexes is not None else None]
    number_of_instance = np.unique([i for i in len_of_parameters if i is not None])
    if len(number_of_instance) > 1:
        raise ValueError("Different length of instances and labels found.")
    else:
        number_of_instance = number_of_instance[0]
    if query_type is None:
        query_type = 'AllLabels'
    else:
        if not check_query_type(query_type):
            raise NotImplementedError("Query type %s is not implemented." % type)
    if instance_indexes is not None:
        if not isinstance(instance_indexes, (list, np.ndarray)):
            raise TypeError("An array-like object is expected, but received: %s" % str(type(instance_indexes)))
        instance_indexes = np.array(instance_indexes)
    else:
        instance_indexes = np.arange(number_of_instance)

    # split
    train_idx = []
    test_idx = []
    label_idx = []
    unlabel_idx = []
    for i in range(split_count):
        if (not all_class) or y is None:
            rp = randperm(number_of_instance - 1)
            cutpoint = round((1 - test_ratio) * len(rp))
            tp_train = instance_indexes[rp[0:cutpoint]]
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            cutpoint = round(initial_label_rate * len(tp_train))
            if cutpoint <= 1:
                cutpoint = 1
            label_idx.append(tp_train[0:cutpoint])
            unlabel_idx.append(tp_train[cutpoint:])
        else:
            if y is None:
                raise Exception("y must be provided when all_class flag is True.")
            y = check_array(y, ensure_2d=False, dtype=None)
            if y.ndim == 1:
                label_num = len(np.unique(y))
            else:
                label_num = y.shape[1]
            if round((1 - test_ratio) * initial_label_rate * number_of_instance) < label_num:
                raise ValueError(
                    "The initial rate is too small to guarantee that each "
                    "split will contain at least one instance for each class.")

            # check validaty
            while 1:
                rp = randperm(number_of_instance - 1)
                cutpoint = round((1 - test_ratio) * len(rp))
                tp_train = instance_indexes[rp[0:cutpoint]]
                cutpointlabel = round(initial_label_rate * len(tp_train))
                if cutpointlabel <= 1:
                    cutpointlabel = 1
                label_id = tp_train[0:cutpointlabel]
                if y.ndim == 1:
                    if len(np.unique(y[label_id])) == label_num:
                        break
                else:
                    temp = np.sum(y[label_id], axis=0)
                    if not np.any(temp == 0):
                        break
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            label_idx.append(tp_train[0:cutpointlabel])
            unlabel_idx.append(tp_train[cutpointlabel:])

    split_save(train_idx=train_idx, test_idx=test_idx, label_idx=label_idx,
               unlabel_idx=unlabel_idx, path=saving_path)
    return train_idx, test_idx, unlabel_idx, label_idx


def split_multi_label(y=None, label_shape=None, test_ratio=0.3, initial_label_rate=0.05,
                      split_count=10, all_class=True, partially_labeled=False, saving_path='.'):
    """Split given label matrix in multi label setting.

    Parameters
    ----------
    y: array-like, optional
        labels of given data, shape like [n_samples, n_labels]

    label_shape: tuple, optional (default=None)
        the shape of y, should be a tuple with 2 elements.
        The first one is the number of instances, and the other is the
        number of labels.

    test_ratio: float, optional (default=0.3)
        ratio of test set

    initial_label_rate: float, optional (default=0.05)
        ratio of initial label set
        e.g. initial_labelset*(1-test_ratio)*n_samples

    split_count: int, optional (default=10)
        random split data split_count times

    all_class: bool, optional (default=True)
        whether each split will contain at least one instance for each class.
        If False, a totally random split will be performed.

    partially_labeled: bool, optional (default=False)
        Whether split the data as partially labeled in the multi-label setting.
        If False, the labeled set is fully labeled, otherwise, only part of labels of each
        instance will be labeled initialized.
        Only available in multi-label setting.

    saving_path: str, optional (default='.')

    Returns
    -------
    train_idx: array-like
        index of training set, shape like [n_split_count, n_training_samples]

    test_idx: array-like
        index of testing set, shape like [n_split_count, n_testing_samples]

    label_idx: array-like
        index of labeling set, shape like [n_split_count, n_labeling_samples]

    unlabel_idx: array-like
        index of unlabeling set, shape like [n_split_count, n_unlabeling_samples]

    """

    # check parameters
    if y is None and label_shape is None:
        raise Exception("Must provide one of y or label_shape.")
    data_shape = None
    if y is not None:
        y = check_matrix(y)
        ytype = type_of_target(y)
        if ytype not in ['multilabel-indicator', 'multilabel-sequences']:
            raise ValueError("y must be a 2D array with the shape like [n_samples, n_labels]")
        data_shape = y.shape
    if label_shape is not None:
        if not isinstance(label_shape, tuple) and len(label_shape) == 2:
            raise TypeError("the shape of y, should be a tuple with 2 elements."
                            "The first one is the number of instances, and the other is the"
                            "number of labels.")
        data_shape = label_shape
    instance_indexes = np.arange(data_shape[0])

    # split
    train_idx = []
    test_idx = []
    label_idx = []
    unlabel_idx = []
    for i in range(split_count):
        if partially_labeled:
            # split train test
            rp = randperm(data_shape[0] - 1)
            cutpoint = round((1 - test_ratio) * len(rp))
            tp_train = instance_indexes[rp[0:cutpoint]]

            # split label & unlabel
            train_size = len(tp_train)
            lab_ind = randperm((0, train_size * data_shape[1] - 1), round(initial_label_rate * train_size))
            if all_class:
                if round(initial_label_rate * train_size) < data_shape[1]:
                    raise ValueError("The initial rate is too small to guarantee that each "
                                     "split will contain at least one instance for each class.")
                while len(np.unique([item % data_shape[1] for item in lab_ind])) != data_shape[1]:
                    # split train test
                    rp = randperm(data_shape[0] - 1)
                    cutpoint = round((1 - test_ratio) * len(rp))
                    tp_train = instance_indexes[rp[0:cutpoint]]
                    # split label & unlabel
                    train_size = len(tp_train)
                    lab_ind = randperm((0, train_size * data_shape[1] - 1), round(initial_label_rate * train_size))
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            unlab_ind = set(np.arange(train_size * data_shape[1]))
            unlab_ind.difference_update(set(lab_ind))
            label_idx.append([(tp_train[item // data_shape[1]], item % data_shape[1]) for item in lab_ind])
            unlabel_idx.append([(tp_train[item // data_shape[1]], item % data_shape[1]) for item in unlab_ind])
        else:
            rp = randperm(data_shape[0] - 1)
            cutpoint = round((1 - test_ratio) * len(rp))
            tp_train = instance_indexes[rp[0:cutpoint]]

            cutpoint = round(initial_label_rate * len(tp_train))
            if cutpoint <= 1:
                cutpoint = 1
            if all_class:
                if cutpoint < data_shape[1]:
                    raise ValueError(
                        "The initial rate is too small to guarantee that each "
                        "split will contain at least one instance-label pair for each class.")
                while 1:
                    label_id = tp_train[0:cutpoint]
                    temp = np.sum(y[label_id], axis=0)
                    if not np.any(temp == 0):
                        break
                    rp = randperm(data_shape[0] - 1)
                    cutpoint = round((1 - test_ratio) * len(rp))
                    tp_train = instance_indexes[rp[0:cutpoint]]

                    cutpoint = round(initial_label_rate * len(tp_train))
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            label_idx.append([(i,) for i in tp_train[0:cutpoint]])
            unlabel_idx.append([(i,) for i in tp_train[cutpoint:]])
    split_save(train_idx=train_idx, test_idx=test_idx, label_idx=label_idx,
               unlabel_idx=unlabel_idx, path=saving_path)
    return train_idx, test_idx, unlabel_idx, label_idx


def split_load(path):
    """Load split from path.

    Parameters
    ----------
    path: str
        Path to a specific file, not a dir.

    Returns
    -------
    setting: experiment_saver.al_experiment.ExperimentSetting
        Object of ExperimentSetting.
    """
    if not isinstance(path, str):
        raise TypeError("A string is expected, but received: %s" % str(type(path)))
    import pickle
    f = open(os.path.abspath(path), 'rb')
    split_setting = pickle.load(f)
    f.close()
    # return train_idx, test_idx, label_idx, unlabel_idx
    return split_setting


def split_save(train_idx, test_idx, label_idx, unlabel_idx, path):
    """Save the split to file for auditting or loading for other methods.

    Parameters
    ----------
    saving_path: str
        path to save the settings. If a dir is provided, it will generate a file called
        'al_split.pkl' for saving.

    """
    if path is None:
        return
    else:
        if not isinstance(path, str):
            raise TypeError("A string is expected, but received: %s" % str(type(path)))
    import pickle
    saving_path = os.path.abspath(path)
    if os.path.isdir(saving_path):
        f = open(os.path.join(saving_path, 'al_split.pkl'), 'wb')
    else:
        f = open(os.path.abspath(saving_path), 'wb')
    pickle.dump((train_idx, test_idx, label_idx, unlabel_idx), f)
    f.close()


if __name__ == '__main__':
    train_idx, test_idx, unlabel_idx, label_idx = split(X=np.random.random((10, 10)), y=np.random.randn(10, 10),
                                                        all_class=False)
    print(train_idx)
    print(test_idx)
    print(label_idx)
    print(unlabel_idx)
    train_idx, test_idx, unlabel_idx, label_idx = split_multi_label(y=np.random.randint(0, 2, 800).reshape(100, -1),
                                                                    initial_label_rate=0.15, partially_labeled=True)
    print(train_idx)
    print(test_idx)
    print(label_idx)
    print(unlabel_idx)
    pass
