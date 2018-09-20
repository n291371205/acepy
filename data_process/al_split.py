"""
Data Split
Split the original dataset into train/test label/unlabelset
Accept not only datamat, but also shape/list of instance name (for image datasets)
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

import numpy as np
import copy
import os

from utils.query_type import check_query_type
from utils.tools import randperm
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels, is_multilabel, check_classification_targets, type_of_target
from utils.ace_warnings import *
from utils.al_collections import IndexCollection
from oracle.oracle import Oracle, OracleQueryMultiLabel
from query_strategy.query_strategy import QueryInstanceUncertainty, QueryInstanceRandom
from experiment_saver.state_io import StateIO


class ExperimentSetting:
    """Experiment Manager is a tool class which initializes the active learning
    elements according to the setting in order to reduce the error and improve
    the usability.

    Elements of active learning experiments includes:
    1. Split data set and return the initialized container
    2. Oracle
    3. Intermediate results saver
    4. Classical query methods
    5. Stopping criteria

    Parameters
    ----------
    X: array-like, optional
        data matrix with [n_samples, n_features]

    y: array-like, optional
        labels of given data [n_samples, n_labels] or [n_samples]

    instance_indexes: array-like, optional (default=None)
        indexes of instances, it should be one-to-one correspondence of
        X, if not provided, it will be generated automatically for each
        xi started from 0.
        It also can be a list contains names of instances, used for image datasets.
        The split will only depend on the indexes if X is not provided.

    model: object, optional (default=None)
        The baseline model for evaluating the active learning strategy.

    query_type: str, optional (default=query_instance)
        active learning settings. It will determine how to split data.

    test_ratio: float, optional (default=0.3)
        ratio of test set

    initial_label_rate: float, optional (default=0.05)
        ratio of initial label set
        e.g. initial_labelset*(1-test_ratio)*n_samples

    split_count: int, optional (default=10)
        random split data split_count times

    performance: str, optional (default='Accuracy')
        The performance index of the experiment.

    saving_path: str, optional (default=None)
        path to save current settings. if None is provided, then it will not
        save the path


    Attributes
    ----------


    Examples
    ----------

    """

    def __init__(self, X=None, y=None, instance_indexes=None, model=None,
                 query_type=None, test_ratio=0.3, initial_label_rate=0.05,
                 split_count=10, performance='Accuracy', saving_path=None):
        if X is None and y is None and instance_indexes is None:
            raise ValueError("Must provide one of X, y or instance_indexes.")
        self._index_len = None

        # check and record parameters
        if y is None:
            warnings.warn("Label matrix is not given, Oracle class can not be initialized.",
                          category=FunctionWarning)
            self._label_flag = False
        else:
            self._label_flag = True
            self._y = check_array(y, ensure_2d=False, dtype=None)
            self._index_len = len(self._y)
        if X is None:
            warnings.warn("Instances matrix or acceptable model is not given, The initial point can not "
                          "be calculated automatically.", category=FunctionWarning)
            self._instance_flag = False
        else:
            self._instance_flag = True
            self._X = check_array(X, accept_sparse='csr', ensure_2d=True, order='C')
            n_samples = self._X.shape[0]
            if self._label_flag:
                if n_samples != self._index_len:
                    raise ValueError("Different length of instances and labels found.")
            else:
                self._index_len = n_samples
        if instance_indexes is None:
            self._indexes = [i for i in range(self._index_len)]
        else:
            if self._index_len is None:  # both X and y is not provided.
                self._index_len = len(self._indexes)
            else:  # one of X and y is provided
                if len(instance_indexes) != self._index_len:
                    raise ValueError("Length of given instance_indexes do not accord the data set.")
            self._indexes = copy.copy(instance_indexes)
        if model is None:
            warnings.warn("Instances matrix or acceptable model is not given, The initial point can not "
                          "be calculated automatically.", category=FunctionWarning)
            self._model_flag = False
        else:
            # check if the model is acceptable
            if not (hasattr(model, 'fit') and hasattr(model, 'predict')):
                warnings.warn("Instances matrix or acceptable model is not given, The initial point can not "
                              "be calculated automatically.", category=FunctionWarning)
                self._model_flag = False
            else:
                self._model_flag = True
        # still in progress
        if query_type is None:  # the most popular setting is used.
            self.query_type = 'AllLabels'
        else:
            if check_query_type(query_type):
                self.query_type = query_type
            else:
                raise NotImplemented("Query type %s is not implemented." % type)
        self.saving_path = saving_path
        self.split_count = split_count
        self.test_ratio = test_ratio
        self.initial_label_rate = initial_label_rate
        # should support other query types in the future
        self.train_idx, self.test_idx, self.unlabel_idx, self.label_idx = split(
            X=self._X if self._instance_flag else None,
            y=self._y if self._label_flag else None,
            query_type=self.query_type, test_ratio=self.test_ratio,
            initial_label_rate=self.initial_label_rate,
            split_count=self.split_count,
            instance_indexes=self._indexes)
        self.save_settings(saving_path)

    def get_split(self, round=None):
        if round is not None:
            assert (0 <= round < self.split_count)
            return copy.copy(self.train_idx[round]), copy.copy(self.test_idx[round]), IndexCollection(
                self.unlabel_idx[round]), IndexCollection(self.label_idx[round])
        else:
            return copy.deepcopy(self.train_idx), copy.deepcopy(self.test_idx), self.unlabel_idx, self.label_idx

    def get_clean_oracle(self):
        ytype =  type_of_target(self._y)
        if ytype in ['multilabel-indicator', 'multilabel-sequences']:
            return OracleQueryMultiLabel(self._y)
        elif ytype in ['binary', 'multiclass']:
            return Oracle(self._y)

    def get_saver(self, round):
        assert (0 <= round < self.split_count)
        train_id, test_id, Ucollection, Lcollection = self.get_split(round)
        if self._label_flag and self._instance_flag and self._model_flag:
            # performance is not implemented yet.
            # return StateIO(round, train_id, test_id, Ucollection, Lcollection, initial_point=accuracy)
            return StateIO(round, train_id, test_id, Ucollection, Lcollection)
        else:
            return StateIO(round, train_id, test_id, Ucollection, Lcollection)

    def uncertainty_selection(self, measure='entropy', scenario='pool'):
        return QueryInstanceUncertainty(X=self._X, y=self._y, measure=measure, scenario=scenario)

    def random_selection(self):
        return QueryInstanceRandom()

    def save_settings(self, saving_path):
        """Save the experiment settings to file for auditting or loading for other methods.

        Parameters
        ----------
        saving_path: str
            path to save the settings. If a dir is provided, it will generate a file called
            'al_settings.pkl' for saving.

        """
        if saving_path is None:
            return
        else:
            if not isinstance(saving_path, str):
                raise TypeError("A string is expected, but received: %s" % str(type(saving_path)))
        import pickle
        saving_path = os.path.abspath(saving_path)
        if os.path.isdir(saving_path):
            f = open(os.path.join(saving_path, 'al_settings.pkl'), 'wb')
        else:
            f = open(os.path.abspath(saving_path), 'wb')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load_settings(cls, path):
        """Loading ExperimentSetting object from path.

        Parameters
        ----------
        path: str
            Path to a specific file, not a dir.

        Returns
        -------
        setting: ExperimentSetting
            Object of ExperimentSetting.
        """
        if not isinstance(path, str):
            raise TypeError("A string is expected, but received: %s" % str(type(path)))
        import pickle
        f = open(os.path.abspath(path), 'rb')
        setting_from_file = pickle.load(f)
        f.close()
        return setting_from_file


def split(X=None, y=None, instance_indexes=None, query_type=None, test_ratio=0.3, initial_label_rate=0.05,
          split_count=10, all_class=True, saving_path='.'):
    """Split given data according to the config

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
        raise ValueError("Must provide one of X, y or instance_indexes.")
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
            raise NotImplemented("Query type %s is not implemented." % type)
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
        if not all_class:
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
                raise ValueError("y must be provided when all_class flag is True.")
            y = check_array(y, ensure_2d=False, dtype=None)
            ytype = type_of_target(y)
            if ytype in ['multilabel-indicator', 'multilabel-sequences']:
                multi_label_flag = True
            elif ytype in ['binary', 'multiclass']:
                multi_label_flag = False
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
            if not multi_label_flag:
                train_idx.append(tp_train)
                test_idx.append(instance_indexes[rp[cutpoint:]])
                label_idx.append(tp_train[0:cutpointlabel])
                unlabel_idx.append(tp_train[cutpointlabel:])
            else:
                train_idx.append([(i,) for i in tp_train])
                test_idx.append([(i,) for i in instance_indexes[rp[cutpoint:]]])
                label_idx.append([(i,) for i in tp_train[0:cutpointlabel]])
                unlabel_idx.append([(i,) for i in tp_train[cutpointlabel:]])

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
    setting: ExperimentSetting
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
    # train_idx, test_idx, label_idx, unlabel_idx = split(X=np.random.random((10, 10)), y=np.random.randint(0, 10, 10),
    #                                                     config=QueryConfig())
    # print(train_idx)
    # print(test_idx)
    # print(label_idx)
    # print(unlabel_idx)
    pass
