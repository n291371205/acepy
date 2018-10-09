"""
Class to store necessary information of
an active learning experiment for analysis
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

import copy
import os
import pickle
import warnings

from sklearn.svm import SVC
from sklearn.utils import check_array
from sklearn.utils.multiclass import unique_labels, type_of_target

import experiment_saver
from data_process.al_split import split, split_multi_label
from experiment_saver.state_io import StateIO
from oracle.oracle import OracleQueryMultiLabel, Oracle
from query_strategy.query_strategy import QueryInstanceUncertainty, QueryRandom
from utils.ace_warnings import *
from utils.ace_warnings import FunctionWarning
from utils.al_collections import IndexCollection, MultiLabelIndexCollection
from utils.knowledge_db import MatrixKnowledgeDB
from utils.query_type import check_query_type


class AlExperiment:
    """
    Class to store necessary information of
    an active learning experiment for analysis
    """

    def __init__(self, method_name, remark=None):
        self.method_name = method_name
        self.remark = remark
        self.__results = list()

    def add_fold(self, src):
        """
        Add one fold of active learning experiment.

        Parameters
        ----------
        src: object or str
            StateIO object or path to the intermediate results file.
        """
        if isinstance(src, experiment_saver.state_io.StateIO):
            if not src.check_batch_size():
                warnings.warn('Checking validity fails, different batch size is found.', category=ValidityWarning)
            self.__add_fold_by_object(src)
        elif isinstance(src, str):
            self.__add_fold_from_file(src)
        else:
            raise TypeError('StateIO object or str is expected, but received:%s' % str(type(src)),
                            category=UnexpectedParameterWarning)

    def add_folds(self, folds):
        """Add multiple folds.

        Parameters
        ----------
        folds: list
            The list contains n StateIO objects.
        """
        for item in folds:
            self.add_fold(item)

    def __add_fold_by_object(self, result):
        """
        Add one fold of active learning experiment

        Parameters
        ----------
        result: utils.StateIO
            object stored a complete fold of active learning experiment
        """
        self.__results.append(copy.deepcopy(result))

    def __add_fold_from_file(self, path):
        """
        Add one fold of active learning experiment from file

        Parameters
        ----------
        path: str
            path of result file.
        """
        f = open(os.path.abspath(path), 'rb')
        result = pickle.load(f)
        f.close()
        assert (isinstance(result, experiment_saver.StateIO))
        if not result.check_batch_size():
            warnings.warn('Checking validity fails, different batch size is found.',
                          category=ValidityWarning)
        self.__results.append(copy.deepcopy(result))

    def check_experiments(self):
        """
        Check results stored in this object that:
        1. whether all folds have the same length. If not, calc the shortest one
        2. whether the batch size is the same.
        3. calculate additional information.

        Returns
        -------

        """
        if self.check_batch_size and self.check_length:
            return True
        else:
            return False

    def check_batch_size(self):
        """
        if all queries have the same batch size.

        Returns
        -------

        """
        bs = set()
        for item in self.__results:
            if not item.check_batch_size():
                return False
            else:
                bs.add(item.batch_size)
        if len(bs) == 1:
            return True
        else:
            return False

    def check_length(self):
        """
        if all folds have the same numbers of query.

        Returns
        -------
        bool
        """
        ls = set()
        for item in self.__results:
            ls.add(len(item))
        if len(ls) == 1:
            return True
        else:
            return False

    def get_batch_size(self):
        """

        Returns
        -------
        -1 if not the same batch size
        """
        if self.check_batch_size:
            return self.__results[0].batch_size
        else:
            return -1

    def get_length(self):
        ls = list()
        for item in self.__results:
            ls.append(len(item))
        return min(ls)

    def __len__(self):
        return len(self.__results)

    def __getitem__(self, item):
        return self.__results.__getitem__(item)

    def __contains__(self, other):
        return other in self.__results

    def __iter__(self):
        return iter(self.__results)

    def __repr__(self):
        return


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
    y: array-like
        labels of given data [n_samples, n_labels] or [n_samples]

    X: array-like, optional
        data matrix with [n_samples, n_features]

    instance_indexes: array-like, optional (default=None)
        indexes of instances, it should be one-to-one correspondence of
        X, if not provided, it will be generated automatically for each
        xi started from 0.
        It also can be a list contains names of instances, used for image datasets.
        The split will only depend on the indexes if X is not provided.

    model: object, optional (default=SVC)
        The baseline model for evaluating the active learning strategy.

    query_type: str, optional (default='AllLabels')
        active learning settings. It will determine how to split data.

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

    performance: str, optional (default='Accuracy')
        The performance index of the experiment.

    saving_path: str, optional (default='.')
        path to save current settings. if None is provided, then it will not
        save the path


    Attributes
    ----------


    Examples
    ----------

    """

    def __init__(self, y, X=None, instance_indexes=None, model=SVC(),
                 query_type='AllLabels', test_ratio=0.3, initial_label_rate=0.05,
                 split_count=10, all_class=True, partially_labeled=False, performance='Accuracy', saving_path='.'):
        if X is None and y is None and instance_indexes is None:
            raise Exception("Must provide one of X, y or instance_indexes.")
        self._index_len = None

        # check and record parameters
        self._y = check_array(y, ensure_2d=False, dtype=None)
        self._index_len = len(self._y)
        self._label_num = len(unique_labels(self._y))
        ytype = type_of_target(y)
        if ytype in ['multilabel-indicator', 'multilabel-sequences']:
            self._target_type = 'multilabel'
        else:
            self._target_type = ytype
        if X is None:
            warnings.warn("Instances matrix or acceptable model is not given, The initial point can not "
                          "be calculated automatically.", category=FunctionWarning)
            self._instance_flag = False
        else:
            self._instance_flag = True
            self._X = check_array(X, accept_sparse='csr', ensure_2d=True, order='C')
            n_samples = self._X.shape[0]
            if n_samples != self._index_len:
                raise ValueError("Different length of instances and labels found.")
            else:
                self._index_len = n_samples
        if instance_indexes is None:
            self._indexes = [i for i in range(self._index_len)]
        else:
            if len(instance_indexes) != self._index_len:
                raise ValueError("Length of given instance_indexes do not accord the data set.")
            self._indexes = copy.copy(instance_indexes)
        # check if the model is acceptable
        if not (hasattr(model, 'fit') and hasattr(model, 'predict')):
            warnings.warn("Instances matrix or acceptable model is not given, The initial point can not "
                            "be calculated automatically.", category=FunctionWarning)
            self._model_flag = False
        else:
            self._model_flag = True
            self.model = model
        # still in progress
        if check_query_type(query_type):
            self.query_type = query_type
        else:
            raise NotImplementedError("Query type %s is not implemented." % type)
        self.saving_path = saving_path
        self.split_count = split_count
        self.test_ratio = test_ratio
        self.initial_label_rate = initial_label_rate
        # should support other query types in the future
        if self._target_type != 'multilabel':
            self.train_idx, self.test_idx, self.unlabel_idx, self.label_idx = split(
                X=self._X if self._instance_flag else None,
                y=self._y if self._label_flag else None,
                query_type=self.query_type, test_ratio=self.test_ratio,
                initial_label_rate=self.initial_label_rate,
                split_count=self.split_count,
                instance_indexes=self._indexes,
                all_class=all_class)
        else:
            self.train_idx, self.test_idx, self.unlabel_idx, self.label_idx = split_multi_label(
                y=self._y,
                test_ratio=self.test_ratio,
                initial_label_rate=self.initial_label_rate,
                split_count=self.split_count,
                all_class=all_class,
                partially_labeled=partially_labeled
            )
        self.save_settings(saving_path)

    def get_split(self, round=None):
        if round is not None:
            assert (0 <= round < self.split_count)
            if self._target_type != 'multilabel':
                return copy.copy(self.train_idx[round]), copy.copy(self.test_idx[round]), IndexCollection(
                    self.unlabel_idx[round]), IndexCollection(self.label_idx[round])
            else:
                return copy.copy(self.train_idx[round]), copy.copy(self.test_idx[round]), MultiLabelIndexCollection(
                    self.unlabel_idx[round], self._label_num), MultiLabelIndexCollection(self.label_idx[round],
                                                                                         self._label_num)
        else:
            return copy.deepcopy(self.train_idx), copy.deepcopy(self.test_idx), self.unlabel_idx, self.label_idx

    def get_clean_oracle(self):
        ytype = type_of_target(self._y)
        if ytype in ['multilabel-indicator', 'multilabel-sequences']:
            return OracleQueryMultiLabel(self._y)
        elif ytype in ['binary', 'multiclass']:
            return Oracle(self._y)

    def get_saver(self, round):
        assert (0 <= round < self.split_count)
        train_id, test_id, Ucollection, Lcollection = self.get_split(round)
        if self._instance_flag and self._model_flag:
            # performance is not implemented yet.
            # return StateIO(round, train_id, test_id, Ucollection, Lcollection, initial_point=accuracy)
            return StateIO(round, train_id, test_id, Ucollection, Lcollection)
        else:
            return StateIO(round, train_id, test_id, Ucollection, Lcollection)

    def get_knowledge_db(self, round):
        assert (0 <= round < self.split_count)
        train_id, test_id, Ucollection, Lcollection = self.get_split(round)
        if self._target_type != 'multilabel':
            return MatrixKnowledgeDB(labels=self._y[Lcollection.index], examples=self._X[Lcollection.index, :],
                                     indexes=Lcollection.index)
        else:
            raise NotImplementedError("Knowledge DB for multi label is not implemented yet.")

    def uncertainty_selection(self, measure='entropy', scenario='pool'):
        return QueryInstanceUncertainty(X=self._X, y=self._y, measure=measure, scenario=scenario)

    def random_selection(self):
        return QueryRandom()

    def get_model(self):
        return self.model

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