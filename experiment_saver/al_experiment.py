"""
Class to encapsulate various tools
and implement the main loop of active learning.

To run the experiment with only one class,
we have to impose some restrictions to make
sure the robustness of the code.
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

import copy
import os
import pickle
import inspect

from sklearn.svm import SVC
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import unique_labels, type_of_target
from sklearn.linear_model import LogisticRegression

from data_process.al_split import split, split_multi_label, split_features
from experiment_saver.state_io import StateIO
from oracle.oracle import OracleQueryMultiLabel, Oracle, OracleQueryFeatures
from experiment_saver.state import State
from query_strategy.query_strategy import QueryInstanceUncertainty, QueryRandom
from utils.ace_warnings import *
from utils.al_collections import IndexCollection, MultiLabelIndexCollection, FeatureIndexCollection
from utils.knowledge_db import MatrixKnowledgeDB, ElementKnowledgeDB
from utils.query_type import check_query_type
from utils.tools import get_labelmatrix_in_multilabel
from utils.stopping_criteria import StoppingCriteria
from analyser.experiment_analyser import ExperimentAnalyser
from utils.multi_thread import aceThreading


class ToolBox:
    """Tool box is a tool class which initializes the active learning
    elements according to the setting in order to reduce the error and improve
    the usability.

    In initializing, necessary information to initialize various tool classes
    must be given. You can set the split setting in initializing or generate a new
    split by ToolBox.split.

    Note that, using ToolBox to initialize other tools is optional, you may use
    each modules independently.

    Parameters
    ----------
    y: array-like
        _labels of given data [n_samples, n_labels] or [n_samples]

    X: array-like, optional (default=None)
        data matrix with [n_samples, n_features].

    instance_indexes: array-like, optional (default=None)
        indexes of instances, it should be one-to-one correspondence of
        X, if not provided, it will be generated automatically for each
        x_i started from 0.
        It also can be a list contains names of instances, used for image datasets.
        The split will only depend on the indexes if X is not provided.

    query_type: str, optional (default='AllLabels')
        active learning settings. It will determine how to split data.
        should be one of ['AllLabels', 'Partlabels', 'Features']:

        AllLabels: query all _labels of an selected instance.
            Support scene: binary classification, multi-class classification, multi-label classification, regression

        Partlabels: query part of _labels of an instance.
            Support scene: multi-label classification

        Features: query part of features of an instance.
            Support scene: missing features

    saving_path: str, optional (default='.')
        path to save current settings. if None is provided, then it will not
        save the path

    train_idx: array-like, optional (default=None)
        index of training set, shape like [n_split_count, n_training_indexex]

    test_idx: array-like, optional (default=None)
        index of testing set, shape like [n_split_count, n_testing_indexex]

    label_idx: array-like, optional (default=None)
        index of labeling set, shape like [n_split_count, n_labeling_indexex]

    unlabel_idx: array-like, optional (default=None)
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexex]


    Attributes
    ----------


    Examples
    ----------

    """

    def __init__(self, y, X=None, instance_indexes=None,
                 query_type='AllLabels', saving_path='.', **kwargs):
        self._index_len = None
        # check and record parameters
        self._y = check_array(y, ensure_2d=False, dtype=None)
        ytype = type_of_target(y)
        if ytype in ['multilabel-indicator', 'multilabel-sequences']:
            self._target_type = 'multilabel'
        else:
            self._target_type = ytype
        self._index_len = len(self._y)
        self._label_space = unique_labels(self._y)
        self._label_num = len(self._label_space)

        self._instance_flag = False
        if X is not None:
            self._instance_flag = True
            self._X = check_array(X, accept_sparse='csr', ensure_2d=True, order='C')
            n_samples = self._X.shape[0]
            if n_samples != self._index_len:
                raise ValueError("Different length of instances and _labels found.")
            else:
                self._index_len = n_samples

        if instance_indexes is None:
            self._indexes = [i for i in range(self._index_len)]
        else:
            if len(instance_indexes) != self._index_len:
                raise ValueError("Length of given instance_indexes do not accord the data set.")
            self._indexes = copy.copy(instance_indexes)

        if check_query_type(query_type):
            self.query_type = query_type
            if self.query_type == 'Features' and not self._instance_flag:
                raise Exception("In feature querying, feature matrix must be given.")
        else:
            raise NotImplementedError("Query type %s is not implemented." % type)

        self._split = False
        train_idx = kwargs.pop('_train_idx', None)
        test_idx = kwargs.pop('_test_idx', None)
        label_idx = kwargs.pop('_label_idx', None)
        unlabel_idx = kwargs.pop('_unlabel_idx', None)
        if train_idx is not None and test_idx is not None and label_idx is not None and unlabel_idx is not None:
            if not (len(train_idx) == len(test_idx) == len(label_idx) == len(unlabel_idx)):
                raise ValueError("_train_idx, _test_idx, _label_idx, _unlabel_idx "
                                 "should have the same split count (length)")
            self._split = True
            self.train_idx = train_idx
            self.test_idx = test_idx
            self.label_idx = label_idx
            self.unlabel_idx = unlabel_idx
            self.split_count = len(train_idx)

        self._saving_path = saving_path
        if saving_path is not None:
            if not isinstance(self._saving_path, str):
                raise TypeError("A string is expected, but received: %s" % str(type(self._saving_path)))
            self.save(saving_path)

    def split_AL(self, test_ratio=0.3, initial_label_rate=0.05,
                 split_count=10, all_class=True):
        """split dataset for active learning experiment.
        The labeled set for multi-label setting is fully labeled.

        Parameters
        ----------
        test_ratio: float, optional (default=0.3)
            ratio of test set

        initial_label_rate: float, optional (default=0.05)
            ratio of initial label set or the existed features (missing rate = 1-initial_label_rate)
            e.g. initial_labelset*(1-test_ratio)*n_samples

        split_count: int, optional (default=10)
            random split data _split_count times

        all_class: bool, optional (default=True)
            whether each split will contain at least one instance for each class.
            If False, a totally random split will be performed.

        Returns
        -------
        train_idx: array-like
            index of training set, shape like [n_split_count, n_training_indexex]

        test_idx: array-like
            index of testing set, shape like [n_split_count, n_testing_indexex]

        label_idx: array-like
            index of labeling set, shape like [n_split_count, n_labeling_indexex]

        unlabel_idx: array-like
            index of unlabeling set, shape like [n_split_count, n_unlabeling_indexex]

        """
        # should support other query types in the future
        self.split_count = split_count
        if self._target_type != 'Features':
            if self._target_type != 'multilabel':
                self.train_idx, self.test_idx, self.label_idx, self.unlabel_idx = split(
                    X=self._X if self._instance_flag else None,
                    y=self._y,
                    query_type=self.query_type, test_ratio=test_ratio,
                    initial_label_rate=initial_label_rate,
                    split_count=split_count,
                    instance_indexes=self._indexes,
                    all_class=all_class)
            else:
                self.train_idx, self.test_idx, self.label_idx, self.unlabel_idx = split_multi_label(
                    y=self._y,
                    label_shape=self._y.shape,
                    test_ratio=test_ratio,
                    initial_label_rate=initial_label_rate,
                    split_count=split_count,
                    all_class=all_class,
                    )
        else:
            self.train_idx, self.test_idx, self.label_idx, self.unlabel_idx = split_features(
                feature_matrix=self._X,
                test_ratio=test_ratio,
                missing_rate=1 - initial_label_rate,
                split_count=split_count,
                all_features=all_class
            )
        self._split = True
        return self.train_idx, self.test_idx, self.label_idx, self.unlabel_idx

    def get_split(self, round=None):
        if not self._split:
            raise Exception("The split setting is unknown, use split_AL() first.")
        if round is not None:
            assert (0 <= round < self.split_count)
            if self.query_type == 'Features':
                return copy.copy(self.train_idx[round]), copy.copy(self.test_idx[round]), FeatureIndexCollection(
                    self.label_idx[round], self._X.shape[1]), FeatureIndexCollection(self.unlabel_idx[round],
                                                                                     self._X.shape[1])
            else:
                if self._target_type == 'multilabel':
                    return copy.copy(self.train_idx[round]), copy.copy(self.test_idx[round]), MultiLabelIndexCollection(
                        self.label_idx[round], self._label_num), MultiLabelIndexCollection(self.unlabel_idx[round],
                                                                                           self._label_num)
                else:
                    return copy.copy(self.train_idx[round]), copy.copy(self.test_idx[round]), IndexCollection(
                        self.label_idx[round]), IndexCollection(self.unlabel_idx[round])
        else:
            return copy.deepcopy(self.train_idx), copy.deepcopy(self.test_idx), \
                   copy.deepcopy(self.label_idx), copy.deepcopy(self.unlabel_idx)

    def clean_oracle(self):
        if self.query_type == 'Features':
            return OracleQueryFeatures(feature_mat=self._X)
        elif self.query_type == 'AllLabels':
            if self._target_type == 'multilabel':
                return OracleQueryMultiLabel(self._y)
            else:
                return Oracle(self._y)

    def StateIO(self, round):
        assert (0 <= round < self.split_count)
        train_id, test_id, Lcollection, Ucollection = self.get_split(round)
        return StateIO(round, train_id, test_id, Lcollection, Ucollection)

    def __knowledge_db(self, round):
        assert (0 <= round < self.split_count)
        train_id, test_id, Ucollection, Lcollection = self.get_split(round)
        if self.query_type == 'AllLabels':
            return MatrixKnowledgeDB(labels=self._y[Lcollection.index],
                                     examples=self._X[Lcollection.index, :] if self._instance_flag else None,
                                     indexes=Lcollection.index)
        else:
            raise NotImplemented("other query types for knowledge DB is not implemented yet.")

    def query_strategy(self, strategy_name="random"):
        """Return the query strategy object.

        Parameters
        ----------
        strategy_name: str, optional (default='random')

        Returns
        -------

        """
        if self.query_type != "AllLabels":
            raise NotImplemented("Query strategy for other query types is not implemented yet.")
        pass

    def get_multilabel_matrix_by_index(self, index, missing_value=0):
        """Index multilabel matrix by index. It can have missing value (query example-label pairs),
        The unknown elements will be set to missing_value

        Parameters
        ----------
        index
        missing_value

        Returns
        -------

        """
        if self._target_type != "multilabel":
            raise Exception("This function is only available in multi label setting.")
        return get_labelmatrix_in_multilabel(index=index, data_matrix=self._y, unknown_element=missing_value)

    def default_model(self):
        return SVC(probability=True)

    def stopping_criterion(self, stopping_criteria=None, value=None):
        """Return example stopping criterion.

        Parameters
        __________
        stopping_criteria: str, optional (default=None)
            stopping criteria, must be one of: [None, 'num_of_queries', 'cost_limit', 'percent_of_unlabel', 'time_limit']

            None: stop when no unlabeled samples available
            'num_of_queries': stop when preset number of quiries is reached
            'cost_limit': stop when cost reaches the limit.
            'percent_of_unlabel': stop when specific percentage of unlabeled data pool is labeled.
            'time_limit': stop when CPU time reaches the limit.
        """
        return StoppingCriteria(stopping_criteria=stopping_criteria, value=value)

    def experiment_analyser(self):
        """Return ExperimentAnalyser object"""
        return ExperimentAnalyser()

    def aceThreading(self, target_function=None, max_thread=None, refresh_interval=1, saving_path='.'):
        """Return the multithreading tool class

        Parameters
        __________
        __max_thread: int, optional (default=None)
            The max __threads for running at the same time. If not provided, it will run all rounds simultaneously.

        _refresh_interval: float, optional (default=1.0)
            how many seconds to refresh the current state output, default is 1.0.

        _saving_path: str, optional (default='.')
            the path to save the result files.
        """
        if not self._instance_flag:
            raise Exception("instance matrix is necessary for initializing aceThreading object.")
        return aceThreading(examples=self._X, labels=self._y,
                            train_idx=self.train_idx, test_idx=self.test_idx,
                            label_index=self.label_idx,
                            unlabel_index=self.unlabel_idx,
                            refresh_interval=refresh_interval,
                            max_thread=max_thread,
                            saving_path=saving_path,
                            target_func=target_function)

    def save(self):
        """Save the experiment settings to file for auditting or loading for other methods."""
        if self._saving_path is None:
            return
        saving_path = os.path.abspath(self._saving_path)
        if os.path.isdir(saving_path):
            f = open(os.path.join(saving_path, 'al_settings.pkl'), 'wb')
        else:
            f = open(os.path.abspath(saving_path), 'wb')
        pickle.dump(self, f)
        f.close()

    def IndexCollection(self, array=None):
        """Return an IndexCollection object initialized with array"""
        return IndexCollection(array)

    @classmethod
    def load(cls, path):
        """Loading ExperimentSetting object from path.

        Parameters
        ----------
        path: str
            Path to a specific file, not a dir.

        Returns
        -------
        setting: ToolBox
            Object of ExperimentSetting.
        """
        if not isinstance(path, str):
            raise TypeError("A string is expected, but received: %s" % str(type(path)))
        import pickle
        f = open(os.path.abspath(path), 'rb')
        setting_from_file = pickle.load(f)
        f.close()
        return setting_from_file


class AlExperiment:
    """AlExperiment is a  class to encapsulate various tools
    and implement the main loop of active learning.

    Only support the most commonly used scenario: query label of an instance

    To run the experiment with only one class,
    we have to impose some restrictions to make
    sure the robustness of the code:
    1. Your model object should accord scikit-learn api
    2. If a custom query strategy is given, you should implement
        the BaseQueryStrategy api. Additional parameters should be static.
    3. The data split should be given if you are comparing multiple methods.
        You may also generate new split with split_AL()


    Parameters
    ----------
    X,y : array
        The data matrix

    model: object
        An model object which accord the scikit-learn api

    performance_metric: str, optional (default='accuracy')
        The performance metric

    stopping_criteria: str, optional (default=None)
        stopping criteria, must be one of: [None, 'num_of_queries', 'cost_limit', 'percent_of_unlabel', 'time_limit']

        None: stop when no unlabeled samples available
        'num_of_queries': stop when preset number of quiries is reached
        'cost_limit': stop when cost reaches the limit.
        'percent_of_unlabel': stop when specific percentage of unlabeled data pool is labeled.
        'time_limit': stop when CPU time reaches the limit.

    batch_size: int, optional (default=1)
        batch size of AL

    train_idx: array-like, optional (default=None)
        index of training set, shape like [n_split_count, n_training_indexex]

    test_idx: array-like, optional (default=None)
        index of testing set, shape like [n_split_count, n_testing_indexex]

    label_idx: array-like, optional (default=None)
        index of labeling set, shape like [n_split_count, n_labeling_indexex]

    unlabel_idx: array-like, optional (default=None)
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexex]
    """

    def __init__(self, X, y, model=SVC(), performance_metric='accuracy',
                 stopping_criteria=None, stopping_value=None, batch_size=1, **kwargs):
        self.__custom_strategy_flag = False
        self._split = False
        self._split_count = 0

        self._X, self._y = check_X_y(X, y, accept_sparse='csc', multi_output=True)
        self._model = model
        self._performance_metric = performance_metric

        # set split in the initial
        train_idx = kwargs.pop('_train_idx', None)
        test_idx = kwargs.pop('_test_idx', None)
        label_idx = kwargs.pop('_label_idx', None)
        unlabel_idx = kwargs.pop('_unlabel_idx', None)
        if train_idx is not None and test_idx is not None and label_idx is not None and unlabel_idx is not None:
            if not (len(train_idx) == len(test_idx) == len(label_idx) == len(unlabel_idx)):
                raise ValueError("_train_idx, _test_idx, _label_idx, _unlabel_idx "
                                 "should have the same split count (length)")
            self._split = True
            self._train_idx = train_idx
            self._test_idx = test_idx
            self._label_idx = label_idx
            self._unlabel_idx = unlabel_idx
            self._split_count = len(train_idx)

        self._stopping_criterion = StoppingCriteria(stopping_criteria, stopping_value)
        self._batch_size = 1

    def set_query_strategy(self, strategy="Uncertainty", **kwargs):
        """

        Parameters
        ----------
        strategy: {str, callable}, optional (default='Uncertainty')
            The query strategy function.
            Giving str to use a pre-defined strategy
            Giving callable to use a user-defined strategy.

        kwargs: dict, optional
            The args used in user-defined strategy.
            Note that, each parameters should be static.
            The parameters will be fed to the callable object automatically.
        """
        if callable(strategy):
            self.__custom_strategy_flag = True
            self._query_function = strategy
            self.__custom_func_arg = kwargs
            return
        if strategy not in []:
            raise NotImplementedError('Strategy %s is not implemented. Specify a valid '
                                      'method name or privide a callable object.', str(strategy))
        pass

    def set_data_split(self, train_idx, test_idx, label_idx, unlabel_idx):
        """set the data split indexes.

        Parameters
        ----------
        train_idx: array-like, optional (default=None)
            index of training set, shape like [n_split_count, n_training_indexex]

        test_idx: array-like, optional (default=None)
            index of testing set, shape like [n_split_count, n_testing_indexex]

        label_idx: array-like, optional (default=None)
            index of labeling set, shape like [n_split_count, n_labeling_indexex]

        unlabel_idx: array-like, optional (default=None)
            index of unlabeling set, shape like [n_split_count, n_unlabeling_indexex]

        Returns
        -------

        """
        if not (len(train_idx) == len(test_idx) == len(label_idx) == len(unlabel_idx)):
            raise ValueError("_train_idx, _test_idx, _label_idx, _unlabel_idx "
                             "should have the same split count (length)")
        self._split = True
        self._train_idx = train_idx
        self._test_idx = test_idx
        self._label_idx = label_idx
        self._unlabel_idx = unlabel_idx
        self._split_count = len(train_idx)

    def split_AL(self, test_ratio=0.3, initial_label_rate=0.05,
                 split_count=10, all_class=True):
        """split dataset for active learning experiment.

        Parameters
        ----------
        test_ratio: float, optional (default=0.3)
            ratio of test set

        initial_label_rate: float, optional (default=0.05)
            ratio of initial label set or the existed features (missing rate = 1-initial_label_rate)
            e.g. initial_labelset*(1-test_ratio)*n_samples

        split_count: int, optional (default=10)
            random split data _split_count times

        all_class: bool, optional (default=True)
            whether each split will contain at least one instance for each class.
            If False, a totally random split will be performed.

        Returns
        -------
        _train_idx: array-like
            index of training set, shape like [n_split_count, n_training_indexex]

        _test_idx: array-like
            index of testing set, shape like [n_split_count, n_testing_indexex]

        label_idx: array-like
            index of labeling set, shape like [n_split_count, n_labeling_indexex]

        unlabel_idx: array-like
            index of unlabeling set, shape like [n_split_count, n_unlabeling_indexex]

        """
        self._split_count = split_count
        self._split = True
        self._train_idx, self._test_idx, self._label_idx, self._unlabel_idx = split(
            X=self._X,
            y=self._y,
            test_ratio=test_ratio,
            initial_label_rate=initial_label_rate,
            split_count=split_count,
            all_class=all_class)
        return self._train_idx, self._test_idx, self._label_idx, self._unlabel_idx

    def start_query(self, multi_thread=True):
        """Start the active learning main loop
        If using implemented query strategy, It will run in multi-thread default"""
        if not self._split:
            raise Exception("Data split is unknown. Use set_data_split() to set an existed split, "
                            "or use split_AL() to generate new split.")

        if multi_thread:
            aceThreading()
        else:
            pass

    def __al_main_loop(self, round, train_id, test_id, Lcollection, Ucollection,
                       saver, examples, labels, global_parameters):
        self._model.fit(X=self._X[Lcollection.index, :], y=self.y[Lcollection.index])
        pred = self._model.predict(self._X[test_id, :])

        # performance calc
        accuracy = sum(pred == self._y[test_id]) / len(test_id)

        saver.set_initial_point(accuracy)
        while not self._stopping_criterion:
            if not self.__custom_strategy_flag:
                if 'model' in inspect.getfullargspec(self._query_function.select)[0]:
                    select_ind = self._query_function.select(Lcollection, Ucollection, batch_size=self._batch_size,
                                                             model=self._model)
                else:
                    select_ind = self._query_function.select(Lcollection, Ucollection, batch_size=self._batch_size)
            else:
                select_ind = self._query_function.select(Lcollection, Ucollection, batch_size=self._batch_size,
                                                         **self.__custom_func_arg)
            Lcollection.update(select_ind)
            Ucollection.difference_update(select_ind)
            # update model
            self._model.fit(X=self._X[Lcollection.index, :], y=self._y[Lcollection.index])
            pred = self._model.predict(self._X[test_id, :])

            # performance calc
            accuracy = sum(pred == self._y[test_id]) / len(test_id)

            # save intermediate results
            st = State(select_index=select_ind, performance=accuracy)
            saver.add_state(st)
            saver.save()
