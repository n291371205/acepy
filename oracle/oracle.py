"""
Pre-defined oracle class
Implement classical situation
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

import collections

import numpy as np
import os
from sklearn.utils.validation import check_array

import utils.base


class Oracle(utils.base.BaseVirtualOracle):
    """Oracle in active learning whose role is to label the given query.

    This class implements basic definition of oracle used in experiment.
    Oracle can provide information given both instance or index. The returned
    information is depending on specific scenario.

    Parameters
    ----------
    labels:  array-like
        label matrix. shape like [n_samples, n_classes] or [n_samples]

    examples: array-like, optional (default=None)
        array of examples, initialize with this parameter to make
        "query by instance" available. Shape like [n_samples, n_features]

    indexes: array-like, optional (default=None)
        index of examples, if not provided, it will be generated
        automatically started from 0.

    cost: array_like, optional (default=None)
        costs of each queried instance, should have the same length
        and is one-to-one correspondence of y, default is 1.
    """

    def __init__(self, labels, examples=None, indexes=None, cost=None):
        labels = check_array(labels, ensure_2d=False, dtype=None)
        if isinstance(labels[0], np.generic):
            self._label_type = type(np.asscalar(labels[0]))
        else:
            self._label_type = type(labels[0])
        if len(np.shape(labels)) == 2:
            self._label_dim = 2
        else:
            self._label_dim = 1

        # check parameters
        if indexes is None:
            self._indexes = [i for i in range(len(labels))]
        else:
            self._indexes = indexes
        if cost is None:
            self._cost_flag = False
        else:
            if not (len(cost) == len(labels)):
                raise ValueError("Different length of labels and cost detected.")
            self._cost_flag = True
        if examples is None:
            self._instance_flag = False
        else:
            if not (len(examples) == len(labels)):
                raise ValueError("Different length of labels and instances detected.")
            self._instance_flag = True
        if not (len(self._indexes) == len(labels)):
            raise ValueError("Different length of labels and indexes detected.")

        # several _indexes construct
        if self._instance_flag:
            self._exa2ind = dict(zip(examples, self._indexes))
        if self._cost_flag:
            self._ind2all = dict(zip(self._indexes, zip(labels, cost)))
        else:
            self._ind2all = dict(zip(self._indexes, labels))

    @property
    def index_keys(self):
        return self._ind2all.keys()

    @property
    def example_keys(self):
        if self._instance_flag:
            return self._exa2ind.keys()
        else:
            return None

    def _add_one_entry(self, label, index, example=None, cost=None):
        """Adding entry to the oracle.

        Add new entry to the oracle for future querying where index is the queried elements,
        label is the returned data. Index should not be in the oracle. Example and cost should
        accord with the initializing (If exists in initializing, then must be provided.)

        The data provided must have the same type with the initializing data. If different, a
        transform is attempted.

        Parameters
        ----------
        label:  array-like
            label matrix.

        index: object
            index of examples, should not in the oracle.

        example: array-like, optional (default=None)
            array of examples, initialize with this parameter to turn
            "query by instance" on.

        cost: array_like, optional (default=None)
            costs of each queried instance, should have the same length
            and is one-to-one correspondence of y, default is 1.
        """
        if isinstance(label, np.generic):
            label = np.asscalar(label)
        if isinstance(label, list):
            label = np.array(label)
        if not isinstance(label, self._label_type):
            raise TypeError("Different types of labels found when adding entries: %s is expected but received: %s" %
                            (str(self._label_type), str(type(label))))
        if self._instance_flag:
            if example is None:
                raise ValueError("This oracle has the instance information,"
                                 "must provide example parameter when adding entry")
            self._exa2ind[example] = index
        if self._cost_flag:
            if cost is None:
                raise ValueError("This oracle has the cost information,"
                                 "must provide cost parameter when adding entry")
            self._ind2all[index] = (label, cost)
        else:
            self._ind2all[index] = label

    def add_knowledge(self, labels, indexes, examples=None, cost=None):
        """Adding entries to the oracle.

        Add new entries to the oracle for future querying where _indexes are the queried elements,
        labels are the returned data. Indexes should not be in the oracle. Examples and cost should
        accord with the initializing (If exists in initializing, then must be provided.)

        Parameters
        ----------
        labels: array-like or object
            label matrix.

        indexes: array-like or object
            index of examples, should not in the oracle.
            if update multiple entries to the oracle, a list or np.ndarray type is expected,
            otherwise, it will be cheated as only one entry.

        examples: array-like, optional (default=None)
            array of examples, initialize with this parameter to turn
            "query by instance" on.

        cost: array_like, optional (default=None)
            costs of each queried instance, should have the same length
            and is one-to-one correspondence of y, default is 1.
        """
        if not isinstance(indexes, (list, np.ndarray)):
            self._add_one_entry(labels, indexes, examples, cost)
        else:
            assert (len(indexes) == len(labels))
            for i in range(len(labels)):
                self._add_one_entry(labels[i], indexes[i], example=examples[i] if examples is not None else None,
                                    cost=cost[i] if cost is not None else None)

    def query_by_index(self, indexes):
        """Query function

        Parameters
        ----------
        indexes: list or int
            index to query, if only one index to query (batch_size = 1),
            an int is expected.

        Returns
        -------
        sup_info: list
            supervised information of queried index.

        costs: list
            corresponding costs produced by query.
        """
        if not isinstance(indexes, (list, np.ndarray)):
            indexes = [indexes]
        sup_info = []
        costs = []
        for k in indexes:
            if k in self._ind2all.keys():
                if self._cost_flag:
                    sup_info.append(self._ind2all[k][0])
                    costs.append(self._ind2all[k][1])
                else:
                    sup_info.append(self._ind2all[k])
                    costs.append(1)
            else:
                self._do_missing(k)
        return sup_info, costs

    def query_by_example(self, queried_examples):
        """Query function, query information giving an instance.
        Note that, this function only available if initializes with
         data matrix.

        Parameters
        ----------
        queried_examples: array_like
            [n_samples, n_features]

        Returns
        -------
        sup_info: list
            supervised information of queried instance.

        costs: list
            corresponding costs produced by query.
        """
        if not self._instance_flag:
            raise ValueError("This oracle do not have the instance information, query_by_instance is not supported")
        if not isinstance(queried_examples, (list, np.ndarray)):
            raise TypeError("An list or numpy.ndarray is expected, but received:%s" % str(type(queried_examples)))
        if len(np.shape(queried_examples)) == 1:
            queried_examples = [queried_examples]
        q_id = []
        for k in queried_examples:
            if k in self._exa2ind.keys():
                q_id.append(self._exa2ind[k])
            else:
                self._do_missing(k, 'instance pool')
        return self.query_by_index(q_id)

    def _do_missing(self, key, dict_name='index pool'):
        """

        Parameters
        ----------
        key

        Returns
        -------

        """
        raise KeyError("%s is not in the " + dict_name + " of this oracle" % str(key))

    def __repr__(self):
        return str(self.supervised_pool)

    def save_oracle(self, saving_path):
        """Save the oracle to file.

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
            f = open(os.path.join(saving_path, 'oracle.pkl'), 'wb')
        else:
            f = open(os.path.abspath(saving_path), 'wb')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load_oracle(cls, path):
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


class OracleQueryInstance(Oracle):
    """Oracle to label all labels of an instance.
    """


class OracleQueryMultiLabel(Oracle):
    """Oracle to label part of labels of instance in multi-label setting.

    When initializing, a 2D array of labels and cost of EACH label should be given.

    Parameters
    ----------
    labels:  array-like
        label matrix. Shape like [n_samples, n_classes]

    examples: array-like, optional (default=None)
        array of examples, initialize with this parameter to make
        "query by instance" available. Shape like [n_samples, n_features]

    indexes: array-like, optional (default=None)
        index of examples, if not provided, it will be generated
        automatically started from 0.

    cost: array_like, optional (default=None)
        cost of each queried instance, should be one-to-one correspondence of each label,
        default is all 1. Shape like [n_samples, n_classes]

    """

    def __init__(self, labels, examples=None, indexes=None, cost=None):
        labels = check_array(labels, ensure_2d=True, dtype=None)
        self._label_shape = np.shape(labels)
        super().__init__(labels, examples, indexes, cost)
        if self._cost_flag:
            if np.shape(cost) != np.shape(labels):
                raise ValueError("Different shapes of cost and labels found. Cost of each queried"
                                 " instance should be one-to-one correspondence of each label.\n"
                                 "Shape of labels:%s\nShape of cost:%s" % (str(np.shape(labels)), str(np.shape(cost))))

    def _add_one_entry(self, label, index, example=None, cost=None):
        """Adding entry to the oracle.

        Add new entry to the oracle for future querying where index is the queried elements,
        label is the returned data. Index should not be in the oracle. Example and cost should
        accord with the initializing (If exists in initializing, then must be provided.)

        Parameters
        ----------
        label:  array-like
            label matrix.

        index: int
            index of examples, should not in the oracle.

        example: array-like, optional (default=None)
            array of examples, initialize with this parameter to turn
            "query by instance" on.

        cost: array_like, optional (default=None)
            costs of each queried instance, should have the same length
            and is one-to-one correspondence of y, default is 1.
        """
        if len(label) != self._label_shape[1]:
            raise TypeError("Different dimension of labels found when adding entries: %s is expected but received: %s" %
                            (str(self._label_shape[1]), str(len(label))))
        if self._instance_flag:
            if example is None:
                raise ValueError("This oracle has the instance information,"
                                 "must provide example parameter when adding entry")
            self._exa2ind[example] = index
        if self._cost_flag:
            if cost is None:
                raise ValueError("This oracle has the cost information,"
                                 "must provide cost parameter when adding entry")
            if len(cost) != self._label_shape[1]:
                raise TypeError(
                    "Different dimension of cost found when adding entries: %s is expected but received: %s" %
                    (str(self._label_shape[1]), str(len(cost))))
            self._ind2all[index] = (np.array(label), cost)
        else:
            self._ind2all[index] = np.array(label)

    def query_by_index(self, indexes):
        """Query function in multi-label setting

        In multi-label setting, a query index is a tuple.
        A single index should only have 1 element (example_index, ) to query all labels or
        2 elements (example_index, [label_indexes]) to query specific labels.
        A list of index can be provided.

        Parameters
        ----------
        indexes: list or tuple or int
            index to query, if only one index to query (batch_size = 1),a tuple or an int is expected.
            e.g., in 10 class classification setting, queried_index = (1, [3,4])
            means query the 2nd instance's 4th,5th labels.
            some legal single index examples:
            queried_index = (1, [3,4])
            queried_index = (1, [3])
            queried_index = (1, 3)
            queried_index = (1, (3))
            queried_index = (1, (3,4))
            queried_index = (1, )   # query all labels

            One or more indexes could be provided.

        Returns
        -------
        sup_info: list
            supervised information of queried index.

        costs: list
            corresponding costs produced by query.
        """
        # check validity of the given indexes
        if not isinstance(indexes, (list, np.ndarray)):
            indexes = [indexes]
        datatype = collections.Counter([type(i) for i in indexes])
        if len(datatype) != 1:
            raise TypeError("Different types found in the given indexes.")
        if not datatype.popitem()[0] == tuple:
            raise TypeError("Each index should be a tuple.")

        # prepare queried labels
        sup_info = []
        costs = []
        for k in indexes:
            # k is a tuple with 2 elements
            k_len = len(k)
            if k_len != 1 and k_len != 2:
                raise ValueError(
                    "A single index should only have 1 element (example_index, ) to query all labels or"
                    "2 elements (example_index, [label_indexes]) to query specific labels. But found %d in %s" %
                    (len(k), str(k)))
            example_ind = k[0]
            if k_len == 1:
                label_ind = [i for i in range(self._label_shape[1])]
            else:
                if isinstance(k[1], collections.Iterable):
                    label_ind = [i for i in k[1] if 0 <= i < self._label_shape[1]]
                else:
                    assert (0 <= k[1] < self._label_shape[1])
                    label_ind = [k[1]]

            # fetch data
            if example_ind in self._ind2all.keys():
                if self._cost_flag:
                    sup_info.append(self._ind2all[example_ind][0][label_ind])
                    costs.append(self._ind2all[example_ind][1][label_ind])
                else:
                    sup_info.append(self._ind2all[example_ind][label_ind])
                    costs.append(np.ones(len(label_ind)))
            else:
                self._do_missing(k)
        return sup_info, costs

    def query_by_example(self, indexes):
        """Query function, query information giving an instance.

        Note that, this function only available if initializes with
        data matrix.

        In multi-label setting, a query index is a tuple.
        A single index should only have 1 element (feature_vector, ) to query all labels or
        2 elements (feature_vector, [label_indexes]) to query specific labels.
        A list of index can be provided.

        Parameters
        ----------
        indexes: array_like
            [n_samples, n_features]

        Returns
        -------
        sup_info: list
            supervised information of queried instance.

        costs: list
            corresponding costs produced by query.
        """
        if not self._instance_flag:
            raise ValueError("This oracle do not have the instance information, query_by_instance is not supported")
        # check validity of the given indexes
        if not isinstance(indexes, (list, np.ndarray)):
            indexes = [indexes]

        q_id = []
        for k in indexes:
            # k is a tuple with 2 elements
            k_len = len(k)
            if k_len != 1 and k_len != 2:
                raise ValueError(
                    "A single index should only have 1 element (feature_vector, ) to query all labels or"
                    "2 elements (feature_vector, [label_indexes]) to query specific labels. But found %d in %s" %
                    (len(k), str(k)))
            example_fea = k[0]

            # fetch data
            if example_fea in self._exa2ind.keys():
                if k_len == 1:
                    q_id.append((example_fea,))
                else:
                    q_id.append((example_fea, k[1]))
            else:
                self._do_missing(example_fea, 'instance pool')
        return self.query_by_index(q_id)


class Oracles:
    """Class to support noisy oracles setting.

    This class is a container that support multiple oracles work together.
    It will store the cost in all and cost for each oracle for analysing.
    also can return row labels for user-defined strategy.
    """

    def __init__(self):
        self.cost_inall = 0
        self.cost_arr = []

    def add_oracle(self, oracle):
        pass

    def query_from(self, index_for_querying, index_of_oracle):
        pass

    def get_oracle(self, index_of_oracle):
        pass

    def get_summary(self):
        pass

    def __repr__(self):
        """Print summaries of each oracle.

        This function returns the content of this object.
        """
        pass


if __name__ == '__main__':
    a = Oracle([1, 2, 3])
    print(a.query_by_index(1))
    a.add_knowledge(labels=0, indexes=3)
    print(a.query_by_index([3]))
    a.add_knowledge(labels=[4,5], indexes=[4,5])
    print(a.query_by_index([5]))

    a = Oracle([[1, 2, 3], [4, 5, 6]])
    print(a.query_by_index(1))
    a.add_knowledge(labels=[0,2,1], indexes=2)
    print(a.query_by_index([2]))
    a.add_knowledge(labels=[[1, 2, 3], [4, 5, 6]], indexes=[3, 4])
    print(a.query_by_index([4]))

    b = OracleQueryMultiLabel([[1, 2, 3], [4, 5, 6]])
    print(b.query_by_index((0,)))
    print(b.query_by_index((0, (0, 1))))
    print(b.query_by_index((0, [0, 1])))
    print(b.query_by_index((0, 0)))
    b.add_knowledge([7,8,9], 2)
    print(b.query_by_index((2, 0)))
    print(b.query_by_index([(2, 0),(1, )]))
