"""
Knowledge database
Have the similar function with oracle
but retrieving from DB will not incur a cost

Also provide the function to return the
labeled set for training
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause
from __future__ import division

import numpy as np
import prettytable as pt
from sklearn.utils.validation import check_array

from utils.ace_warnings import *
from utils.base import BaseDB
from utils.tools import _is_arraylike


class ElementKnowledgeDB(BaseDB):
    """Class to store fine-grained (element-wise) data.

    Both the example AND label are not required to be an array-like object,
    they can be complicated object.
    """

    def __init__(self, labels, indexes, examples=None):
        """initialize supervised information.
        """
        # check and record parameters
        if not _is_arraylike(labels):
            raise TypeError("An array like parameter is expected.")
        if not _is_arraylike(indexes):
            raise TypeError("An array like parameter is expected.")
        # self._y = copy.copy(labels)
        self._index_len = len(labels)
        if len(indexes) != self._index_len:
            raise ValueError("Length of given instance_indexes do not accord the data set.")
        self._indexes = list(indexes)
        if examples is None:
            self._instance_flag = False
        else:
            if not _is_arraylike(examples):
                raise TypeError("An array like parameter is expected.")
            # self._X = copy.copy(examples)
            n_samples = len(examples)
            if n_samples != self._index_len:
                raise ValueError("Different length of instances and labels found.")
            self._instance_flag = True

        # several _indexes construct
        if self._instance_flag:
            self._exa2ind = dict(zip(examples, self._indexes))
            self._ind2exa = dict(zip(self._indexes, examples))
        self._ind2label = dict(zip(self._indexes, labels))

        # record
        self.cost_inall = 0
        self.cost_arr = []
        self.num_of_queries = 0
        self.query_history = []

    def __len__(self):
        return self._index_len

    def add(self, select_index, label, cost, example=None):
        """add an element to the DB.

        Parameters
        ----------
        select_index: int or tuple
            the selected index in active learning.

        label: object
            supervised information given by oracle.

        cost: object, optional (default=1)
            costs produced by query, given by the oracle.

        example: object
            same type of the element already in the set.
            Raise if unknown type is given.
        """
        if self._instance_flag:
            if example is None:
                raise Exception("This oracle has the instance information,"
                                "must provide example parameter when adding entry")
            self._exa2ind[example] = select_index
            self._ind2exa[select_index] = example
        self._ind2label[select_index] = label
        self._indexes.append(select_index)
        self.cost_inall += np.sum(cost)
        self.cost_arr.append(cost)

    def discard(self, index=None, example=None):
        """discard element either by index or example.

        Must provide at least one of them.

        Parameters
        ----------
        index: int or tuple
            index to discard.

        example: object
            example to discard, must be one of the data base.
        """
        """Remove an element."""
        if index is None and example is None:
            raise Exception("Must provide one of index or example")
        if index is not None:
            if index not in self._indexes:
                warnings.warn("Index %s is not in the data base, skipped." % str(index),
                              category=ValidityWarning,
                              stacklevel=3)
                return self
            self._indexes.pop(index)
        if example is not None:
            if not self._instance_flag:
                raise Exception("This data base is not initialized with examples, discard by example is illegal.")
            else:
                if example not in self._exa2ind:
                    warnings.warn("example %s is not in the data base, skipped." % str(example),
                                  category=ValidityWarning,
                                  stacklevel=3)
                    return self
                ind = self._exa2ind[example]
                self._indexes.remove(ind)
                self._exa2ind.pop(example)
                self._ind2exa.pop(ind)
                self._ind2label.pop(ind)
        return self

    def update_query(self, labels, indexes, cost, examples=None):
        """Updating data base with queried information.

        Parameters
        ----------
        labels: array-like or object
            labels to be updated.

        indexes: array-like or object
            if multiple example-label pairs are provided, it should be a list or np.ndarray type
            otherwise, it will be treated as only one pair for adding.

        cost: array-like or object
            cost corresponds to the query.

        examples: array-like or object
            examples to be updated.
        """
        if not isinstance(indexes, (list, np.ndarray)):
            self.add(labels, indexes, examples, cost)
        else:
            if len(indexes) == 1:
                self.add(labels, indexes[0], examples, cost)
            else:
                assert (len(indexes) == len(labels))
                for i in range(len(labels)):
                    self.add(labels[i], indexes[i], example=examples[i] if examples is not None else None,
                             cost=cost[i] if cost is not None else None)
        self.num_of_queries += 1
        self._update_query_history(labels, indexes, cost)
        return self

    def retrieve_by_indexes(self, indexes):
        """retrieve by indexes

        Parameters
        ----------
        indexes: array-like or object
            if 2 or more indexes to retrieve, a list or np.ndarray is expected
            otherwise, it will be treated as only one index.

        Returns
        -------
        X,y: array-like
            the retrieved data
        """
        if not isinstance(indexes, (list, np.ndarray)):
            indexes = [indexes]
        example_arr = []
        label_arr = []
        for k in indexes:
            if k in self._ind2label.keys():
                label_arr.append(self._ind2label[k])
                if self._instance_flag:
                    example_arr.append(self._ind2exa[k])
            else:
                warnings.warn("Index %s for retrieving is not in the data base, skip." % str(k),
                              category=ValidityWarning)
        return example_arr, label_arr

    def retrieve_by_examples(self, examples):
        """retrieve by examples

        Parameters
        ----------
        examples: array-like or object
            if 2 or more examples to retrieve, a 2D array is expected
            otherwise, it will be treated as only one index.

        Returns
        -------
        X,y: array-like
            the retrieved data
        """
        if not self._instance_flag:
            raise Exception("This oracle do not have the instance information, query_by_instance is not supported")
        if not isinstance(examples, (list, np.ndarray)):
            raise TypeError("An list or numpy.ndarray is expected, but received:%s" % str(type(examples)))
        if len(np.shape(examples)) == 1:
            examples = [examples]
        q_id = []
        for k in examples:
            if k in self._exa2ind.keys():
                q_id.append(self._exa2ind[k])
            else:
                warnings.warn("Example for retrieving is not in the data base, skip.",
                              category=ValidityWarning)
        return self.retrieve_by_indexes(q_id)

    def get_examples(self):
        """Get all examples in the data base

        If this object is a MatrixKnowledgeDB, it will return the feature matrix,
        otherwise, A dict will be returned.
        """
        return [self._ind2exa[i] for i in self._indexes]

    def get_labels(self, *args):
        """Get all labels in the data base

        If this object is a MatrixKnowledgeDB, it will return the label matrix,
        otherwise, unknown elements will be set to a specific value (query a single label in multi-label setting).
        """
        return [self._ind2label[i] for i in self._indexes]

    def clear(self):
        self._indexes.clear()
        self._exa2ind.clear()
        self._ind2label.clear()
        self._indexes.clear()
        self._instance_flag = False
        self.cost_inall = 0
        self.cost_arr = []
        self.num_of_queries = 0
        self.query_history.clear()

    def _update_query_history(self, labels, indexes, cost):
        """record the query history"""
        self.query_history.append(((labels, cost), indexes))

    def full_history(self):
        """return full version of query history"""
        tb = pt.PrettyTable()
        # tb.set_style(pt.MSWORD_FRIENDLY)
        for query_ind in range(len(self.query_history)):
            query_result = self.query_history[query_ind]
            tb.add_column(str(query_ind), "query_index:%s\nresponse:%s\ncost:%s" % (
                          str(query_result[1]), str(query_result[0][0]), str(query_result[0][1])))
        tb.add_column('in all', "number_of_queries:%s\ncost:%s"%(str(len(self.query_history)), str(self.cost_inall)))
        return str(tb)


# this class can only deal with the query-all-labels setting
class MatrixKnowledgeDB(BaseDB):
    """Knowledge DataBase.

    Knowledge DataBase. class is a basic data type of setting operation. This
    class can further store the supervised information given by oracle.
    Multiple different types of elements are supported for Active learning.
    Also check the validity of given operation.

    Parameters
    ----------
    labels:  array-like
        label matrix. shape like [n_samples, n_classes] or [n_samples]

    indexes: array-like
        index of examples, should have the same length
        and is one-to-one correspondence of y.

    examples: array-like, optional (default=None)
        array of examples, initialize with this parameter to make
        "query by instance" available. Shape like [n_samples, n_features]

    Examples
    --------
    """

    def __init__(self, labels, indexes, examples=None):
        """initialize supervised information.
        """
        # check and record parameters
        self._y = check_array(labels, ensure_2d=False, dtype=None)
        if isinstance(labels[0], np.generic):
            self._label_type = type(np.asscalar(labels[0]))
        else:
            self._label_type = type(labels[0])
        self._index_len = len(self._y)
        assert (_is_arraylike(indexes))
        if len(indexes) != self._index_len:
            raise ValueError("Length of given instance_indexes do not accord the data set.")
        self._indexes = np.asarray(indexes)
        if examples is None:
            self._instance_flag = False
        else:
            self._instance_flag = True
            self._X = check_array(examples, accept_sparse='csr', ensure_2d=True, order='C')
            n_samples = self._X.shape[0]
            if n_samples != self._index_len:
                raise ValueError("Different length of instances and labels found.")

        # record
        self.cost_inall = 0
        self.cost_arr = []
        self.num_of_queries = 0
        self.query_history = []

    def __len__(self):
        return self._index_len

    def add(self, label, select_index, cost, example=None):
        """add an element to the DB.

        Parameters
        ----------
        select_index: int or tuple
            the selected index in active learning.

        label: object
            supervised information given by oracle.

        cost: object
            costs produced by query, given by the oracle.

        example: object
            same type of the element already in the set.
            Raise if unknown type is given.
        """
        # check validation
        if self._y.ndim == 1:
            if hasattr(label, '__len__'):
                raise TypeError("The initialized label array only have 1 dimension, "
                                "but received an array like label: %s." % str(label))
            self._y = np.append(self._y, [label])
        else:
            # this operation will check the validity automatically.
            self._y = np.append(self._y, [label], axis=0)
        if select_index in self._indexes:
            warnings.warn("Repeated index is found when adding element to knowledge data base. Skipp this item")
            return self
        self._indexes = np.append(self._indexes, select_index)
        self.cost_arr.append(cost)
        self.cost_inall += np.sum(cost)
        if self._instance_flag:
            if example is None:
                raise Exception("Example must be provided in a database initialized with examples.")
            else:
                self._X = np.append(self._X, [example], axis=0)
        return self

    def update_query(self, labels, indexes, cost, examples=None):
        """Updating data base with queried information.

        Parameters
        ----------
        labels: array-like or object
            labels to be updated.

        indexes: array-like or object
            if multiple example-label pairs are provided, it should be a list or np.ndarray type
            otherwise, it will be treated as only one pair for adding.

        cost: array-like or object
            cost corresponds to the query.

        examples: array-like or object
            examples to be updated.
        """
        if not isinstance(indexes, (list, np.ndarray)):
            self.add(labels, indexes, cost, examples)
        else:
            assert (len(indexes) == len(labels))
            for i in range(len(indexes)):
                self.add(labels[i], indexes[i], cost=cost[i],
                         example=examples[i] if examples is not None else None)
        self.num_of_queries += 1
        self._update_query_history(labels, indexes, cost, examples)
        return self

    def discard(self, index=None, example=None):
        """Remove an element."""
        if index is None and example is None:
            raise Exception("Must provide one of index or example")
        if index is not None:
            if index not in self._indexes:
                warnings.warn("Index %s is not in the data base, skipped." % str(index),
                              category=ValidityWarning,
                              stacklevel=3)
                return self
            ind = np.argwhere(self._indexes == index)
        if example is not None:
            if not self._instance_flag:
                raise Exception("This data base is not initialized with examples, discard by example is illegal.")
            else:
                ind = np.argwhere(self._X == example)
        mask = np.ones(len(self._indexes), dtype=bool)
        mask[ind] = False
        self._y = self._y[mask]
        self._indexes = self._indexes[mask]
        if self._instance_flag:
            self._X = self._X[mask]
        return self

    def retrieve_by_indexes(self, indexes):
        """retrieve by indexes

        Parameters
        ----------
        indexes: array-like or object
            if 2 or more indexes to retrieve, a list or np.ndarray is expected
            otherwise, it will be treated as only one index.

        Returns
        -------
        X,y: array-like
            the retrieved data
        """
        # check if in the dict?
        if not isinstance(indexes, (list, np.ndarray)):
            ind = np.argwhere(self._indexes == indexes)  # will return empty array if not found.
            return self._X[ind,] if self._instance_flag else None, self._y[ind,]
        else:
            if len(indexes) == 1:
                ind = np.argwhere(self._indexes == indexes[0]).flatten()
                return self._X[ind,] if self._instance_flag else None, self._y[ind,]
            else:
                ind = [np.argwhere(self._indexes == indexes[i]).flatten() for i in range(len(indexes))]
                return self._X[ind,] if self._instance_flag else None, self._y[ind,]

    def retrieve_by_examples(self, examples):
        """retrieve by examples

        Parameters
        ----------
        examples: array-like or object
            if 2 or more examples to retrieve, a 2D array is expected
            otherwise, it will be treated as only one index.

        Returns
        -------
        X,y: array-like
            the retrieved data
        """
        if not self._instance_flag:
            raise Exception("This data base is not initialized with examples, retrieve by example is illegal.")
        examples = np.asarray(examples)
        if examples.ndim == 1:
            ind = np.argwhere(self._X == examples).flatten()  # will return empty array if not found.
            return self._X[ind,], self._y[ind,]
        elif examples.ndim == 2:
            ind = []
            for i in range(len(examples)):
                for ins in self._X:
                    if np.all(ins == examples[i]):
                        ind.append(i)
            # ind = [np.argwhere(self._X == examples[i]).flatten() for i in range(len(examples))]
            return self._X[ind, ], self._y[ind, ]
        else:
            raise Exception("A 1D or 2D array is expected. But received: %d" % examples.ndim)

    def get_examples(self):
        """Get all examples in the data base

        If this object is a MatrixKnowledgeDB, it will return the feature matrix,
        otherwise, A dict will be returned.
        """
        if self._instance_flag:
            return self._X.copy()
        else:
            return None

    def get_labels(self):
        """Get all labels in the data base

        If this object is a MatrixKnowledgeDB, it will return the label matrix,
        otherwise, unknown elements will be set to a specific value (query a single label in multi-label setting).
        """
        return self._y.copy()

    def clear(self):
        self.cost_inall = 0
        self.cost_arr = []
        self.num_of_queries = 0
        self._instance_flag = False
        self._X = None
        self._y = None
        self._indexes = None
        self.query_history.clear()

    def _update_query_history(self, labels, indexes, cost):
        """record the query history"""
        self.query_history.append(((labels, cost), indexes))

    def full_history(self):
        """return full version of query history"""
        tb = pt.PrettyTable()
        # tb.set_style(pt.MSWORD_FRIENDLY)
        for query_ind in range(len(self.query_history)):
            query_result = self.query_history[query_ind]
            tb.add_column(str(query_ind), "query_index:%s\nresponse:%s\ncost:%s" % (
                str(query_result[1]), str(query_result[0][0]), str(query_result[0][1])))
        tb.add_column('in all', "number_of_queries:%s\ncost:%s" % (str(len(self.query_history)), str(self.cost_inall)))
        return str(tb)

# class MultiLabelKnowledgeDB(MatrixKnowledgeDB):
#     """Using a sparse matrix to save the data
#     """
