"""
Base
ABC for AL
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod
import collections.abc
import copy


class BaseQueryStrategy(metaclass=ABCMeta):
    """
    Base query class
    should set the parameters and global const in __init__
    implement query function with data matrix
    should return element in unlabel_index set
    """

    @abstractmethod
    def select(self, *args):
        """Select instance to query

        Returns
        -------
        queried keys, key should be in unlabel_index
        """
        pass


class BaseOracle(metaclass=ABCMeta):
    @abstractmethod
    def query_by_instance(self, instance):
        """Return cost and queried info

        Parameters
        ----------
        instance: object
            queried instance

        Returns
        -------
        labels of queried _indexes AND cost
        """
        pass


class BaseVirtualOracle(metaclass=ABCMeta):
    """
    Basic class of Virtual Oracle for experiment

    1.record basic information of oracle
    2.define return type and return pool(can be noisy)
    """

    @abstractmethod
    def query_by_index(self, collection):
        """Return cost and queried info

        Parameters
        ----------
        collection: array-like
            queried index

        Returns
        -------
        labels of queried _indexes AND cost
        """
        pass


class BaseCollection(metaclass=ABCMeta):
    _innercontainer = None
    _element_type = None

    def __contains__(self, other):
        return other in self._innercontainer

    def __iter__(self):
        return iter(self._innercontainer)

    def __len__(self):
        return len(self._innercontainer)

    def __repr__(self):
        return self._innercontainer.__repr__()

    @abstractmethod
    def add(self, *args):
        """
        add element to the container
        """
        pass

    @abstractmethod
    def discard(self, *args):
        """
        discard element to the container
        """
        pass

    def remove(self, value):
        """Remove an element. If not a member, raise a KeyError."""
        self.discard(value)

    def clear(self):
        """This is slow (creates N new iterators!) but effective."""
        self._innercontainer.clear()


class BaseDB(metaclass=ABCMeta):
    """Knowledge database
    Have the similar function with oracle
    but retrieving from DB will not incur a cost

    Also provide the function to return the feature
    and label matrix of labeled set for training
    """
    def __getitem__(self, index):
        """Same function with retrieve by index.

        Raise if item is not in the index set.

        Parameters
        ----------
        index: object
            index of example and label

        Returns
        -------
        example: np.ndarray
            the example corresponding the index.

        label: object
            the label corresponding the index.
            The type of returned object is the same with the
            initializing.
        """
        return self.retrieve_by_indexes(index)

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_examples(self):
        """Get all examples in the data base

        If this object is a MatrixKnowledgeDB, it will return the feature matrix,
        otherwise, A dict will be returned.
        """
        pass

    @abstractmethod
    def get_labels(self, *args):
        """Get all labels in the data base

        If this object is a MatrixKnowledgeDB, it will return the label matrix,
        otherwise, unknown elements will be set to a specific value (query a single label in multi-label setting).
        """
        pass

    @abstractmethod
    def clear(self):
        pass