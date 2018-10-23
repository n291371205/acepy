"""
ABC for acepy
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod
import collections.abc
import copy
import numpy as np
from sklearn.utils.validation import check_X_y


class BaseQueryStrategy(metaclass=ABCMeta):
    """
    Base query class.
    The parameters and global const are set in __init__()
    The instance to query can be obtained by select(), the labeled and unlabeled
    indexes of instances should be given. An array of selected elements in unlabeled indexes
    should be returned.
    """
    def __init__(self, X=None, y=None, **kwargs):
        if X is not None and y is not None:
            if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                # will not use additional memory
                check_X_y(X, y, accept_sparse='csc', multi_output=True)
                self.X = X
                self.y = y
            else:
                self.X, self.y = check_X_y(X, y, accept_sparse='csc', multi_output=True)
        else:
            self.X = X
            self.y = y

    @abstractmethod
    def select(self, label_index, unlabel_index, **kwargs):
        """Select instances to query.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled instances.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled instances.

        Returns
        -------
        selected_index: list
            The elements of selected_index should be in unlabel_index.
        """
        pass

    def select_by_prediction_mat(self, unlabel_index, predict, **kwargs):
        """select in a model-independent way.

        Parameters
        ----------
        prediction_mat: array, shape [n_examples, n_classes]
            The probability prediction matrix.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled instances. Should be one-to-one
            correspondence to the prediction_mat

        Returns
        -------
        selected_index: list
            The elements of selected_index should be in unlabel_index.
        """
        pass


class BaseVirtualOracle(metaclass=ABCMeta):
    """
    Basic class of virtual Oracle for experiment

    This class will build a dictionary between index-label in the __init__().
    When querying, the queried_index should be one of the key in the dictionary.
    And the label which corresponds to the key will be returned.
    """

    @abstractmethod
    def query_by_index(self, indexes):
        """Return cost and queried info.

        Parameters
        ----------
        indexes: array
            Queried indexes.

        Returns
        -------
        Labels of queried indexes AND cost
        """
        pass


class BaseCollection(metaclass=ABCMeta):
    """The basic container of indexes.

    Functions include:
    1. Update new indexes.
    2. Discard existed indexes.
    3. Validity checking. (Repeated element, etc.)
    """
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
        """Add element to the container."""
        pass

    @abstractmethod
    def discard(self, *args):
        """Discard element in the container."""
        pass

    @abstractmethod
    def update(self, *args):
        """Update multiple elements to the container."""
        pass

    @abstractmethod
    def difference_update(self, *args):
        """Discard multiple elements in the container"""
        pass

    def remove(self, value):
        """Remove an element. If not a member, raise a KeyError."""
        self.discard(value)

    def clear(self):
        """Clear the container."""
        self._innercontainer.clear()


class BaseRepository(metaclass=ABCMeta):
    """Knowledge repository
    Store the information given by the oracle (labels, cost, etc.).

    Functions include:
    1. Retrieving
    2. History recording
    3. Get labeled set for training model
    """
    def __getitem__(self, index):
        """Same function with retrieve by index.

        Raise if item is not in the index set.

        Parameters
        ----------
        index: object
            Index of example and label.

        Returns
        -------
        example: np.ndarray
            The example corresponding the index.

        label: object
            The corresponding label of the index.
            The type of returned object is the same with the
            initializing.
        """
        return self.retrieve_by_indexes(index)

    @abstractmethod
    def add(self, select_index, label, cost=None, example=None):
        """Add an element to the repository."""
        pass

    @abstractmethod
    def discard(self, index=None, example=None):
        """Discard element either by index or example."""
        pass

    @abstractmethod
    def update_query(self, labels, indexes, cost=None, examples=None):
        """Updating data base with queried information."""
        pass

    @abstractmethod
    def retrieve_by_indexes(self, indexes):
        """Retrieve by indexes."""
        pass

    @abstractmethod
    def retrieve_by_examples(self, examples):
        """Retrieve by examples."""
        pass

    @abstractmethod
    def get_training_data(self):
        """Get training set."""
        pass

    @abstractmethod
    def clear(self):
        """Clear this container."""
        pass
