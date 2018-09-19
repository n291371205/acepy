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

    def __getitem__(self, item):
        return self._innercontainer.__getitem__(item)

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
