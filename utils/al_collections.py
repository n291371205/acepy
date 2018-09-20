"""
Data_Collection
Serve as the basic type of 'set' operation
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division

import collections
import copy

import numpy as np

import utils.tools
from utils.ace_warnings import *
from utils.base import BaseCollection
from utils.tools import _is_arraylike
from sklearn.utils.validation import check_array


class IndexCollection(BaseCollection):
    """Index Collection.

    Index Collection class is a basic data type of setting operation.
    Only support for
    Multiple different type of element is supported for Active learning.
    Also check the validity of given operation.

    Parameters
    ----------
    data : list or np.ndarray , shape like [n_element]
        element should be int or tuple, if tuple, it may represent index
        of (sample,label) for instance-label pair query,
        (sample, feature) for feature query, (sample, sample) for active clustering;
        if int, it is treated as the index of example, and each query will return
        ALL labels of the selected example.

    Examples
    --------
    """

    def __init__(self, data=None):
        """Initialize the container.

        Parameters
        ----------
        data: collections.Iterable
        """
        if data is not None:
            assert (_is_arraylike(data))
            self._innercontainer = list(np.unique([i for i in data], axis=0))
            if len(self._innercontainer) != len(data):
                warnings.warn("There are %d same elements in the given data" % (len(data) - len(self._innercontainer)),
                              category=RepeatElementWarning,
                              stacklevel=3)
            datatype = collections.Counter([type(i) for i in self._innercontainer])
            if len(datatype) != 1:
                raise TypeError("Different types found in the given _indexes.")
            tmp_data = self._innercontainer[0]
            if isinstance(tmp_data, np.generic):
                self._element_type = type(np.asscalar(tmp_data))
            else:
                self._element_type = type(tmp_data)
        else:
            self._innercontainer = []

    @property
    def index(self):
        return copy.deepcopy(self._innercontainer)

    def pop(self):
        """Return the popped value. Raise KeyError if empty."""
        return self._innercontainer.pop()

    def add(self, key):
        """add element.

        Parameters
        ----------
        key: object
            same type of the element already in the set.
            Raise if unknown type is given.

        value: object, optional (default=None)
            supervised information given by oracle.
        """
        if self._element_type is None:
            self._element_type = type(key)
        # check validation
        if isinstance(key, np.generic):
            key = np.asscalar(key)
        if not isinstance(key, self._element_type):
            raise TypeError("A %s parameter is expected, but received: %s" % (str(self._element_type), str(type(key))))
        if key in self._innercontainer:
            warnings.warn("Adding element %s has already in the collection, skip." % (key.__str__()),
                          category=RepeatElementWarning,
                          stacklevel=3)
        else:
            self._innercontainer.append(key)
        return self

    def discard(self, value):
        """Remove an element.  Do not raise an exception if absent."""
        if value not in self._innercontainer:
            warnings.warn("Element %s to discard is not in the collection, skip." % (value.__str__()),
                          category=RepeatElementWarning,
                          stacklevel=3)
        else:
            self._innercontainer.remove(value)
        return self

    def difference_update(self, other):
        """ Remove all elements of another set from this set. """
        if not _is_arraylike(other):
            if isinstance(other, (list, collections.Iterable)):
                other = list(other)
            else:
                other = [other]
        for item in other:
            self.discard(item)
        return self

    def update(self, other):
        """ Update a set with the union of itself and others. """
        if not _is_arraylike(other):
            if isinstance(other, (list, collections.Iterable)):
                other = list(other)
            else:
                other = [other]
        for item in other:
            self.add(item)
        return self

    def random_sampling(self, rate=0.3):
        """
        return a random sampled subset of this collection.

        Parameters
        ----------
        rate: float, optional (default=None)
            the rate of sampling. Must be a number in [0,1].

        Returns
        -------
        array: IndexCollection
            the sampled index collection.
        """
        assert (0 < rate < 1)
        perm = utils.tools.randperm(len(self) - 1, round(rate * len(self)))
        return IndexCollection([self.index[i] for i in perm])


class MultiLabelIndexCollection(IndexCollection):
    """Class for multi-label index.

    Mainly solve the difference between index in querying all labels and specific labels
    """


if __name__ == '__main__':
    a = IndexCollection([1, 2, 2, 3])
    b = IndexCollection([1, 2, 2, 3])
    c = set([1, 2, 3])
    print(a == b)
    print(a == c)
    print(1 in a)
    a.add(3)
    a.add(4)
    print(a.pop())
    a.discard(5)
    a.discard(4)
    a.update(set([2, 6, 7, 8]))
    a.update(IndexCollection([2, 9, 10]))
    for i in a:
        print(i)
    print(len(a))
    print(a)
    a.difference_update(set([1, 200]))
    a.difference_update(IndexCollection([2, 100]))
    print(a)

    d = IndexCollection(np.array([1, 2, 3, 4]))
    print(d)
    # d = DataCollection([[1, 2], [3, 4]],
    #                    ['instance', 'label']) # fail
    d = IndexCollection([(1, 2), (3, 4)])
    print(d)
    # d = DataCollection(np.array([(1, 2), (3, 4)]),
    #                    ['instance', 'label'])  # fail ,because the ndarray is like [[1,2],[3,4]]
    print(d)
    print(a.random_sampling(0.5))
    print(type(a.random_sampling(0.5)))
