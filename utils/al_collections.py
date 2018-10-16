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
from utils.tools import check_index_multilabel, infer_label_size_multilabel, flattern_multilabel_index


class IndexCollection(BaseCollection):
    """Index Collection.

    Index Collection class is a basic data type of setting operation.
    Only support for
    Multiple different type of element is supported for Active learning.
    Also check the validity of given operation.

    Parameters
    ----------
    data : list or np.ndarray or object, shape like [n_element]
        element should be int or tuple, if tuple, it may represent index
        of (sample,label) for instance-label pair query,
        (sample, feature) for feature query, (sample, sample) for active clustering;
        if int, it is treated as the index of example, and each query will return
        ALL labels of the selected example.

        Note that, if multiple indexes are contained, a list or np.ndarray should be given.
        Otherwise, it will be cheated as an object.

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
            if isinstance(data, IndexCollection):
                self._innercontainer = copy.deepcopy(data)
                self._element_type = data.get_elementType()
                return
            if not isinstance(data, (list, np.ndarray)):
                data = [data]
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

    def __getitem__(self, item):
        return self._innercontainer.__getitem__(item)

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
        if not isinstance(other, (list, np.ndarray, IndexCollection)):
            other = [other]
        for item in other:
            self.discard(item)
        return self

    def update(self, other):
        """ Update a set with the union of itself and others. """
        # if not _is_arraylike(other):
        #     if isinstance(other, (list, collections.Iterable)):
        #         other = list(other)
        #     else:
        #         other = [other]
        if not isinstance(other, (list, np.ndarray, IndexCollection)):
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

    def __init__(self, data=None, label_size=None):
        """Initialize the container.

        Parameters
        ----------
        data: collections.Iterable
        """
        if data is not None:
            # check given indexes
            data = check_index_multilabel(data)
            if label_size is None:
                self.label_size = infer_label_size_multilabel(data, check_arr=False)
            else:
                self.label_size = label_size

            # decompose all labels queries.
            decomposed_data = flattern_multilabel_index(data, self.label_size, check_arr=False)

            self._innercontainer = set(decomposed_data)
            if len(self._innercontainer) != len(decomposed_data):
                warnings.warn(
                    "There are %d same elements in the given data" % (len(data) - len(self._innercontainer)),
                    category=RepeatElementWarning,
                    stacklevel=3)
        else:
            self._innercontainer = set()
            if label_size is None:
                warnings.warn("This collection does not have a label_size value, set it manually or "
                              "it will raise when decomposing indexes.")
            self.label_size = label_size

    @property
    def index(self):
        return list(self._innercontainer)

    def set_label_size(self, label_size):
        self.label_size = label_size

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
        # check validation
        assert(isinstance(key, tuple))
        if len(key) == 1:
            key = [(key[0], i) for i in range(self.label_size)]
            return self.update(key)
        if key in self._innercontainer:
            warnings.warn("Adding element %s has already in the collection, skip." % (key.__str__()),
                          category=RepeatElementWarning,
                          stacklevel=3)
        else:
            self._innercontainer.add(key)
        return self

    def get_elementType(self):
        """Return the type of the elements in the container."""
        return self._element_type

    def discard(self, value):
        """Remove an element.  Do not raise an exception if absent."""
        assert (isinstance(value, tuple))
        if len(value) == 1:
            value = [(value[0], i) for i in range(self.label_size)]
            return self.difference_update(value)
        if value not in self._innercontainer:
            warnings.warn("Element %s to discard is not in the collection, skip." % (value.__str__()),
                          category=RepeatElementWarning,
                          stacklevel=3)
        else:
            self._innercontainer.discard(value)
        return self

    def difference_update(self, other):
        """ Remove all elements of another set from this set. """
        if isinstance(other, (list, np.ndarray, MultiLabelIndexCollection)):
            label_ind = flattern_multilabel_index(other, self.label_size)
            for j in label_ind:
                self.discard(j)
        elif isinstance(other, tuple):
            self.discard(other)
        else:
            raise TypeError(
                "A list or np.ndarray is expected if multiple indexes are "
                "contained. Otherwise, a tuple should be provided")
        return self

    def update(self, other):
        """ Update a set with the union of itself and others. """
        if isinstance(other, (list, np.ndarray, MultiLabelIndexCollection)):
            label_ind = flattern_multilabel_index(other, self.label_size)
            for j in label_ind:
                self.add(j)
        elif isinstance(other, tuple):
            self.add(other)
        else:
            raise TypeError(
                "A list or np.ndarray is expected if multiple indexes are "
                "contained. Otherwise, a tuple should be provided")
        return self

class FeatureIndexCollection(MultiLabelIndexCollection):
    """container to store the indexes in feature querying scenario."""


if __name__ == '__main__':
    a = IndexCollection([1, 2, 2, 3])
    b = IndexCollection([1, 2, 2, 3])
    print(a == b)
    print(1 in a)
    a.add(3)
    a.add(4)
    print(a.pop())
    a.discard(5)
    a.discard(4)
    a.update(IndexCollection([2, 9, 10]))
    for i in a:
        print(i)
    print(len(a))
    print(a)
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

    from data_process.al_split import split_multi_label
    train_idx, test_idx, unlabel_idx, label_idx = split_multi_label(y=np.random.randint(0, 2, 800).reshape(100, -1),
                                                                    initial_label_rate=0.15, partially_labeled=True)
    print(label_idx)
    a = MultiLabelIndexCollection(label_idx[0])
    print(a)
    a.update((0,4))
    a.discard((0,4))
    a.update([(0,4),(0,5)])
    print(a)
    a.update([(0, ), (0, 5)])
    print(a)
    a.difference_update([(0,)])
    print(a)
    train_idx, test_idx, unlabel_idx, label_idx = split_multi_label(y=np.random.randint(0, 2, 800).reshape(100, -1),
                                                                    initial_label_rate=0.15, partially_labeled=False)
    # b = MultiLabelIndexCollection(label_idx[0])

    b = MultiLabelIndexCollection(label_idx[0], label_size=8)
    print(b)
    b.update((4,6))
    b.difference_update((0,))
    print(b)


