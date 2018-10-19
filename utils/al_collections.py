"""
The container to store indexes in active learning.
Serve as the basic type of 'set' operation.
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
    Multiple different type of element is supported for Active learning.
    Also check the validity of given operation.

    Note that:
    1. The types of elements should be same
    1. If multiple elements to update, it should be a list, numpy.ndarray or IndexCollection
        object, otherwise, it will be cheated as one single element. (If single element
        contains multiple values, take tuple as the type of element.)

    Parameters
    ----------
    data : list or np.ndarray or object, optional (default=None)
        shape [n_element].  Element should be int or tuple.
        The meaning of elements can be defined by users.

        Some examples of elements:
        (example_index, label_index) for instance-label pair query.
        (example_index, feature_index) for feature query,
        (example_index, example_index) for active clustering;
        If int, it may be the index of an instance, for example.

    Attributes
    ----------
    index: list, shape (1, n_elements)
        A list contains all elements in this container.

    Examples
    --------
    >>> a = IndexCollection([1, 2, 3])
    >>> a.update([4,5])
    >>> a
    [1,2,3,4,5]
    >>> a.difference_update([1,2])
    >>> a
    [3,4,5]
    """

    def __init__(self, data=None):
        if data is not None:
            if isinstance(data, IndexCollection):
                self._innercontainer = copy.deepcopy(data.index)
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

    def get_elementType(self):
        return self._element_type

    def pop(self):
        """Return the popped value. Raise KeyError if empty."""
        return self._innercontainer.pop()

    def add(self, value):
        """Add element.

        It will warn if the value to add is existent.

        Parameters
        ----------
        value: object
            same type of the element already in the set.
            Raise if unknown type is given.

        Returns
        -------
        self: object
            return self.
        """
        if self._element_type is None:
            self._element_type = type(value)
        # check validation
        if isinstance(value, np.generic):
            value = np.asscalar(value)
        if not isinstance(value, self._element_type):
            raise TypeError("A %s parameter is expected, but received: %s" % (str(self._element_type), str(type(value))))
        if value in self._innercontainer:
            warnings.warn("Adding element %s has already in the collection, skip." % (value.__str__()),
                          category=RepeatElementWarning,
                          stacklevel=3)
        else:
            self._innercontainer.append(value)
        return self

    def discard(self, value):
        """Remove an element.

        It will warn if the value to discard is inexistent.

        Parameters
        ----------
        value: object
            Value to discard.

        Returns
        -------
        self: object
            Return self.
        """
        if value not in self._innercontainer:
            warnings.warn("Element %s to discard is not in the collection, skip." % (value.__str__()),
                          category=InexistentElementWarning,
                          stacklevel=3)
        else:
            self._innercontainer.remove(value)
        return self

    def difference_update(self, other):
        """Remove all elements of another array from this container.

        Parameters
        ----------
        other: object
            Elements to discard. Note that, if multiple indexes are contained,
            a list, numpy.ndarray or IndexCollection should be given. Otherwise,
            it will be cheated as an object.

        Returns
        -------
        self: object
            Return self.
        """
        if not isinstance(other, (list, np.ndarray, IndexCollection)):
            other = [other]
        for item in other:
            self.discard(item)
        return self

    def update(self, other):
        """Update self with the union of itself and others.

        Parameters
        ----------
        other: object
            Elements to add. Note that, if multiple indexes are contained,
            a list, numpy.ndarray or IndexCollection should be given. Otherwise,
            it will be cheated as an object.

        Returns
        -------
        self: object
            Return self.
        """
        if not isinstance(other, (list, np.ndarray, IndexCollection)):
            other = [other]
        for item in other:
            self.add(item)
        return self

    def random_sampling(self, rate=0.3):
        """Return a random sampled subset of this collection.

        Parameters
        ----------
        rate: float, optional (default=None)
            The rate of sampling. Must be a number in [0,1].

        Returns
        -------
        array: IndexCollection
            The sampled index collection.
        """
        assert (0 < rate < 1)
        perm = utils.tools.randperm(len(self) - 1, round(rate * len(self)))
        return IndexCollection([self.index[i] for i in perm])


class MultiLabelIndexCollection(IndexCollection):
    """Class for managing multi-label indexes.

    This class stores indexes in multi-label. Each element should be a tuple.
    A single index should only have 1 element (example_index, ) to query all _labels or
    2 elements (example_index, [label_indexes]) to query specific _labels.

    Some examples of valid multi-label indexes include:
    queried_index = (1, [3,4])
    queried_index = (1, [3])
    queried_index = (1, 3)
    queried_index = (1, (3))
    queried_index = (1, (3,4))
    queried_index = (1, )   # query all _labels

    Several validity checking are implemented in this class.
    Such as repeated elements, Index out of bound.

    Parameters
    ----------
    data : list or np.ndarray of a single tuple, optional (default=None)
        shape [n_element]. All elements should be tuples.

    label_size: int, optional (default=None)
        The number of classes. If not provided, an infer is attempted, raise if fail.

    Attributes
    ----------
    index: list, shape (1, n_elements)
        A list contains all elements in this container.

    Examples
    --------
    >>> multi_lab_ind1 = MultiLabelIndexCollection([(0, 1), (0, 2), (0, (3, 4)), (1, (0, 1))], label_size=5)
    >>> multi_lab_ind1.update((0, 0))
    >>> multi_lab_ind1.update([(1, 2), (1, (3, 4))])
    >>> multi_lab_ind1.update([(2,)])
    >>> multi_lab_ind1.difference_update([(0,)])
    """

    def __init__(self, data=None, label_size=None):
        """Initialize the container.

        Parameters
        ----------
        data: collections.Iterable
        """
        if data is not None:
            if isinstance(data, MultiLabelIndexCollection):
                self._innercontainer = copy.deepcopy(data.index)
                self._label_size = data._label_size
                return
            # check given indexes
            data = check_index_multilabel(data)
            if label_size is None:
                self._label_size = infer_label_size_multilabel(data, check_arr=False)
            else:
                self._label_size = label_size

            # decompose all label queries.
            decomposed_data = flattern_multilabel_index(data, self._label_size, check_arr=False)

            self._innercontainer = set(decomposed_data)
            if len(self._innercontainer) != len(decomposed_data):
                warnings.warn(
                    "There are %d same elements in the given data" % (len(data) - len(self._innercontainer)),
                    category=RepeatElementWarning,
                    stacklevel=3)
        else:
            self._innercontainer = set()
            if label_size is None:
                warnings.warn("This collection does not have a _label_size value, set it manually or "
                              "it will raise when decomposing indexes.",
                              category=ValidityWarning)
            self._label_size = label_size

    @property
    def index(self):
        return list(self._innercontainer)

    def add(self, value):
        """Add element.

        It will warn if the value to add is existent. Raise if
        invalid type of value is given.

        Parameters
        ----------
        value: tuple
            Index for adding. Raise if index is out of bound.

        Returns
        -------
        self: object
            return self.
        """
        # check validation
        assert(isinstance(value, tuple))
        if len(value) == 1:
            value = [(value[0], i) for i in range(self._label_size)]
            return self.update(value)
        elif len(value) == 2:
            if isinstance(value[1], collections.Iterable):
                for item in value[1]:
                    if item >= self._label_size:
                        raise ValueError("Index %s is out of bound %s" % (str(item), str(self._label_size)))
            else:
                if value[1] >= self._label_size:
                    raise ValueError("Index %s is out of bound %s" % (str(value[1]), str(self._label_size)))
        else:
            raise ValueError("A tuple with 1 or 2 elements is expected, but received: %s" % str(value))
        if value in self._innercontainer:
            warnings.warn("Adding element %s has already in the collection, skip." % (value.__str__()),
                          category=RepeatElementWarning,
                          stacklevel=3)
        else:
            self._innercontainer.add(value)
        return self

    def discard(self, value):
        """Remove an element.

        It will warn if the value to discard is inexistent. Raise if
        invalid type of value is given.

        Parameters
        ----------
        value: tuple
            Index for adding. Raise if index is out of bound.

        Returns
        -------
        self: object
            return self.
        """
        assert (isinstance(value, tuple))
        if len(value) == 1:
            value = [(value[0], i) for i in range(self._label_size)]
            return self.difference_update(value)
        if value not in self._innercontainer:
            warnings.warn("Element %s to discard is not in the collection, skip." % (value.__str__()),
                          category=InexistentElementWarning,
                          stacklevel=3)
        else:
            self._innercontainer.discard(value)
        return self

    def difference_update(self, other):
        """Remove all elements of another array from this container.

        Parameters
        ----------
        other: object
            Elements to discard. Note that, if multiple indexes are contained,
            a list, numpy.ndarray or MultiLabelIndexCollection should be given. Otherwise,
            a tuple should be given.

        Returns
        -------
        self: object
            Return self.
        """
        if isinstance(other, (list, np.ndarray, MultiLabelIndexCollection)):
            label_ind = flattern_multilabel_index(other, self._label_size)
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
        """Update self with the union of itself and others.

        Parameters
        ----------
        other: object
            Elements to add. Note that, if multiple indexes are contained,
            a list, numpy.ndarray or MultiLabelIndexCollection should be given. Otherwise,
            a tuple should be given.

        Returns
        -------
        self: object
            Return self.
        """
        if isinstance(other, (list, np.ndarray, MultiLabelIndexCollection)):
            label_ind = flattern_multilabel_index(other, self._label_size)
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
    """container to store the indexes in feature querying scenario.

    This class stores indexes in incomplete feature matrix setting. Each element should be a tuple.
    A single index should only have 1 element (example_index, ) to query all features or
    2 elements (example_index, [feature_indexes]) to query specific features.

    Some examples of valid indexes include:
    queried_index = (1, [3,4])
    queried_index = (1, [3])
    queried_index = (1, 3)
    queried_index = (1, (3))
    queried_index = (1, (3,4))
    queried_index = (1, )   # query all _labels

    Several validity checking are implemented in this class.
    Such as repeated elements, Index out of bound.

    Parameters
    ----------
    data : list or np.ndarray of a single tuple, optional (default=None)
        shape [n_element]. All elements should be tuples.

    feature_size: int, optional (default=None)
        The number of features. If not provided, an infer is attempted, raise if fail.

    Attributes
    ----------
    index: list, shape (1, n_elements)
        A list contains all elements in this container.

    Examples
    --------
    >>> fea_ind1 = FeatureIndexCollection([(0, 1), (0, 2), (0, (3, 4)), (1, (0, 1))], feature_size=5)
    >>> fea_ind1.update((0, 0))
    >>> fea_ind1.update([(1, 2), (1, (3, 4))])
    >>> fea_ind1.update([(2,)])
    >>> fea_ind1.difference_update([(0,)])
    """

    def __init__(self, data, feature_size=None):
        try:
            super(FeatureIndexCollection, self).__init__(data=data, label_size=feature_size)
        except(Exception, ValueError):
            raise Exception("The inference of feature_size is failed, please set a specific value.")
