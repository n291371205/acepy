"""
State
Save all information in one AL iteration
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

import copy
import numpy as np
from utils.ace_warnings import *


class State:
    """A class to store information in one iteration of active learning
    for auditting and analysing.

    Parameters
    ----------
    select_index: array-like or object
        if multiple select_index are provided, it should be a list or np.ndarray type
        otherwise, it will be treated as only one pair for adding.

    performance: array-like or object
        performance after querying.

    queried_label: array-like or object, optional
        The queried label.

    cost: array-like or object, optional
        cost corresponds to the query.
    """

    def __init__(self, select_index, performance, queried_label=None, cost=None):
        self.__save_seq = dict()
        if not isinstance(select_index, (list, np.ndarray)):
            select_index = [select_index]
        self.__save_seq['select_index'] = copy.deepcopy(select_index)
        self.__save_seq['performance'] = copy.copy(performance)
        if queried_label is not None:
            self.__save_seq['queried_label'] = copy.deepcopy(queried_label)
        if cost is not None:
            self.__save_seq['cost'] = copy.copy(cost)

        if isinstance(select_index, (list, np.ndarray)):
            self.batch_size = len(select_index)
        else:
            self.batch_size = 1

    def keys(self):
        return self.__save_seq.keys()

    def add_element(self, key, value):
        self.__save_seq[key] = copy.deepcopy(value)

    def del_element(self, key):
        if key in ['select_index', 'queried_info', 'performance', 'cost']:
            warnings.warn("Critical information %s can not be discarded." % str(key),
                          category=ValidityWarning)
        elif key not in self.__save_seq.keys():
            warnings.warn("Key %s be discarded is not in the State, skip." % str(key),
                          category=ValidityWarning)
        else:
            self.__save_seq.pop(key)

    def get_value(self, key):
        return self.__save_seq[key]

    def set_value(self, key, value):
        """modify the value of an existed item.

        Parameters
        ----------
        key : string
            key in the State, must a existed key

        value : object,
            value to cover the original value
        """
        if key not in self.__save_seq.keys():
            raise KeyError('key must be an existed one in State')
        self.__save_seq[key] = copy.deepcopy(value)

    def __repr__(self):
        return self.__save_seq.__repr__()

if __name__ == '__main__':
    st = State(1)
    print(st)
