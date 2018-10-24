"""
Test the functions in repository modules
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division
import pytest
import numpy as np
from utils.knowledge_repository import ElementRepository, MatrixRepository
from utils.ace_warnings import *

# initialize
X = np.array(range(100))  # 100 instances in total with 2 features
X = np.tile(X, (2, 1))
X = X.T
# print(X)
y = np.array([0] * 50 + [1] * 50)  # 0 for first 50, 1 for the others.
# print(y)
label_ind = [11, 32, 0, 6, 74]

ele_exa = ElementRepository(labels=y[label_ind], indexes=label_ind, examples=X[label_ind])
ele = ElementRepository(labels=y[label_ind], indexes=label_ind)


def test_ele_raise_no_example():
    with pytest.warns(ValidityWarning, match=r'.*is not in the repository.*'):
        ele.discard(index=7)
    with pytest.raises(ValueError, match=r'Different length of parameters found.*'):
        ele.update_query(labels=[1], indexes=[10, 9])
    with pytest.warns(ValidityWarning, match=r'.*is not in the repository.*'):
        ele.retrieve_by_indexes(indexes=7)
    with pytest.raises(Exception, match=r'This repository do not have the instance information.*'):
        ele.retrieve_by_examples(examples=[4,4])


def test_ele_raise_example():
    with pytest.raises(Exception, match=r'This repository has the instance information.*'):
        ele_exa.update_query(labels=[1], indexes=[9])
    with pytest.warns(ValidityWarning, match=r'.*is not in the repository.*'):
        ele_exa.discard(index=7)
    with pytest.raises(ValueError, match=r'Different length of parameters found.*'):
        ele_exa.update_query(labels=[1], indexes=[10, 9])
    with pytest.warns(ValidityWarning, match=r'.*is not in the repository.*'):
        ele_exa.retrieve_by_indexes(indexes=7)
    with pytest.warns(ValidityWarning, match=r'Example for retrieving is not in the repository.*'):
        ele_exa.retrieve_by_examples(examples=[4,4])


def test_ele_basic_no_example():
    ele.add(select_index=1, label=0)
    assert (1 in ele)
    ele.update_query(labels=[1], indexes=[60])
    ele.update_query(labels=[1], indexes=61)
    assert (60 in ele)
    assert (61 in ele)
    ele.update_query(labels=[1, 1], indexes=[63, 64])
    assert (63 in ele)
    assert (64 in ele)
    ele.discard(index=61)
    assert (61 not in ele)
    _, a = ele.retrieve_by_indexes(60)
    assert (a == 1)
    _, b = ele.retrieve_by_indexes([63, 64])
    assert (np.all(b == [1, 1]))
    print(ele.get_training_data())
    print(ele.full_history())
    """
    (array([], dtype=float64), array([0, 0, 0, 0, 1, 0, 1, 1, 1]), array([11, 32,  0,  6, 74,  1, 60, 63, 64]))
    +----------------+----------------+----------------------+---------------------+
    |       0        |       1        |          2           |        in all       |
    +----------------+----------------+----------------------+---------------------+
    | query_index:60 | query_index:61 | query_index:[63, 64] | number_of_queries:3 |
    |   response:1   |   response:1   |   response:[1, 1]    |        cost:0       |
    |   cost:None    |   cost:None    |      cost:None       |                     |
    +----------------+----------------+----------------------+---------------------+
    """


def test_ele_basic_example():
    ele_exa.add(select_index=1, label=0, example=X[1])
    assert 1 in ele_exa
    x,y = ele_exa.retrieve_by_indexes(1)
    assert x == [1, 1]

# if __name__ == '__main__':
#     test_ele_basic_no_example()
