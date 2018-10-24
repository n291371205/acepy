"""
Test the functions in repository modules
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division
import pytest
import numpy as np
from utils.knowledge_repository import ElementRepository, MatrixRepository

# initialize
X = np.array(range(100))  # 100 instances in total with 2 features
X = np.tile(X, (2, 1))
X = X.T
# print(X)
y = np.array([0]*50 + [1]*50)   # 0 for first 50, 1 for the others.
# print(y)
label_ind = [11, 32, 0, 6, 74]

ele_exa = ElementRepository(labels=y[label_ind], indexes=label_ind, examples=X[label_ind])
ele = ElementRepository(labels=y[label_ind], indexes=label_ind)


def test_ele_basic_no_example():
    ele.add(select_index=1, label=0)
    assert(1 in ele)
    ele.update_query(labels=[1], indexes=[60])
    ele.update_query(labels=[1], indexes=61)
    assert(60 in ele)
    assert(61 in ele)
    ele.update_query(labels=[1, 1], indexes=[63, 64])
    assert(63 in ele)
    assert(64 in ele)
    ele.discard(index=61)
    assert(61 not in ele)
    _, a = ele.retrieve_by_indexes(60)
    assert (a == 1)
    _, b = ele.retrieve_by_indexes([63,64])
    assert(np.all(b==[1,1]))
    print(ele.get_training_data())
