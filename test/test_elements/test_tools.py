from __future__ import division
import pytest
import numpy as np
import random
from utils.tools import check_index_multilabel,check_one_to_one_correspondence


def test_check_index_multilabel():

    assert [(1, 2, 3, 4, 5)] == check_index_multilabel((1, 2, 3, 4, 5))
    print(check_index_multilabel((1, 2, 0.5, True)))
    with pytest.raises(TypeError):
        check_index_multilabel([0])
    with pytest.raises(TypeError):
        check_index_multilabel((1, 2, 0.5, True))


def test_check_one_to_one_correspondence():
    assert check_one_to_one_correspondence([i for i in range(10)], [i for i in range(10)], [i for i in range(10)])
    assert not check_one_to_one_correspondence([i for i in range(10)], [i for i in range(9)], [i for i in range(10)])
    assert check_one_to_one_correspondence([i for i in range(10)], [i for i in range(10)], [i for i in range(10)])
    a = np.array([i for i in range(50)])
    assert not check_one_to_one_correspondence(np.reshape(a,(5,10)), np.reshape(a,(10,5)))


def test_