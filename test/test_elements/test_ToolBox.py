from __future__ import division
import numpy as np
import pytest
from experiment_saver.al_experiment import ToolBox
from sklearn.datasets import load_iris
from utils.tools import check_one_to_one_correspondence
X, y = load_iris(return_X_y=True)

split_count = 5
tb = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None)

def test_init():
    with pytest.raises(ValueError):
        ToolBox(y[0:5], X, query_type='AllLabels', saving_path=None)
    with pytest.raises(NotImplementedError):
        ToolBox(y, X, query_type='AllLabel', saving_path=None)
    with pytest.raises(Exception):
        ToolBox(y, x=None, query_type='Features', saving_path=None)
    with pytest.raises(TypeError):
        ToolBox(X=X, y=y, query_type='AllLabels', saving_path='asdfasf')
    

def test_al_split():
    train_idx, test_idx, Lind, Uind = tb.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=split_count)
    assert (check_one_to_one_correspondence(train_idx, test_idx, Lind, Uind))


def test_get_split(): 
    with pytest.raises(Exception):
        t = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None)
        a, b, c, d = t.get_split()
    tb.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=split_count)
    a, b, c, d = tb.get_split()
    assert (check_one_to_one_correspondence(a, b, c, d))


# def test_aceThreading():


# def test_load():


# def test_save():

