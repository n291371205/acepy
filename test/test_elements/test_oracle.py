'''
Test functions of oracle modules
'''

from __future__ import division
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels, type_of_target
from data_process.al_split import *
from utils.tools import check_index_multilabel, integrate_multilabel_index, flattern_multilabel_index


