"""
Test the functions in StateIO class
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division
import pytest
import numpy as np
from experiment_saver.state import State
from experiment_saver.state_io import StateIO
import copy
from sklearn.datasets import load_iris
from experiment_saver.state import State

from query_strategy.query_strategy import (QueryInstanceQBC,
                                           QueryInstanceUncertainty,
                                           QueryRandom,
                                           QureyExpectedErrorReduction)
from query_strategy.third_party_methods import QueryInstanceQUIRE, QueryInstanceGraphDensity
from utils.al_collections import IndexCollection
from experiment_saver.al_experiment import ToolBox


X, y = load_iris(return_X_y=True)
split_count = 5
acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None)

# split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=split_count)
saver = acebox.StateIO(round=0)

