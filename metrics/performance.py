"""
Pre-defined Performance
Implement classical methods
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division
import numpy as np

__all__ = [
    'accuracy'
]


def accuracy(predict, ground_truth):
    assert (np.shape(predict) == np.shape(ground_truth))
    count = 0.0
    for i in range(len(predict)):
        if predict[i] == ground_truth[i]:
            count = count + 1
    return count / len(predict)
