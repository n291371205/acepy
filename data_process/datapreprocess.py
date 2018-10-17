'''
'''
from __future__ import division
import numpy as np
from scipy import sparse
__all__ = [
    'minmax_scale'
]


def minmax_scale(X, feature_range=(0, 1)):
    """Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, i.e. between
    zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    """

    if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))
    
    if sparse.issparse(X):
            raise TypeError("MinMaxScaler does no support sparse input. "
                            "You may consider to use MaxAbsScaler instead.")

    data_min = np.nanmin(X, axis=0)
    data_max = np.nanmax(X, axis=0)
    data_dis = data_max - data_min
    x_std = (X - data_min) / data_dis
    x_scaled = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return x_scaled


if __name__ == '__main__':
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    d=minmax_scale(np.array(data),(0,10))
    print(d)
