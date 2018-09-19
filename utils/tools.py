"""
get_Matrix() #get data mat according to the index
kernel()  # claculate kernel matrix
feature_scale() #normalization
ttest()  #ttest

"""
from __future__ import division
from sklearn.utils.validation import check_array
import numpy as np
from sklearn.utils.validation import check_X_y


def getmat(index, X, y, config=None):
    """get data matrix by giving collection and config object
    collection contains index or examples, config represent
    the active learning scenario.
    if collection contains label information, it will return
    the labels in the collection, other wise, it will return
    the ground truth label according to the index.

    Parameters
    ----------
    index: object or list or set
        Contains a set of index, maybe also contains label information.

    X:  array-like
        data matrix with [n_samples, n_features].

    y:  array-like
        label matrix with [n_samples, n_classes] or [n_samples]

    config: object, optional (default=QueryConfig())
        QueryConfig type, contains the meaning of the index, default
        is querying all-labels of an instance.

    Returns
    -------
    X_get: np.ndarray
        data matrix given index

    y_get: np.ndarray
        label matrix given index
    """
    # if not isinstance(config, QueryConfig):
    #     raise TypeError('config must be a QeuryConfig object.')
    # X, y = check_X_y(X, y, accept_sparse='csc', multi_output=True)
    # if not isinstance(index, (BaseCollection, set, list, np.ndarray)):
    #     raise TypeError('index should be a DataCollection, Set or List Type')
    #
    # # func
    # if isinstance(index, LabelCollection):
    #     # using supervised information in collection.
    #     gt_flag = 0
    # else:
    #     # using ground_truth supervised information
    #     gt_flag = 1
    # element_list = list(index.data)
    # if config.query_type == ['instance', 'all_labels']:
    #     if gt_flag == 1:
    #         X_ret = X[element_list,:]
    #         y_ret = y[element_list,:]
    #         return X_ret, y_ret
    #     else:
    #         X_ret = X[element_list, :]
    #         # concat supervised information matrix
    #         y_ret = np.array([index[i] for i in element_list])
    #         return X_ret, y_ret
    pass


def get_gaussian_kernel_mat(X, sigma=1.0, check_arr=True):
    """Calculate kernel matrix between X and X.

    Parameters
    ----------
    X: np.ndarray
        data matrix with [n_samples, n_features]

    sigma: float, optional (default=1.0)
        the width in gaussian kernel.

    check_arr: bool, optional (default=True)
        whether to check the given feature matrix.

    Returns
    -------
    K: np.ndarray
        Kernel matrix between X and X.
    """
    if check_arr:
        X = check_array(X, accept_sparse='csr', ensure_2d=True, order='C')
    else:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
    n = X.shape[0]
    tmp = np.sum(X ** 2, axis=1).reshape(1, -1)
    return np.exp((-tmp.T.dot(np.ones((1, n))) - np.ones((n, 1)).dot(tmp) + 2 * (X.dot(X.T))) / (2 * (sigma ** 2)))


def randperm(n, k=None):
    """Generate a random array which contains k elements range from (n[0]:n[1])

    Parameters
    ----------
    n: int or tuple
        range from [n[0]:n[1]], include n[0] and n[1].
        if an int is given, then n[0] = 0

    k: int, optional (default=end - start + 1)
        how many numbers will be generated. should not larger than n[1]-n[0]+1,
        default=n[1] - n[0] + 1.

    Returns
    -------
    perm: list
        the generated array.
    """
    if isinstance(n, np.generic):
        n = np.asscalar(n)
    if isinstance(n, tuple):
        if n[0] is not None:
            start = n[0]
        else:
            start = 0
        end = n[1]
    elif isinstance(n, int):
        start = 0
        end = n
    else:
        raise TypeError("n must be tuple or int.")

    if k is None:
        k = end - start + 1
    if not isinstance(k, int):
        raise TypeError("k must be an int.")
    if k > end - start + 1:
        raise ValueError("k should not larger than n[1]-n[0]+1")

    randarr = np.arange(start, end + 1)
    np.random.shuffle(randarr)
    return randarr[0:k]


def _is_arraylike(x):
    """Returns whether the input is array-like"""
    return (hasattr(x, '__len__') or
            hasattr(x, 'shape') or
            hasattr(x, '__array__'))


if __name__ == '__main__':
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(get_gaussian_kernel_mat(a))
