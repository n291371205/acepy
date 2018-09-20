"""
get_Matrix() #get data mat according to the index
kernel()  # claculate kernel matrix
feature_scale() #normalization
ttest()  #ttest

"""
from __future__ import division
from sklearn.utils.validation import check_array
import numpy as np
import collections
from sklearn.utils.validation import check_X_y


def check_index_multilabel(index):
    """check if the given index is legal."""
    if not isinstance(index, (list, np.ndarray)):
        index = [index]
    datatype = collections.Counter([type(i) for i in index])
    if len(datatype) != 1:
        raise TypeError("Different types found in the given indexes.")
    if not datatype.popitem()[0] == tuple:
        raise TypeError("Each index should be a tuple.")
    return index


def check_matrix(matrix):
    """check if the given matrix is legal."""
    matrix = check_array(matrix, accept_sparse='csr', ensure_2d=False, order='C')
    if matrix.ndim != 2:
        if matrix.ndim == 1 and len(matrix) == 1:
            matrix = matrix.reshape(1, -1)
        else:
            raise TypeError("Matrix should be a 2D array with [n_samples, n_features] or [n_samples, n_classes].")
    return matrix


def get_labelmatrix_in_multilabel(index, label_matrix, unknown_element=0):
    """get data matrix by giving index in multi-label setting.

    Note:
    Each index should be a tuple, with the first element representing instance index.
    e.g.
    queried_index = (1, [3,4])  # 1st instance, 3rd,4t labels
    queried_index = (1, [3])    # 1st instance, 3rd labels
    queried_index = (1, 3)
    queried_index = (1, (3))
    queried_index = (1, (3,4))
    queried_index = (1, )   # query all labels

    Parameters
    ----------
    index: list, np.ndarray or tuple
        if only one index, a tuple is expected.
        Otherwise, it should be a list type with n tuples.

    label_matrix:  array-like
        matrix with [n_samples, n_features] or [n_samples, n_classes].

    unknown_element: object
        value to fill up the unknown part of the matrix_clip.

    Returns
    -------
    Matrix_clip: np.ndarray
        data matrix given index

    index_arr: list
        index of examples correspond to the each row of Matrix_clip
    """
    # check validity
    index = check_index_multilabel(index)
    label_matrix = check_matrix(label_matrix)

    ins_bound = label_matrix.shape[0]
    ele_bound = label_matrix.shape[1]

    index_arr = []  # record if a row is already constructed
    current_rows = 0  # record how many rows have been constructed
    label_indexed = None
    for k in index:
        # k is a tuple with 2 elements
        k_len = len(k)
        if k_len != 1 and k_len != 2:
            raise ValueError(
                "A single index should only have 1 element (example_index, ) to query all labels or"
                "2 elements (example_index, [label_indexes]) to query specific labels. But found %d in %s" %
                (len(k), str(k)))
        example_ind = k[0]
        assert (example_ind < ins_bound)
        if example_ind in index_arr:
            ind_row = index_arr.index(example_ind)
        else:
            index_arr.append(example_ind)
            ind_row = -1  # new row
            current_rows += 1
        if k_len == 1:  # all labels
            label_ind = [i for i in range(ele_bound)]
        else:
            if isinstance(k[1], collections.Iterable):
                label_ind = [i for i in k[1] if 0 <= i < ele_bound]
            else:
                assert (0 <= k[1] < ele_bound)
                label_ind = [k[1]]

        # construct mat
        if ind_row == -1:
            tmp = np.zeros((1, ele_bound))
            tmp[0, label_ind] = label_matrix[example_ind, label_ind]
            if label_indexed is None:
                label_indexed = tmp.copy()
            else:
                label_indexed = np.append(label_indexed, tmp, axis=0)
        else:
            label_indexed[ind_row, label_ind] = label_matrix[example_ind, label_ind]
    return label_indexed, index_arr


def get_Xy_in_multilabel(index, X, y, unknown_element=0):
    """get data matrix by giving index in multi-label setting.

    Note:
    Each index should be a tuple, with the first element representing instance index.
    e.g.
    queried_index = (1, [3,4])  # 1st instance, 3rd,4t labels
    queried_index = (1, [3])    # 1st instance, 3rd labels
    queried_index = (1, 3)
    queried_index = (1, (3))
    queried_index = (1, (3,4))
    queried_index = (1, )   # query all labels

    Parameters
    ----------
    index: list, np.ndarray or tuple
        if only one index, a tuple is expected.
        Otherwise, it should be a list type with n tuples.

    X:  array-like
        array with [n_samples, n_features].

    y:  array-like
        array with [n_samples, n_classes].

    unknown_element: object
        value to fill up the unknown part of the matrix_clip.

    Returns
    -------
    Matrix_clip: np.ndarray
        data matrix given index
    """
    # check validity
    X = check_matrix(X)
    if not len(X) == len(y):
        raise ValueError("Different length of instances and labels found.")

    label_matrix, ins_index = get_labelmatrix_in_multilabel(index, y)


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
    print()
    lm = np.random.randn(2, 4)
    print(lm)
    print(get_labelmatrix_in_multilabel([(0,), (1, 1), (1, 2)], lm))
    print(get_labelmatrix_in_multilabel([(1, (0, 1)), (0, [1, 2]), (1, 2)], lm))
