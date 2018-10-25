"""
Misc functions to be settled
"""

from __future__ import division
from sklearn.utils.validation import check_array
import numpy as np
import collections
import xml.dom.minidom
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, \
    rbf_kernel


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


def infer_label_size_multilabel(index_arr, check_arr=True):
    """Infer the label size from a set of index arr.

    raise if all index are example index only.

    Parameters
    ----------
    index_arr: list or np.ndarray
        index array.

    Returns
    -------
    _label_size: int
        the inferred label size.
    """
    if check_arr:
        index_arr = check_index_multilabel(index_arr)
    data_len = np.array([len(i) for i in index_arr])
    if np.any(data_len == 2):
        label_size = np.max([i[1] for i in index_arr if len(i) == 2]) + 1
    elif np.all(data_len == 1):
        raise Exception(
            "Label_size can not be induced from fully labeled set, label_size must be provided.")
    else:
        raise ValueError(
            "All elements in indexes should be a tuple, with length = 1 (example_index, ) "
            "to query all labels or length = 2 (example_index, [label_indexes]) to query specific labels.")
    return label_size


def flattern_multilabel_index(index_arr, label_size=None, check_arr=True):
    if check_arr:
        index_arr = check_index_multilabel(index_arr)
    if label_size is None:
        label_size = infer_label_size_multilabel(index_arr)
    else:
        assert (label_size > 0)
    decomposed_data = []
    for item in index_arr:
        if len(item) == 1:
            for i in range(label_size):
                decomposed_data.append((item[0], i))
        else:
            if isinstance(item[1], collections.Iterable):
                label_ind = [i for i in item[1] if 0 <= i < label_size]
            else:
                assert (0 <= item[1] < label_size)
                label_ind = [item[1]]
            for j in range(len(label_ind)):
                decomposed_data.append((item[0], label_ind[j]))
    return decomposed_data


def integrate_multilabel_index(index_arr, label_size=None, check_arr=True):
    """ Integrated the indexes of multi-label.

    Parameters
    ----------
    index_arr: list or np.ndarray
        multi-label index array.

    label_size: int, optional (default = None)
        the size of label set. If not provided, an inference is attempted.
        raise if the inference is failed.

    check_arr: bool, optional (default = True)
        whether to check the validity of index array.

    Returns
    -------
    array: list
        the integrated array.
    """
    if check_arr:
        index_arr = check_index_multilabel(index_arr)
    if label_size is None:
        label_size = infer_label_size_multilabel(index_arr)
    else:
        assert (label_size > 0)

    integrated_arr = []
    integrated_dict = {}
    for index in index_arr:
        example_ind = index[0]
        if len(index) == 1:
            integrated_dict[example_ind] = set(range(label_size))
        else:
            # length = 2
            if example_ind in integrated_dict.keys():
                integrated_dict[example_ind].update(
                    set(index[1] if isinstance(index[1], collections.Iterable) else [index[1]]))
            else:
                integrated_dict[example_ind] = set(
                    index[1] if isinstance(index[1], collections.Iterable) else [index[1]])

    for item in integrated_dict.items():
        if len(item[1]) == label_size:
            integrated_arr.append((item[0],))
        else:
            integrated_arr.append((item[0], tuple(item[0])))

    return integrated_arr


def get_labelmatrix_in_multilabel(index, data_matrix, unknown_element=0):
    """get data matrix by giving index in multi-label setting.

    Note:
    Each index should be a tuple, with the first element representing instance index.
    e.g.
    queried_index = (1, [3,4])  # 1st instance, 3rd,4t _labels
    queried_index = (1, [3])    # 1st instance, 3rd _labels
    queried_index = (1, 3)
    queried_index = (1, (3))
    queried_index = (1, (3,4))
    queried_index = (1, )   # query all _labels

    Parameters
    ----------
    index: list, np.ndarray or tuple
        if only one index, a tuple is expected.
        Otherwise, it should be a list type with n tuples.

    data_matrix:  array-like
        matrix with [n_samples, n_features] or [n_samples, n_classes].

    unknown_element: object
        value to fill up the unknown part of the matrix_clip.

    Returns
    -------
    Matrix_clip: np.ndarray
        data matrix given index

    index_arr: list
        index of _examples correspond to the each row of Matrix_clip
    """
    # check validity
    index = check_index_multilabel(index)
    data_matrix = check_matrix(data_matrix)

    ins_bound = data_matrix.shape[0]
    ele_bound = data_matrix.shape[1]

    index_arr = []  # record if a row is already constructed
    current_rows = 0  # record how many rows have been constructed
    label_indexed = None
    for k in index:
        # k is a tuple with 2 elements
        k_len = len(k)
        if k_len != 1 and k_len != 2:
            raise ValueError(
                "A single index should only have 1 element (example_index, ) to query all _labels or"
                "2 elements (example_index, [label_indexes]) to query specific _labels. But found %d in %s" %
                (len(k), str(k)))
        example_ind = k[0]
        assert (example_ind < ins_bound)
        if example_ind in index_arr:
            ind_row = index_arr.index(example_ind)
        else:
            index_arr.append(example_ind)
            ind_row = -1  # new row
            current_rows += 1
        if k_len == 1:  # all _labels
            label_ind = [i for i in range(ele_bound)]
        else:
            if isinstance(k[1], collections.Iterable):
                label_ind = [i for i in k[1] if 0 <= i < ele_bound]
            else:
                assert (0 <= k[1] < ele_bound)
                label_ind = [k[1]]

        # construct mat
        if ind_row == -1:
            tmp = np.zeros((1, ele_bound)) + unknown_element
            tmp[0, label_ind] = data_matrix[example_ind, label_ind]
            if label_indexed is None:
                label_indexed = tmp.copy()
            else:
                label_indexed = np.append(label_indexed, tmp, axis=0)
        else:
            label_indexed[ind_row, label_ind] = data_matrix[example_ind, label_ind]
    return label_indexed, index_arr


def get_Xy_in_multilabel(index, X, y, unknown_element=0):
    """get data matrix by giving index in multi-label setting.

    Note:
    Each index should be a tuple, with the first element representing instance index.
    e.g.
    queried_index = (1, [3,4])  # 1st instance, 3rd,4t _labels
    queried_index = (1, [3])    # 1st instance, 3rd _labels
    queried_index = (1, 3)
    queried_index = (1, (3))
    queried_index = (1, (3,4))
    queried_index = (1, )   # query all _labels

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
        raise ValueError("Different length of instances and _labels found.")

    label_matrix, ins_index = get_labelmatrix_in_multilabel(index, y)
    return X[ins_index, :], label_matrix


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


def nlargestarg(a, n):
    """Return n largest values' indexes of the given array a.

    Parameters
    ----------
    a: array
        Data array.

    n: int
        The number of returned args.

    Returns
    -------
    nlargestarg: list
        The n largest args in array a.
    """
    assert(_is_arraylike(a))
    assert (n > 0)
    argret = np.argsort(a)
    # ascend
    return argret[argret.size - n:]


def nsmallestarg(a, n):
    """Return n smallest values' indexes of the given array a.

    Parameters
    ----------
    a: array
        Data array.

    n: int
        The number of returned args.

    Returns
    -------
    nlargestarg: list
        The n smallest args in array a.
    """
    assert(_is_arraylike(a))
    assert (n > 0)
    argret = np.argsort(a)
    # ascend
    return argret[0:n]


def calc_kernel_matrix(X, kernel, **kwargs):
    """calculate kernel matrix between X and X.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', callable}, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', or a callable.
        If a callable is given it is used to pre-compute the kernel matrix
        from data matrices; that matrix should be an array of shape
        ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default=1.)
        Kernel coefficient for 'rbf', 'poly'.

    coef0 : float, optional (default=1.)
        Independent term in kernel function.
        It is only significant in 'poly'.

    Returns
    -------

    """
    if kernel == 'rbf':
        K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma', 1.))
    elif kernel == 'poly':
        K = polynomial_kernel(X=X,
                              Y=X,
                              coef0=kwargs.pop('coef0', 1),
                              degree=kwargs.pop('degree', 3),
                              gamma=kwargs.pop('gamma', 1.))
    elif kernel == 'linear':
        K = linear_kernel(X=X, Y=X)
    elif hasattr(kernel, '__call__'):
        K = kernel(X=np.array(X), Y=np.array(X))
    else:
        raise NotImplementedError

    return K

def check_one_to_one_correspondence(*args):
    """Check if the parameters are one-to-one correspondence.

    Parameters
    ----------
    args: object
        The parameters to test.

    Returns
    -------
    result: int
        Whether the parameters are one-to-one correspondence.
        1 : yes
        0 : no
        -1: some parameters have the length 1.
    """
    first_not_none = True
    result = True
    for item in args:
        # only check not none object
        if item is not None:
            if first_not_none:
                # record item type
                first_not_none = False
                if_array = isinstance(item, (list, np.ndarray))
                if if_array:
                    itemlen = len(item)
                else:
                    itemlen = 1
            else:
                if isinstance(item, (list, np.ndarray)):
                    if len(item) != itemlen:
                        return False
                else:
                    if itemlen != 1:
                        return False
    return True


def unpack(*args):
    """Unpack the list with only one element."""
    ret_args = []
    for arg in args:
        if isinstance(arg, (list, np.ndarray)):
            if len(arg) == 1:
                ret_args.append(arg[0])
            else:
                ret_args.append(arg)
        else:
            ret_args.append(arg)
    return tuple(ret_args)


# Implement image dataset related function.

def read_voc_like(xml_path, filename):
    """Read annotations of voc like image dataset. The annotation file is .xml"""
    xml_filename = filename.split('.')[0] + '.xml'
    xml_file = xml_path + '\\' + xml_filename
    dom = xml.dom.minidom.parse(xml_file)
    root = dom.documentElement
    element_dict = dict()

    # gathering elements
    bndboxes = root.getElementsByTagName('bndbox')
    for bndbox in bndboxes:
        xmin = bndbox.getElementsByTagName('xmin')[0]
        ymin = bndbox.getElementsByTagName('ymin')[0]
        xmax = bndbox.getElementsByTagName('xmax')[0]
        ymax = bndbox.getElementsByTagName('ymax')[0]

    
# use coco api to implement
def read_coco():
    """Read annotations of coco like image dataset. The annotation file is .json.

    Returns
    -------

    """
    pass


if __name__ == '__main__':
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(get_gaussian_kernel_mat(a))
    print()
    lm = np.random.randn(2, 4)
    print(lm)
    print(get_labelmatrix_in_multilabel([(0,), (1, 1), (1, 2)], lm))
    print(get_labelmatrix_in_multilabel([(1, (0, 1)), (0, [1, 2]), (1, 2)], lm))
