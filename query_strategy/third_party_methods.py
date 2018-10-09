"""
Pre-defined query strategy from third party toolbox.
Included some selected state-of-the-art methods

Implement:
-- libact --

Active Learning by QUerying Informative and Representative Examples (QUIRE)
-- S.-J. Huang, R. Jin, and Z.-H. Zhou. Active learning by querying
   informative and representative examples.

-- active learning --

Graph Density
-- https://www.mpi-inf.mpg.de/fileadmin/inf/d2/Research_projects_files/EbertCVPR2012.pdf




Planning:

Hierarchical Sampling for Active Learning (HS)
-- Sanjoy Dasgupta and Daniel Hsu. "Hierarchical sampling for active
   learning." ICML 2008.

Active Learning by Learning
-- Hsu & Lin 2015, Active Learning by Learning.
https://www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings
import copy

import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
import utils.base
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, \
    rbf_kernel
import utils.tools


class QueryInstanceQUIRE(utils.base.BaseQueryStrategy):
    """Querying Informative and Representative Examples (QUIRE)

    Query the most informative and representative examples where the metrics
    measuring and combining are done using min-max approach.

    Parameters
    ----------
    X: 2D array
        data matrix

    y: array-like
        label matrix

    lambda: float, optional (default=1.0)
        A regularization parameter used in the regularization learning
        framework.

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


    Attributes
    ----------

    Examples
    --------


    References
    ----------
    .. [1] S.-J. Huang, R. Jin, and Z.-H. Zhou. Active learning by querying
           informative and representative examples.
    """

    def __init__(self, X, y, **kwargs):
        # K: kernel matrix
        #

        self.y = np.array(y)
        self.lmbda = kwargs.pop('lambda', 1.)
        self.kernel = kwargs.pop('kernel', 'rbf')
        if self.kernel == 'rbf':
            self.K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma', 1.))
        elif self.kernel == 'poly':
            self.K = polynomial_kernel(X=X,
                                       Y=X,
                                       coef0=kwargs.pop('coef0', 1),
                                       degree=kwargs.pop('degree', 3),
                                       gamma=kwargs.pop('gamma', 1.))
        elif self.kernel == 'linear':
            self.K = linear_kernel(X=X, Y=X)
        elif hasattr(self.kernel, '__call__'):
            self.K = self.kernel(X=np.array(X), Y=np.array(X))
        else:
            raise NotImplementedError

        if not isinstance(self.K, np.ndarray):
            raise TypeError('K should be an ndarray')
        if self.K.shape != (len(X), len(X)):
            raise ValueError(
                'kernel should have size (%d, %d)' % (len(X), len(X)))
        self.L = np.linalg.inv(self.K + self.lmbda * np.eye(len(X)))

    def select(self, label_index, unlabel_index, batch_size=1):
        """

        Parameters
        ----------
        label_index: array or set like
            index of label set

        unlabel_index: array or set like
            index of unlabel set

        Returns
        -------

        """
        L = self.L
        Lindex = list(label_index)
        Uindex = list(unlabel_index)
        query_index = -1
        min_eva = np.inf
        # y_labeled = np.array([label for label in self.y if label is not None])
        y_labeled = self.y[Lindex]
        det_Laa = np.linalg.det(L[np.ix_(Uindex, Uindex)])
        # efficient computation of inv(Laa)
        M3 = np.dot(self.K[np.ix_(Uindex, Lindex)],
                    np.linalg.inv(self.lmbda * np.eye(len(Lindex))))
        M2 = np.dot(M3, self.K[np.ix_(Lindex, Uindex)])
        M1 = self.lmbda * np.eye(len(Uindex)) + self.K[np.ix_(Uindex, Uindex)]
        inv_Laa = M1 - M2
        iList = list(range(len(Uindex)))
        if len(iList) == 1:
            return Uindex[0]
        for i, each_index in enumerate(Uindex):
            # go through all unlabeled instances and compute their evaluation
            # values one by one
            Uindex_r = Uindex[:]
            Uindex_r.remove(each_index)
            iList_r = iList[:]
            iList_r.remove(i)
            inv_Luu = inv_Laa[np.ix_(iList_r, iList_r)] - 1 / inv_Laa[i, i] * \
                                                          np.dot(inv_Laa[iList_r, i], inv_Laa[iList_r, i].T)
            tmp = np.dot(
                L[each_index][Lindex] -
                np.dot(
                    np.dot(
                        L[each_index][Uindex_r],
                        inv_Luu
                    ),
                    L[np.ix_(Uindex_r, Lindex)]
                ),
                y_labeled,
            )
            eva = L[each_index][each_index] - \
                  det_Laa / L[each_index][each_index] + 2 * np.abs(tmp)

            if eva < min_eva:
                query_index = each_index
                min_eva = eva
        return query_index


class QueryInstanceGraphDensity(utils.base.BaseQueryStrategy):
    """Diversity promoting sampling method that uses graph density to determine
    most representative points.
    """

    def __init__(self, X, y):
        self.name = 'graph_density'
        super(QueryInstanceGraphDensity, self).__init__(X, y)
        # Set gamma for gaussian kernel to be equal to 1/n_features
        self.gamma = 1. / self.X.shape[1]
        self.flat_X = X
        self.compute_graph_density()

    def compute_graph_density(self, n_neighbor=10):
        # kneighbors graph is constructed using k=10
        connect = kneighbors_graph(self.flat_X, n_neighbor, p=1)
        # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
        # another point, make it vice versa
        neighbors = connect.nonzero()
        inds = zip(neighbors[0], neighbors[1])
        # Graph edges are weighted by applying gaussian kernel to manhattan dist.
        # By default, gamma for rbf kernel is equal to 1/n_features but may
        # get better results if gamma is tuned.
        for entry in inds:
            i = entry[0]
            j = entry[1]
            distance = pairwise_distances(self.flat_X[[i]], self.flat_X[[j]], metric='manhattan')
            distance = distance[0, 0]
            weight = np.exp(-distance * self.gamma)
            connect[i, j] = weight
            connect[j, i] = weight
        self.connect = connect
        # Define graph density for an observation to be sum of weights for all
        # edges to the node representing the datapoint.  Normalize sum weights
        # by total number of neighbors.
        self.graph_density = np.zeros(self.X.shape[0])
        for i in np.arange(self.X.shape[0]):
            self.graph_density[i] = connect[i, :].sum() / (connect[i, :] > 0).sum()
        self.starting_density = copy.deepcopy(self.graph_density)

    def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
        # If a neighbor has already been sampled, reduce the graph density
        # for its direct neighbors to promote diversity.
        batch = set()
        self.graph_density[label_index] = min(self.graph_density) - 1
        while len(batch) < batch_size:
            selected = np.argmax(self.graph_density)

            if selected not in unlabel_index:
                continue

            neighbors = (self.connect[selected, :] > 0).nonzero()[1]
            self.graph_density[neighbors] = self.graph_density[neighbors] - self.graph_density[selected]
            batch.add(selected)
            self.graph_density[label_index] = min(self.graph_density) - 1
            self.graph_density[list(batch)] = min(self.graph_density) - 1
        return list(batch)

    def to_dict(self):
        output = {}
        output['connectivity'] = self.connect
        output['graph_density'] = self.starting_density
        return output


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from data_process.al_split import split

    X, y = load_iris(return_X_y=True)
    Train_idx, Test_idx, U_pool, L_pool = split(X=X, y=y, test_ratio=0.3, initial_label_rate=0.2, split_count=5)

    # test quire
    qs = QueryInstanceQUIRE(X, y)
    select_index = qs.select(label_index=L_pool[0], unlabel_index=U_pool[0])
    print(U_pool[0])
    print(select_index)

    # test graph density
    qs = QueryInstanceGraphDensity(X, y)
    select_index = qs.select(label_index=L_pool[0], unlabel_index=U_pool[0])
    print(select_index)