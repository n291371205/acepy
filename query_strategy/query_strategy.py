"""
Pre-defined query strategy. Implement classical
methods for various situation
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division

import collections

import numpy as np
from sklearn.utils.validation import check_X_y

import utils.base
import utils.tools


class QueryInstanceUncertainty(utils.base.BaseQueryStrategy):
    """
    Uncertainty query strategy in instance query
    Implement binary, multiclass / margin-based, entropy-based etc

    should set the parameters and global const in __init__
    implement query function with data matrix
    should return element in unlabel_index set
    """

    def __init__(self, X=None, y=None, measure='entropy', scenario='pool'):
        """initializing

        Parameters
        ----------
        X: 2D array
            data matrix

        y: array-like
            label matrix

        measure: string, optional (default='entropy')
            measurement to calculate uncertainty, should be one of
            ['least_confident', 'margin', 'entropy', 'distance_to_margin']
            --'least_confident' x* = argmax 1-P(y_hat|x) ,where y_hat = argmax P(yi|x)
            --'margin' x* = argmax P(y_hat1|x) - P(y_hat2|x), where y_hat1 and y_hat2 are the first and second
                most probable class labels under the model, respectively.
            --'entropy' x* = argmax -sum(P(yi|x)logP(yi|x))
            --'distance_to_margin' Only available in binary classification, x* = argmin |f(x)|

        scenario: string, optional (default='pool')
            should be one of ['pool', 'stream', 'membership']

            note that, the 'least_confident', 'margin', 'entropy' needs the probability output of the model.
            For the 'distance_to_margin',should provide the value of f(x)
        """
        if measure not in ['least_confident', 'margin', 'entropy', 'distance_to_margin']:
            raise ValueError("measure must in ['least_confident', 'margin', 'entropy', 'distance_to_margin']")
        self.measure = measure
        if scenario not in ['pool', 'stream', 'membership']:
            raise ValueError("scenario must in ['pool', 'stream', 'membership']")
        self.scenario = scenario
        if X is not None and y is not None:
            self.X, self.y = check_X_y(X, y, accept_sparse='csc', multi_output=True)
        else:
            # check validity and calculate shape etc.
            self.X = X
            self.y = y

    def select(self, unlabel_index, model, batch_size=1):
        """Select index in unlabel_index to query

        Parameters
        ----------
        unlabel_index: array or set like
            index of unlabel set

        model : object
            current classification model

        batch_size: int
            batch size of AL

        Returns
        -------
        selected_idx: array-like
            queried keys, keys are in unlabel_index
        """
        # if not isinstance(unlabel_index, (BaseCollection,set,list,np.ndarray)) :
        #     raise TypeError('unlabel_index should be a DataCollection, Set, List Type')
        # if not isinstance(label_index,(BaseCollection,set,list,np.ndarray)) :
        #     raise TypeError('unlabel_index should be a DataCollection, Set, List Type')
        # assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        if len(unlabel_index) <= batch_size:
            return np.array([i for i in unlabel_index])
        # assert(isinstance(label_index,collections.Iterable))

        # get unlabel_x
        if self.X is None:
            raise ValueError('Data matrix is not provided, use select_by_prediction_mat() instead.')
        if not isinstance(unlabel_index, np.ndarray):
            unlabel_index = np.array([i for i in unlabel_index])
        unlabel_x = self.X[unlabel_index, :]
        ##################################

        if self.measure == 'distance_to_margin':
            if not hasattr(model, 'predict_value'):
                raise TypeError('model object must implement predict_value methods in distance_to_margin measure.')
            pv = model.predict_value(unlabel_x)
            spv = np.shape(pv)
            assert (len(spv) in [1, 2])
            if len(spv) == 2:
                if spv[1] != 1:
                    raise ValueError('1d or 2d with 1 column array is expected, but received: \n%s' % str(pv))
                else:
                    pv = np.array(pv).flatten()
            argpv = np.argsort(pv)
            return unlabel_index[argpv[0:batch_size]]

        pv, _ = self.__get_proba_pred(unlabel_x, model)
        return self.select_by_prediction_mat(unlabel_index=unlabel_index, predict=pv,
                                             batch_size=batch_size)

    def select_by_prediction_mat(self, unlabel_index, predict, batch_size=1):
        """Select index in unlabel_index to query

        Parameters
        ----------
        unlabel_index: array-like
            index of unlabel set

        predict : array-like
            prediction matrix, shape like [n_samples, n_classes] if probability prediction is needed
            or [n_samples] only if distance_to_margin

        batch_size: int
            batch size of AL

        Returns
        -------
        selected_idx: array-like
            queried keys, keys are in unlabel_index
        """
        # if not isinstance(unlabel_index, (BaseCollection,set,list,np.ndarray)) :
        #     raise TypeError('unlabel_index should be a DataCollection, Set, List Type')
        # if not isinstance(label_index,(BaseCollection,set,list,np.ndarray)) :
        #     raise TypeError('unlabel_index should be a DataCollection, Set, List Type')
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (batch_size > 0)
        if len(unlabel_index) <= batch_size:
            return np.array([i for i in unlabel_index])

        pv = predict
        spv = np.shape(pv)
        # validity check here

        if self.measure == 'distance_to_margin':
            assert (len(spv) in [1, 2])
            if len(spv) == 2:
                if spv[1] != 1:
                    raise ValueError('1d or 2d with 1 column array is expected, but received: \n%s' % str(pv))
                else:
                    pv = np.array(pv).flatten()
            argpv = np.argsort(pv)
            return unlabel_index[argpv[0:batch_size]]

        if self.measure == 'entropy':
            # calc entropy
            pv[pv <= 0] = 0.000001
            entro = [-np.sum(vec * np.log(vec)) for vec in pv]
            assert (len(np.shape(entro)) == 1)
            argentro = np.argsort(entro)
            # descend
            return unlabel_index[argentro[argentro.size - batch_size:]]

        if self.measure == 'margin':
            # calc margin
            pat = np.partition(pv, (spv[1] - 2, spv[1] - 1), axis=1)
            pat = pat[:, spv[1] - 2] - pat[:, spv[1] - 1]
            argret = np.argsort(pat)
            # descend
            return unlabel_index[argret[argret.size - batch_size:]]

        if self.measure == 'least_confident':
            # calc least_confident
            pat = np.partition(pv, spv[1] - 1, axis=1)
            pat = 1 - pat[:, spv[1] - 1]
            argret = np.argsort(pat)
            # descend
            return unlabel_index[argret[argret.size - batch_size:]]

    def __get_proba_pred(self, unlabel_x, model):
        """
        check the model object and get prediction

        Parameters
        ----------
        unlabel_x: array
            data matrix

        model: object
            Model object which has the attribute predict_proba

        Returns
        -------
        pv: np.ndarray
            probability predictions matrix

        spv: tuple
            shape of pv
        """
        if not hasattr(model, 'predict_proba'):
            raise TypeError('model object must implement predict_proba methods in %s measure.' % self.measure)
        pv = model.predict_proba(unlabel_x)
        if not isinstance(pv, np.ndarray):
            pv = np.asarray(pv)
        spv = np.shape(pv)
        if len(spv) != 2 or spv[1] == 1:
            raise ValueError('2d array with [n_samples, n_class] is expected, but received: \n%s' % str(pv))
        return pv, spv

    @classmethod
    def calc_entropy(cls, predict_proba):
        """
        Calc the entropy for each instance.

        Parameters
        ----------
        predict_proba: array-like, [n_samples, n_class]
            probability prediction for each instance

        Returns
        -------
        entropy: list
            1d array, entropy for each instance
        """
        if not isinstance(predict_proba, np.ndarray):
            pv = np.asarray(predict_proba)
        spv = np.shape(pv)
        if len(spv) != 2 or spv[1] == 1:
            raise ValueError('2d array with the shape of [n_samples, n_class]'
                             ' is expected, but received: \n%s' % str(pv))
        # calc entropy
        entropy = [-np.sum(vec * np.log(vec)) for vec in pv]
        return entropy


class QueryRandom(utils.base.BaseQueryStrategy):
    """
    Randomly sample a batch of _indexes.
    """

    def __init__(self, scenario='pool'):
        """

        Parameters
        ----------
        scenario: string, optional (default='pool')
            should be one of ['pool', 'stream', 'membership']
        """
        if scenario not in ['pool', 'stream', 'membership']:
            raise ValueError("scenario must in ['pool', 'stream', 'membership']")
        self.scenario = scenario

    def select(self, unlabel_index, batch_size=1):
        """

        Parameters
        ----------
        unlabel_index: array-like
            the container should be an array-like data structure. If
            other types are given, a transform is attempted.

        Returns
        -------

        """
        if len(unlabel_index) <= batch_size:
            return np.array([i for i in unlabel_index])
        perm = utils.tools.randperm(len(unlabel_index) - 1, batch_size)
        tpl = list(unlabel_index.index)
        return [tpl[i] for i in perm]


class QueryInstanceQBC(utils.base.BaseQueryStrategy):
    """
    query-by-committee (QBC) algorithm (Seung et al., 1992) is minimizing
    the version space, which is the set of hypotheses that are consistent
    with the current labeled training data.
    """

    def __init__(self, X=None, y=None, method='query_by_bagging', disagreement='vote_entropy', scenario='pool'):
        """

        Parameters
        ----------
        X: 2D array
            data matrix

        y: array-like
            label matrix

        method: str
            method name.

        disagreement: str
            method to calculate disagreement. should be one of ['vote_entropy', 'KL_divergence']

        scenario: string, optional (default='pool')
            should be one of ['pool', 'stream', 'membership']
        """
        self.method = method
        if scenario not in ['pool', 'stream', 'membership']:
            raise ValueError("scenario must in ['pool', 'stream', 'membership']")
        self.scenario = scenario
        if X is not None and y is not None:
            self.X, self.y = check_X_y(X, y, accept_sparse='csc', multi_output=True)
        else:
            self.X = X
            self.y = y
        if disagreement in ['vote_entropy', 'KL_divergence']:
            self.disagreement = disagreement
        else:
            raise ValueError("disagreement must be one of ['vote_entropy', 'KL_divergence']")

    def select(self, label_index, unlabel_index, model, batch_size=1):
        """Select index in unlabel_index to query

        Parameters
        ----------
        label_index: array or set like
            index of label set

        unlabel_index: array or set like
            index of unlabel set

        model : object
            current classification model

        batch_size: int
            batch size of AL

        Returns
        -------
        selected_idx: array-like
            queried keys, keys are in unlabel_index
        """
        assert (isinstance(unlabel_index, collections.Iterable))
        if len(unlabel_index) <= batch_size:
            return np.array([i for i in unlabel_index])

        # get unlabel_x
        if self.X is None or self.y is None:
            raise ValueError('Data matrix is not provided, use calc_vote_entropy() instead.')
        if not isinstance(unlabel_index, np.ndarray):
            unlabel_index = np.array([i for i in unlabel_index])
        if not isinstance(label_index, np.ndarray):
            label_index = np.array([i for i in label_index])
        unlabel_x = self.X[unlabel_index, :]
        label_x = self.X[label_index, :]
        if len(np.shape(self.y)) == 1:
            label_y = self.y[label_index]
        else:
            label_y = self.y[label_index, :]
        #####################################

        # bagging
        from sklearn.ensemble import BaggingClassifier
        bagging = BaggingClassifier(model)
        bagging.fit(label_x, label_y)
        est_arr = bagging.estimators_

        # calc score
        if self.disagreement == 'vote_entropy' and self.scenario == 'pool':
            score = self.calc_vote_entropy([estimator.predict(unlabel_x) for estimator in est_arr])
        else:
            score = self.calc_avg_KL_divergence([estimator.predict_proba(unlabel_x) for estimator in est_arr])
        argret = np.argsort(score)
        # descend
        return unlabel_index[argret[argret.size - batch_size:]]

    def _check_committee_results(self, *arys):
        """check the validity of given committee predictions.

        Parameters
        ----------
        arys1, arys2, ... : 2D-array
            Predicted label matrix. Each shape like [n_samples, n_class] or [n_samples]

        Returns
        -------

        """
        arys = arys[0]
        shapes = [np.shape(X) for X in arys if X is not None]
        uniques = np.unique(shapes, axis=0)
        if len(uniques) > 1:
            raise ValueError("Found input variables with inconsistent numbers of"
                             " shapes: %r" % [int(l) for l in shapes])
        committee_size = len(arys)
        input_shape = uniques[0]
        return input_shape, committee_size

    @classmethod
    def calc_vote_entropy(cls, *arys):
        """calculate the vote entropy for measuring the level of disagreement in QBC.

        Parameters
        ----------
        arys1, arys2, ... : 2D-array
            Predicted label matrix. Each shape like [n_samples, n_class] or [n_samples]

        Returns
        -------
        score: array
            score for each instance. Shape like [n_samples]

        References
        -------
        I. Dagan and S. Engelson. Committee-based sampling for training probabilistic
        classifiers. In Proceedings of the International Conference on Machine
        Learning (ICML), pages 150–157. Morgan Kaufmann, 1995.
        """
        arys = arys[0]
        score = []
        input_shape, committee_size = cls()._check_committee_results(arys)
        if len(input_shape) == 2:
            ele_uni = np.unique(arys)
            if not (len(ele_uni) == 2 and 0 in ele_uni and 1 in ele_uni):
                raise ValueError("The predicted label matrix must only contain 0 and 1")
            # calc each instance
            for i in range(input_shape[0]):
                instance_mat = np.array([X[i, :] for X in arys if X is not None])
                voting = np.sum(instance_mat, axis=0)
                tmp = 0
                # calc each label
                for vote in voting:
                    if vote != 0:
                        tmp += vote / len(arys) * np.log(vote / len(arys))
                score.append(-tmp)
        else:
            input_mat = np.array([X for X in arys if X is not None])
            # label_arr = np.unique(input_mat)
            # calc each instance's score
            for i in range(input_shape[0]):
                count_dict = collections.Counter(input_mat[:, i])
                tmp = 0
                for key in count_dict:
                    tmp += count_dict[key] / committee_size * np.log(count_dict[key] / committee_size)
                score.append(-tmp)
        return score

    @classmethod
    def calc_avg_KL_divergence(cls, *arys):
        """calculate the average Kullback-Leibler (KL) divergence for measuring the
        level of disagreement in QBC.

        Parameters
        ----------
        arys1, arys2, ... : 2D-array
            Probabilistic prediction matrix. Each shape like [n_samples, n_class]

        Returns
        -------
        score: array
            score for each instance. Shape like [n_samples]

        References
        -------
        A. McCallum and K. Nigam. Employing EM in pool-based active learning for
        text classification. In Proceedings of the International Conference on Machine
        Learning (ICML), pages 359–367. Morgan Kaufmann, 1998.
        """
        arys = arys[0]
        score = []
        input_shape, committee_size = cls()._check_committee_results(arys)
        if len(input_shape) == 2:
            label_num = input_shape[1]
            # calc kl div for each instance
            for i in range(input_shape[0]):
                instance_mat = np.array([X[i, :] for X in arys if X is not None])
                tmp = 0
                # calc each label
                for lab in range(label_num):
                    committee_consensus = np.sum(instance_mat[:, lab]) / committee_size
                    for committee in range(committee_size):
                        tmp += instance_mat[committee, lab] * np.log(instance_mat[committee, lab] / committee_consensus)
                score.append(tmp)
        else:
            raise ValueError(
                "A 2D probabilistic prediction matrix must be provided, with the shape like [n_samples, n_class]")
        return score


if __name__ == '__main__':
    # Q = QueryInstanceUncertainty(measure='entropy')
    # id = np.arange(20)
    # np.random.shuffle(id)
    # label_idx = id[:5]
    # unlabel_idx = id[5:]
    # print(unlabel_idx)
    # unlabel_x = np.random.rand(15, 5)
    #
    #
    # class Model:
    #     def __init(self):
    #         pass
    #
    #     def predict_proba(self, X):
    #         return np.random.rand(15, 10)
    #
    #
    # m = Model()
    # idx = Q.select(label_index=label_idx, unlabel_index=unlabel_idx,
    #                model=m,
    #                batch_size=1)
    # print(idx)
    pass
