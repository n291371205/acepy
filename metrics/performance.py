"""
Pre-defined Performance
Implement classical methods
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    'accuracy'
]


def check_consistent_length(*arrays):
    """
        Check that all arrays have consistent first dimensions.
    """
    lengths = [X for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def type_of_target(y):
    """Determine the type of data indicated by the target.
    """
    if (len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        return 'multiclass' + suffix  # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
    else:
        return 'binary'  # [1, 2] or [["a"], ["b"]]


def _check_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same classification task

    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.

    Column vectors are squeezed to 1d, while multilabel formats are returned
    as CSR sparse label indicators.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``

    y_true : array or indicator matrix

    y_pred : array or indicator matrix
    """
    check_consistent_length(y_true, y_pred)
    type_true = type_of_target(y_true)
    type_pred = type_of_target(y_pred)

    y_type = set([type_true, type_pred])
    if y_type == set(["binary", "multiclass"]):
        y_type = set(["multiclass"])

    if len(y_type) > 1:
        raise ValueError("Classification metrics can't handle a mix of {0} "
                         "and {1} targets".format(type_true, type_pred))

    # We can't have more than one value on y_type => The set is no more needed
    y_type = y_type.pop()

    # No metrics support "multiclass-multioutput" format
    if (y_type not in ["binary", "multiclass", "multilabel-indicator"]):
        raise ValueError("{0} is not supported".format(y_type))

    if y_type in ["binary", "multiclass"]:
        y_true = column_or_1d(y_true)
        y_pred = column_or_1d(y_pred)
        if y_type == "binary":
            unique_values = np.union1d(y_true, y_pred)
            if len(unique_values) > 2:
                y_type = "multiclass"

    if y_type.startswith('multilabel'):
        y_true = csr_matrix(y_true)
        y_pred = csr_matrix(y_pred)
        y_type = 'multilabel-indicator'

    return y_type, y_true, y_pred


def accuracy_score(y_true, y_pred, sample_weight=None):
    """Accuracy classification score.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    score : float
    """

    # y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    # check_consistent_length(y_true, y_pred, sample_weight)
    if y_type.startswith('multilabel'):
        differing_labels = np.diff(csr_matrix(y_true - y_pred).indptr)
        score = differing_labels == 0
    else:
        score = y_true == y_pred
    return np.average(score, weights=sample_weight)


def get_tps_fps_thresholds(y_true, y_score, pos_label=None, sample_weight=None):
    assert(np.shape(y_score) == np.shape(y_true))
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    thresholds = np.array(y_score)[desc_score_indices]
    y_true = np.array(y_true)[desc_score_indices]
    y_score = np.array(y_score)[desc_score_indices]

    if sample_weight is not None:
        weight = np.array(sample_weight)[desc_score_indices]
    else:
        weight = 1.    
    tps = []
    fps = []
    for threshold in thresholds:
        y_prob = [1 if i >= threshold else 0 for i in y_score]
        result = [i == j for i, j in zip(y_true, y_prob)]
        postive = [i == 1 for i in y_prob]
        tp = [i and j for i, j in zip(result, postive)]
        fp = [(not i) and j for i, j in zip(result, postive)]
        tps.append(tp.count(True))
        fps.append(fp.count(True))
    return np.array(tps), np.array(fps), np.array(thresholds)


def roc_curve(y_true, y_score, pos_label=None, sample_weight=None):
    fps, tps, thresholds = get_tps_fps_thresholds(y_true, y_score,  
                        pos_label=pos_label, sample_weight=sample_weight)

    if np.array(tps).size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        raise ValueError("No negative samples in y_true,false positive value should be meaningless")
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        raise ValueError("No positive samples in y_true,true positive value should be meaningless")
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds


def roc_auc_score(x, y):
    assert(np.shape(x) == np.shape(y))
    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    area = np.trapz(y, x)
    return area
    

def average_precision_score(y_true, y_score, average="macro", pos_label=1,
                            sample_weight=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score,
                                    pos_label=pos_label, sample_weight=sample_weight)
    # order = np.lexsort((precision, recall))
    # x, y = recall[order], precision[order]
    area = np.trapz(precision, recall)
    return area


def precision_recall_curve(y_true, y_score, pos_label=None,
                           sample_weight=None):
    fps, tps, thresholds = get_tps_fps_thresholds(y_true, y_score,
                                    pos_label=pos_label, sample_weight=sample_weight)

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def hinge_loss(y_true, pred_decision, labels=None, sample_weight=None):
    """Average hinge loss (non-regularized)

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True target, consisting of integers of two values. The positive label
        must be greater than the negative label.

    pred_decision : array, shape = [n_samples] or [n_samples, n_classes]
        Predicted decisions, as output by decision_function (floats).

    labels : array, optional, default None
        Contains all the labels for the problem. Used in multiclass hinge loss.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    loss : float
    """
    assert (np.shape(y_true) == np.shape(pred_decision))
    y_true_unique = np.un`                                                                                                                                                                                                                                                                          ique(y_true)
    if y_true_unique.size > 2:
        if (labels is None and pred_decision.ndim > 1 and
                (np.size(y_true_unique) != pred_decision.shape[1])):
            raise ValueError("Please include all labels in y_true "
                             "or pass labels as third argument")
        if labels is None:
            labels = y_true_unique
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        mask = np.ones_like(pred_decision, dtype=bool)
        mask[np.arange(y_true.shape[0]), y_true] = False
        margin = pred_decision[~mask]
        margin -= np.max(pred_decision[mask].reshape(y_true.shape[0], -1),
                         axis=1)

    else:
        # Handles binary class case
        # this code assumes that positive and negative labels
        # are encoded as +1 and -1 respectively
        pred_decision = np.ravel(pred_decision)

        lbin = LabelBinarizer(neg_label=-1)
        y_true = lbin.fit_transform(y_true)[:, 0]

        try:
            margin = y_true * pred_decision
        except TypeError:
            raise TypeError("pred_decision should be an array of floats.")

    losses = 1 - margin
    # The hinge_loss doesn't penalize good enough predictions.
    np.clip(losses, 0, None, out=losses)
    return np.average(losses, weights=sample_weight)


def hamming_loss(y_true, y_pred, labels=None, sample_weight=None):
    """Compute the average Hamming loss.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    labels : array, shape = [n_labels], optional (default=None)
        Integer array of labels. If not provided, labels will be inferred
        from y_true and y_pred.

        .. versionadded:: 0.18

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

        .. versionadded:: 0.18

    Returns
    -------
    loss : float or int,
        Return the average Hamming loss between element of ``y_true`` and
        ``y_pred``.

    """

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    if sample_weight is None:
        weight_average = 1.
    else:
        weight_average = np.mean(sample_weight)

    if y_type.startswith('multilabel'):
        n_differences = count_nonzero(y_true - y_pred,
                                      sample_weight=sample_weight)
        return (n_differences /
                (y_true.shape[0] * len(labels) * weight_average))

    elif y_type in ["binary", "multiclass"]:
        return _weighted_sum(y_true != y_pred, sample_weight, normalize=True)
    else:
        raise ValueError("{0} is not supported".format(y_type))


if __name__ == '__main__':
    # y = np.array([1, 1, 2, 2])
    # scores = np.array([0.1, 0.4, 0.35, 0.8])
    # tps, fps, t = get_tps_fps_thresholds(y, scores, pos_label=2)
    # print('fps is ', fps)
    # print('tps is ', tps)
    # print('thresholds is ', t)

    # fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)

    # print('fpr is ', fpr)
    # print('tpr is ', tpr)
    # print('thresholds is ', thresholds)

    # p, r, th = precision_recall_curve(y, scores, pos_label=2)
    # print('percision is ', p)
    # print('recall is ', r)
    # print('thresholds is ', thresholds)
    # print('auc is ', get_auc(fpr, tpr))

    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    p, r, th = precision_recall_curve(y_true, y_scores)
    print('percision is ', p)
    print('recall is ', r)
    print(average_precision_score(y_true, y_scores))
