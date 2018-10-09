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

def _weighted_sum(sample_score, sample_weight, normalize=False):
    if normalize:
        return np.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()


def accuracy(predict, ground_truth):
    assert(np.shape(predict) == np.shape(ground_truth))
    count = 0.0
    for i in range(len(predict)):
        if predict[i] == ground_truth[i]:
            count = count + 1
    return count/len(predict)



def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    """Accuracy classification score.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    score : float
        If ``normalize == True``, return the fraction of correctly
        classified samples (float), else returns the number of correctly
        classified samples (int).

        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.

    See also

    """

    # Compute accuracy for each possible representation
    # y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    # check_consistent_length(y_true, y_pred, sample_weight)
    if y_type.startswith('multilabel'):
        differing_labels = count_nonzero(y_true - y_pred, axis=1)
        score = differing_labels == 0
    else:
        score = y_true == y_pred

    return _weighted_sum(score, sample_weight, normalize)


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


def get_auc(x, y):
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


def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None):
    """Compute confusion matrix to evaluate the accuracy of a classification

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` but
    predicted to be in group :math:`j`.

    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.

    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    C : array, shape = [n_classes, n_classes]
        Confusion matrix

    Examples
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    In the binary case, we can extract true positives, etc as follows:

    >>> tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    >>> (tn, fp, fn, tp)
    (0, 2, 1, 1)

    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)
        if np.all([l not in y_true for l in labels]):
            raise ValueError("At least one label specified must be in y_true")

    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)

    check_consistent_length(y_true, y_pred, sample_weight)

    n_labels = labels.size
    label_to_ind = dict((y, x) for x, y in enumerate(labels))
    # convert yt, yp into index
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    # also eliminate weights of eliminated items
    sample_weight = sample_weight[ind]

    # Choose the accumulator dtype to always have high precision
    if sample_weight.dtype.kind in {'i', 'u', 'b'}:
        dtype = np.int64
    else:
        dtype = np.float64

    CM = coo_matrix((sample_weight, (y_true, y_pred)),
                    shape=(n_labels, n_labels), dtype=dtype,
                    ).toarray()

    return CM

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
