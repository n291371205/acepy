"""
Pre-defined Performance
Implement classical methods
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

__all__ = [
    'accuracy'，
    'auc',
    'get_tps_fps_thresholds',


    'hamming_loss',
    'one_error',
    'coverage_error',
    'label_ranking_loss',
    'label_ranking_average_precision_score'
]


def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def check_consistent_length(*arrays):
    """
        Check that all arrays have consistent first dimensions.
    """
    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def type_of_target(y):
    """Determine the type of data indicated by the target.
    """
    y = np.asarray(y)
    if(len(np.unique(y)) <= 2):
        if(y.ndim >= 2 and len(y[0]) > 1):
            return 'multilabel'
        else:
            return 'binary'
    elif(len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        return 'multiclass'
    return 'unknown'
        

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
    if (y_type not in ["binary", "multiclass", "multilabel"]):
        raise ValueError("{0} is not supported".format(y_type))

    if y_type in ["binary", "multiclass"]:
        #Ravel column or 1d numpy array
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        if y_type == "binary":
            unique_values = np.union1d(y_true, y_pred)
            if len(unique_values) > 2:
                y_type = "multiclass"

    if y_type.startswith('multilabel'):
        y_true = csr_matrix(y_true)
        y_pred = csr_matrix(y_pred)
        y_type = 'multilabel'

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
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred, sample_weight)
    if y_type.startswith('multilabel'):
        differing_labels = np.diff((y_true - y_pred).indptr)
        score = differing_labels == 0
    else:
        score = y_true == y_pred
    return np.average(score, weights=sample_weight)


def auc(x, y):
    assert(np.shape(x) == np.shape(y))
    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    area = np.trapz(y, x)
    return area


def _average_binary_score(binary_metric, y_true, y_score, average,
                          sample_weight=None):
    """Average a binary metric for multilabel classification

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels in binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.

    average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    binary_metric : callable, returns shape [n_classes]
        The binary metric function to use.

    Returns
    -------
    score : float or array of shape [n_classes]
        If not ``None``, average the score, else return the score for each
        classes.

    """
    average_options = (None, 'micro', 'macro', 'weighted', 'samples')
    if average not in average_options:
        raise ValueError('average has to be one of {0}'
                         ''.format(average_options))

    y_type = type_of_target(y_true)
    if y_type not in ("binary", "multilabel"):
        raise ValueError("{0} format is not supported".format(y_type))

    if y_type == "binary":
        return binary_metric(y_true, y_score, sample_weight=sample_weight)

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = check_array(y_true)
    y_score = check_array(y_score)

    not_average_axis = 1
    score_weight = sample_weight
    average_weight = None

    if average == "micro":
        if score_weight is not None:
            score_weight = np.repeat(score_weight, y_true.shape[1])
        y_true = y_true.ravel()
        y_score = y_score.ravel()

    elif average == 'weighted':
        if score_weight is not None:
            average_weight = np.sum(np.multiply(
                y_true, np.reshape(score_weight, (-1, 1))), axis=0)
        else:
            average_weight = np.sum(y_true, axis=0)
        if np.isclose(average_weight.sum(), 0.0):
            return 0

    elif average == 'samples':
        # swap average_weight <-> score_weight
        average_weight = score_weight
        score_weight = None
        not_average_axis = 0

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))

    n_classes = y_score.shape[not_average_axis]
    score = np.zeros((n_classes,))
    for c in range(n_classes):
        y_true_c = y_true.take([c], axis=not_average_axis).ravel()
        y_score_c = y_score.take([c], axis=not_average_axis).ravel()
        score[c] = binary_metric(y_true_c, y_score_c,
                                 sample_weight=score_weight)

    # Average the results
    if average is not None:
        return np.average(score, weights=average_weight)
    else:
        return score


def get_tps_fps_thresholds(y_true, y_score, pos_label=None, sample_weight=None):
    
    check_consistent_length(y_true, y_score, sample_weight)
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
    tps=[]
    fps=[]
    if sample_weight is not None:
        weight = np.array(sample_weight)[desc_score_indices]
    else:
        weight = 1.    
    for threshold in thresholds:
        y_prob = [1 if i >= threshold else 0 for i in y_score]
        result = [i == j for i, j in zip(y_true, y_prob)]
        postive = [i == 1 for i in y_prob]
        tp = [i and j for i, j in zip(result, postive)]
        fp = [(not i) and j for i, j in zip(result, postive)]
        tps.append(tp.count(True))
        fps.append(fp.count(True))
    return np.array(tps), np.array(fps), thresholds


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


def roc_auc_score(y_true, y_score, average="macro", sample_weight=None,
                  max_fpr=None):
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.

    Note: this implementation is restricted to the binary classification task
    or multilabel classification task in label indicator format.

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels or binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers). For binary
        y_true, y_score is supposed to be the score of the class with greater
        label.

    average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    max_fpr : float > 0 and <= 1, optional
        If not ``None``, the standardized partial AUC [3]_ over the range
        [0, max_fpr] is returned.

    Returns
    -------
    auc : float
    """
    def _binary_roc_auc_score(y_true, y_score, sample_weight=None):
        if len(np.unique(y_true)) != 2:
            raise ValueError("Only one class present in y_true. ROC AUC score "
                             "is not defined in that case.")

        fpr, tpr, _ = roc_curve(y_true, y_score,
                                sample_weight=sample_weight)
        if max_fpr is None or max_fpr == 1:
            return auc(fpr, tpr)
        if max_fpr <= 0 or max_fpr > 1:
            raise ValueError("Expected max_frp in range ]0, 1], got: %r"
                             % max_fpr)

        # Add a single point at max_fpr by linear interpolation
        stop = np.searchsorted(fpr, max_fpr, 'right')
        x_interp = [fpr[stop - 1], fpr[stop]]
        y_interp = [tpr[stop - 1], tpr[stop]]
        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
        fpr = np.append(fpr[:stop], max_fpr)
        partial_auc = auc(fpr, tpr)

        # McClish correction: standardize result to be 0.5 if non-discriminant
        # and 1 if maximal
        min_area = 0.5 * max_fpr**2
        max_area = max_fpr
        return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))

    y_type = type_of_target(y_true)
    if y_type == "binary":
        labels = np.unique(y_true)
        y_true = label_binarize(y_true, labels)[:, 0]

    return _average_binary_score(
        _binary_roc_auc_score, y_true, y_score, average,
        sample_weight=sample_weight)


def average_precision_score(y_true, y_score, average="macro", pos_label=1,
                            sample_weight=None):
    """Compute average precision (AP) from prediction scores
    """
    def _binary_uninterpolated_average_precision(
            y_true, y_score, pos_label=1, sample_weight=None):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

    y_type = type_of_target(y_true)
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError("Parameter pos_label is fixed to 1 for "
                         "multilabel-indicator y_true. Do not set "
                         "pos_label or set pos_label to 1.")
    average_precision = partial(_binary_uninterpolated_average_precision,
                                pos_label=pos_label)
    return _average_binary_score(average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)


def precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None,
                                    pos_label=1, average=None,
                                    warn_for=('precision', 'recall',
                                              'f-score'),
                                    sample_weight=None):
    """Compute precision, recall, F-measure and support for each class

    """
    average_options = (None, 'micro', 'macro', 'weighted', 'samples')
    if average not in average_options and average != 'binary':
        raise ValueError('average has to be one of ' +
                         str(average_options))
    if beta <= 0:
        raise ValueError("beta should be >0 in the F-beta score")

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred, sample_weight)
    present_labels = unique_labels(y_true, y_pred)

    if average == 'binary':
        if y_type == 'binary':
            if pos_label not in present_labels:
                if len(present_labels) < 2:
                    # Only negative labels
                    return (0., 0., 0., 0)
                else:
                    raise ValueError("pos_label=%r is not a valid label: %r" %
                                     (pos_label, present_labels))
            labels = [pos_label]
        else:
            raise ValueError("Target is %s but average='binary'. Please "
                             "choose another average setting." % y_type)
    elif pos_label not in (None, 1):
        warnings.warn("Note that pos_label (set to %r) is ignored when "
                      "average != 'binary' (got %r). You may use "
                      "labels=[pos_label] to specify a single positive class."
                      % (pos_label, average), UserWarning)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels,
                                                 assume_unique=True)])

    # Calculate tp_sum, pred_sum, true_sum ###

    if y_type.startswith('multilabel'):
        sum_axis = 1 if average == 'samples' else 0

        # All labels are index integers for multilabel.
        # Select labels:
        if not np.all(labels == present_labels):
            if np.max(labels) > np.max(present_labels):
                raise ValueError('All labels must be in [0, n labels). '
                                 'Got %d > %d' %
                                 (np.max(labels), np.max(present_labels)))
            if np.min(labels) < 0:
                raise ValueError('All labels must be in [0, n labels). '
                                 'Got %d < 0' % np.min(labels))

        if n_labels is not None:
            y_true = y_true[:, labels[:n_labels]]
            y_pred = y_pred[:, labels[:n_labels]]

        # calculate weighted counts
        true_and_pred = y_true.multiply(y_pred)
        tp_sum = count_nonzero(true_and_pred, axis=sum_axis,
                               sample_weight=sample_weight)
        pred_sum = count_nonzero(y_pred, axis=sum_axis,
                                 sample_weight=sample_weight)
        true_sum = count_nonzero(y_true, axis=sum_axis,
                                 sample_weight=sample_weight)

    elif average == 'samples':
        raise ValueError("Sample-based precision, recall, fscore is "
                         "not meaningful outside multilabel "
                         "classification. See the accuracy_score instead.")
    else:
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(tp_bins, weights=tp_bins_weights,
                              minlength=len(labels))
        else:
            # Pathological case
            true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        if len(y_pred):
            pred_sum = np.bincount(y_pred, weights=sample_weight,
                                minlength=len(labels))
        if len(y_true):
            true_sum = np.bincount(y_true, weights=sample_weight,
                                minlength=len(labels))

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #

    beta2 = beta ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        # Divide, and on zero-division, set scores to 0 and warn:

        # Oddly, we may get an "invalid" rather than a "divide" error
        # here.
        precision = _prf_divide(tp_sum, pred_sum,
                                'precision', 'predicted', average, warn_for)
        recall = _prf_divide(tp_sum, true_sum,
                             'recall', 'true', average, warn_for)
        # Don't need to warn for F: either P or R warned, or tp == 0 where pos
        # and true are nonzero, in which case, F is well-defined and zero
        f_score = ((1 + beta2) * precision * recall /
                   (beta2 * precision + recall))
        f_score[tp_sum == 0] = 0.0

    # Average the results

    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            return 0, 0, 0, None
    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != 'binary' or len(precision) == 1
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = None  # return no support

    return precision, recall, f_score, true_sum


def hamming_loss(y_true, y_pred):
    """Compute the average Hamming loss.

    """
    y_type, _, _ = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_type.startswith('multilabel'):
        num_samples, num_classses = np.array(y_true).shape
        n_differences = np.sum(y_true != y_pred)
        return n_differences / (num_samples * num_classses)
    elif y_type in ["binary", "multiclass"]:
        return np.sum(y_true != y_pred) / y_true.shape[0]
    else:
        raise ValueError("{0} is not supported".format(y_type))


def one_error(y_true, y_pred, sample_weight=None):
    check_consistent_length(y_true, y_pred, sample_weight)
    y_type = type_of_target(y_true)
    n_samples, n_labels = y_true.shape
    if y_type != "multilabel":
        raise ValueError("{0} format is not supported".format(y_type))
    n_differents = np.sum((y_true - y_pred), axis=1) == 0

    return n_differents.sum() / n_samples


def coverage_error(y_true, y_score, sample_weight=None):
    """Coverage error measure
    """
    check_consistent_length(y_true, y_score, sample_weight)
    y_type = type_of_target(y_true)
    if y_type != "multilabel":
        raise ValueError("{0} format is not supported".format(y_type))
    y_score_mask = np.ma.masked_array(y_score, mask=np.logical_not(y_true))
    y_min_relevant = y_score_mask.min(axis=1).reshape((-1, 1))
    coverage = (y_score >= y_min_relevant).sum(axis=1)
    coverage = coverage.filled(0)
    return np.average(coverage, weights=sample_weight)


def label_ranking_loss(y_true, y_score, sample_weight=None):
    """Compute Ranking loss measure

    """
    check_consistent_length(y_true, y_score, sample_weight)

    y_type = type_of_target(y_true)
    if y_type not in ("multilabel",):
        raise ValueError("{0} format is not supported".format(y_type))

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shape")

    n_samples, n_labels = y_true.shape

    y_true = csr_matrix(y_true)

    loss = np.zeros(n_samples)
    for i, (start, stop) in enumerate(zip(y_true.indptr, y_true.indptr[1:])):
        # Sort and bin the label scores
        unique_scores, unique_inverse = np.unique(y_score[i],
                                                  return_inverse=True)
        true_at_reversed_rank = np.bincount(
            unique_inverse[y_true.indices[start:stop]],
            minlength=len(unique_scores))
        all_at_reversed_rank = np.bincount(unique_inverse,
                                        minlength=len(unique_scores))
        false_at_reversed_rank = all_at_reversed_rank - true_at_reversed_rank

        # if the scores are ordered, it's possible to count the number of
        # incorrectly ordered paires in linear time by cumulatively counting
        # how many false labels of a given score have a score higher than the
        # accumulated true labels with lower score.
        loss[i] = np.dot(true_at_reversed_rank.cumsum(),
                         false_at_reversed_rank)

    n_positives = np.diff(y_true.indptr)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss /= ((n_labels - n_positives) * n_positives)

    # When there is no positive or no negative labels, those values should
    # be consider as correct, i.e. the ranking doesn't matter.
    loss[np.logical_or(n_positives == 0, n_positives == n_labels)] = 0.

    return np.average(loss, weights=sample_weight)


def label_ranking_average_precision_score(y_true, y_score, sample_weight=None):
    """Compute ranking-based average precision

    Parameters
    ----------
    y_true : array or sparse matrix, shape = [n_samples, n_labels]
        True binary labels in binary indicator format.

    y_score : array, shape = [n_samples, n_labels]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    score : float

    """
    check_consistent_length(y_true, y_score, sample_weight)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shape")

    # Handle badly formatted array and the degenerate case with one label
    y_type = type_of_target(y_true)
    if (y_type != "multilabel" and
            not (y_type == "binary" and y_true.ndim == 2)):
        raise ValueError("{0} format is not supported".format(y_type))

    y_true = csr_matrix(y_true)
    y_score = -y_score

    n_samples, n_labels = y_true.shape

    out = 0.
    for i, (start, stop) in enumerate(zip(y_true.indptr, y_true.indptr[1:])):
        relevant = y_true.indices[start:stop]

        if (relevant.size == 0 or relevant.size == n_labels):
            # If all labels are relevant or unrelevant, the score is also
            # equal to 1. The label ranking has no meaning.
            out += 1.
            continue

        scores_i = y_score[i]
        rank = rankdata(scores_i, 'max')[relevant]
        L = rankdata(scores_i[relevant], 'max')
        aux = (L / rank).mean()
        if sample_weight is not None:
            aux = aux * sample_weight[i]
        out += aux

    if sample_weight is None:
        out /= n_samples
    else:
        out /= np.sum(sample_weight)

    return out


def avgprec(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    temp_outputs = []
    temp_test_target = []
    instance_num = 0
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            instance_num = instance_num + 1
            temp_outputs.append(outputs[i])
            temp_test_target.append(test_target[i])
            labels_size.append(sum(test_target[i] == 1))
            index1, index2 = find(test_target[i], 1, 0)            
            labels_index.append(index1)
            not_labels_index.append(index2)
    
    aveprec = 0
    for i in range(instance_num):
        tempvalue, index = sort(temp_outputs[i])
        indicator = np.zeros((class_num,))     
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            indicator[loc] = 1
        summary = 0
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            #print(loc)
            summary = summary + sum(indicator[loc:class_num])*1.0/(class_num-loc);
        aveprec = aveprec + summary*1.0/labels_size[i]
    return aveprec*1.0/test_data_num


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
    # y_true = np.array([[1, 0, 1, 0],[0, 1, 0, 1],[1, 0, 0, 1],[0, 1, 1, 0],[1, 0, 0, 0]])
    # y_socre = np.array([[0.9, 0.0, 0.4, 0.6],[0.1, 0.8, 0.0, 0.8],[0.8, 0.0, 0.1, 0.7],[0.1, 0.7, 0.1, 0.2],[1.0, 0, 0, 1.0]])
    y_true = np.array([[1, 0, 1, 0],[0, 1, 0, 1]])
    y_socre = np.array([[0.9, 0.0, 0.4, 0.6],[0.1, 0.8, 0.0, 0.8]])

    print(label_ranking_average_precision_score(y_true,y_socre))

