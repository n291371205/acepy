class BaseModel(metaclass=ABCMeta):
    """
    Implement predict, fit, and predict_prob mehtods

    constrains:
    predcit class is in the given label set
    predict is a list : binary_[n_samples] multi_[n_samples, n_class]
    """

    @abstractmethod
    def predict(self, X):
        """output the class of X

        Parameters
        ----------
        X: data matrix

        Returns
        -------
        array-like, [n_samples, n_class]
        """
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict_proba(self, X, y):
        """output the probability of prediction
        if not supported, attribute has_prob should be set to False

        Parameters
        ----------
        X: data matrix

        Returns
        -------
        array-like, [n_samples, n_class]
        """
        pass


class BasePerformance(metaclass=ABCMeta):
    """
    Base class of performance function
    """

    @abstractmethod
    def calc_perf(self, predict, ground_truth):
        """

        Parameters
        ----------
        predict
        ground_truth

        Returns
        -------
        one float number or one tuple
        """
        pass


class BaseStoppingCriteria(metaclass=ABCMeta):
    """
    Base class of Stopping Criteria
    """

    @abstractmethod
    def stopping_criteria(self, state):
        """

        Parameters
        ----------
        state : an object of base_types.State

        Returns
        -------
        bool : True if stop, otherwise False
        """
        pass

# python2
# label_idx.append(IndexCollection(zip(tp_train[0:cutpoint],[-1]*cutpoint)))
# unlabel_idx.append(IndexCollection(zip(tp_train[cutpoint:],[-1]*(len(tp_train)-cutpoint))))
# python3
# label_idx.append(IndexCollection(list(zip(tp_train[0:cutpoint], [-1 ] * cutpoint))))
# unlabel_idx.append(IndexCollection(list(zip(tp_train[cutpoint:], [-1 ]* (Xsh[0] - cutpoint)))))

# elif query_type == '':
#     if y is None:
#         raise ValueError("y must be provided in splitting multi-label dataset.")
#     y = check_array(y, ensure_2d=False, dtype=None)
#     if y.dim != 2:
#         raise TypeError("y must be a 2D array. instance_label split only available in multi label setting")
#     elif y.shape[1] == 1:
#         raise TypeError("y must be a 2D array. instance_label split only available in multi label setting")
#     pass