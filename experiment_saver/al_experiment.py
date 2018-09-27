"""
Class to store necessary information of
an active learning experiment for analysis
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

import copy
import os
import pickle
import experiment_saver
from utils.ace_warnings import *


class AlExperiment:
    """
    Class to store necessary information of
    an active learning experiment for analysis
    """

    def __init__(self, method_name, remark=None):
        self.method_name = method_name
        self.remark = remark
        self.__results = list()

    def add_fold(self, src):
        """
        Add one fold of active learning experiment.

        Parameters
        ----------
        src: object or str
            StateIO object or path to the intermediate results file.
        """
        if isinstance(src, experiment_saver.state_io.StateIO):
            if not src.check_batch_size():
                warnings.warn('Checking validity fails, different batch size is found.', category=ValidityWarning)
            self.__add_fold_by_object(src)
        elif isinstance(src, str):
            self.__add_fold_from_file(src)
        else:
            raise TypeError('StateIO object or str is expected, but received:%s' % str(type(src)),
                            category=UnexpectedParameterWarning)

    def add_folds(self, folds):
        """Add multiple folds.

        Parameters
        ----------
        folds: list
            The list contains n StateIO objects.
        """
        for item in folds:
            self.add_fold(item)

    def __add_fold_by_object(self, result):
        """
        Add one fold of active learning experiment

        Parameters
        ----------
        result: utils.StateIO
            object stored a complete fold of active learning experiment
        """
        self.__results.append(copy.deepcopy(result))

    def __add_fold_from_file(self, path):
        """
        Add one fold of active learning experiment from file

        Parameters
        ----------
        path: str
            path of result file.
        """
        f = open(os.path.abspath(path), 'rb')
        result = pickle.load(f)
        f.close()
        assert (isinstance(result, experiment_saver.StateIO))
        if not result.check_batch_size():
            warnings.warn('Checking validity fails, different batch size is found.',
                          category=ValidityWarning)
        self.__results.append(copy.deepcopy(result))

    def check_experiments(self):
        """
        Check results stored in this object that:
        1. whether all folds have the same length. If not, calc the shortest one
        2. whether the batch size is the same.
        3. calculate additional information.

        Returns
        -------

        """
        if self.check_batch_size and self.check_length:
            return True
        else:
            return False

    def check_batch_size(self):
        """
        if all queries have the same batch size.

        Returns
        -------

        """
        bs = set()
        for item in self.__results:
            if not item.check_batch_size():
                return False
            else:
                bs.add(item.batch_size)
        if len(bs) == 1:
            return True
        else:
            return False

    def check_length(self):
        """
        if all folds have the same numbers of query.

        Returns
        -------
        bool
        """
        ls = set()
        for item in self.__results:
            ls.add(len(item))
        if len(ls) == 1:
            return True
        else:
            return False

    def get_batch_size(self):
        """

        Returns
        -------
        -1 if not the same batch size
        """
        if self.check_batch_size:
            return self.__results[0].batch_size
        else:
            return -1

    def get_length(self):
        ls = list()
        for item in self.__results:
            ls.append(len(item))
        return min(ls)

    def __len__(self):
        return len(self.__results)

    def __getitem__(self, item):
        return self.__results.__getitem__(item)

    def __contains__(self, other):
        return other in self.__results

    def __iter__(self):
        return iter(self.__results)

    def __repr__(self):
        return
