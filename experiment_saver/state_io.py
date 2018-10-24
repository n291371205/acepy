"""
StateIO
Container to store state object.
Several useful functions are implemented in this class:
1. Saving intermediate results to files.
2. Recover workspace at any iteration (label set and unlabel set).
3. Recover workspace from the intermediate result file in case the program exits unexpectedly.
4. Gathering and checking the information stored in State object.
5. Print active learning progress: current_iteration, current_mean_performance, current_cost, etc.
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division
import experiment_saver.state
import collections.abc
import copy
import os
import pickle
import sys
import prettytable as pt
import experiment_saver.state
from utils.ace_warnings import *
import numpy as np
from utils.al_collections import IndexCollection


class StateIO:
    """
    A class to store states.
    Functions including:
    1. Saving intermediate results to files.
    2. Recover workspace at any iteration (label set and unlabel set).
    3. Recover workspace from the intermediate result file in case the program exits unexpectedly.
    4. Gathering and checking the information stored in State object.
    5. Print active learning progress: current_iteration, current_mean_performance, current_cost, etc.

    Parameters
    ----------
    round: int
        Number of k-fold experiments loop. 0 <= round < k

    train_idx: array_like
        Training index of one fold experiment.

    test_idx: array_like
        Testing index of one fold experiment.

    init_L: array_like
        Initial labeled index of one fold experiment.

    init_U: array_like
        Initial unlabeled index of one fold experiment.

    initial_point: object, optional (default=None)
        The performance before any querying.
        If not specify, the initial point of different methods will be different.

    saving_path: str, optional (default='.')
        Path to save the intermediate files. If None is given, it will
        not save the intermediate result.

    check_flag: bool, optional (default=True)
        Whether to check the validity of states.

    verbose: bool, optional (default=True)
        Whether to print query information during the AL process.

    print_interval: int optional (default=1)
        How many queries will trigger a print when verbose is True.
    """

    def __init__(self, round, train_idx, test_idx, init_L, init_U, initial_point=None, saving_path='.',
                 check_flag=True, verbose=True, print_interval=1):
        assert (isinstance(check_flag, bool))
        assert (isinstance(verbose, bool))
        self.__check_flag = check_flag
        self.__verbose = verbose
        self.__print_interval = print_interval
        if self.__check_flag:
            # check validity
            assert (isinstance(train_idx, collections.Iterable))
            assert (isinstance(test_idx, collections.Iterable))
            assert (isinstance(init_U, collections.Iterable))
            assert (isinstance(init_L, collections.Iterable))
            assert (isinstance(round, int) and round >= 0)

        self.round = round
        self.train_idx = copy.copy(train_idx)
        self.test_idx = copy.copy(test_idx)
        self.init_U = IndexCollection(init_U)
        self.init_L = IndexCollection(init_L)
        self.initial_point = initial_point
        self.batch_size = 0
        if saving_path is not None:
            self.saving_path = os.path.abspath(saving_path)
            if not os.path.exists(self.saving_path):
                os.mkdir(self.saving_path)
        else:
            self.saving_path = None
        self.__state_list = []
        self._first_print = True
        self.cost_inall = 0
        self._numqdata = 0

    @classmethod
    def load(cls, path):
        """Load StateIO object from file.

        Parameters
        ----------
        path: str
            The path should be a specific .pkl file.

        Returns
        -------
        object: StateIO
            The StateIO object in the file.
        """
        f = open(os.path.abspath(path), 'rb')
        saver_from_file = pickle.load(f)
        f.close()
        return saver_from_file

    def set_initial_point(self, perf):
        """The initial point of performance before querying.

        Parameters
        ----------
        perf: object
            The performance value.
        """
        self.initial_point = perf

    def save(self, file_name=None):
        """Saving intermediate results to file.

        Parameters
        ----------
        file_name: str, optional (default=None)
            File name for saving, if not given ,then default name will be used.
        """
        if self.saving_path is None:
            return
        if file_name is None:
            f = open(os.path.join(self.saving_path, 'AL_round_' + str(self.round) + '.pkl'), 'wb')
        else:
            f = open(os.path.join(self.saving_path, file_name), 'wb')
        pickle.dump(self, f)
        f.close()

    def add_state(self, state):
        """Add a State object to the container.

        Parameters
        ----------
        state: State
            State object to be added.
        """
        assert (isinstance(state, experiment_saver.state.State))
        self.__state_list.append(copy.deepcopy(state))
        self.__update_info()

        if self.__verbose and len(self) % self.__print_interval == 0:
            if self._first_print:
                print('\n' + self.__repr__(), end='')
                self._first_print = False
            else:
                print('\r' + self._refresh_dataline(), end='')
                sys.stdout.flush()

    def get_state(self, index):
        """Get a State object in the container.

        Parameters
        ----------
        index: int
            The index of the State object. 0 <= index < len(self)

        Returns
        -------
        st: State
            The State object in the previous iteration.
        """
        assert(0 <= index < len(self))
        return copy.deepcopy(self.__state_list[index])

    def check_batch_size(self):
        """Check if all queries have the same batch size.

        Returns
        -------
        result: bool
            Whether all the states have the same batch size.
        """
        ind_uni = np.unique(
            [self.__state_list[i].batch_size for i in range(len(self.__state_list) - 1)], axis=0)
        if len(ind_uni) == 1:
            self.batch_size = ind_uni[0]
            return True
        else:
            return False

    def pop(self, i=None):
        """remove and return item at index (default last)."""
        return self.__state_list.pop(i)

    def recovery(self, iteration=None):
        """Recovery workspace of a given iteration before querying.
        For example, if 1 is given, the initial workspace without any querying will be recovered.
        Note that, the object itself will be recovered, the information after the iteration will be discarded.

        Parameters
        ----------
        iteration: int, optional(default=None)
            Number of iteration to recover, start from 1.
            If nothing given, it will recover the last query.

        Returns
        -------
        train_idx: array-like
            Index of training set, shape like [n_training_samples]

        test_idx: array-like
            Index of testing set, shape like [n_testing_samples]

        label_idx: array-like
            Index of labeling set, shape like [n_labeling_samples]

        unlabel_idx: array-like
            Index of unlabeling set, shape like [n_unlabeling_samples]
        """
        if iteration is None:
            iteration = len(self.__state_list)
        work_U = copy.deepcopy(self.init_U)
        work_L = copy.deepcopy(self.init_L)
        for i in range(iteration - 1):
            state = self.__state_list[i]
            work_U.difference_update(state.get_value('select_index'))
            work_L.update(state.get_value('select_index'))
        self.__state_list = self.__state_list[0:iteration - 1]
        return copy.copy(self.train_idx), copy.copy(self.test_idx), copy.deepcopy(work_L), copy.deepcopy(work_U)

    def get_workspace(self, iteration=None):
        """Get workspace of a given iteration before querying.
        For example, if 1 is given, the initial workspace without any querying will be recovered.

        Parameters
        ----------
        iteration: int
            Number of iteration.

        Returns
        -------
        train_idx: array-like
            Index of training set, shape like [n_training_samples]

        test_idx: array-like
            Index of testing set, shape like [n_testing_samples]

        label_idx: array-like
            Index of labeling set, shape like [n_labeling_samples]

        unlabel_idx: array-like
            Index of unlabeling set, shape like [n_unlabeling_samples]
        """
        if iteration is None:
            iteration = len(self.__state_list)
        work_U = copy.deepcopy(self.init_U)
        work_L = copy.deepcopy(self.init_L)
        for i in range(iteration - 1):
            state = self.__state_list[i]
            work_U.difference_update(state.get_value('select_index'))
            work_L.update(state.get_value('select_index'))
        return copy.copy(self.train_idx), copy.copy(self.test_idx), copy.deepcopy(work_L), copy.deepcopy(work_U)

    def num_of_query(self):
        """Return the number of queries"""
        return len(self.__state_list)

    def get_current_performance(self):
        """Return the mean performance of all existed states."""
        if len(self) == 0:
            return 0, 0
        else:
            tmp = [self[i].get_value('performance') for i in range(self.__len__())]
            return np.mean(tmp), np.std(tmp)

    def __len__(self):
        return len(self.__state_list)

    def __getitem__(self, item):
        return self.__state_list.__getitem__(item)

    def __contains__(self, other):
        return other in self.__state_list

    def __iter__(self):
        return iter(self.__state_list)

    def refresh_info(self):
        """re-calculate current active learning progress."""
        numqdata = 0
        cost = 0.0
        for state in self.__state_list:
            numqdata += len(state.get_value('select_index'))
            if 'cost' in state.keys():
                cost += np.sum(state.get_value('cost'))
        self.cost_inall = cost
        self._numqdata = numqdata
        return numqdata, cost

    def __update_info(self):
        """Update current active learning progress"""
        state = self.__state_list[len(self)-1]
        if 'cost' in state.keys():
            self.cost_inall += np.sum(state.get_value('cost'))
        self._numqdata += len(state.get_value('select_index'))

    def __repr__(self):
        numqdata = self._numqdata
        cost = self.cost_inall
        tb = pt.PrettyTable()
        tb.set_style(pt.MSWORD_FRIENDLY)
        tb.add_column('round', [self.round])
        tb.add_column('initially labeled data', [
            " %d (%.2f%% of all)" % (len(self.init_L), 100 * len(self.init_L) / (len(self.init_L) + len(self.init_U)))])
        tb.add_column('number of queries', [len(self.__state_list)])
        # tb.add_column('queried data', ["%d (%.2f%% of unlabeled data)" % (numqdata, self.queried_percentage)])
        tb.add_column('cost', [cost])
        tb.add_column('saving path', [self.saving_path])
        return str(tb)

    def _refresh_dataline(self):
        tb = self.__repr__()
        return tb.splitlines()[1]

# class StateIO_all_labels(StateIO):
#     """StateIO for all _labels querying"""
#     def add_state(self, state):
#         assert (isinstance(state, experiment_saver.state.State))
#         self.__state_list.append(copy.deepcopy(state))
#         if self.__check_flag:
#             res, err_st, err_ind = self.check_select_index()
#             if res == -1:
#                 warnings.warn(
#                     'Checking validity fails, there is a queried instance not in set_U in '
#                     'State:%d, index:%s.' % (err_st, str(err_ind)),
#                     category=ValidityWarning)
#             if res == -2:
#                 warnings.warn('Checking validity fails, there are instances already queried '
#                               'in previous iteration in State:%d, index:%s.' % (err_st, str(err_ind)),
#                               category=ValidityWarning)
#         self.__update_info()
#
#
#         if self.__verbose and len(self) % self.__print_interval == 0:
#             if self._first_print:
#                 print('\n' + self.__repr__(), end='')
#                 self._first_print = False
#             else:
#                 print('\r' + self._refresh_dataline(), end='')
#                 sys.stdout.flush()
#
#     def check_select_index(self):
#         """
#         check:
#         - Q has no repeating elements
#         - Q in U
#         Returns
#         -------
#         result: int
#             check result
#             - if -1 is returned, there is a queried instance not in U
#             - if -2 is returned, there are repeated instances in Q
#             - if 1 is returned, CHECK OK
#
#         state_index: int
#             the state index when checking fails (start from 0)
#             if CHECK OK, None is returned.
#
#         select_index: object
#             the select_index when checking fails.
#             if CHECK OK, None is returned.
#         """
#         repeat_dict = dict()
#         ind = -1
#         for st in self.__state_list:
#             ind += 1
#             for instance in st.get_value('select_index'):
#                 if instance not in self.init_U:
#                     return -1, ind, instance
#                 if instance not in repeat_dict.keys():
#                     repeat_dict[instance] = 1
#                 else:
#                     return -2, ind, instance
#         return 1, None, None
#
#     @property
#     def queried_percentage(self):
#         """return the queried percentage of unlabeled data"""
#         return 100 * self._numqdata / len(self.init_U)

if __name__ == '__main__':
    saver = StateIO()
