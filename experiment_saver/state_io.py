from __future__ import division

"""
State
Save all information in one AL iteration
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

"""
由于state与stateIO均采用
copy实现，很可能导致结果文件过大
需按流读取，并每次查询N次就写一次文件释放内存
"""

import experiment_saver.state
import collections.abc
import copy
import os
import pickle
import experiment_saver.state
from utils.ace_warnings import *
import numpy as np


class StateIO:
    """
    A class to store states
    functions including:
    1.saving intermediate results to files
    2.loading data from file
    3.return specific State
    4.recover workspace from specific State

    Parameters
    ----------
    round: int
        number of experiments loop

    train_idx: array_like
        training index

    test_idx: array_like
        testing index

    init_U: array_like
        initial unlabeled set

    init_L: array_like
        initial labeled set

    initial_point: object
        the performance index

    saving_path: str, optional (default='.')
        path to save the intermediate files.

    check_flag: bool, optional (default=True)
        whether to check the validaty of states.

    verbose: bool, optional (default=True)
        whether to print query information during the AL process
    """

    def __init__(self, round, train_idx, test_idx, init_U, init_L, initial_point=None, saving_path=None,
                 check_flag=True,
                 verbose=True):
        """When init this class,
        should store the basic info of this active learning info,
        include: thread_id, train/test_idx, round, etc.
        """
        assert (isinstance(check_flag, bool))
        assert (isinstance(verbose, bool))
        self.__check_flag = check_flag
        self.__verbose = verbose
        if self.__check_flag:
            # check validity
            assert (isinstance(train_idx, collections.Iterable))
            assert (isinstance(test_idx, collections.Iterable))
            assert (isinstance(init_U, collections.Iterable))
            assert (isinstance(init_L, collections.Iterable))
            assert (isinstance(round, int) and round >= 0)
            # if not (len(train_idx) == len(init_L) + len(init_U)):
            #     warnings.warn("Length of train_idx is not equal len(init_L) + len(init_U).")

        self.round = round
        self.train_idx = copy.copy(train_idx)
        self.test_idx = copy.copy(test_idx)
        self.init_U = copy.deepcopy(init_U)
        self.init_L = copy.deepcopy(init_L)
        self.initial_point = initial_point
        self.batch_size = 0
        if saving_path is not None:
            self.saving_path = os.path.abspath(saving_path)
        else:
            tp_path = os.path.join(os.getcwd(), 'AL_result')
            if not os.path.exists(tp_path):
                os.makedirs(tp_path)
            self.saving_path = tp_path
        self.__state_list = []

    @classmethod
    def load(cls, path):
        f = open(os.path.abspath(path), 'rb')
        saver_from_file = pickle.load(f)
        f.close()
        return saver_from_file

    def set_check_flag(self, flag):
        """

        Parameters
        ----------
        flag: bool
            whether to check the validaty of states.
        """
        assert (isinstance(flag, bool))
        self.__check_flag = flag

    def set_verbose_flag(self, flag):
        """

        Parameters
        ----------
        flag: bool
            whether to print summary of states.
        """
        assert (isinstance(flag, bool))
        self.__verbose = flag

    def set_initial_point(self, perf):
        """
        The initial point before querying.

        Parameters
        ----------
        perf: object
            the performance index
        """
        self.initial_point = perf

    # use txt file to save for better generalization ability
    def save(self, file_name=None):
        """
        Saving intermediate results to file.

        Parameters
        ----------
        file_name: str, optional (default=None)
            file_name for saving, if not given ,then default name will be used.
        """
        if file_name is None:
            f = open(os.path.join(self.saving_path, 'experiment_result_file_round_' + str(self.round)), 'wb')
        else:
            f = open(os.path.join(self.saving_path, file_name), 'wb')
        pickle.dump(self, f)
        f.close()

    def add_state(self, state):
        assert (isinstance(state, experiment_saver.state.State))
        self.__state_list.append(copy.deepcopy(state))
        # if self.__check_flag:
        #     res, err_st, err_ind = self.check_select_index()
        #     if res == -1:
        #         warnings.warn(
        #             'Checking validity fails, there is a queried instance not in set_U in '
        #             'State:%d, index:%s.' % (err_st, str(err_ind)),
        #             category=ValidityWarning)
        #     if res == -2:
        #         warnings.warn('Checking validity fails, there are instances already queried '
        #                       'in previous iteration in State:%d, index:%s.' % (err_st, str(err_ind)),
        #                       category=ValidityWarning)
        if self.__verbose:
            print(self.__repr__())

    def get_state(self, index):
        return copy.deepcopy(self.__state_list[index])

    # def check_select_index(self):
    #     """
    #     check:
    #     - Q has no repeating elements
    #     - Q in U
    #     Returns
    #     -------
    #     result: int
    #         check result
    #         - if -1 is returned, there is a queried instance not in U
    #         - if -2 is returned, there are repeated instances in Q
    #         - if 1 is returned, CHECK OK
    #
    #     state_index: int
    #         the state index when checking fails (start from 0)
    #         if CHECK OK, None is returned.
    #
    #     select_index: object
    #         the select_index when checking fails.
    #         if CHECK OK, None is returned.
    #     """
    #     repeat_dict = dict()
    #     ind = -1
    #     for st in self.__state_list:
    #         ind += 1
    #         for instance in st.get_value('select_index'):
    #             if instance not in self.init_U:
    #                 return -1, ind, instance
    #             if instance not in repeat_dict.keys():
    #                 repeat_dict[instance] = 1
    #             else:
    #                 return -2, ind, instance
    #     return 1, None, None

    def check_batch_size(self):
        """
        Check if all queries have the same batch size.

        Returns
        -------
        bool
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
        """
        Recovery workspace of given iteration before querying.
        For example, if 1 is given, original State will be returned.
        Note that, the object itself will be recovered, some information
        will be discarded.

        Parameters
        ----------
        iteration: int, optional(default=None)
            number of iteration to recover, start from 1.
            if nothing given, it will recover the last query.

        Returns
        -------
        Ucollection: Indexcollection
            Unlabel index collection in iteration before querying.

        Lcollection: Indexcollection
            Label index collection in iteration before querying.
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
        return copy.copy(self.train_idx), copy.copy(self.test_idx), copy.deepcopy(work_U), copy.deepcopy(work_L)

    def get_workspace(self, iteration):
        """
        get workspace of given iteration before querying.
        For example, if 1 is given, original State will be returned.

        Parameters
        ----------
        iteration: int
            number of iteration to recover, start from 1.

        Returns
        -------
        Ucollection: Indexcollection
            Unlabel index collection in iteration before querying.

        Lcollection: Indexcollection
            Label index collection in iteration before querying.
        """
        if iteration is None:
            iteration = len(self.__state_list)
        work_U = copy.deepcopy(self.init_U)
        work_L = copy.deepcopy(self.init_L)
        for i in range(iteration - 1):
            state = self.__state_list[i]
            work_U.difference_update(state.get_value('select_index'))
            work_L.update(state.get_value('select_index'))
        return copy.copy(self.train_idx), copy.copy(self.test_idx), copy.deepcopy(work_U), copy.deepcopy(work_L)

    def __len__(self):
        return len(self.__state_list)

    def __getitem__(self, item):
        return self.__state_list.__getitem__(item)

    def __contains__(self, other):
        return other in self.__state_list

    def __iter__(self):
        return iter(self.__state_list)

    def __repr__(self):
        numqdata = 0
        cost = 0.0
        for state in self.__state_list:
            numqdata += len(state.get_value('select_index'))
            if 'cost' in state.keys():
                cost += np.sum(state.get_value('cost'))
        return '''\rActive selection summary:
_____________________________________________
round: %d
initially labeled data: %d (%.2f%% of all)
number of queries: %d
queried data: %d (%.2f%% of unlabeled data)
cost: %.2f
saving path: %s
''' % (self.round, len(self.init_L), 100 * len(self.init_L) / (len(self.init_L) + len(self.init_U)),
       len(self.__state_list), numqdata, 100 * numqdata / len(self.init_U), cost, self.saving_path)



if __name__ == '__main__':
    saver = StateIO()
