"""
Class to gathering, process and visualize active learning experiment results.
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

import copy
import os
import pickle
import experiment_saver
import matplotlib.pyplot as plt
import prettytable as pt
import numpy as np
import scipy.stats
import scipy.io as scio
import collections
from utils.base import BaseAnalyser
from utils.ace_warnings import *


def ExperimentAnalyser(x_axis='num_of_queries'):
    """Class to gathering, process and visualize active learning experiment results.

    Normally, the results should be a list which contains k elements. Each element represents
    one fold experiment result.
    Legal result object includes:
        - StateIO object.
        - A list contains n performances for n queries.
        - A list contains n tuples with 2 elements, in which, the first
          element is the x_axis (e.g., iteration, cost),
          and the second element is the y_axis (e.g., the performance)

    Functions include:
        - Line chart (different x,y,axis, mean±std bars)
        - Paired t-test

    Parameters
    ----------
    x_axis: str, optional (default='num_of_queries')
        The x_axis when analysing the result.
        x_axis should be one of ['num_of_queries', 'cost'],
        if 'cost' is given, your experiment results must contains the
        cost value for each performance value.

    Returns
    -------
    analyser: object
        The experiment analyser object

    """
    if x_axis not in ['num_of_queries', 'cost']:
        raise ValueError("x_axis should be one of ['num_of_queries', 'cost'].")
    if x_axis == 'num_of_queries':
        return _NumOfQueryAnalyser()
    else:
        return _CostSensitiveAnalyser()


class StateIOContainer:
    """Class to process StateIO object.

    If a list of StateIO objects is given, the data stored
    in each StateIO object can be extracted with this class.
    """

    def __init__(self, method_name, method_results):
        self.method_name = method_name
        self.__results = list()
        self.add_folds(method_results)

    def add_fold(self, src):
        """
        Add one fold of active learning experiment.

        Parameters
        ----------
        src: object or str
            StateIO object or path to the intermediate results file.
        """
        if isinstance(src, experiment_saver.state_io.StateIO):
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

    def extract_matrix(self, extract_keys='performance'):
        """Extract the data stored in the StateIO obejct.

        Parameters
        ----------
        extract_keys: str or list of str, optional (default='performance')
            Extract what value in the State object.
            The extract_keys should be a subset of the keys of each State object.
            Such as: 'select_index', 'performance', 'queried_label', 'cost', etc.

            Note that, the extracted matrix is associated with the extract_keys.
            If more than 1 key is given, each element in the matrix is a tuple.
            The values in tuple are one-to-one correspondence to the extract_keys.

        Returns
        -------
        extracted_matrix: list
            The extracted matrix.
        """
        extracted_matrix = []
        if isinstance(extract_keys, str):
            for stateio in self:
                stateio_line = []
                for state in stateio:
                    if extract_keys not in state.keys():
                        raise ValueError('The extract_keys should be a subset of the keys of each State object.\n'
                                         'But keys in the state are: %s' % str(state.keys()))
                    stateio_line.append(state.get_value(extract_keys))
                extracted_matrix.append(copy.copy(stateio_line))

        elif isinstance(extract_keys, list):
            for stateio in self:
                stateio_line = []
                for state in stateio:
                    state_line = []
                    for key in extract_keys:
                        if key not in state.keys():
                            raise ValueError('The extract_keys should be a subset of the keys of each State object.\n'
                                             'But keys in the state are: %s' % str(state.keys()))
                        state_line.append(state.get_value(key))
                    stateio_line.append(tuple(state_line))
                extracted_matrix.append(copy.copy(stateio_line))

        else:
            raise TypeError("str or list of str is expected, but received: %s" % str(type(extract_keys)))

        return extracted_matrix

    def __len__(self):
        return len(self.__results)

    def __getitem__(self, item):
        return self.__results.__getitem__(item)

    def __iter__(self):
        return iter(self.__results)


class _ContentSummary:
    """
    store summary info of a given method experiment result
    """

    def __init__(self, method_results, method_type):
        self.method_type = method_type
        # basic info
        self.mean = 0
        self.std = 0
        self.folds = len(method_results)

        # for stateio object only
        self.batch_flag = True
        self.length_flag = True
        self.ip = None
        self.batch_size = 0

        # Only for num of query
        self.effective_length = 0

        # Only for Cost
        self.cost_inall = []

        if isinstance(method_results, StateIOContainer):
            self.stateio_summary(method_results)
        else:
            self.list_summary(method_results)

    def stateio_summary(self, method_results):
        """
        calculate summary of a method
        Parameters
        ----------
        method_results: utils.AlExperiment.AlExperiment
            experiment results of a method.
        """
        # examine the AlExperiment object
        self.batch_flag = True
        self.length_flag = True
        if not method_results.check_batch_size():
            warnings.warn('Checking validity fails, different batch size is found.',
                          category=ValidityWarning)
            self.batch_flag = False
        if not method_results.check_length():
            warnings.warn('Checking validity fails, different length of folds is found.',
                          category=ValidityWarning)
            self.length_flag = False
        self.effective_length = method_results.get_length()
        if self.batch_flag:
            self.batch_size = method_results.get_batch_size()
        # get matrix
        ex_data = []
        for result in method_results:
            self.ip = result.initial_point
            one_fold_perf = [result[i].get_value('performance') for i in range(self.effective_length)]
            one_fold_cost = [result[i].get_value('cost') if 'cost' in result[i].keys() else 0 for i in range(self.effective_length)]
            self.cost_inall.append(one_fold_cost)
            if self.ip is not None:
                one_fold_perf.insert(0, self.ip)
            ex_data.append(one_fold_perf)
        mean_ex = np.mean(ex_data, axis=1)
        self.mean = np.mean(mean_ex)
        self.std = np.std(mean_ex)

    def list_summary(self, method_results):
        # Only for num of query
        self.effective_length = np.min([len(i) for i in method_results])
        if self.method_type == 1:
            # basic info
            self.mean = np.mean(method_results)
            self.std = np.std(method_results)
        else:
            perf_mat = [[tup[1] for tup in line] for line in method_results]
            cost_mat = [[tup[0] for tup in line] for line in method_results]
            mean_perf_for_each_fold = np.mean(perf_mat, axis=1)
            self.mean = np.mean(mean_perf_for_each_fold)
            self.std = np.std(mean_perf_for_each_fold)
            # Only for Cost
            self.cost_inall = np.sum(cost_mat, axis=1)

class _NumOfQueryAnalyser(BaseAnalyser):
    """Class to process the data whose x_axis is the number of query.

    The validity checking will depend only on the number of query.
    """

    def __init__(self):
        super(_NumOfQueryAnalyser, self).__init__()

    def add_method(self, method_name, method_results):
        """
        Add results of a method.

        Parameters
        ----------
        method_results: {list, np.ndarray}
            experiment results of a method. contains k stateIO object with k-fold experiment results.

        method_name: str
            Name of the given method.
        """
        assert (isinstance(method_results, (list, np.ndarray)))
        # StateIO object
        # The type must be one of [0,1,2], otherwise, it will raise in that function.
        self._is_all_stateio = True
        result_type = self._type_of_data(method_results)
        if result_type == 0:
            method_container = StateIOContainer(method_results, method_name)
            self.__data_extracted[method_name] = method_container.extract_matrix()
            # get method summary
            # The summary, however, can not be inferred from a list of performances.
            # So if any lists of extracted data are given, we assume all the results are legal,
            # and thus we will not do further checking on it.
            self.__data_summary[method_name] = _ContentSummary(method_results=method_results, method_type=result_type)
        elif result_type == 1:
            self.__data_extracted[method_name] = copy.copy(method_results)
            self._is_all_stateio = False
            self.__data_summary[method_name] = _ContentSummary(method_results=method_results, method_type=result_type)
        else:
            raise ValueError("The element in each list should be a single performance value.")

    def _check_plotting(self):
        """
        check:
        1.NaN, Inf etc.
        2.methods_continuity
        """
        if not self._check_methods_continuity:
            warnings.warn('Settings among all methods are not the same. The difference will be ignored.',
                          category=ValidityWarning)
        for i in self.__data_extracted.keys():
            if np.isnan(self.__data_extracted[i]).any() != 0:
                raise ValueError('NaN is found in methods %s in %s.' % (
                    i, str(np.argwhere(np.isnan(self.__data_extracted[i]) == True))))
            if np.isinf(self.__data_extracted[i]).any() != 0:
                raise ValueError('Inf is found in methods %s in %s.' % (
                    i, str(np.argwhere(np.isinf(self.__data_extracted[i]) == True))))
        return True

    def _check_methods_continuity(self):
        """
        check if all methods have the same batch size, length and folds

        Returns
        -------
        result: bool
            True if the same, False otherwise.
        """
        first_flag = True
        bs = 0
        el = 0
        folds = 0
        ip = None
        for i in self.__data_extracted.keys():
            summary = self.__data_summary[i]
            if first_flag:
                bs = summary.batch_size
                el = summary.effective_length
                folds = summary.folds
                ip = summary.ip
                first_flag = False
            else:
                if bs != summary.batch_size or el != summary.effective_length or folds != summary.folds or not isinstance(
                        ip, type(summary.ip)):
                    return False
        return True

    def simple_plot(self, start_point=None, title=None, std_bar=False, saving_path='.'):
        """plotting the performance curves.

        Parameters
        ----------
        start_point: int, optional (default=None)
            The start value of x_axis. If not specify, it will infer from the data.
            Set this parameter to force the start point to be a specific value.

        title
        std_bar
        saving_path

        Returns
        -------

        """
        if self._is_all_stateio:
            self._check_plotting()

        # plotting
        for i in self.methods():
            points = np.mean(self.__data_extracted[i], axis=0)
            if std_bar:
                std_points = np.std(self.__data_extracted[i], axis=0)
            if start_point is None:
                if not self._is_all_stateio or self.__data_summary[i].ip is None:
                    start_point = 1
                else:
                    start_point = 0
            plt.plot(np.arange(len(points))+start_point, points, label=i)
            if std_bar:
                plt.fill_between(np.arange(len(points))+start_point, points - std_points, points + std_points,
                                 interpolate=True, alpha=0.3)

        # axis & title
        plt.legend(fancybox=True, framealpha=0.5)
        plt.xlabel("Number of queries")
        plt.ylabel("Performance")
        if title is not None:
            plt.title(str(title))

        # saving
        if saving_path is not None:
            saving_path = os.path.abspath(saving_path)
            if os.path.isdir(saving_path):
                plt.savefig(os.path.join(saving_path, 'acepy_plotting.jpg'))
            else:
                plt.savefig(saving_path)
        plt.show()

    def __repr__(self):
        """
        summary of current methods.

        print:
        1. methods name, numbers
        2. batch size, if the same
        3. length of each method
        4. round
        5. mean+-std performances (whole length)
        """
        tb = pt.PrettyTable()
        tb.field_names = ['Methods', 'batch_size', 'number_of_queries', 'number_of_different_split', 'performance']
        for i in self.__data_extracted.keys():
            summary = self.__data_summary[i]
            if summary.batch_flag:
                tb.add_row([i, summary.batch_size, summary.effective_length, summary.folds,
                            "%.3f ± %.2f" % (summary.mean, summary.std)])
            else:
                tb.add_row([i, 'NotSameSize', summary.effective_length, summary.folds,
                            "%.3f ± %.2f" % (summary.mean, summary.std)])
        return '\n' + str(tb)


class _CostSensitiveAnalyser(BaseAnalyser):
    """Class to process the cost sensitive experiment results.

    The validity checking will depend only on the cost.
    """
    def __init__(self):
        super(_CostSensitiveAnalyser, self).__init__()

    def add_method(self, method_name, method_results):
        """
        Add results of a method.

        Parameters
        ----------
        method_results: {list, np.ndarray}
            experiment results of a method. contains k stateIO object or
            a list contains n tuples with 2 elements, in which, the first
            element is the x_axis (e.g., iteration, cost),
            and the second element is the y_axis (e.g., the performance)

        method_name: str
            Name of the given method.
        """
        assert (isinstance(method_results, (list, np.ndarray)))
        # StateIO object
        # The type must be one of [0,1,2], otherwise, it will raise in that function.
        self._is_all_stateio = True
        result_type = self._type_of_data(method_results)
        if result_type == 0:
            method_container = StateIOContainer(method_results, method_name)
            self.__data_extracted[method_name] = method_container.extract_matrix()
            # get method summary
            # The summary, however, can not be inferred from a list of performances.
            # So if any lists of extracted data are given, we assume all the results are legal,
            # and thus we will not do further checking on it.
            self.__data_summary[method_name] = _ContentSummary(method_results=method_results, method_type=result_type)
        elif result_type == 2:
            self.__data_extracted[method_name] = copy.copy(method_results)
            self._is_all_stateio = False
            self.__data_summary[method_name] = _ContentSummary(method_results=method_results, method_type=result_type)
        else:
            raise ValueError("Illegal result data in cost sensitive setting is given.\n"
                             "Legal result object includes:\n"
                             "\t- StateIO object.\n"
                             "\t- A list contains n tuples with 2 elements, in which, "
                             "the first element is the x_axis (e.g., iteration, cost),"
                             "and the second element is the y_axis (e.g., the performance)")


if __name__ == "__main__":
    a = [1.2, 2, 3]
    b = [1.6, 2.5, 1.1]
    print(ExperimentAnalyser.paired_ttest(a, b))
    print(ExperimentAnalyser.paired_ttest(a, a))
