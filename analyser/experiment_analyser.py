"""
Class to analyse active learning experiments.
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

import copy
import os
import warnings
import matplotlib.pyplot as plt
import prettytable as pt
import numpy as np
import scipy.stats
import scipy.io as scio

import experiment_saver.al_experiment
from utils.ace_warnings import *


class ExperimentAnalyser:
    """
    Class to analyse active learning experiment.
    Include:
    1. Tools:
        1.1. A container to store results of varies methods (AlExperiment object)
        1.2. Extract each segment of results.
        1.3. Extract experiment settings.
    2. Plotting:
        2.1. line chart (different x,y,axis, mean+-std bars)
    3. Statistic and tools:
        3.1. Paired t-test
        3.2. Sort
    """

    class __MethodSummary:
        """
        store summary info of a given method experiment result
        """

        def __init__(self, method_results):
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
            self.folds = len(method_results)
            if self.batch_flag:
                self.batch_size = method_results.get_batch_size()
            # get matrix
            ex_data = []
            for round in method_results:
                self.ip = round.initial_point
                tmp = [round[i].get_value('performance') for i in range(self.effective_length)]
                if self.ip is not None:
                    tmp.insert(0, self.ip)
                ex_data.append(tmp)
            mean_ex = np.mean(ex_data, axis=1)
            self.mean = np.mean(mean_ex)
            self.std = np.std(mean_ex)

    def __init__(self):
        self.__result_raw = dict()
        self.__result_data = dict()
        self.__method_summary = dict()

    def add_method(self, method_results):
        """
        Add results of a method
        Parameters
        ----------
        method_results: utils.AlExperiment.AlExperiment
            experiment results of a method.
        """
        assert (isinstance(method_results, experiment_saver.al_experiment.AlExperiment))
        self.__result_raw[method_results.method_name] = copy.deepcopy(method_results)
        _, self.__result_data[method_results.method_name] = self.__extract_data(method_results)
        self.__method_summary[method_results.method_name] = self.__MethodSummary(method_results)

    def __extract_data(self, method_results):
        """
        Extract data for analysis.

        Parameters
        ----------
        method_results: utils.AlExperiment.AlExperiment
            experiment results of a method.

        Returns
        -------
        data: np.ndarray
            Extracted performance matrix
        """
        round_idx = []
        ex_data = []
        # examine the AlExperiment object
        batch_flag = True
        length_flag = True
        if not method_results.check_batch_size():
            warnings.warn('Checking validity fails, different batch size is found.',
                          category=ValidityWarning)
            batch_flag = False
        if not method_results.check_length():
            warnings.warn('Checking validity fails, different length of folds is found.',
                          category=ValidityWarning)
            length_flag = False
        effective_length = method_results.get_length()
        # get matrix
        for round in method_results:
            round_idx.append(round.round)
            ip = round.initial_point
            tmp = [round[i].get_value('performance') for i in range(effective_length)]
            if ip is not None:
                tmp.insert(0, ip)
            ex_data.append(tmp)
        return round_idx, np.asarray(ex_data)  # in case that when multi-threading, index of round is unordered

    def check_plotting(self):
        """
        check:
        1.NaN, Inf etc.
        2.methods_continuity
        """
        if not self.check_methods_continuity:
            warnings.warn('Settings among all methods are not the same. The difference will be ignored.',
                          category=ValidityWarning)
        for i in self.methods():
            if np.isnan(self.__result_data[i]).any() != 0:
                raise ValueError('NaN is found in methods %s in %s.' % (
                    i, str(np.argwhere(np.isnan(self.__result_data[i]) == True))))
            if np.isinf(self.__result_data[i]).any() != 0:
                raise ValueError('Inf is found in methods %s in %s.' % (
                    i, str(np.argwhere(np.isinf(self.__result_data[i]) == True))))
        return True

    def simple_plot(self, xlabel='queries', ylabel='performance', saving_path='.'):
        """plotting the performance curves.

        Parameters
        ----------
        xlabel

        ylabel

        saving_path: str, optional (default='.')
            path to save the figure. If None is given, the saving will be disabled.
        """
        self.check_plotting()
        for i in self.methods():
            points = np.mean(self.__result_data[i], axis=0)
            if self.__method_summary[i].ip is None:
                plt.plot(np.arange(start=1, stop=len(points) + 1), points, label=i)
            else:
                plt.plot(np.arange(len(points)), points, label=i)
        plt.legend()
        plt.xlabel("Number of queries")
        plt.ylabel("Performance")
        if saving_path is not None:
            saving_path = os.path.abspath(saving_path)
            if os.path.isdir(saving_path):
                plt.savefig(os.path.join(saving_path, 'acepy_plotting.jpg'))
            else:
                plt.savefig(saving_path)
        plt.show()

    def check_methods_continuity(self):
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
        for i in self.methods():
            summary = self.__method_summary[i]
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

    def methods(self):
        return self.__result_raw.keys()

    def get_method_data(self, method):
        return self.__result_raw[method]

    def get_extracted_data(self):
        return self.__result_data.copy()

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
        for i in self.methods():
            summary = self.__method_summary[i]
            if summary.batch_flag:
                tb.add_row([i, summary.batch_size, summary.effective_length, summary.folds,
                            "%.3f ± %.2f" % (summary.mean, summary.std)])
            else:
                tb.add_row([i, 'NotSameSize', summary.effective_length, summary.folds,
                            "%.3f ± %.2f" % (summary.mean, summary.std)])
        return '\n' + str(tb)

    def __len__(self):
        return len(self.__result_raw)

    def __getitem__(self, key):
        return self.__result_raw.__getitem__(key)

    def __contains__(self, other):
        return other in self.__result_raw

    def __iter__(self):
        return iter(self.__result_raw)

    @classmethod
    def load_result(self, path):
        """load result from file"""
        pass

    # some commonly used tool function for experiment analysing.
    @classmethod
    def paired_ttest(cls, a, b, alpha=0.05):
        """Performs a two-tailed paired t-test of the hypothesis that two
        matched samples, in the arrays a and b, come from distributions with
        equal means. The difference a-b is assumed to come from a normal
        distribution with unknown variance.  a and b must have the same length.

        Parameters
        ----------
        a: array-like
            array to t-test.

        b: array-like
            array for t-test.

        alpha: float, optional (default=0.05)
            A value alpha between 0 and 1 specifying the
            significance level as (100*alpha)%. Default is
            0.05 for 5% significance.

        Returns
        -------
        H: int
            the result of the test.
            H=0     -- indicates that the null hypothesis ("mean is zero")
                    cannot be rejected at the alpha% significance level
                    (No significance difference between a and b).
            H=1     -- indicates that the null hypothesis can be rejected at the alpha% level
                    (a and b have significance difference).

        Examples
        -------
        >>> from analyser.experiment_analyser import ExperimentAnalyser
        >>> a = [1.2, 2, 3]
        >>> b = [1.6, 2.5, 1.1]
        >>> print(ExperimentAnalyser.paired_ttest(a, b))
        1
        """
        rava = a
        ravb = b
        # check a,b
        sh = np.shape(a)
        if len(sh) == 1:
            pass
        elif sh[0] == 1 or sh[1] == 1:
            rava = np.ravel(a)
            ravb = np.ravel(b)
        else:
            raise Exception("a or b must be a 1-D array. but received: %s" % str(sh))
        assert(len(a)==len(b))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, pvalue = scipy.stats.ttest_rel(rava, ravb)
        H = int(pvalue <= alpha)
        return H


class BaseAnalyser:
    """class to plotting, analysing the experiment results only depends on the
    data array. It may also deal with other types of data file, such as:
    .mat
    .txt
    .csv
    """
    pass


if __name__ == "__main__":
    a = [1.2, 2, 3]
    b = [1.6, 2.5, 1.1]
    print(ExperimentAnalyser.paired_ttest(a, b))
    print(ExperimentAnalyser.paired_ttest(a, a))
