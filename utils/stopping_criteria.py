"""
Heuristic:
1. preset number of quiries
2. preset limitation of cost
3. preset percent of unlabel pool is labeled
4. preset running time (CPU time) is reached
5. No unlabeled samples available
Formal:
5. the accuracy of a learner has reached a plateau
6. the cost of acquiring new training data is greater than the cost of the errors made by the current model
"""

from __future__ import division
import numpy as np
import time
from experiment_saver.state_io import StateIO

class StoppingCriteria:
    """class to implement stopping criteria.

    Initialize it with a certain string to determine when to stop.

    It will collect the information in each iteration of active learning.
    Such as number of iterations, cost, number of queries, performances, etc.

    example of supported stopping criteria:
    1. No unlabeled samples available (default)
    2. preset number of quiries is reached
    3. preset limitation of cost is reached
    4. preset percent of unlabel pool is labeled
    5. preset running time (CPU time) is reached

    Parameters
    __________
    stopping_criteria: str, optional (default=None)
        stopping criteria, must be one of: [None, 'num_of_queries', 'cost_limit', 'percent_of_unlabel', 'time_limit']

        None: stop when no unlabeled samples available
        'num_of_queries': stop when preset number of quiries is reached
        'cost_limit': stop when cost reaches the limit.
        'percent_of_unlabel': stop when specific percentage of unlabeled data pool is labeled.
        'time_limit': stop when CPU time reaches the limit.
    """

    def __init__(self, stopping_criteria=None, value=None):
        if stopping_criteria not in [None, 'num_of_queries', 'cost_limit', 'percent_of_unlabel', 'time_limit']:
            raise ValueError("Stopping criteria must be one of: [None, 'num_of_queries', 'cost_limit', 'percent_of_unlabel', 'time_limit']")
        self._stopping_criteria = stopping_criteria
        if isinstance(value, np.generic):
            value = np.asscalar(value)

        if stopping_criteria == 'num_of_queries':
            if not isinstance(value, int):
                value = int(value)
        else:
            if not isinstance(value, float):
                value = float(value)
        if stopping_criteria == 'time_limit':
            self._start_time = time.clock()
        self.value = value

        # collect information
        self._current_iter = 0
        self._accum_cost = 0
        self._current_unlabel = 100
        self._percent = 1.0

        self._init_value = value

    def is_stop(self):
        if self._current_unlabel == 0:
            return True
        elif self._stopping_criteria == 'num_of_queries':
            if self._current_iter >= self.value:
                return True
            else:
                return False
        elif self._stopping_criteria == 'cost_limit':
            if self._accum_cost >= self.value:
                return True
            else:
                return False
        elif self._stopping_criteria == 'percent_of_unlabel':
            if self._percent >= self.value:
                return True
            else:
                return False
        elif self._stopping_criteria == 'time_limit':
            if time.clock() - self._start_time >= self.value:
                return True
            else:
                return False
        return False

    def update_information(self, saver):
        """update value according to the specific criterion

        Parameters
        ----------
        saver: StateIO
            StateIO object
        """
        _,_,_, Uindex = saver.get_workspace()
        _, _, _, ini_Uindex = saver.get_workspace(iteration=0)
        self._current_unlabel = len(Uindex)
        if self._stopping_criteria == 'num_of_queries':
            self._current_iter = len(saver)
        elif self._stopping_criteria == 'cost_limit':
            self._accum_cost = saver.cost_inall
        elif self._stopping_criteria == 'percent_of_unlabel':
            self._percent = (len(ini_Uindex)-len(Uindex))/len(ini_Uindex)
        return self

    def reset(self):
        self.value = self._init_value
        self._start_time = time.clock()
        self._current_iter = 0
        self._accum_cost = 0
        self._current_unlabel = 100
        self._percent = 1.0
