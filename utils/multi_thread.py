import threading
import random
import numpy as np
import time
import os
import pickle
import inspect
import prettytable as pt
import copy
from experiment_saver.state_io import StateIO


class aceThreading:
    """This class implement multi-threading in active learning for multiple 
    random splits experiments.

    It will display the progress of each thead. When all threads reach the 
    end points, it will return k StateIO objects for analysis.

    Once initialized, it can store and recover from any iterations and breakpoints.

    Note that, this class only provides visualization and file IO for threads, but
    not implement any threads. You should construct different threads by your own, 
    and then provide them as parameters for visualization.

    Specifically, the parameters of thread function must be:
    (round, train_id, test_id, Ucollection, Lcollection, saver, examples, labels, global_parameters)
    in which, the global_parameters is a dict which contains the other variables for user-defined function.

    Parameters
    ----------
    examples: array-like
        data matrix, shape like [n_samples, n_features].

    labels:: array-like
        labels of examples. shape like [n_samples] or [n_samples, n_classes] if in the multi-label setting.

    train_idx: array-like
        index of training examples. shape like [n_round, n_training_examples].

    test_idx: array-like
        index of training examples. shape like [n_round, n_testing_examples].

    label_index: array-like
        index of initially labeled examples. shape like [n_round, n_labeled_examples].

    unlabel_index: array-like
        index of unlabeled examples. shape like [n_round, n_unlabeled_examples].

    max_thread: int, optional (default=None)
        The max threads for running at the same time. If not provided, it will run all rounds simultaneously.

    refresh_interval: float, optional (default=1.0)
        how many seconds to refresh the current state output, default is 1.0.

    saving_path: str, optional (default='.')
        the path to save the result files.
    """

    def __init__(self, examples, labels, train_idx, test_idx, unlabel_index, label_index, target_func,
                 max_thread=None, refresh_interval=1, saving_path='.'):
        self.examples = examples
        self.labels = labels
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.label_index = label_index
        self.unlabel_index = unlabel_index
        self.refresh_interval = refresh_interval
        self.saving_path = os.path.abspath(saving_path)
        self.recover_arr = None

        # the path to store results of each thread.
        tp_path = os.path.join(self.saving_path, 'AL_result')
        if not os.path.exists(tp_path):
            os.makedirs(tp_path)

        assert (len(train_idx) == len(test_idx) ==
                len(label_index) == len(unlabel_index))
        self._round_num = len(label_index)
        self.threads = []
        # for monitoring threads' progress
        self.saver = [
            StateIO(round=i, train_idx=self.train_idx[i], test_idx=self.test_idx[i], init_U=self.unlabel_index[i],
                    init_L=self.label_index[i], saving_path=os.path.join(self.saving_path, 'AL_result'),
                    verbose=False) for i in range(self._round_num)]
        if max_thread is None:
            self.max_thread = self._round_num
        else:
            self.max_thread = max_thread
        # for controlling the print frequency
        self._start_time = time.clock()
        # for displaying the time elapse
        self._thread_time_elapse = [-1] * self._round_num
        # for recovering the workspace
        self.alive_thread = [False] * self._round_num

        # check target function validity
        argname = inspect.getfullargspec(target_func)[0]
        for name1 in ['round', 'train_id', 'test_id', 'Ucollection', 'Lcollection', 'saver', 'examples', 'labels',
                      'global_parameters']:
            if name1 not in argname:
                raise ValueError(
                    "the parameters of target_func must be (round, train_id, test_id, "
                    "Ucollection, Lcollection, saver, examples, labels, global_parameters)")
        self._target_func = target_func

    def start_all_threads(self, global_parameters=None):
        """Start multi-threading.

        this function will automatically invoke the thread_func function with the parameters:
        (round, train_id, test_id, Ucollection, Lcollection, saver, examples, labels, **global_parameters),
        in which, the global_parameters should be provided by the user for additional variables.

        It is necessary that the parameters of your thread_func accord the appointment, otherwise,
        it will raise an error.

        Parameters
        ----------
        target_func: function object
            the function to parallel, the parameters must accord the appointment.

        global_parameters: dict, optional (default=None)
            the additional variables to implement user-defined query-strategy.
        """
        self.__init_threads(global_parameters)
        # start thread
        self.__start_all_threads()

    def __init_threads(self, global_parameters=None):
        if global_parameters is not None:
            assert (isinstance(global_parameters, dict))
        self._global_parameters = global_parameters

        # init thread objects
        for i in range(self._round_num):
            t = threading.Thread(target=self._target_func, name=str(i), kwargs={
                'round': i, 'train_id': self.train_idx[i], 'test_id': self.test_idx[i],
                'Ucollection': copy.deepcopy(self.unlabel_index[i]), 'Lcollection': copy.deepcopy(self.label_index[i]),
                'saver': self.saver[i], 'examples': self.examples, 'labels': self.labels,
                'global_parameters': global_parameters})
            self.threads.append(t)

    def __start_all_threads(self):
        if self.recover_arr is None:
            self.recover_arr = [True] * self._round_num
        else:
            assert (len(self.recover_arr) == self._round_num)
        # start thread
        available_thread = self._round_num
        for i in range(self._round_num):
            if not self.recover_arr[i]:
                continue
            if available_thread > 0:
                self.threads[i].start()
                self._thread_time_elapse[i] = time.time()
                self.alive_thread[i] = True
                available_thread -= 1

                # saving
                self._update_thread_state()
                self.save()
            else:
                while True:
                    if self._if_refresh():
                        print(self)
                    if threading.active_count() < self.max_thread:
                        break
                available_thread += self.max_thread - threading.active_count()
                self.threads[i].start()
                self._thread_time_elapse[i] = time.time()
                self.alive_thread[i] = True
                available_thread -= 1

                # saving
                self._update_thread_state()
                self.save()

        # waiting for other threads.
        for i in range(self._round_num):
            if not self.recover_arr[i]:
                continue
            while self.threads[i].is_alive():
                if self._if_refresh():
                    print(self)
            self._update_thread_state()
            self.save()
        print(self)

    def __repr__(self):
        tb = pt.PrettyTable()
        tb.field_names = ['round', 'number_of_queries', 'time_elapse', 'performance (mean ± std)']

        for i in range(len(self.saver)):
            if self._thread_time_elapse[i] == -1:
                time_elapse = '0'
            else:
                time_elapse = time.time() - self._thread_time_elapse[i]
                m, s = divmod(time_elapse, 60)
                h, m = divmod(m, 60)
                time_elapse = "%02d:%02d:%02d" % (h, m, s)
            tb.add_row([self.saver[i].round, self.saver[i].get_current_progress(),
                        time_elapse,
                        "%.3f ± %.2f" % self.saver[i].get_current_performance()])
        return str(tb)

    def _if_refresh(self):
        if time.clock() - self._start_time > self.refresh_interval:
            self._start_time = time.clock()
            return True
        else:
            return False

    def _update_thread_state(self):
        for i in range(len(self.threads)):
            if self.threads[i].is_alive():
                self.alive_thread[i] = True
            else:
                self.alive_thread[i] = False

    def __getstate__(self):
        pickle_seq = (
            self.examples,
            self.labels,
            self.train_idx,
            self.test_idx,
            self.label_index,
            self.unlabel_index,
            self.refresh_interval,
            self.saving_path,
            self._round_num,
            self.max_thread,
            self._target_func,
            self._global_parameters,
            self.alive_thread,
            self.saver
        )
        return pickle_seq

    def __setstate__(self, state):
        self.examples, self.labels, self.train_idx, self.test_idx, self.label_index, self.unlabel_index, self.refresh_interval, self.saving_path, self._round_num, self.max_thread, self._target_func, self._global_parameters, self.alive_thread, self.saver = state

    def save(self):
        if os.path.isdir(self.saving_path):
            f = open(os.path.join(self.saving_path, 'multi_thread_state.pkl'), 'wb')
        else:
            f = open(self.saving_path, 'wb')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def recover(cls, path):
        # load breakpoint
        if not isinstance(path, str):
            raise TypeError("A string is expected, but received: %s" % str(type(path)))
        f = open(os.path.abspath(path), 'rb')
        breakpoint = pickle.load(f)
        f.close()
        if not isinstance(breakpoint, aceThreading):
            raise TypeError("Please enter the correct path to the multi-threading saving file.")

        # recover the workspace
        # init self
        recover_thread = cls(breakpoint.examples, breakpoint.labels, breakpoint.train_idx,
                                        breakpoint.test_idx, breakpoint.unlabel_index, breakpoint.label_index,
                                        breakpoint._target_func, breakpoint.max_thread,
                                        breakpoint.refresh_interval, breakpoint.saving_path)
        # loading tmp files
        state_path = os.path.join(breakpoint.saving_path, 'AL_result')
        recover_arr = [True] * breakpoint._round_num
        for i in range(breakpoint._round_num):
            file_dir = os.path.join(state_path, 'experiment_result_file_round_' + str(i) + '.pkl')
            if not breakpoint.alive_thread[i]:
                if os.path.exists(file_dir) and os.path.getsize(file_dir) != 0:
                    # all finished
                    recover_arr[i] = False
                else:
                    # not started
                    pass
            else:
                if os.path.getsize(file_dir) == 0:
                    # not saving, but started
                    continue
                # still running
                recover_thread.saver[i] = StateIO.load(
                    os.path.join(state_path, 'experiment_result_file_round_' + str(i) + '.pkl'))
                tmp = recover_thread.saver[i].get_workspace()
                recover_thread.unlabel_index[i] = tmp[2]
                recover_thread.label_index[i] = tmp[3]
        recover_thread.recover_arr = recover_arr
        # recover_thread.__init_threads(breakpoint._global_parameters)
        # recover_thread.__start_all_threads()
        return recover_thread


if __name__ == '__main__':
    pass
