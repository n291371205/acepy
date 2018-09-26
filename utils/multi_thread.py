import threading
import random
import numpy as np
import time
import prettytable
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
    (round, train_id, test_id, Ucollection, Lcollection, saver, **global_parameters)
    """
    def __init__(self, examples, labels, train_idx, test_idx, label_index, unlabel_index, max_thread=None):
        self.examples = examples
        self.labels = labels
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.label_index = label_index
        self.unlabel_index = unlabel_index

        assert(len(train_idx) == len(test_idx) ==
               len(label_index) == len(unlabel_index))
        self._round_num = len(label_index)
        self.threads = []
        self.saver = [StateIO(round=i, train_idx=self.train_idx[i], test_idx=self.test_idx[i], init_U=self.unlabel_index[i],
                              init_L=self.label_index[i], verbose=False) for i in range(self._round_num)]
        if max_thread is None:
            self.max_thread = self._round_num
        else:
            self.max_thread = max_thread
        self._start_time = time.clock()

    def start_threads(self, thread_func, global_parameters=None):
        if global_parameters is not None:
            assert(isinstance(global_parameters, dict))
        # init thread objects
        for i in range(self._round_num):
            t = threading.Thread(target=thread_func, name=str(i), kwargs={
                'round': i, 'train_idx': self.train_idx[i], 'test_idx': self.test_idx[i],
                'unlabel_index': self.unlabel_index[i], 'label_index': self.label_index[i],
                'saver': self.saver[i],
                'global_parameters': global_parameters})
            self.threads.append(t)

        # start thread
        available_thread = self._round_num
        started_thread = 0
        for i in range(self._round_num):
            if available_thread > 0:
                self.threads[i].start()
                started_thread += 1
                available_thread -= 1
            else:
                while True:
                    if threading.active_count() < self.max_thread:
                        break
                available_thread += self.max_thread-threading.active_count()
                self.threads[i].start()
                started_thread += 1
                available_thread -= 1
        for i in range(self._round_num):
            self.threads[i].join()

    def show_state(self):
        pass

    def _if_refresh(self):
        if time.clock() - self._start_time > 1:
            return True
        else:
            return False

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from data_process.al_split import ExperimentSetting
    from analyser.experiment_analyser import ExperimentAnalyser
    from experiment_saver.al_experiment import AlExperiment
    from sklearn.ensemble import RandomForestClassifier
    from utils.knowledge_db import MatrixKnowledgeDB
    from experiment_saver.state import State
    from sklearn.preprocessing import LabelBinarizer

    X, y = load_iris(return_X_y=True)
    es = ExperimentSetting(X=X, y=y)
    ea = ExperimentAnalyser()
    reg = RandomForestClassifier()

    oracle = es.get_clean_oracle()
    qs = es.uncertainty_selection()

    ae = AlExperiment(method_name='uncertainty')

    train_id, test_id, Ucollection, Lcollection = es.get_split(round)

    def run_thread(round, train_id, test_id, Ucollection, Lcollection, saver, **global_parameters):
        db = es.get_knowledge_db(round)
        reg.fit(X=db.get_examples(), y=db.get_labels())

        while len(Ucollection) > 10:
            select_index = qs.select(Ucollection, reg)
            queried_labels, cost = oracle.query_by_index(select_index)
            Ucollection.difference_update(select_index)
            Lcollection.update(select_index)
            db.update_query(labels=queried_labels, indexes=select_index,
                            cost=cost, examples=X[select_index, :])
            # print(db.retrieve_by_indexes(select_index))
            # print(db.retrieve_by_examples(X[Lcollection.index, :]))

            # update model
            reg.fit(X=db.get_examples(), y=db.get_labels())
            pred = reg.predict(X[test_id, :])
            accuracy = sum(pred == y[test_id]) / len(test_id)

            st = State(select_index=select_index, performance=accuracy)
            saver.add_state(st)
            saver.save()
        return saver
