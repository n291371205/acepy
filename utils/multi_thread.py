import threading
import random
import numpy as np


class aceThreading:
    """This class implement multi-threading in active learning for multiple 
    random splits experiments.

    It will display the progress of each thead. When all threads reach the 
    end points, it will return k StateIO objects for analysis.

    Once initialized, it can store and recover from any iterations and breakpoints.

    Note that, this class only provides visualization and file IO for threads, but
    not implement any threads. You should construct different threads by your own, 
    and then provide them as parameters for visualization.
    """
    pass


def query(U):
    print(U)
    return U[1]


def main_loop(query_func=object, U=None, L=None):
    Q = query_func(U)
    print(Q)
    L = [L, Q]


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

    def run_thread(round, train_id, test_id, Ucollection, Lcollection, 
    model, examples, labels, oracle, query_strategy, performance, global_parameters):
        saver = es.get_saver(round)
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

    # threads = []
    # # 创建线程
    # for i in range(round):
    #     t = threading.Thread(target=main_loop, args=(query, U[i, :], L[i, :]))
    #     threads.append(t)
    # # 启动线程
    # for i in range(round):
    #     threads[i].start()
    # for i in range(round):
    #     threads[i].join()
