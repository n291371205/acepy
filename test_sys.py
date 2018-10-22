import copy
from sklearn.datasets import load_iris
from experiment_saver.state import State

from query_strategy.query_strategy import (QueryInstanceQBC,
                                           QueryInstanceUncertainty,
                                           QueryRandom,
                                           QureyExpectedErrorReduction)
from query_strategy.third_party_methods import QueryInstanceQUIRE, QueryInstanceGraphDensity
from utils.al_collections import IndexCollection
from experiment_saver.al_experiment import ToolBox
from metrics.performance import accuracy_score

X, y = load_iris(return_X_y=True)
acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None)

# split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)

# use the default SVM classifier
model = acebox.default_model()

# query 50 times
stopping_criterion = acebox.stopping_criterion('num_of_queries', 50)

# use pre-defined strategy, The data matrix is a reference which will not use additional space

EER = QureyExpectedErrorReduction(X, y,)
uncertainStrategy = QueryInstanceUncertainty(X, y)

EER_result = []
for round in range(10):
    train_idx, test_idx, Lind, Uind = acebox.get_split(round)
    saver = acebox.StateIO(round)

    # calc the initial point
    model.fit(X=X[Lind.index, :], y=y[Lind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)

    saver.set_initial_point(accuracy)
    while not stopping_criterion.is_stop():
        select_ind = EER.select(Lind, Uind, model=model)
        Lind.update(select_ind)
        Uind.difference_update(select_ind)

        # update model and calc performance
        model.fit(X=X[Lind.index, :], y=y[Lind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = sum(pred == y[test_idx]) / len(test_idx)

        # save intermediate result
        st = State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # update stopping_criteria
        stopping_criterion.update_information(saver)
    stopping_criterion.reset()
    EER_result.append(copy.deepcopy(saver))

uncertainty_result = []
for round in range(5):
    train_idx, test_idx, Lind, Uind = acebox.get_split(round)
    saver = acebox.StateIO(round)

    # calc the initial point
    model.fit(X=X[Lind.index, :], y=y[Lind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)

    saver.set_initial_point(accuracy)
    while not stopping_criterion.is_stop():
        select_ind = uncertainStrategy.select(Lind, Uind, model=model)
        Lind.update(select_ind)
        Uind.difference_update(select_ind)

        # update model and calc performance
        model.fit(X=X[Lind.index, :], y=y[Lind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = sum(pred == y[test_idx]) / len(test_idx)
        a = accuracy_score(y[test_idx], pred)
        # print('round{0},accuarcy: {1}'.format(round, a))
        # save intermediate result
        st = State(select_index=select_ind, performance=accuracy)
        st2 = State(select_index=select_ind, performance=a)
        saver.add_state(st)
        saver.add_state(st2)
        saver.save()

        # update stopping_criteria
        stopping_criterion.update_information(saver)
    stopping_criterion.reset()
    uncertainty_result.append(copy.deepcopy(saver))


analyser = acebox.experiment_analyser()
analyser.add_method(EER_result, 'ExpectedErrorReduction')
analyser.add_method(uncertainty_result, 'uncertainty')
print(analyser)
analyser.simple_plot(title='Iris')