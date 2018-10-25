import copy
from sklearn.datasets import load_iris,make_classification
from experiment_saver.state import State

from query_strategy.query_strategy import (QueryInstanceQBC,
                                           QueryInstanceUncertainty,
                                           QueryRandom,
                                           QureyExpectedErrorReduction)
from query_strategy.third_party_methods import QueryInstanceQUIRE, QueryInstanceGraphDensity
from utils.al_collections import IndexCollection
from experiment_saver.al_experiment import ToolBox

X, y = make_classification(n_samples=150, n_features=20, n_informative=2, n_redundant=2, 
    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, 
    hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
split_count = 5
acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None)

# split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=split_count)

# use the default Logistic Regression classifier
model = acebox.default_model()

# query 50 times
stopping_criterion = acebox.stopping_criterion('num_of_queries', 50)

# use pre-defined strategy, The data matrix is a reference which will not use additional memory
QBCStrategy = QueryInstanceQBC(X, y)
randomStrategy = QueryRandom()
uncertainStrategy = QueryInstanceUncertainty(X, y)
QUIREStrategy = QueryInstanceQUIRE(X, y)
EER = QureyExpectedErrorReduction(X, y)

QBC_result = []
for round in range(split_count):
    train_idx, test_idx, Lind, Uind = acebox.get_split(round)
    saver = acebox.StateIO(round)

    # calc the initial point
    model.fit(X=X[Lind.index, :], y=y[Lind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)

    saver.set_initial_point(accuracy)
    while not stopping_criterion.is_stop():
        select_ind = QBCStrategy.select(Lind, Uind, model=model)
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
    QBC_result.append(copy.deepcopy(saver))

random_result = []
for round in range(split_count):
    train_idx, test_idx, Lind, Uind = acebox.get_split(round)
    saver = acebox.StateIO(round)

    # calc the initial point
    model.fit(X=X[Lind.index, :], y=y[Lind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)

    saver.set_initial_point(accuracy)
    while not stopping_criterion.is_stop():
        select_ind = randomStrategy.select(Lind, Uind)
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
    random_result.append(copy.deepcopy(saver))

uncertainty_result = []
for round in range(split_count):
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

        # save intermediate result
        st = State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # update stopping_criteria
        stopping_criterion.update_information(saver)
    stopping_criterion.reset()
    uncertainty_result.append(copy.deepcopy(saver))

QUIRE_result = []
for round in range(split_count):
    train_idx, test_idx, Lind, Uind = acebox.get_split(round)
    saver = acebox.StateIO(round)

    # calc the initial point
    model.fit(X=X[Lind.index, :], y=y[Lind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)

    saver.set_initial_point(accuracy)
    while not stopping_criterion.is_stop():
        select_ind = QUIREStrategy.select(Lind, Uind)
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
    QUIRE_result.append(copy.deepcopy(saver))

density_result = []
for round in range(split_count):
    train_idx, test_idx, Lind, Uind = acebox.get_split(round)
    saver = acebox.StateIO(round)
    densityStrategy = QueryInstanceGraphDensity(X, y, train_idx=train_idx)

    # calc the initial point
    model.fit(X=X[Lind.index, :], y=y[Lind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)

    saver.set_initial_point(accuracy)
    while not stopping_criterion.is_stop():
        select_ind = densityStrategy.select(Lind, Uind)
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
    density_result.append(copy.deepcopy(saver))


EER_result = []
for round in range(split_count):
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


analyser = acebox.experiment_analyser()
analyser.add_method(QBC_result, 'QBC')
analyser.add_method(random_result, 'random')
analyser.add_method(uncertainty_result, 'uncertainty')
analyser.add_method(QUIRE_result, 'QUIRE')
analyser.add_method(EER_result, 'ExpectedErrorReduction')
analyser.add_method(density_result, 'density_graph')
print(analyser)
analyser.simple_plot(title='Iris')
