from sklearn import linear_model
from sklearn.datasets import load_iris

from analyser.experiment_analyser import ExperimentAnalyser
from data_process.al_split import split
from experiment_saver.al_experiment import AlExperiment
from experiment_saver.state import State
from experiment_saver.state_io import StateIO
from oracle.oracle import Oracle
# QBC
# QBC_ve
# random
# uncertainty
from query_strategy.query_strategy import (QueryInstanceQBC,
                                           QueryInstanceUncertainty,
                                           QueryRandom)
from utils.al_collections import IndexCollection

# X, y, _ = load_csv_data('C:\\Code\\altools\\dataset\\iris.csv')
X, y = load_iris(return_X_y=True)
Train_idx, Test_idx, U_pool, L_pool = split(X=X, y=y, test_ratio=0.3, initial_label_rate=0.2, split_count=5)
ea = ExperimentAnalyser()


qs = QueryInstanceUncertainty(X, y)
oracle = Oracle(y)

reg = linear_model.LogisticRegression()
ae = AlExperiment(method_name='uncertainty')

for round in range(5):
    train_id = Train_idx[round]
    test_id = Test_idx[round]
    Ucollection = IndexCollection(U_pool[round])
    Lcollection = IndexCollection(L_pool[round])
    # sup_db = KnowledgeDB(Lcollection, y)

    # initialize object
    reg.fit(X=X[Lcollection.index, :], y=y[Lcollection.index])
    pred = reg.predict(X[test_id, :])
    accuracy = sum(pred == y[test_id]) / len(test_id)
    # initialize StateIO module
    saver = StateIO(round, train_id, test_id, Ucollection, Lcollection, initial_point=accuracy)
    # saver = StateIO(round, train_id, test_id, Ucollection, Lcollection)
    while len(Ucollection) > 10:
        select_index = qs.select(Lcollection, Ucollection, reg)
        # accerlate version is available
        # sub_U = Ucollection.random_sampling()
        values, costs = oracle.query_by_index(select_index)
        Ucollection.difference_update(select_index)
        Lcollection.update(select_index)
        # db is optional
        # sup_db.update_query(select_index,values,costs)
        # reg.fit(X=X[Lcollection.index,:], y=sup_db.get_supervise_info(Lcollection))

        # update model
        reg.fit(X=X[Lcollection.index, :], y=y[Lcollection.index])
        pred = reg.predict(X[test_id, :])
        accuracy = sum(pred == y[test_id]) / len(test_id)

        # save intermediate results
        st = State(select_index=select_index, queried_label=values, cost=costs, performance=accuracy)
        # add user defined information
        # st.add_element(key='sub_ind', value=sub_ind)
        saver.add_state(st)
        saver.save()

        # get State and workspace
        # stored queried instances, labels, costs, performance and user defined items
        # using like a dict
        # st = saver.get_state(5)
        # Lcollection, Ucollection = saver.get_workspace(5)
    ae.add_fold(saver)
ea.add_method(ae)


qs = QueryRandom()
ae = AlExperiment(method_name='random')

for round in range(5):
    train_id = Train_idx[round]
    test_id = Test_idx[round]
    Ucollection = IndexCollection(U_pool[round])
    Lcollection = IndexCollection(L_pool[round])
    # sup_db = KnowledgeDB(Lcollection, y)

    # initialize object
    reg.fit(X=X[Lcollection.index, :], y=y[Lcollection.index])
    pred = reg.predict(X[test_id, :])
    accuracy = sum(pred == y[test_id]) / len(test_id)
    saver = StateIO(round, train_id, test_id, Ucollection, Lcollection, initial_point=accuracy)
    while len(Ucollection) > 10:
        select_index = qs.select(None, Ucollection)
        # accerlate version is available
        # sub_U = Ucollection.random_sampling()
        values, costs = oracle.query_by_index(select_index)
        Ucollection.difference_update(select_index)
        Lcollection.update(select_index)
        # db is optional
        # sup_db.update_query(select_index,values,costs)
        # reg.fit(X=X[Lcollection.index,:], y=sup_db.get_supervise_info(Lcollection))

        # update model
        reg.fit(X=X[Lcollection.index, :], y=y[Lcollection.index])
        pred = reg.predict(X[test_id, :])
        accuracy = sum(pred == y[test_id]) / len(test_id)

        # save intermediate results
        st = State(select_index=select_index, queried_label=values, cost=costs, performance=accuracy)
        # add user defined information
        # st.add_element(key='sub_ind', value=sub_ind)
        saver.add_state(st)
        saver.save()

        # get State and workspace
        # stored queried instances, labels, costs, performance and user defined items
        # using like a dict
        # st = saver.get_state(5)
        # Lcollection, Ucollection = saver.get_workspace(5)
    ae.add_fold(saver)
ea.add_method(ae)


qs = QueryInstanceQBC(X,y,disagreement='vote_entropy')
ae = AlExperiment(method_name='QBC_ve')

for round in range(5):
    train_id = Train_idx[round]
    test_id = Test_idx[round]
    Ucollection = IndexCollection(U_pool[round])
    Lcollection = IndexCollection(L_pool[round])
    # sup_db = KnowledgeDB(Lcollection, y)

    # initialize object
    reg.fit(X=X[Lcollection.index, :], y=y[Lcollection.index])
    pred = reg.predict(X[test_id, :])
    accuracy = sum(pred == y[test_id]) / len(test_id)
    saver = StateIO(round, train_id, test_id, Ucollection, Lcollection, initial_point=accuracy)
    while len(Ucollection) > 10:
        select_index = qs.select(Lcollection, Ucollection, reg)
        # accerlate version is available
        # sub_U = Ucollection.random_sampling()
        values, costs = oracle.query_by_index(select_index)
        Ucollection.difference_update(select_index)
        Lcollection.update(select_index)
        # db is optional
        # sup_db.update_query(select_index,values,costs)
        # reg.fit(X=X[Lcollection.index,:], y=sup_db.get_supervise_info(Lcollection))

        # update model
        reg.fit(X=X[Lcollection.index, :], y=y[Lcollection.index])
        pred = reg.predict(X[test_id, :])
        accuracy = sum(pred == y[test_id]) / len(test_id)

        # save intermediate results
        st = State(select_index=select_index, queried_label=values, cost=costs, performance=accuracy)
        # add user defined information
        # st.add_element(key='sub_ind', value=sub_ind)
        saver.add_state(st)
        saver.save()

        # get State and workspace
        # stored queried instances, labels, costs, performance and user defined items
        # using like a dict
        # st = saver.get_state(5)
        # Lcollection, Ucollection = saver.get_workspace(5)
    ae.add_fold(saver)
ea.add_method(ae)

qs = QueryInstanceQBC(X,y,disagreement='KL_divergence')
ae = AlExperiment(method_name='QBC_kl')

for round in range(5):
    train_id = Train_idx[round]
    test_id = Test_idx[round]
    Ucollection = IndexCollection(U_pool[round])
    Lcollection = IndexCollection(L_pool[round])
    # sup_db = KnowledgeDB(Lcollection, y)

    # initialize object
    reg.fit(X=X[Lcollection.index, :], y=y[Lcollection.index])
    pred = reg.predict(X[test_id, :])
    accuracy = sum(pred == y[test_id]) / len(test_id)
    saver = StateIO(round, train_id, test_id, Ucollection, Lcollection, initial_point=accuracy)
    while len(Ucollection) > 10:
        select_index = qs.select(Lcollection, Ucollection, reg)
        # accerlate version is available
        # sub_U = Ucollection.random_sampling()
        values, costs = oracle.query_by_index(select_index)
        Ucollection.difference_update(select_index)
        Lcollection.update(select_index)
        # db is optional
        # sup_db.update_query(select_index,values,costs)
        # reg.fit(X=X[Lcollection.index,:], y=sup_db.get_supervise_info(Lcollection))

        # update model
        reg.fit(X=X[Lcollection.index, :], y=y[Lcollection.index])
        pred = reg.predict(X[test_id, :])
        accuracy = sum(pred == y[test_id]) / len(test_id)

        # save intermediate results
        st = State(select_index=select_index, queried_label=values, cost=costs, performance=accuracy)
        # add user defined information
        # st.add_element(key='sub_ind', value=sub_ind)
        saver.add_state(st)
        saver.save()

        # get State and workspace
        # stored queried instances, labels, costs, performance and user defined items
        # using like a dict
        # st = saver.get_state(5)
        # Lcollection, Ucollection = saver.get_workspace(5)
    ae.add_fold(saver)
ea.add_method(ae)

print(ea)
# result analyse module
ea.simple_plot()
