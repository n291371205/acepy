from sklearn.datasets import load_iris
from data_process.al_split import ExperimentSetting
from analyser.experiment_analyser import ExperimentAnalyser
from experiment_saver.al_experiment import AlExperiment
from sklearn import linear_model
from utils.knowledge_db import MatrixKnowledgeDB
from experiment_saver.state import State

X, y = load_iris(return_X_y=True)
es = ExperimentSetting(X=X, y=y)
ea = ExperimentAnalyser()
reg = linear_model.LogisticRegression()

oracle = es.get_clean_oracle()
qs = es.uncertainty_selection()

ae = AlExperiment(method_name='uncertainty')
for round in range(5):
    saver = es.get_saver(round)
    train_id, test_id, Ucollection, Lcollection= es.get_split(round)
    db = MatrixKnowledgeDB(labels=y[Lcollection.index], examples=X[Lcollection.index,:], indexes=Lcollection.index)
    reg.fit(X=X[Lcollection.index, :], y=y[Lcollection.index])

    while len(Ucollection) > 10:
        select_index = qs.select(Ucollection, reg)
        queried_labels, cost = oracle.query_by_index(select_index)
        Ucollection.difference_update(select_index)
        Lcollection.update(select_index)
        db.update_query(labels=queried_labels, indexes=select_index, cost=cost, examples=X[select_index, :])
        # print(db.retrieve_by_indexes(select_index))
        # print(db.retrieve_by_examples(X[Lcollection.index, :]))

        # update model
        reg.fit(X=db.get_examples(), y=db.get_labels())
        pred = reg.predict(X[test_id, :])
        accuracy = sum(pred == y[test_id]) / len(test_id)

        st = State(select_index, accuracy)
        saver.add_state(st)
        saver.save()
    ae.add_fold(saver)

ea.add_method(ae)
ea.simple_plot()
