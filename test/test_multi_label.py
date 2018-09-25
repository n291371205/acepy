from sklearn.datasets import load_iris
from data_process.al_split import ExperimentSetting
from analyser.experiment_analyser import ExperimentAnalyser
from experiment_saver.al_experiment import AlExperiment
from sklearn.ensemble import RandomForestClassifier
from utils.knowledge_db import MatrixKnowledgeDB
from experiment_saver.state import State
from sklearn.preprocessing import LabelBinarizer

X, y = load_iris(return_X_y=True)
y = LabelBinarizer().fit_transform(y=y)
# print(y)
es = ExperimentSetting(X=X, y=y, partially_labeled=False)
ea = ExperimentAnalyser()
reg = RandomForestClassifier()

oracle = es.get_clean_oracle()
qs = es.random_selection()

ae = AlExperiment(method_name='uncertainty')
for round in range(5):
    saver = es.get_saver(round)
    train_id, test_id, Ucollection, Lcollection= es.get_split(round)
    # db = MatrixKnowledgeDB(labels=y[Lcollection.index,:], examples=X[Lcollection.index,:], indexes=Lcollection.index)
    # print(db.get_labels())
    # reg.fit(X=db.get_examples(), y=y[Lcollection.index])

    while len(Ucollection) > 10:
        select_index = qs.select(Ucollection, batch_size=5)
        queried_labels, cost = oracle.query_by_index(select_index)
        Ucollection.difference_update(select_index)
        Lcollection.update(select_index)
        # db.update_query(labels=queried_labels, indexes=select_index, cost=cost, examples=X[select_index, :])
        # print(db.retrieve_by_indexes(select_index))
        # print(db.retrieve_by_examples(X[Lcollection.index, :]))

        # update model
        # reg.fit(X=db.get_examples(), y=y[Lcollection.index])
        # pred = reg.predict(X[test_id, :])
        # accuracy = sum(pred == y[test_id]) / len(test_id)

        st = State(select_index, performance=0.5)
        saver.add_state(st)
        saver.save()
    ae.add_fold(saver)

ea.add_method(ae)
ea.simple_plot()
