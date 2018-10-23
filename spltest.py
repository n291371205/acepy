from sklearn.svm import SVC
from sklearn.datasets import samples_generator
from data_process.al_split import *
import numpy as np

# X, y = load_breast_cancer(return_X_y=True)
# n_samples = len(X)
# X = X[50:]
# y = y[50:]
# y = np.asarray([0]*50 + [1]*50)
# np.random.shuffle(y)

n_samples = 200
X, y = samples_generator.make_classification(n_samples=n_samples, flip_y=0.2)

# perm = list(range(n_samples))
# np.random.shuffle(perm)
# train_idx = np.asarray(perm[0:round(0.7*n_samples)])
# test_idx = np.asarray(perm[round(0.7*n_samples):])
train_idx,test_idx,_,_ = split(X, y, test_ratio=0.3, split_count=1)
train_idx = train_idx[0]
test_idx = test_idx[0]


model = SVC()
model.fit(X[train_idx], y[train_idx])
pred = model.predict(X[test_idx])
accuracy = sum(pred == y[test_idx]) / len(test_idx)
print('All accuracy:' + str(accuracy))

select_len = [round(n_samples*0.7*i) for i in [0.5,0.6,0.7,0.8,0.9,1.0]]
for length in select_len:
    val = model.decision_function(X[train_idx])
    hinge_loss = 1-y[train_idx]*val
    hinge_loss[hinge_loss<0] = 0
    # print('hinge loss:' + str(hinge_loss))
    sorted = np.argsort(hinge_loss)

    # select index
    next_train = train_idx[sorted[0:length]]
    model = SVC()
    model.fit(X[next_train], y[next_train])

    # test
    pred = model.predict(X[test_idx])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)
    print('Accuracy with %s samples is %s:' % (str(length), str(accuracy)))
