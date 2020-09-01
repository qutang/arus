

from ..extensions.sklearn import ELM
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.estimator_checks import check_estimator
import numpy as np


def test_elm():
    bunch = datasets.load_iris(return_X_y=False)
    X = bunch['data']
    y = np.array([bunch['target_names'][i] for i in bunch['target']])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=None)
    print(X_train.shape)
    print(X_test.shape)
    print(bunch['target'][:5])
    print(y[:5])
    print(bunch['target_names'])

    elm = ELM(hidden_layer_sizes=(10000,))
    elm.fit(X_train, y_train, labels=bunch['target_names'])
    pred_y = elm.predict(X_test)
    pred_probs = elm.predict_proba(X_test)
    print(accuracy_score(y_test, pred_y))
    print(pred_y[:5])
    print(bunch['target_names'])
    print(pred_probs[:5, :])
    check_estimator(elm)
    assert True
