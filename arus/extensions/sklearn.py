from joblib import Parallel, delayed
from collections.abc import Iterable
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv2
import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def _val_predict(clf, X_train, y_train, X_test, y_test, fit_params=None, method='predict'):
    fit_params = fit_params or {}
    clf.fit(X_train, y_train, **fit_params)
    predict = getattr(clf, method)
    y_pred = predict(X_test)
    return y_test, y_pred


def cross_val_predict(clf, X, y, groups=None, cv=None, n_jobs=None, fit_params=None, method='predict'):
    if isinstance(cv, Iterable):
        cv = cv
    elif hasattr(cv, "split") and hasattr(cv, "get_n_splits"):
        cv = cv.split(X, y, groups=groups)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(_val_predict)(
            clf=clf,
            X_train=X[train_idx, :],
            y_train=y[train_idx],
            X_test=X[test_idx, :],
            y_test=y[test_idx],
            fit_params=fit_params,
            method=method) for train_idx, test_idx in cv)
    preds = []
    tests = []
    for test, pred in results:
        tests += test.tolist()
        preds += pred.tolist()
    return tests, preds


class ELM(BaseEstimator):
    def __init__(self, hidden_layer_sizes=(10000,), random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        if random_state is None:
            self.random_state = np.random.randint(100)

    def _init_hidden_layers(self, input_size):
        np.random.seed(self.random_state)
        hidden_layer_sizes = self.hidden_layer_sizes
        self._input_layer_weights = np.random.normal(
            size=[input_size, hidden_layer_sizes[0]])
        self._input_layer_biases = np.random.normal(
            size=[hidden_layer_sizes[0]])

    def _activate(self, fn='relu'):
        if fn == 'relu':
            return lambda x: np.maximum(x, 0, x)

    def _feedforward(self, X):
        out1 = np.dot(X, self._input_layer_weights)
        out1 = out1 + self._input_layer_biases
        out2 = self._activate(fn='relu')(out1)
        return out2

    def _encode_y(self, y, labels=None, from_fit=True):
        if from_fit:
            if labels is None:
                labels = 'auto'
            else:
                labels = [labels]
            self._y_encoder = OneHotEncoder(categories=labels)
            return self._y_encoder.fit_transform(np.expand_dims(y, axis=1)).toarray().astype(int)
        else:
            return self._y_encoder.transform(y).toarray().astype(int)

    def _get_labels(self, y):
        self._labels = np.unique(y)

    def fit(self, X, y, labels=None):
        X = np.real(X).astype(float)
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        self._init_hidden_layers(X.shape[1])
        encoded_y = self._encode_y(y, labels=labels, from_fit=True)
        out = self._feedforward(X)
        self._output_layer_weights = np.dot(pinv2(out), encoded_y)
        return self

    def _predict_scores(self, X):
        check_is_fitted(self)
        X = np.real(X).astype(float)
        X = check_array(X)
        out = self._feedforward(X)
        out = np.dot(out, self._output_layer_weights)
        return out

    def predict(self, X):
        scores = self._predict_scores(X)
        encoded_scores = np.zeros(scores.shape)
        for i in range(scores.shape[0]):
            encoded_scores[i] = (scores[i] == np.max(scores[i])).astype(int)
        pred_y = self._y_encoder.inverse_transform(encoded_scores)
        return pred_y

    def predict_proba(self, X):
        scores = self._predict_scores(X)
        pred_probs = softmax(scores, axis=1)
        return pred_probs

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(
            list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x ==
                                 unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

    def inverse_transform(self, id_list):
        unknown_id = self.label_encoder.transform(['Unknown'])[0]
        new_id_list = list(id_list)
        for unique_item in np.unique(id_list):
            if unique_item == -1:
                new_id_list = [unknown_id if x ==
                               unique_item else x for x in new_id_list]

        return self.label_encoder.inverse_transform(new_id_list)
