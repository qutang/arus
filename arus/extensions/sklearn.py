from joblib import Parallel, delayed
from collections.abc import Iterable


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
