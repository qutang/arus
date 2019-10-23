

from ..fill_nan import fill_nan
import numpy as np


def test_fill_nan():
    # test signal without nan
    X = np.random.rand(10, 3)
    X_new = fill_nan(X)
    np.testing.assert_array_almost_equal(X, X_new)

    X = np.random.rand(10, 1)
    X_new = fill_nan(X)
    np.testing.assert_array_almost_equal(X, X_new)

    # test signal with single sample without nan
    X = np.array([[0.]])
    X_new = fill_nan(X)
    np.testing.assert_array_almost_equal(X, X_new)

    X = np.array([[0., 0, 0, ]])
    X_new = fill_nan(X)
    np.testing.assert_array_almost_equal(X, X_new)

    # test signal with nan
    X = np.atleast_2d(np.sin(2*np.pi * 1 * np.arange(0, 1, 1.0 / 100))).T
    X_nan = np.copy(X)
    X_nan[5:10, 0] = np.nan
    X_new = fill_nan(X_nan)
    np.testing.assert_array_almost_equal(X, X_new, decimal=4)

    X = np.tile(np.sin(2*np.pi * 1 *
                       np.arange(0, 1, 1.0 / 100.)), (3, 1)).T
    X_nan = np.copy(X)
    X_nan[5:10, 0:3] = np.nan
    X_new = fill_nan(X_nan)
    np.testing.assert_array_almost_equal(X, X_new, decimal=4)
