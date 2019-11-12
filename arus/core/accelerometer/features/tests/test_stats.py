import numpy as np
from .. import stats


def test_mean():
    # test on a single row
    X = np.array([[1., 1., 1., ]])
    result = stats.mean(X)
    np.testing.assert_array_equal(result[0], X)
    np.testing.assert_array_equal(result[1], ['MEAN_0', 'MEAN_1', 'MEAN_2'])

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.mean(X)
    np.testing.assert_array_equal(result[0], X)
    np.testing.assert_array_equal(result[1], ['MEAN_0', 'MEAN_1', 'MEAN_2'])

    # test on an array
    X = np.array([[1., 1., 1., ], [2., 2., 2.]])
    result = stats.mean(X)
    np.testing.assert_array_equal(result[0], np.array([[1.5, 1.5, 1.5]]))
    np.testing.assert_array_equal(result[1], ['MEAN_0', 'MEAN_1', 'MEAN_2'])

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
    result = stats.mean(X)
    np.testing.assert_array_equal(result[0], np.array([[1.5, 1, 1.5]]))
    np.testing.assert_array_equal(result[1], ['MEAN_0', 'MEAN_1', 'MEAN_2'])


def test_median():
    # test on a single row
    X = np.array([[1., 1., 1., ]])
    result = stats.median(X)
    np.testing.assert_array_equal(result[0], X)
    np.testing.assert_array_equal(
        result[1], ['MEDIAN_0', 'MEDIAN_1', 'MEDIAN_2'])

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.median(X)
    np.testing.assert_array_equal(result[0], X)
    np.testing.assert_array_equal(
        result[1], ['MEDIAN_0', 'MEDIAN_1', 'MEDIAN_2'])

    # test on an array
    X = np.array([[1., 1., 1., ], [2., 2., 2.]])
    result = stats.median(X)
    np.testing.assert_array_equal(result[0], np.array([[1.5, 1.5, 1.5]]))
    np.testing.assert_array_equal(
        result[1], ['MEDIAN_0', 'MEDIAN_1', 'MEDIAN_2'])

    # test on larger array
    X = np.array([[1., 1., 1., ], [2., 2., 2.], [5., 5., 5.]])
    result = stats.median(X)
    np.testing.assert_array_equal(result[0], np.array([[2, 2, 2]]))
    np.testing.assert_array_equal(
        result[1], ['MEDIAN_0', 'MEDIAN_1', 'MEDIAN_2'])

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
    result = stats.median(X)
    np.testing.assert_array_equal(result[0], np.array([[1.5, 1, 1.5]]))
    np.testing.assert_array_equal(
        result[1], ['MEDIAN_0', 'MEDIAN_1', 'MEDIAN_2'])


def test_std():
    # test on a single row
    X = np.array([[1., 1., 1., ]])
    result = stats.std(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[np.nan, np.nan, np.nan, ]]))
    np.testing.assert_array_equal(result[1], ['STD_0', 'STD_1', 'STD_2'])

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.std(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[np.nan, np.nan, np.nan]]))
    np.testing.assert_array_equal(result[1], ['STD_0', 'STD_1', 'STD_2'])

    # test on an array
    X = np.array([[1., 1., 1., ], [2., 2., 2.]])
    result = stats.std(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)]]))
    np.testing.assert_array_equal(result[1], ['STD_0', 'STD_1', 'STD_2'])

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
    result = stats.std(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[np.sqrt(0.5), np.nan, np.sqrt(0.5)]]))
    np.testing.assert_array_equal(result[1], ['STD_0', 'STD_1', 'STD_2'])

    # test on an array with more than two rows with nan
    X = np.array([[1., 1., 1., ], [2., 2, 2.], [1.5, np.nan, 1.5]])
    result = stats.std(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[np.sqrt(0.25), np.sqrt(0.5), np.sqrt(0.25)]]))
    np.testing.assert_array_equal(result[1], ['STD_0', 'STD_1', 'STD_2'])


def test_skew():
    # test on a single row
    X = np.array([[1., 1., 1., ]])
    result = stats.skew(X)
    np.testing.assert_array_equal(result[0], np.atleast_2d([0, 0, 0]))
    np.testing.assert_array_equal(
        result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.skew(X)
    np.testing.assert_array_equal(result[0], np.atleast_2d([0, np.nan, 0]))
    np.testing.assert_array_equal(
        result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])

    # test on an array
    X = np.array([[1., 1., 1., ], [2., 2., 2.]])
    result = stats.skew(X)
    np.testing.assert_array_equal(result[0], np.array([[0, 0, 0]]))
    np.testing.assert_array_equal(
        result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])

    # test on normal distribution
    X = np.random.randn(1000000, 3)
    result = stats.skew(X)
    np.testing.assert_array_almost_equal(
        result[0], np.array([[0, 0, 0]]), decimal=1)
    np.testing.assert_array_equal(
        result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])

    # test on normal distribution with nan
    X = np.random.randn(1000000, 3)
    X[np.random.randint(0, 49, 1), 0:3] = np.nan
    result = stats.skew(X)
    np.testing.assert_array_almost_equal(
        result[0], np.array([[0, 0, 0]]), decimal=1)
    np.testing.assert_array_equal(
        result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])

    # test on exponential distribution
    X = np.random.standard_exponential(size=(1000000, 3))
    result = stats.skew(X)
    np.testing.assert_array_almost_equal(
        result[0], np.array([[2, 2, 2]]), decimal=1)
    np.testing.assert_array_equal(
        result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])


def test_kurtosis():
    # test on a single row
    X = np.array([[1., 1., 1., ]])
    result = stats.kurtosis(X)
    np.testing.assert_array_equal(result[0], np.atleast_2d([-3, -3, -3]))
    np.testing.assert_array_equal(
        result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.kurtosis(X)
    np.testing.assert_array_equal(result[0], np.atleast_2d([-3, np.nan, -3]))
    np.testing.assert_array_equal(
        result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])

    # test on uniform distribution
    X = np.random.uniform(low=0, high=1, size=(1000000, 3))
    result = stats.kurtosis(X)
    np.testing.assert_array_almost_equal(
        result[0], np.array([[-1.2, -1.2, -1.2]]), decimal=2)
    np.testing.assert_array_equal(
        result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])

    # test on normal distribution
    X = np.random.randn(1000000, 3)
    result = stats.kurtosis(X)
    np.testing.assert_array_almost_equal(
        result[0], np.array([[0, 0, 0]]), decimal=1)
    np.testing.assert_array_equal(
        result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])

    # test on normal distribution with nan
    X = np.random.randn(1000000, 3)
    X[np.random.randint(0, 49, 1), 0:3] = np.nan
    result = stats.kurtosis(X)
    np.testing.assert_array_almost_equal(
        result[0], np.array([[0, 0, 0]]), decimal=1)
    np.testing.assert_array_equal(
        result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])

    # test on laplace distribution
    X = np.random.laplace(loc=0, scale=1, size=(1000000, 3))
    result = stats.kurtosis(X)
    np.testing.assert_array_almost_equal(
        result[0], np.array([[3, 3, 3]]), decimal=1)
    np.testing.assert_array_equal(
        result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])


def test_max_value():
    # test on a single row
    X = np.array([[1., 1., 1., ]])
    result = stats.max_value(X)
    np.testing.assert_array_equal(result[0], X)
    np.testing.assert_array_equal(result[1], ['MAX_0', 'MAX_1', 'MAX_2'])

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.max_value(X)
    np.testing.assert_array_equal(result[0], X)
    np.testing.assert_array_equal(result[1], ['MAX_0', 'MAX_1', 'MAX_2'])

    # test on an array
    X = np.array([[1., 1., 1., ], [2., 2., 2.]])
    result = stats.max_value(X)
    np.testing.assert_array_equal(result[0], np.array([[2, 2, 2]]))
    np.testing.assert_array_equal(result[1], ['MAX_0', 'MAX_1', 'MAX_2'])

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
    result = stats.max_value(X)
    np.testing.assert_array_equal(result[0], np.array([[2, 1, 2]]))
    np.testing.assert_array_equal(result[1], ['MAX_0', 'MAX_1', 'MAX_2'])


def test_min_value():
    # test on a single row
    X = np.array([[-1., -1., -1., ]])
    result = stats.min_value(X)
    np.testing.assert_array_equal(result[0], X)
    np.testing.assert_array_equal(result[1], ['MIN_0', 'MIN_1', 'MIN_2'])

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.min_value(X)
    np.testing.assert_array_equal(result[0], X)
    np.testing.assert_array_equal(result[1], ['MIN_0', 'MIN_1', 'MIN_2'])

    # test on an array
    X = np.array([[1., 1., 1., ], [2., 2., 2.]])
    result = stats.min_value(X)
    np.testing.assert_array_equal(result[0], np.array([[1, 1, 1]]))
    np.testing.assert_array_equal(result[1], ['MIN_0', 'MIN_1', 'MIN_2'])

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
    result = stats.min_value(X)
    np.testing.assert_array_equal(result[0], np.array([[1, 1, 1]]))
    np.testing.assert_array_equal(result[1], ['MIN_0', 'MIN_1', 'MIN_2'])


def test_range():
    # test on a single row
    X = np.array([[-1., -1., -1.]])
    result = stats.max_minus_min(X)
    np.testing.assert_array_equal(result[0], np.array([[0., 0., 0.]]))
    np.testing.assert_array_equal(result[1], ['RANGE_0', 'RANGE_1', 'RANGE_2'])

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.max_minus_min(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[0., np.nan, 0., ]]))
    np.testing.assert_array_equal(result[1], ['RANGE_0', 'RANGE_1', 'RANGE_2'])

    # test on an array
    X = np.array([[1., 1., 1., ], [3., 3., 3.]])
    result = stats.max_minus_min(X)
    np.testing.assert_array_equal(result[0], np.array([[2, 2, 2]]))
    np.testing.assert_array_equal(result[1], ['RANGE_0', 'RANGE_1', 'RANGE_2'])

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [3., np.nan, 3.]])
    result = stats.max_minus_min(X)
    np.testing.assert_array_equal(result[0], np.array([[2, 0, 2]]))
    np.testing.assert_array_equal(result[1], ['RANGE_0', 'RANGE_1', 'RANGE_2'])


def test_abs_max_value():
    # test on a single row
    X = np.array([[1., -1., 1., ]])
    result = stats.abs_max_value(X)
    np.testing.assert_array_equal(result[0], np.array([[1., 1., 1., ]]))
    np.testing.assert_array_equal(
        result[1], ['ABS_MAX_0', 'ABS_MAX_1', 'ABS_MAX_2'])

    # test on a single row with nan
    X = np.array([[1., np.nan, -1., ]])
    result = stats.abs_max_value(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[1., np.nan, 1., ]]))
    np.testing.assert_array_equal(
        result[1], ['ABS_MAX_0', 'ABS_MAX_1', 'ABS_MAX_2'])

    # test on an array
    X = np.array([[1., 1., -1., ], [-2., 2., 2.]])
    result = stats.abs_max_value(X)
    np.testing.assert_array_equal(result[0], np.array([[2, 2, 2]]))
    np.testing.assert_array_equal(
        result[1], ['ABS_MAX_0', 'ABS_MAX_1', 'ABS_MAX_2'])

    # test on an array with nan
    X = np.array([[1., -1., 1., ], [-2., np.nan, 2.]])
    result = stats.abs_max_value(X)
    np.testing.assert_array_equal(result[0], np.array([[2, 1, 2]]))
    np.testing.assert_array_equal(
        result[1], ['ABS_MAX_0', 'ABS_MAX_1', 'ABS_MAX_2'])


def test_abs_min_value():
    # test on a single row
    X = np.array([[-1., -1., -1.]])
    result = stats.abs_min_value(X)
    np.testing.assert_array_equal(result[0], np.array([[1., 1., 1.]]))
    np.testing.assert_array_equal(
        result[1], ['ABS_MIN_0', 'ABS_MIN_1', 'ABS_MIN_2'])

    # test on a single row with nan
    X = np.array([[-1., np.nan, -1., ]])
    result = stats.abs_min_value(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[1., np.nan, 1., ]]))
    np.testing.assert_array_equal(
        result[1], ['ABS_MIN_0', 'ABS_MIN_1', 'ABS_MIN_2'])

    # test on an array
    X = np.array([[1., -1., 1., ], [-2., 2., 2.]])
    result = stats.abs_min_value(X)
    np.testing.assert_array_equal(result[0], np.array([[1, 1, 1]]))
    np.testing.assert_array_equal(
        result[1], ['ABS_MIN_0', 'ABS_MIN_1', 'ABS_MIN_2'])

    # test on an array with nan
    X = np.array([[1., -1., 1., ], [2., np.nan, -2.]])
    result = stats.abs_min_value(X)
    np.testing.assert_array_equal(result[0], np.array([[1, 1, 1]]))
    np.testing.assert_array_equal(
        result[1], ['ABS_MIN_0', 'ABS_MIN_1', 'ABS_MIN_2'])


def test_correlation():
    # test on a single row
    X = np.array([[1., 1., 1.]])
    result = stats.correlation(X)
    np.testing.assert_array_equal(
        result[0], np.array([[np.nan, np.nan, np.nan]]))
    np.testing.assert_array_equal(
        result[1], ['CORRELATION_0', 'CORRELATION_1', 'CORRELATION_2'])

    # test on a single row with nan
    X = np.array([[-1., np.nan, -1., ]])
    result = stats.correlation(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[np.nan, np.nan, np.nan]]))
    np.testing.assert_array_equal(
        result[1], ['CORRELATION_0', 'CORRELATION_1', 'CORRELATION_2'])

    # test on an array
    x = np.transpose(np.atleast_2d(np.arange(10)))
    y = np.transpose(-np.atleast_2d(np.arange(10)))
    z = np.transpose(np.atleast_2d(np.arange(10)/5))
    X = np.hstack((x, y, z))
    result = stats.correlation(X)
    np.testing.assert_array_almost_equal(result[0], np.array([[-1, 1, -1]]))
    np.testing.assert_array_equal(
        result[1], ['CORRELATION_0', 'CORRELATION_1', 'CORRELATION_2'])

    # test on an array with nan
    x = np.transpose(np.atleast_2d(np.arange(10)))
    y = np.transpose(-np.atleast_2d(np.arange(10)))
    z = np.transpose(np.atleast_2d(np.arange(10)/5))
    X = np.hstack((x, y, z))
    X[np.random.randint(0, 10, 1), 0:3] = np.nan
    result = stats.correlation(X)
    np.testing.assert_array_equal(
        result[0], np.array([[np.nan, np.nan, np.nan]]))
    np.testing.assert_array_equal(
        result[1], ['CORRELATION_0', 'CORRELATION_1', 'CORRELATION_2'])
