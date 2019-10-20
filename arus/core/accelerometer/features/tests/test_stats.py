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
