import numpy as np
from .. import stats


def test_mean():
    # test on a single row
    X = np.array([[1., 1., 1., ]])
    result = stats.mean(X)
    assert np.allclose(result[0], X, atol=0.001)
    assert result[1] == 'MEAN'

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.mean(X)
    assert np.allclose(result[0], X, atol=0.001, equal_nan=True)
    assert result[1] == 'MEAN'

    # test on an array
    X = np.array([[1., 1., 1., ], [2., 2., 2.]])
    result = stats.mean(X)
    assert np.allclose(result[0], np.array([[1.5, 1.5, 1.5]]), atol=0.001)
    assert result[1] == 'MEAN'

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
    result = stats.mean(X)
    assert np.allclose(result[0], np.array([[1.5, 1, 1.5]]), atol=0.001)
    assert result[1] == 'MEAN'


def test_std():
    # test on a single row
    X = np.array([[1., 1., 1., ]])
    result = stats.std(X)
    assert np.allclose(result[0], np.array(
        [[np.nan, np.nan, np.nan, ]]), atol=0.001, equal_nan=True)
    assert result[1] == 'STD'

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.std(X)
    assert np.allclose(result[0], np.array(
        [[np.nan, np.nan, np.nan]]), atol=0.001, equal_nan=True)
    assert result[1] == 'STD'

    # test on an array
    X = np.array([[1., 1., 1., ], [2., 2., 2.]])
    result = stats.std(X)
    assert np.allclose(result[0], np.array(
        [[np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)]]), atol=0.001)
    assert result[1] == 'STD'

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
    result = stats.std(X)
    assert np.allclose(result[0], np.array(
        [[np.sqrt(0.5), np.nan, np.sqrt(0.5)]]), atol=0.001, equal_nan=True)
    assert result[1] == 'STD'

    # test on an array with more than two rows with nan
    X = np.array([[1., 1., 1., ], [2., 2, 2.], [1.5, np.nan, 1.5]])
    result = stats.std(X)
    assert np.allclose(result[0], np.array(
        [[np.sqrt(0.25), np.sqrt(0.5), np.sqrt(0.25)]]), atol=0.001, equal_nan=True)
    assert result[1] == 'STD'


def test_max_value():
    # test on a single row
    X = np.array([[1., 1., 1., ]])
    result = stats.max_value(X)
    assert np.allclose(result[0], X, atol=0.001)
    assert result[1] == 'MAX'

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.max_value(X)
    assert np.allclose(result[0], X, atol=0.001, equal_nan=True)
    assert result[1] == 'MAX'

    # test on an array
    X = np.array([[1., 1., 1., ], [2., 2., 2.]])
    result = stats.max_value(X)
    assert np.allclose(result[0], np.array([[2, 2, 2]]), atol=0.001)
    assert result[1] == 'MAX'

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
    result = stats.max_value(X)
    assert np.allclose(result[0], np.array([[2, 1, 2]]), atol=0.001)
    assert result[1] == 'MAX'


def test_min_value():
    # test on a single row
    X = np.array([[-1., -1., -1., ]])
    result = stats.min_value(X)
    assert np.allclose(result[0], X, atol=0.001)
    assert result[1] == 'MIN'

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.min_value(X)
    assert np.allclose(result[0], X, atol=0.001, equal_nan=True)
    assert result[1] == 'MIN'

    # test on an array
    X = np.array([[1., 1., 1., ], [2., 2., 2.]])
    result = stats.min_value(X)
    assert np.allclose(result[0], np.array([[1, 1, 1]]), atol=0.001)
    assert result[1] == 'MIN'

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
    result = stats.min_value(X)
    assert np.allclose(result[0], np.array([[1, 1, 1]]), atol=0.001)
    assert result[1] == 'MIN'


def test_range():
    # test on a single row
    X = np.array([[-1., -1., -1.]])
    result = stats.range(X)
    assert np.allclose(result[0], np.array([[0., 0., 0.]]), atol=0.001)
    assert result[1] == 'RANGE'

    # test on a single row with nan
    X = np.array([[1., np.nan, 1., ]])
    result = stats.range(X)
    assert np.allclose(result[0], np.array(
        [[0., np.nan, 0., ]]), atol=0.001, equal_nan=True)
    assert result[1] == 'RANGE'

    # test on an array
    X = np.array([[1., 1., 1., ], [3., 3., 3.]])
    result = stats.range(X)
    assert np.allclose(result[0], np.array([[2, 2, 2]]), atol=0.001)
    assert result[1] == 'RANGE'

    # test on an array with nan
    X = np.array([[1., 1., 1., ], [3., np.nan, 3.]])
    result = stats.range(X)
    assert np.allclose(result[0], np.array([[2, 0, 2]]), atol=0.001)
    assert result[1] == 'RANGE'


def test_abs_max_value():
    # test on a single row
    X = np.array([[1., -1., 1., ]])
    result = stats.abs_max_value(X)
    assert np.allclose(result[0], np.array([[1., 1., 1., ]]), atol=0.001)
    assert result[1] == 'ABS_MAX'

    # test on a single row with nan
    X = np.array([[1., np.nan, -1., ]])
    result = stats.abs_max_value(X)
    assert np.allclose(result[0], np.array(
        [[1., np.nan, 1., ]]), atol=0.001, equal_nan=True)
    assert result[1] == 'ABS_MAX'

    # test on an array
    X = np.array([[1., 1., -1., ], [-2., 2., 2.]])
    result = stats.abs_max_value(X)
    assert np.allclose(result[0], np.array([[2, 2, 2]]), atol=0.001)
    assert result[1] == 'ABS_MAX'

    # test on an array with nan
    X = np.array([[1., -1., 1., ], [-2., np.nan, 2.]])
    result = stats.abs_max_value(X)
    assert np.allclose(result[0], np.array([[2, 1, 2]]), atol=0.001)
    assert result[1] == 'ABS_MAX'


def test_abs_min_value():
    # test on a single row
    X = np.array([[-1., -1., -1.]])
    result = stats.abs_min_value(X)
    assert np.allclose(result[0], np.array([[1., 1., 1.]]), atol=0.001)
    assert result[1] == 'ABS_MIN'

    # test on a single row with nan
    X = np.array([[-1., np.nan, -1., ]])
    result = stats.abs_min_value(X)
    assert np.allclose(result[0], np.array(
        [[1., np.nan, 1., ]]), atol=0.001, equal_nan=True)
    assert result[1] == 'ABS_MIN'

    # test on an array
    X = np.array([[1., -1., 1., ], [-2., 2., 2.]])
    result = stats.abs_min_value(X)
    assert np.allclose(result[0], np.array([[1, 1, 1]]), atol=0.001)
    assert result[1] == 'ABS_MIN'

    # test on an array with nan
    X = np.array([[1., -1., 1., ], [2., np.nan, -2.]])
    result = stats.abs_min_value(X)
    assert np.allclose(result[0], np.array([[1, 1, 1]]), atol=0.001)
    assert result[1] == 'ABS_MIN'
