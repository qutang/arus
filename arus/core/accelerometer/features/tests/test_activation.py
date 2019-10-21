import numpy as np
from ..activation import stats_active_samples


def test_stats_active_samples():
    # test on single sample multi-channel signal
    X = np.array([[0, 1, 0]])
    result = stats_active_samples(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]))

    # test on single sample single-channel signal
    X = np.array([[0, ]])
    result = stats_active_samples(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[0, 0, 0, 0]]))
    X = np.array([[1, ]])
    result = stats_active_samples(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[1, 1, 1, 0]]))

    # test on multi sample multi-channel signal edge case
    X = np.concatenate(
        (np.zeros((5, 3)), np.ones((5, 3)), np.zeros((5, 3))), axis=0)
    result = stats_active_samples(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[5/15, 5/15, 5/15, 1/5, 1/5, 1/5, 5/15, 5/15, 5/15, 0, 0, 0]]))

    X = np.concatenate(
        (np.zeros((5, 3)), np.ones((5, 3))), axis=0)
    result = stats_active_samples(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[5/10, 5/10, 5/10, 1/5, 1/5, 1/5, 5/10, 5/10, 5/10, 0, 0, 0]]))

    # test on multi sample multi-channel signal edge case
    X = np.concatenate(
        (np.ones((5, 3)), np.zeros((5, 3))), axis=0)
    result = stats_active_samples(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[5/10, 5/10, 5/10, 1/5, 1/5, 1/5, 5/10, 5/10, 5/10, 0, 0, 0]]))

    # test on multi sample multi-channel signal edge case
    X = np.ones((10, 3))
    result = stats_active_samples(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[10/10, 10/10, 10/10, 1/10, 1/10, 1/10, 10/10, 10/10, 10/10, 0, 0, 0]]))

    X = np.zeros((10, 3))
    result = stats_active_samples(X)
    np.testing.assert_array_equal(result[0], np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    # test multiple activations
    X = np.concatenate(
        (np.zeros((5, 3)), np.ones((5, 3)), np.zeros((5, 3)), np.ones((10, 3))), axis=0)
    result = stats_active_samples(X)
    np.testing.assert_array_almost_equal(result[0], np.array(
        [[15/25, 15/25, 15/25, 2/15, 2/15, 2/15, 7.5/25, 7.5/25, 7.5/25, 3.53553/25, 3.53553/25, 3.53553/25]]))
