import numpy as np
from ..apply_over_subwins import apply_over_subwins
from ...num import format_arr


def test_apply_over_subwins():
    func = np.mean
    # test on single row array with subwins and subwin_samples not set
    X = format_arr(np.array([1., 1., 1.]))
    result = apply_over_subwins(
        X, func, subwin_samples=None, subwins=None, axis=0)
    assert np.array_equal(result, X)

    # test on single row array with subwin_samples not set
    X = format_arr(np.array([1., 1., 1.]))
    result = apply_over_subwins(
        X, func, subwin_samples=None, subwins=1, axis=0)
    assert np.array_equal(result, X)

    # test on single row array with subwins not set
    X = format_arr(np.array([1., 1., 1.]))
    result = apply_over_subwins(
        X, func, subwin_samples=1, subwins=None, axis=0)
    assert np.array_equal(result, X)

    # test on single row array with subwins to be zero
    X = format_arr(np.array([1., 1., 1.]))
    result = apply_over_subwins(
        X, func, subwin_samples=2, subwins=None, axis=0)
    assert np.array_equal(result, X)
    result = apply_over_subwins(
        X, func, subwin_samples=None, subwins=0, axis=0)
    assert np.array_equal(result, X)

    # test on single row array with subwin_samples to be zero
    X = format_arr(np.array([1., 1., 1.]))
    result = apply_over_subwins(
        X, func, subwin_samples=0, subwins=None, axis=0)
    assert np.array_equal(result, X)
    result = apply_over_subwins(
        X, func, subwin_samples=None, subwins=2, axis=0)
    assert np.array_equal(result, X)

    # test on 2d array
    X = format_arr(np.ones((10, 3)))
    result = apply_over_subwins(
        X, func, subwin_samples=2, subwins=None, axis=0)
    assert np.array_equal(result, np.ones((5, 3)))
    result = apply_over_subwins(
        X, func, subwin_samples=None, subwins=2, axis=0)
    assert np.array_equal(result, np.ones((2, 3)))

    # test on 2d array use subwins at first
    X = format_arr(np.ones((10, 3)))
    result = apply_over_subwins(
        X, func, subwin_samples=2, subwins=2, axis=0)
    assert np.array_equal(result, np.ones((2, 3)))

    # test on 2d array use window parameters that are not fully dividable
    X = format_arr(
        np.concatenate((
            np.ones((1, 3)) * 2,
            np.ones((8, 3)),
            np.ones((1, 3)) * 2),
            axis=0)
    )
    result = apply_over_subwins(
        X, func, subwin_samples=4, subwins=None, axis=0)
    assert np.array_equal(result, np.ones((2, 3)))

    # test on 2d array use window parameters that are not fully dividable
    X = format_arr(
        np.concatenate((
            np.ones((1, 3)) * 2,
            np.ones((3, 3)),
            np.ones((6, 3)) * 2),
            axis=0)
    )
    result = apply_over_subwins(
        X, func, subwin_samples=3, subwins=None, axis=0)
    assert np.array_equal(result, np.array([[1, 1, 1], [2, 2, 2], [2, 2, 2]]))
