import numpy as np
from ..orientation import gravity_angles, gravity_angle_stats, _gravity_angles


def test__gravity_angles():
    # test angles (0, 90, 90) with unit in degree on 1d array
    X = np.array([[1., 0., 0.]])
    result = _gravity_angles(X, unit='deg')
    np.testing.assert_array_equal(result, np.array([[0, 90, 90]]))
    # test angles (90, 0, 90) with unit in degree on 1d array
    X = np.array([[0, 1., 0.]])
    result = _gravity_angles(X, unit='deg')
    np.testing.assert_array_equal(result, np.array([[90, 0, 90]]))
    # test angles (90, 90, 0) with unit in degree on 1d array
    X = np.array([[0, 0., 1.]])
    result = _gravity_angles(X, unit='deg')
    np.testing.assert_array_equal(result, np.array([[90, 90, 0]]))
    # test angles (45, 45, 90) with unit in degree on 1d array
    X = np.array([[np.sqrt(2)/2, np.sqrt(2)/2, 0.]])
    result = _gravity_angles(X, unit='deg')
    np.testing.assert_array_equal(
        result, np.array([[45, 45, 90]]))

    # test angles (0, 90, 90) with unit in radian on 1d array
    X = np.array([[1., 0., 0.]])
    result = _gravity_angles(X, unit='rad')
    np.testing.assert_array_equal(result, np.array(
        [[np.deg2rad(0), np.deg2rad(90), np.deg2rad(90)]]))

    # test angles (0, 90, 90) with unit in degree on 2d array
    X = np.array([[1., 0., 0.], [1., 0., 0.]])
    result = _gravity_angles(X, unit='deg')
    np.testing.assert_array_equal(result, np.array(
        [[0, 90, 90]]))

    # test angles (0, 90, 90) with unit in degree on 2d array containing nan
    X = np.array([[1., 0., 0.], [1., np.nan, 0.]])
    result = _gravity_angles(X, unit='deg')
    np.testing.assert_array_equal(result, np.array(
        [[0, 90, 90]]))

    # test angles (0, 90, 90) with unit in degree on 2d array containing nan in the whole column
    X = np.array([[1., np.nan, 0.], [1., np.nan, 0.]])
    result = _gravity_angles(X, unit='deg')
    np.testing.assert_array_equal(result, np.array([[np.nan] * 3]))


def test_gravity_angles():
    # test on a single row
    X = np.array([[1., 0., 0., ]])
    result = gravity_angles(X, unit='deg')
    np.testing.assert_array_equal(result[0], np.array(
        [[0, 90, 90]]))
    np.testing.assert_array_equal(
        result[1], ["G_ANGLE_X_0", "G_ANGLE_Y_0", "G_ANGLE_Z_0"])

    # test on a single row with nan
    X = np.array([[1., np.nan, 0., ]])
    result = gravity_angles(X, unit='deg')
    np.testing.assert_array_equal(result[0], np.array(
        [[np.nan, np.nan, np.nan]]))
    np.testing.assert_array_equal(
        result[1], ["G_ANGLE_X_0", "G_ANGLE_Y_0", "G_ANGLE_Z_0"])

    # test on a 2d array with subwins
    X = np.concatenate([np.tile([[0., -1., 0., ]], (5, 1)),
                        np.tile([[1., 0., 0., ]], (5, 1)),
                        np.tile([[0., 0., 1., ]], (5, 1))], axis=0)
    result = gravity_angles(X, subwins=3, unit='deg')
    np.testing.assert_array_almost_equal(result[0], np.array(
        [[90, 180, 90, 0, 90, 90, 90, 90, 0]]))
    np.testing.assert_array_equal(result[1], ["G_ANGLE_X_0", "G_ANGLE_Y_0", "G_ANGLE_Z_0", "G_ANGLE_X_1",
                                              "G_ANGLE_Y_1", "G_ANGLE_Z_1", "G_ANGLE_X_2", "G_ANGLE_Y_2", "G_ANGLE_Z_2"])


def test_gravity_angle_stats():
    # test on a single row
    X = np.array([[1., 0., 0., ]])
    result = gravity_angle_stats(X, unit='deg')
    np.testing.assert_array_equal(result[0], np.array(
        [[0, 90, 90, 0, 0, 0, np.nan, np.nan, np.nan]]))
    np.testing.assert_array_equal(result[1], ["MEDIAN_G_ANGLE_X", "MEDIAN_G_ANGLE_Y", "MEDIAN_G_ANGLE_Z", "RANGE_G_ANGLE_X",
                                              "RANGE_G_ANGLE_Y", "RANGE_G_ANGLE_Z", "STD_G_ANGLE_X", "STD_G_ANGLE_Y", "STD_G_ANGLE_Z"])

    # test on a single row with nan
    X = np.array([[1., np.nan, 0., ]])
    result = gravity_angle_stats(X, unit='deg')
    np.testing.assert_array_equal(result[0], np.full((1, 9), np.nan))
    np.testing.assert_array_equal(result[1], ["MEDIAN_G_ANGLE_X", "MEDIAN_G_ANGLE_Y", "MEDIAN_G_ANGLE_Z", "RANGE_G_ANGLE_X",
                                              "RANGE_G_ANGLE_Y", "RANGE_G_ANGLE_Z", "STD_G_ANGLE_X", "STD_G_ANGLE_Y", "STD_G_ANGLE_Z"])

    # test on a 2d array with subwins
    X = np.concatenate([np.tile([[0., -1., 0., ]], (5, 1)),
                        np.tile([[1., 0., 0., ]], (5, 1)),
                        np.tile([[0., 0., 1., ]], (5, 1))], axis=0)
    result = gravity_angle_stats(X, subwins=3, unit='deg')
    np.testing.assert_array_almost_equal(result[0], np.array(
        [[90, 90, 90, 90, 90, 90, 51.961524, 51.961524, 51.961524]]))
    np.testing.assert_array_equal(result[1], ["MEDIAN_G_ANGLE_X", "MEDIAN_G_ANGLE_Y", "MEDIAN_G_ANGLE_Z", "RANGE_G_ANGLE_X",
                                              "RANGE_G_ANGLE_Y", "RANGE_G_ANGLE_Z", "STD_G_ANGLE_X", "STD_G_ANGLE_Y", "STD_G_ANGLE_Z"])
