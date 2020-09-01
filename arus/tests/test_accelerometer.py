import numpy as np
from .. import accelerometer as accel


class TestActivationFeatures:
    def test_activation(self):
        # test on single sample multi-channel signal
        X = np.array([[0, 1, 0]])
        result = accel.activation_features(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]))

        # test on single sample single-channel signal
        X = np.array([[0, ]])
        result = accel.activation_features(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[0, 0, 0, 0]]))
        X = np.array([[1, ]])
        result = accel.activation_features(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[1, 1, 1, 0]]))

        # test on multi sample multi-channel signal edge case
        X = np.concatenate(
            (np.zeros((5, 3)), np.ones((5, 3)), np.zeros((5, 3))), axis=0)
        result = accel.activation_features(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[5/15, 5/15, 5/15, 1/5, 1/5, 1/5, 5/15, 5/15, 5/15, 0, 0, 0]]))

        X = np.concatenate(
            (np.zeros((5, 3)), np.ones((5, 3))), axis=0)
        result = accel.activation_features(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[5/10, 5/10, 5/10, 1/5, 1/5, 1/5, 5/10, 5/10, 5/10, 0, 0, 0]]))

        # test on multi sample multi-channel signal edge case
        X = np.concatenate(
            (np.ones((5, 3)), np.zeros((5, 3))), axis=0)
        result = accel.activation_features(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[5/10, 5/10, 5/10, 1/5, 1/5, 1/5, 5/10, 5/10, 5/10, 0, 0, 0]]))

        # test on multi sample multi-channel signal edge case
        X = np.ones((10, 3))
        result = accel.activation_features(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[10/10, 10/10, 10/10, 1/10, 1/10, 1/10, 10/10, 10/10, 10/10, 0, 0, 0]]))

        X = np.zeros((10, 3))
        result = accel.activation_features(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

        # test multiple activations
        X = np.concatenate(
            (np.zeros((5, 3)), np.ones((5, 3)), np.zeros((5, 3)), np.ones((10, 3))), axis=0)
        result = accel.activation_features(X)
        np.testing.assert_array_almost_equal(result[0], np.array(
            [[15/25, 15/25, 15/25, 2/15, 2/15, 2/15, 7.5/25, 7.5/25, 7.5/25, 3.53553/25, 3.53553/25, 3.53553/25]]))


class TestCounts:
    def test_enmo(self):
        # test on a single row
        X = np.array([[1., 1., 1., ]])
        result = accel.enmo(X)
        assert np.allclose(result[0], np.sqrt(3) - 1, atol=0.001)
        assert result[1] == 'ENMO_0'

        # test on a single row with nan
        X = np.array([[1., np.nan, 1., ]])
        result = accel.enmo(X)
        assert np.allclose(result[0], np.nan, atol=0.001, equal_nan=True)
        assert result[1] == 'ENMO_0'

        # test on an array
        X = np.array([[1., 1., 1., ], [1., 1., 1.]])
        result = accel.enmo(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[np.sqrt(3) - 1]]))
        assert result[1] == 'ENMO_0'

        # test on an array with nan
        X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
        result = accel.enmo(X)
        np.testing.assert_array_equal(result[0], np.array([[np.sqrt(3) - 1]]))
        assert result[1] == 'ENMO_0'


class TestOrientation:
    def test_gravity_angles(self):
        # test without subwins
        # test angles (0, 90, 90) with unit in degree on 1d array
        X = np.array([[1., 0., 0.]])
        result = accel.gravity_angles(X, unit='deg')
        np.testing.assert_array_equal(result[0], np.array([[0, 90, 90]]))
        # test angles (90, 0, 90) with unit in degree on 1d array
        X = np.array([[0, 1., 0.]])
        result = accel.gravity_angles(X, unit='deg')
        np.testing.assert_array_equal(result[0], np.array([[90, 0, 90]]))
        # test angles (90, 90, 0) with unit in degree on 1d array
        X = np.array([[0, 0., 1.]])
        result = accel.gravity_angles(X, unit='deg')
        np.testing.assert_array_equal(result[0], np.array([[90, 90, 0]]))
        # test angles (45, 45, 90) with unit in degree on 1d array
        X = np.array([[np.sqrt(2)/2, np.sqrt(2)/2, 0.]])
        result = accel.gravity_angles(X, unit='deg')
        np.testing.assert_array_equal(
            result[0], np.array([[45, 45, 90]]))

        # test angles (0, 90, 90) with unit in radian on 1d array
        X = np.array([[1., 0., 0.]])
        result = accel.gravity_angles(X, unit='rad')
        np.testing.assert_array_equal(result[0], np.array(
            [[np.deg2rad(0), np.deg2rad(90), np.deg2rad(90)]]))

        # test angles (0, 90, 90) with unit in degree on 2d array
        X = np.array([[1., 0., 0.], [1., 0., 0.]])
        result = accel.gravity_angles(X, unit='deg')
        np.testing.assert_array_equal(result[0], np.array(
            [[0, 90, 90]]))

        # test angles (0, 90, 90) with unit in degree on 2d array containing nan
        X = np.array([[1., 0., 0.], [1., np.nan, 0.]])
        result = accel.gravity_angles(X, unit='deg')
        np.testing.assert_array_equal(result[0], np.array(
            [[0, 90, 90]]))

        # test angles (0, 90, 90) with unit in degree on 2d array containing nan in the whole column
        X = np.array([[1., np.nan, 0.], [1., np.nan, 0.]])
        result = accel.gravity_angles(X, unit='deg')
        np.testing.assert_array_equal(result[0], np.array([[np.nan] * 3]))

        # test on a single row with nan
        X = np.array([[1., np.nan, 0., ]])
        result = accel.gravity_angles(X, unit='deg')
        np.testing.assert_array_equal(result[0], np.array(
            [[np.nan, np.nan, np.nan]]))
        np.testing.assert_array_equal(
            result[1], ["G_ANGLE_X_0", "G_ANGLE_Y_0", "G_ANGLE_Z_0"])

        # test on a 2d array with subwins
        X = np.concatenate([np.tile([[0., -1., 0., ]], (5, 1)),
                            np.tile([[1., 0., 0., ]], (5, 1)),
                            np.tile([[0., 0., 1., ]], (5, 1))], axis=0)
        result = accel.gravity_angles(X, subwins=3, unit='deg')
        np.testing.assert_array_almost_equal(result[0], np.array(
            [[90, 180, 90, 0, 90, 90, 90, 90, 0]]))
        np.testing.assert_array_equal(result[1], ["G_ANGLE_X_0", "G_ANGLE_Y_0", "G_ANGLE_Z_0", "G_ANGLE_X_1",
                                                  "G_ANGLE_Y_1", "G_ANGLE_Z_1", "G_ANGLE_X_2", "G_ANGLE_Y_2", "G_ANGLE_Z_2"])

    def test_orientation_features(self):
        # test on a single row
        X = np.array([[1., 0., 0., ]])
        result = accel.orientation_features(X, unit='deg')
        np.testing.assert_array_equal(result[0], np.array(
            [[0, 90, 90, 0, 0, 0, np.nan, np.nan, np.nan]]))
        np.testing.assert_array_equal(result[1], ["MEDIAN_G_ANGLE_0",
                                                  "MEDIAN_G_ANGLE_1", "MEDIAN_G_ANGLE_2", "RANGE_G_ANGLE_0",
                                                  "RANGE_G_ANGLE_1", "RANGE_G_ANGLE_2", "STD_G_ANGLE_0", "STD_G_ANGLE_1", "STD_G_ANGLE_2"])

        # test on a single row with nan
        X = np.array([[1., np.nan, 0., ]])
        result = accel.orientation_features(X, unit='deg')
        np.testing.assert_array_equal(result[0], np.full((1, 9), np.nan))
        np.testing.assert_array_equal(result[1], ["MEDIAN_G_ANGLE_0",
                                                  "MEDIAN_G_ANGLE_1", "MEDIAN_G_ANGLE_2", "RANGE_G_ANGLE_0",
                                                  "RANGE_G_ANGLE_1", "RANGE_G_ANGLE_2", "STD_G_ANGLE_0", "STD_G_ANGLE_1", "STD_G_ANGLE_2"])

        # test on a 2d array with subwins
        X = np.concatenate([np.tile([[0., -1., 0., ]], (5, 1)),
                            np.tile([[1., 0., 0., ]], (5, 1)),
                            np.tile([[0., 0., 1., ]], (5, 1))], axis=0)
        result = accel.orientation_features(X, subwins=3, unit='deg')
        np.testing.assert_array_almost_equal(result[0], np.array(
            [[90, 90, 90, 90, 90, 90, 51.961524, 51.961524, 51.961524]]))
        np.testing.assert_array_equal(result[1], ["MEDIAN_G_ANGLE_0",
                                                  "MEDIAN_G_ANGLE_1", "MEDIAN_G_ANGLE_2", "RANGE_G_ANGLE_0",
                                                  "RANGE_G_ANGLE_1", "RANGE_G_ANGLE_2", "STD_G_ANGLE_0", "STD_G_ANGLE_1", "STD_G_ANGLE_2"])


class TestSpectrum:
    def test_spectrum_features(self):
        names = ['DOM_FREQ_0',
                 'DOM_FREQ_POWER_0',
                 'TOTAL_FREQ_POWER_0',
                 'FREQ_POWER_ABOVE_3DOT5_0',
                 'FREQ_POWER_RATIO_ABOVE_3DOT5_0',
                 'DOM_FREQ_POWER_RATIO_0',
                 'FREQ_POWER_RATIO_BEWTEEN_DOT5_2DOT5_0',
                 'DOM_FREQ_POWER_BETWEEN_DOT6_2DOT6_0',
                 'DOM_FREQ_RATIO_PREV_BOUT_0',
                 'SPECTRAL_ENTROPY_0']
        # test on single sample single channel signal
        X = np.array([[0, ]])
        result = accel.spectrum_features(
            X, sr=80, freq_range=None, prev_spectrum_features=None)
        np.testing.assert_array_equal(
            result[0], np.array(
                [[0, 0, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan]]))
        np.testing.assert_array_equal(result[1], names)

        # test on multiple sample single channel signal
        dom_freq = 1
        sr = 100
        X = np.atleast_2d(np.sin(2*np.pi * dom_freq *
                                 np.arange(0, 1, 1.0 / sr))).T
        result = accel.spectrum_features(
            X, sr=100, freq_range=None, prev_spectrum_features=None)
        np.testing.assert_array_almost_equal(
            result[0], np.array(
                [[1, 4.283017e-01, 6.107265e-01, 0, 0, 4.283017e-01 / 6.107265e-01, 1, 4.283017e-01, np.nan, 0.15508315]]))
        np.testing.assert_array_equal(result[1], names)

        # test on multiple sample single channel signal with nan
        dom_freq = 1
        sr = 100
        X = np.atleast_2d(np.sin(2*np.pi * dom_freq *
                                 np.arange(0, 1, 1.0 / sr))).T
        X[5:10, 0] = np.nan
        result = accel.spectrum_features(
            X, sr=100, freq_range=None, prev_spectrum_features=None)
        np.testing.assert_array_almost_equal(
            result[0], np.array(
                [[1, 4.283017e-01, 6.107265e-01, 0, 0, 4.283017e-01 / 6.107265e-01, 1, 4.283017e-01, np.nan, 0.15508315]]), decimal=4)
        np.testing.assert_array_equal(result[1], names)

        # test on single sample multi channel signal
        names = ['DOM_FREQ_0', 'DOM_FREQ_1', 'DOM_FREQ_2'] + \
            ['DOM_FREQ_POWER_0', 'DOM_FREQ_POWER_1', 'DOM_FREQ_POWER_2'] + \
            ['TOTAL_FREQ_POWER_0', 'TOTAL_FREQ_POWER_1', 'TOTAL_FREQ_POWER_2'] + \
            ['FREQ_POWER_ABOVE_3DOT5_0', 'FREQ_POWER_ABOVE_3DOT5_1', 'FREQ_POWER_ABOVE_3DOT5_2'] + \
            ['FREQ_POWER_RATIO_ABOVE_3DOT5_0', 'FREQ_POWER_RATIO_ABOVE_3DOT5_1', 'FREQ_POWER_RATIO_ABOVE_3DOT5_2'] + \
            ['DOM_FREQ_POWER_RATIO_0', 'DOM_FREQ_POWER_RATIO_1', 'DOM_FREQ_POWER_RATIO_2'] + \
            ['FREQ_POWER_RATIO_BEWTEEN_DOT5_2DOT5_0', 'FREQ_POWER_RATIO_BEWTEEN_DOT5_2DOT5_1', 'FREQ_POWER_RATIO_BEWTEEN_DOT5_2DOT5_2'] + \
            ['DOM_FREQ_POWER_BETWEEN_DOT6_2DOT6_0', 'DOM_FREQ_POWER_BETWEEN_DOT6_2DOT6_1', 'DOM_FREQ_POWER_BETWEEN_DOT6_2DOT6_2'] + \
            ['DOM_FREQ_RATIO_PREV_BOUT_0', 'DOM_FREQ_RATIO_PREV_BOUT_1',
                'DOM_FREQ_RATIO_PREV_BOUT_2', 'SPECTRAL_ENTROPY_0', 'SPECTRAL_ENTROPY_1', 'SPECTRAL_ENTROPY_2']
        sr = 100
        X = np.array([[0, 0, 0]])
        result = accel.spectrum_features(
            X, sr=100, freq_range=None, prev_spectrum_features=None)
        np.testing.assert_array_almost_equal(
            result[0], np.array(
                [[0] * 3 +
                 [0] * 3 +
                 [0] * 3 +
                 [0] * 3 +
                 [0] * 3 +
                 [0] * 3 +
                 [0] * 3 +
                 [np.nan] * 3 +
                 [np.nan] * 3 +
                 [np.nan] * 3]))
        np.testing.assert_array_equal(result[1], names)

        # test on multiple sample multi channel signal
        dom_freq = 1
        sr = 100
        X = np.tile(np.sin(2*np.pi * dom_freq *
                           np.arange(0, 1, 1.0 / sr)), (3, 1)).T
        result = accel.spectrum_features(
            X, sr=100, freq_range=None, prev_spectrum_features=None)
        np.testing.assert_array_almost_equal(
            result[0],
            [[1] * 3 +
             [4.283017e-01] * 3 +
             [6.107265e-01] * 3 +
             [0] * 3 +
             [0] * 3 +
             [4.283017e-01 / 6.107265e-01] * 3 +
             [1] * 3 +
             [4.283017e-01] * 3 +
             [np.nan] * 3 +
             [0.15508315] * 3]
        )
        np.testing.assert_array_equal(result[1], names)


class TestStats:
    def test_mean(self):
        # test on a single row
        X = np.array([[1., 1., 1., ]])
        result = accel.mean(X)
        np.testing.assert_array_equal(result[0], X)
        np.testing.assert_array_equal(
            result[1], ['MEAN_0', 'MEAN_1', 'MEAN_2'])

        # test on a single row with nan
        X = np.array([[1., np.nan, 1., ]])
        result = accel.mean(X)
        np.testing.assert_array_equal(result[0], X)
        np.testing.assert_array_equal(
            result[1], ['MEAN_0', 'MEAN_1', 'MEAN_2'])

        # test on an array
        X = np.array([[1., 1., 1., ], [2., 2., 2.]])
        result = accel.mean(X)
        np.testing.assert_array_equal(result[0], np.array([[1.5, 1.5, 1.5]]))
        np.testing.assert_array_equal(
            result[1], ['MEAN_0', 'MEAN_1', 'MEAN_2'])

        # test on an array with nan
        X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
        result = accel.mean(X)
        np.testing.assert_array_equal(result[0], np.array([[1.5, 1, 1.5]]))
        np.testing.assert_array_equal(
            result[1], ['MEAN_0', 'MEAN_1', 'MEAN_2'])

    def test_median(self):
        # test on a single row
        X = np.array([[1., 1., 1., ]])
        result = accel.median(X)
        np.testing.assert_array_equal(result[0], X)
        np.testing.assert_array_equal(
            result[1], ['MEDIAN_0', 'MEDIAN_1', 'MEDIAN_2'])

        # test on a single row with nan
        X = np.array([[1., np.nan, 1., ]])
        result = accel.median(X)
        np.testing.assert_array_equal(result[0], X)
        np.testing.assert_array_equal(
            result[1], ['MEDIAN_0', 'MEDIAN_1', 'MEDIAN_2'])

        # test on an array
        X = np.array([[1., 1., 1., ], [2., 2., 2.]])
        result = accel.median(X)
        np.testing.assert_array_equal(result[0], np.array([[1.5, 1.5, 1.5]]))
        np.testing.assert_array_equal(
            result[1], ['MEDIAN_0', 'MEDIAN_1', 'MEDIAN_2'])

        # test on larger array
        X = np.array([[1., 1., 1., ], [2., 2., 2.], [5., 5., 5.]])
        result = accel.median(X)
        np.testing.assert_array_equal(result[0], np.array([[2, 2, 2]]))
        np.testing.assert_array_equal(
            result[1], ['MEDIAN_0', 'MEDIAN_1', 'MEDIAN_2'])

        # test on an array with nan
        X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
        result = accel.median(X)
        np.testing.assert_array_equal(result[0], np.array([[1.5, 1, 1.5]]))
        np.testing.assert_array_equal(
            result[1], ['MEDIAN_0', 'MEDIAN_1', 'MEDIAN_2'])

    def test_std(self):
        # test on a single row
        X = np.array([[1., 1., 1., ]])
        result = accel.std(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[np.nan, np.nan, np.nan, ]]))
        np.testing.assert_array_equal(result[1], ['STD_0', 'STD_1', 'STD_2'])

        # test on a single row with nan
        X = np.array([[1., np.nan, 1., ]])
        result = accel.std(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[np.nan, np.nan, np.nan]]))
        np.testing.assert_array_equal(result[1], ['STD_0', 'STD_1', 'STD_2'])

        # test on an array
        X = np.array([[1., 1., 1., ], [2., 2., 2.]])
        result = accel.std(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)]]))
        np.testing.assert_array_equal(result[1], ['STD_0', 'STD_1', 'STD_2'])

        # test on an array with nan
        X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
        result = accel.std(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[np.sqrt(0.5), np.nan, np.sqrt(0.5)]]))
        np.testing.assert_array_equal(result[1], ['STD_0', 'STD_1', 'STD_2'])

        # test on an array with more than two rows with nan
        X = np.array([[1., 1., 1., ], [2., 2, 2.], [1.5, np.nan, 1.5]])
        result = accel.std(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[np.sqrt(0.25), np.sqrt(0.5), np.sqrt(0.25)]]))
        np.testing.assert_array_equal(result[1], ['STD_0', 'STD_1', 'STD_2'])

    def test_skew(self):
        # test on a single row
        X = np.array([[1., 1., 1., ]])
        result = accel.skew(X)
        np.testing.assert_array_equal(result[0], np.atleast_2d([0, 0, 0]))
        np.testing.assert_array_equal(
            result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])

        # test on a single row with nan
        X = np.array([[1., np.nan, 1., ]])
        result = accel.skew(X)
        np.testing.assert_array_equal(result[0], np.atleast_2d([0, np.nan, 0]))
        np.testing.assert_array_equal(
            result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])

        # test on an array
        X = np.array([[1., 1., 1., ], [2., 2., 2.]])
        result = accel.skew(X)
        np.testing.assert_array_equal(result[0], np.array([[0, 0, 0]]))
        np.testing.assert_array_equal(
            result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])

        # test on normal distribution
        X = np.random.randn(1000000, 3)
        result = accel.skew(X)
        np.testing.assert_array_almost_equal(
            result[0], np.array([[0, 0, 0]]), decimal=1)
        np.testing.assert_array_equal(
            result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])

        # test on normal distribution with nan
        X = np.random.randn(1000000, 3)
        X[np.random.randint(0, 49, 1), 0:3] = np.nan
        result = accel.skew(X)
        np.testing.assert_array_almost_equal(
            result[0], np.array([[0, 0, 0]]), decimal=1)
        np.testing.assert_array_equal(
            result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])

        # test on exponential distribution
        X = np.random.standard_exponential(size=(1000000, 3))
        result = accel.skew(X)
        np.testing.assert_array_almost_equal(
            result[0], np.array([[2, 2, 2]]), decimal=1)
        np.testing.assert_array_equal(
            result[1], ['SKEW_0', 'SKEW_1', 'SKEW_2'])

    def test_kurtosis(self):
        # test on a single row
        X = np.array([[1., 1., 1., ]])
        result = accel.kurtosis(X)
        np.testing.assert_array_equal(result[0], np.atleast_2d([-3, -3, -3]))
        np.testing.assert_array_equal(
            result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])

        # test on a single row with nan
        X = np.array([[1., np.nan, 1., ]])
        result = accel.kurtosis(X)
        np.testing.assert_array_equal(
            result[0], np.atleast_2d([-3, np.nan, -3]))
        np.testing.assert_array_equal(
            result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])

        # test on uniform distribution
        X = np.random.uniform(low=0, high=1, size=(1000000, 3))
        result = accel.kurtosis(X)
        np.testing.assert_array_almost_equal(
            result[0], np.array([[-1.2, -1.2, -1.2]]), decimal=2)
        np.testing.assert_array_equal(
            result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])

        # test on normal distribution
        X = np.random.randn(1000000, 3)
        result = accel.kurtosis(X)
        np.testing.assert_array_almost_equal(
            result[0], np.array([[0, 0, 0]]), decimal=1)
        np.testing.assert_array_equal(
            result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])

        # test on normal distribution with nan
        X = np.random.randn(1000000, 3)
        X[np.random.randint(0, 49, 1), 0:3] = np.nan
        result = accel.kurtosis(X)
        np.testing.assert_array_almost_equal(
            result[0], np.array([[0, 0, 0]]), decimal=1)
        np.testing.assert_array_equal(
            result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])

        # test on laplace distribution
        X = np.random.laplace(loc=0, scale=1, size=(1000000, 3))
        result = accel.kurtosis(X)
        np.testing.assert_array_almost_equal(
            result[0], np.array([[3, 3, 3]]), decimal=1)
        np.testing.assert_array_equal(
            result[1], ['KURTOSIS_0', 'KURTOSIS_1', 'KURTOSIS_2'])

    def test_max_value(self):
        # test on a single row
        X = np.array([[1., 1., 1., ]])
        result = accel.max_value(X)
        np.testing.assert_array_equal(result[0], X)
        np.testing.assert_array_equal(result[1], ['MAX_0', 'MAX_1', 'MAX_2'])

        # test on a single row with nan
        X = np.array([[1., np.nan, 1., ]])
        result = accel.max_value(X)
        np.testing.assert_array_equal(result[0], X)
        np.testing.assert_array_equal(result[1], ['MAX_0', 'MAX_1', 'MAX_2'])

        # test on an array
        X = np.array([[1., 1., 1., ], [2., 2., 2.]])
        result = accel.max_value(X)
        np.testing.assert_array_equal(result[0], np.array([[2, 2, 2]]))
        np.testing.assert_array_equal(result[1], ['MAX_0', 'MAX_1', 'MAX_2'])

        # test on an array with nan
        X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
        result = accel.max_value(X)
        np.testing.assert_array_equal(result[0], np.array([[2, 1, 2]]))
        np.testing.assert_array_equal(result[1], ['MAX_0', 'MAX_1', 'MAX_2'])

    def test_min_value(self):
        # test on a single row
        X = np.array([[-1., -1., -1., ]])
        result = accel.min_value(X)
        np.testing.assert_array_equal(result[0], X)
        np.testing.assert_array_equal(result[1], ['MIN_0', 'MIN_1', 'MIN_2'])

        # test on a single row with nan
        X = np.array([[1., np.nan, 1., ]])
        result = accel.min_value(X)
        np.testing.assert_array_equal(result[0], X)
        np.testing.assert_array_equal(result[1], ['MIN_0', 'MIN_1', 'MIN_2'])

        # test on an array
        X = np.array([[1., 1., 1., ], [2., 2., 2.]])
        result = accel.min_value(X)
        np.testing.assert_array_equal(result[0], np.array([[1, 1, 1]]))
        np.testing.assert_array_equal(result[1], ['MIN_0', 'MIN_1', 'MIN_2'])

        # test on an array with nan
        X = np.array([[1., 1., 1., ], [2., np.nan, 2.]])
        result = accel.min_value(X)
        np.testing.assert_array_equal(result[0], np.array([[1, 1, 1]]))
        np.testing.assert_array_equal(result[1], ['MIN_0', 'MIN_1', 'MIN_2'])

    def test_range(self):
        # test on a single row
        X = np.array([[-1., -1., -1.]])
        result = accel.max_minus_min(X)
        np.testing.assert_array_equal(result[0], np.array([[0., 0., 0.]]))
        np.testing.assert_array_equal(
            result[1], ['RANGE_0', 'RANGE_1', 'RANGE_2'])

        # test on a single row with nan
        X = np.array([[1., np.nan, 1., ]])
        result = accel.max_minus_min(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[0., np.nan, 0., ]]))
        np.testing.assert_array_equal(
            result[1], ['RANGE_0', 'RANGE_1', 'RANGE_2'])

        # test on an array
        X = np.array([[1., 1., 1., ], [3., 3., 3.]])
        result = accel.max_minus_min(X)
        np.testing.assert_array_equal(result[0], np.array([[2, 2, 2]]))
        np.testing.assert_array_equal(
            result[1], ['RANGE_0', 'RANGE_1', 'RANGE_2'])

        # test on an array with nan
        X = np.array([[1., 1., 1., ], [3., np.nan, 3.]])
        result = accel.max_minus_min(X)
        np.testing.assert_array_equal(result[0], np.array([[2, 0, 2]]))
        np.testing.assert_array_equal(
            result[1], ['RANGE_0', 'RANGE_1', 'RANGE_2'])

    def test_abs_max_value(self):
        # test on a single row
        X = np.array([[1., -1., 1., ]])
        result = accel.abs_max_value(X)
        np.testing.assert_array_equal(result[0], np.array([[1., 1., 1., ]]))
        np.testing.assert_array_equal(
            result[1], ['ABS_MAX_0', 'ABS_MAX_1', 'ABS_MAX_2'])

        # test on a single row with nan
        X = np.array([[1., np.nan, -1., ]])
        result = accel.abs_max_value(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[1., np.nan, 1., ]]))
        np.testing.assert_array_equal(
            result[1], ['ABS_MAX_0', 'ABS_MAX_1', 'ABS_MAX_2'])

        # test on an array
        X = np.array([[1., 1., -1., ], [-2., 2., 2.]])
        result = accel.abs_max_value(X)
        np.testing.assert_array_equal(result[0], np.array([[2, 2, 2]]))
        np.testing.assert_array_equal(
            result[1], ['ABS_MAX_0', 'ABS_MAX_1', 'ABS_MAX_2'])

        # test on an array with nan
        X = np.array([[1., -1., 1., ], [-2., np.nan, 2.]])
        result = accel.abs_max_value(X)
        np.testing.assert_array_equal(result[0], np.array([[2, 1, 2]]))
        np.testing.assert_array_equal(
            result[1], ['ABS_MAX_0', 'ABS_MAX_1', 'ABS_MAX_2'])

    def test_abs_min_value(self):
        # test on a single row
        X = np.array([[-1., -1., -1.]])
        result = accel.abs_min_value(X)
        np.testing.assert_array_equal(result[0], np.array([[1., 1., 1.]]))
        np.testing.assert_array_equal(
            result[1], ['ABS_MIN_0', 'ABS_MIN_1', 'ABS_MIN_2'])

        # test on a single row with nan
        X = np.array([[-1., np.nan, -1., ]])
        result = accel.abs_min_value(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[1., np.nan, 1., ]]))
        np.testing.assert_array_equal(
            result[1], ['ABS_MIN_0', 'ABS_MIN_1', 'ABS_MIN_2'])

        # test on an array
        X = np.array([[1., -1., 1., ], [-2., 2., 2.]])
        result = accel.abs_min_value(X)
        np.testing.assert_array_equal(result[0], np.array([[1, 1, 1]]))
        np.testing.assert_array_equal(
            result[1], ['ABS_MIN_0', 'ABS_MIN_1', 'ABS_MIN_2'])

        # test on an array with nan
        X = np.array([[1., -1., 1., ], [2., np.nan, -2.]])
        result = accel.abs_min_value(X)
        np.testing.assert_array_equal(result[0], np.array([[1, 1, 1]]))
        np.testing.assert_array_equal(
            result[1], ['ABS_MIN_0', 'ABS_MIN_1', 'ABS_MIN_2'])

    def test_zcr(self):
        # test on a single row
        X = np.array([[-1., -1., -1.]])
        result = accel.zcr(X)
        np.testing.assert_array_equal(
            result[0], np.array([[np.nan, np.nan, np.nan]]))
        np.testing.assert_array_equal(
            result[1], ['ZCR_0', 'ZCR_1', 'ZCR_2'])

        # test on a single row with nan
        X = np.array([[-1., np.nan, -1., ]])
        result = accel.zcr(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[np.nan, np.nan, np.nan, ]]))
        np.testing.assert_array_equal(
            result[1], ['ZCR_0', 'ZCR_1', 'ZCR_2'])

        # test on an array
        X = np.array([[1., -1., 1., ], [-2., 2., 2.]])
        result = accel.zcr(X)
        np.testing.assert_array_equal(result[0], np.array([[1, 1, 0]]))
        np.testing.assert_array_equal(
            result[1], ['ZCR_0', 'ZCR_1', 'ZCR_2'])

        # test on an array with nan
        X = np.array([[1., -1., 1., ], [2., np.nan, -2.]])
        result = accel.zcr(X)
        np.testing.assert_array_equal(result[0], np.array([[0, 0, 1]]))
        np.testing.assert_array_equal(
            result[1], ['ZCR_0', 'ZCR_1', 'ZCR_2'])

    def test_mcr(self):
        # test on a single row
        X = np.array([[-1., -1., -1.]])
        result = accel.mcr(X)
        np.testing.assert_array_equal(
            result[0], np.array([[np.nan, np.nan, np.nan]]))
        np.testing.assert_array_equal(
            result[1], ['MCR_0', 'MCR_1', 'MCR_2'])

        # test on a single row with nan
        X = np.array([[-1., np.nan, -1., ]])
        result = accel.mcr(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[np.nan, np.nan, np.nan, ]]))
        np.testing.assert_array_equal(
            result[1], ['MCR_0', 'MCR_1', 'MCR_2'])

        # test on an array
        X = np.array([[1., -1., 1., ], [-2., 2., 2.]])
        result = accel.mcr(X)
        np.testing.assert_array_equal(result[0], np.array([[1, 1, 1]]))
        np.testing.assert_array_equal(
            result[1], ['MCR_0', 'MCR_1', 'MCR_2'])

        # test on an array with nan
        X = np.array([[1., -1., 1., ], [2., np.nan, -2.]])
        result = accel.mcr(X)
        np.testing.assert_array_equal(result[0], np.array([[1, 0, 1]]))
        np.testing.assert_array_equal(
            result[1], ['MCR_0', 'MCR_1', 'MCR_2'])

    def test_correlation(self):
        # test on a single row
        X = np.array([[1., 1., 1.]])
        result = accel.correlation(X)
        np.testing.assert_array_equal(
            result[0], np.array([[np.nan, np.nan, np.nan]]))
        np.testing.assert_array_equal(
            result[1], ['CORRELATION_0', 'CORRELATION_1', 'CORRELATION_2'])

        # test on a single row with nan
        X = np.array([[-1., np.nan, -1., ]])
        result = accel.correlation(X)
        np.testing.assert_array_equal(result[0], np.array(
            [[np.nan, np.nan, np.nan]]))
        np.testing.assert_array_equal(
            result[1], ['CORRELATION_0', 'CORRELATION_1', 'CORRELATION_2'])

        # test on an array
        x = np.transpose(np.atleast_2d(np.arange(10)))
        y = np.transpose(-np.atleast_2d(np.arange(10)))
        z = np.transpose(np.atleast_2d(np.arange(10)/5))
        X = np.hstack((x, y, z))
        result = accel.correlation(X)
        np.testing.assert_array_almost_equal(
            result[0], np.array([[-1, 1, -1]]))
        np.testing.assert_array_equal(
            result[1], ['CORRELATION_0', 'CORRELATION_1', 'CORRELATION_2'])

        # test on an array with nan
        x = np.transpose(np.atleast_2d(np.arange(10)))
        y = np.transpose(-np.atleast_2d(np.arange(10)))
        z = np.transpose(np.atleast_2d(np.arange(10)/5))
        X = np.hstack((x, y, z))
        X[np.random.randint(0, 10, 1), 0:3] = np.nan
        result = accel.correlation(X)
        np.testing.assert_array_equal(
            result[0], np.array([[np.nan, np.nan, np.nan]]))
        np.testing.assert_array_equal(
            result[1], ['CORRELATION_0', 'CORRELATION_1', 'CORRELATION_2'])
