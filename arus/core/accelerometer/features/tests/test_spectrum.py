import numpy as np
from .. import spectrum


def test_spectrum_features():
    names = ['DOM_FREQ_0', 'DOM_FREQ_POWER_0', 'TOTAL_FREQ_POWER_0', 'FREQ_POWER_ABOVE_3DOT5_0', 'FREQ_POWER_RATIO_ABOVE_3DOT5_0',
             'DOM_FREQ_POWER_RATIO_0', 'DOM_FREQ_BETWEEN_DOT6_2DOT6_0', 'DOM_FREQ_POWER_BETWEEN_DOT6_2DOT6_0', 'DOM_FREQ_RATIO_PREV_BOUT_0', 'SPECTRAL_ENTROPY_0']
    # test on single sample single channel signal
    X = np.array([[0, ]])
    result = spectrum.spectrum_features(
        X, sr=80, freq_range=None, prev_spectrum_features=None, preset="")
    np.testing.assert_array_equal(
        result[0], np.array(
            [[0, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan]]))
    np.testing.assert_array_equal(result[1], names)

    # test on multiple sample single channel signal
    dom_freq = 1
    sr = 100
    X = np.atleast_2d(np.sin(2*np.pi * dom_freq * np.arange(0, 1, 1.0 / sr))).T
    result = spectrum.spectrum_features(
        X, sr=100, freq_range=None, prev_spectrum_features=None, preset="")
    np.testing.assert_array_almost_equal(
        result[0], np.array(
            [[1, 4.283017e-01, 6.107265e-01, 0, 0, 4.283017e-01 / 6.107265e-01, 1, 4.283017e-01, np.nan, 0.15508315]]))
    np.testing.assert_array_equal(result[1], names)

    # test on multiple sample single channel signal with nan
    dom_freq = 1
    sr = 100
    X = np.atleast_2d(np.sin(2*np.pi * dom_freq * np.arange(0, 1, 1.0 / sr))).T
    X[5:10, 0] = np.nan
    result = spectrum.spectrum_features(
        X, sr=100, freq_range=None, prev_spectrum_features=None, preset="")
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
        ['DOM_FREQ_BETWEEN_DOT6_2DOT6_0', 'DOM_FREQ_BETWEEN_DOT6_2DOT6_1', 'DOM_FREQ_BETWEEN_DOT6_2DOT6_2'] + \
        ['DOM_FREQ_POWER_BETWEEN_DOT6_2DOT6_0', 'DOM_FREQ_POWER_BETWEEN_DOT6_2DOT6_1', 'DOM_FREQ_POWER_BETWEEN_DOT6_2DOT6_2'] + \
        ['DOM_FREQ_RATIO_PREV_BOUT_0', 'DOM_FREQ_RATIO_PREV_BOUT_1',
            'DOM_FREQ_RATIO_PREV_BOUT_2', 'SPECTRAL_ENTROPY_0', 'SPECTRAL_ENTROPY_1', 'SPECTRAL_ENTROPY_2']
    sr = 100
    X = np.array([[0, 0, 0]])
    result = spectrum.spectrum_features(
        X, sr=100, freq_range=None, prev_spectrum_features=None, preset="")
    np.testing.assert_array_almost_equal(
        result[0], np.array(
            [[0] * 3 +
             [0] * 3 +
             [0] * 3 +
             [0] * 3 +
             [0] * 3 +
             [0] * 3 +
             [np.nan] * 3 +
             [np.nan] * 3 +
             [np.nan] * 3 +
             [np.nan] * 3]))
    np.testing.assert_array_equal(result[1], names)

    # test on multiple sample multi channel signal
    dom_freq = 1
    sr = 100
    X = np.tile(np.sin(2*np.pi * dom_freq *
                       np.arange(0, 1, 1.0 / sr)), (3, 1)).T
    result = spectrum.spectrum_features(
        X, sr=100, freq_range=None, prev_spectrum_features=None, preset="")
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
