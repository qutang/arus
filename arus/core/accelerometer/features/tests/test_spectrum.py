import numpy as np
from .. import spectrum


def test_spectrum_features():
    names = ['DOM_FREQ_0', 'DOM_FREQ_POWER_0', 'TOTAL_FREQ_POWER_0', 'FREQ_POWER_ABOVE_3DOT5_0', 'FREQ_POWER_RATIO_ABOVE_3DOT5_0',
             'DOM_FREQ_POWER_RATIO_0', 'DOM_FREQ_BETWEEN_DOT6_2DOT6_0', 'DOM_FREQ_POWER_BETWEEN_DOT6_2DOT6_0', 'DOM_FREQ_RATIO_PREV_BOUT_0']
    # test on single sample single channel signal
    X = np.array([[0, ]])
    result = spectrum.spectrum_features(
        X, sr=80, freq_range=None, prev_spectrum_features=None, preset="")
    np.testing.assert_array_equal(
        result[0], np.array(
            [[0, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan]]))
    np.testing.assert_array_equal(result[1], names)

    # test on multiple sample single channel signal
    dom_freq = 1
    sr = 100
    X = np.atleast_2d(np.sin(2*np.pi * dom_freq * np.arange(0, 1, 1.0 / sr))).T
    result = spectrum.spectrum_features(
        X, sr=100, freq_range=None, prev_spectrum_features=None, preset="")
    np.testing.assert_array_equal(
        result[0], np.array(
            [[0, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan]]))
    np.testing.assert_array_equal(result[1], names)
