import pytest
import numpy as np
import arus.libs.signal_processing.filtering as filtering
import itertools
import matplotlib.pyplot as plt


def _as_2d_waterfall_signal(arg, repeat_n=3):
    return np.transpose(np.tile(arg, (repeat_n, 1)))


@pytest.fixture
def signals():
    sr = 100
    t = np.linspace(0, 1, sr, False)
    mix_10_20_sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    single_10_sig = np.sin(2*np.pi*10*t)
    constant_sig = np.ones(t.shape)
    zero_sig = np.zeros(t.shape)
    sigs = [mix_10_20_sig,
            _as_2d_waterfall_signal(mix_10_20_sig),
            constant_sig,
            _as_2d_waterfall_signal(constant_sig)
            ]

    low_pass_expects = [single_10_sig,
                        _as_2d_waterfall_signal(single_10_sig),
                        constant_sig,
                        _as_2d_waterfall_signal(constant_sig), ]

    low_pass_cases = list(itertools.starmap(lambda sig, expected: ({'X': sig, 'sr': sr, 'cut_offs': 15, 'order': 4,
                                                                    'filter_type': 'low'}, expected), zip(sigs, low_pass_expects)))

    band_pass_expects = [zero_sig,
                         _as_2d_waterfall_signal(zero_sig),
                         zero_sig,
                         _as_2d_waterfall_signal(zero_sig)]

    band_pass_cases = list(itertools.starmap(lambda sig, expected: ({'X': sig, 'sr': sr, 'cut_offs': [13, 17], 'order': 4,
                                                                     'filter_type': 'pass'}, expected), zip(sigs, band_pass_expects)))

    return low_pass_cases + band_pass_cases


def test_butterworth(signals):
    for args, expected in signals:
        result = filtering.butterworth(**args)
        diff = expected[10:90, ] - result[10:90, ]
        assert expected.shape == result.shape
        assert np.all(diff < 0.1)
