import pytest
import numpy as np
import arus.core.libs.dsp.filtering as filtering
import itertools
import matplotlib.pyplot as plt


def _as_2d_waterfall_signal(arg, repeat_n=3):
    return np.transpose(np.tile(arg, (repeat_n, 1)))


@pytest.fixture
def butterworth_test_signals():
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


@pytest.fixture
def resample_test_signals():
    list_sr = [100, 80, 50, 30]
    test_cases = []
    signals = []
    for sr in list_sr:
        t = np.linspace([0, 0], [1, 1], sr, False, axis=0)
        mix_2_5_sig = np.sin(2*np.pi*2*t) + np.sin(2*np.pi*5*t)
        signals.append(mix_2_5_sig)
    sr_signal_pairs = zip(list_sr, signals)
    test_cases = itertools.combinations(sr_signal_pairs, 2)
    return test_cases


def test_butterworth(butterworth_test_signals):
    for args, expected in butterworth_test_signals:
        result = filtering.butterworth(**args)
        diff = expected[10:90, ] - result[10:90, ]
        assert expected.shape == result.shape
        assert np.all(diff < 0.1)


def test_resample(resample_test_signals):
    for test_signal_pair_1, test_signal_pair_2 in resample_test_signals:
        sr1 = test_signal_pair_1[0]
        signal1 = test_signal_pair_1[1]
        sr2 = test_signal_pair_2[0]
        signal2 = test_signal_pair_2[1]
        result12 = filtering.resample(signal1, sr1, sr2)
        expected12 = signal2
        st = int(sr2/10)
        et = int(sr2 - sr2/10)
        diff12 = expected12[st:et, ] - \
            result12[st:et, ]
        print('{}, {}'.format(sr1, sr2))
        assert expected12.shape == result12.shape
        assert np.all(diff12 < 0.1)

        result21 = filtering.resample(signal2, sr2, sr1)
        expected21 = signal1
        st = int(sr1/10)
        et = int(sr1 - sr1/10)
        diff21 = expected21[st:et, ] - \
            result21[st:et, ]
        print('{}, {}'.format(sr2, sr1))
        assert expected21.shape == result21.shape
        assert np.all(diff21 < 0.1)
