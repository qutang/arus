import pandas as pd
import numpy as np
from .. import extensions
import pytest
import itertools


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

    low_pass_cases = list(
        itertools.starmap(
            lambda sig, expected: (
                {'X': sig, 'sr': sr, 'cut_offs': 15, 'order': 4,
                 'filter_type': 'low'},
                expected
            ),
            zip(sigs, low_pass_expects)
        )
    )

    band_pass_expects = [zero_sig,
                         _as_2d_waterfall_signal(zero_sig),
                         zero_sig,
                         _as_2d_waterfall_signal(zero_sig)]

    band_pass_cases = list(
        itertools.starmap(
            lambda sig, expected: (
                {'X': sig, 'sr': sr, 'cut_offs': [13, 17], 'order': 4,
                 'filter_type': 'pass'},
                expected),
            zip(sigs, band_pass_expects)
        )
    )

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


@pytest.fixture
def regularize_test_signals():
    results = []
    for et, expected in zip([1, 1.1, 1.11, 1.01], [80, 88, 88, 80]):
        t = np.linspace(0, et, num=85)
        X = np.random.rand(len(t), 3)
        results.append((t, X, 80, expected))
    return results


class TestPandas:
    def test_merge_all(self):
        df1 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'],
                            'value': [1, 2, 3, 5], 'group': [1, 1, 2, 2]})

        df2 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'],
                            'value': [5, 6, 7, 8], 'group': [1, 1, 2, 2]})

        df3 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'],
                            'value': [9, 10, 11, 12], 'group': [1, 1, 2, 2]})
        dfs = [df1, df2, df3]
        merged, cols_with_suffixes = extensions.pandas.merge_all(*dfs, suffix_names=['DW', 'DA', 'DT'], suffix_cols=[
            'value'], on=['key', 'group'], how='inner', sort=False)

        np.testing.assert_array_equal(
            cols_with_suffixes, ['value_DW', 'value_DA', 'value_DT'])
        np.testing.assert_array_equal(
            merged[['key', 'group']], df1[['key', 'group']]
        )
        np.testing.assert_array_equal(
            set(merged.columns), set(['key', 'value_DW', 'value_DA', 'value_DT', 'group']))

        merged, cols_with_suffixes = extensions.pandas.merge_all(df1, suffix_names=['DW'], suffix_cols=[
            'value'], on=['key', 'group'], how='inner', sort=False)
        np.testing.assert_array_equal(merged.values, df1.values)
        np.testing.assert_array_equal(
            set(merged.columns), set(['key', 'value_DW', 'group']))

    def test_filter_column(self):
        df1 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'],
                            'value': [1, 2, 3, 5], 'group': [1, 1, 2, 2]})
        filtered = extensions.pandas.filter_column(
            df1, col='key', values_to_filter_out=['foo'])
        np.testing.assert_array_equal(filtered['value'].values, [2, 3])
        np.testing.assert_array_equal(filtered['key'].values, ['bar', 'baz'])

        filtered = extensions.pandas.filter_column(
            df1, col='value', values_to_filter_out=[2, 5])
        np.testing.assert_array_equal(filtered['value'].values, [1, 3])
        np.testing.assert_array_equal(filtered['key'].values, ['foo', 'baz'])


class TestNumpy:
    def test_mutate_nan(self):
        # test signal without nan
        X = np.random.rand(10, 3)
        X_new = extensions.numpy.mutate_nan(X)
        np.testing.assert_array_almost_equal(X, X_new)

        X = np.random.rand(10, 1)
        X_new = extensions.numpy.mutate_nan(X)
        np.testing.assert_array_almost_equal(X, X_new)

        # test signal with single sample without nan
        X = np.array([[0.]])
        X_new = extensions.numpy.mutate_nan(X)
        np.testing.assert_array_almost_equal(X, X_new)

        X = np.array([[0., 0, 0, ]])
        X_new = extensions.numpy.mutate_nan(X)
        np.testing.assert_array_almost_equal(X, X_new)

        # test signal with nan
        X = np.atleast_2d(np.sin(2*np.pi * 1 * np.arange(0, 1, 1.0 / 100))).T
        X_nan = np.copy(X)
        X_nan[5:10, 0] = np.nan
        X_new = extensions.numpy.mutate_nan(X_nan)
        np.testing.assert_array_almost_equal(X, X_new, decimal=4)

        X = np.tile(np.sin(2*np.pi * 1 *
                           np.arange(0, 1, 1.0 / 100.)), (3, 1)).T
        X_nan = np.copy(X)
        X_nan[5:10, 0:3] = np.nan
        X_new = extensions.numpy.mutate_nan(X_nan)
        np.testing.assert_array_almost_equal(X, X_new, decimal=4)

    def test_butterworth(self, butterworth_test_signals):
        for args, expected in butterworth_test_signals:
            result = extensions.numpy.butterworth(**args)
            diff = expected[10:90, ] - result[10:90, ]
            assert expected.shape == result.shape
            assert np.all(diff < 0.1)

    def test_resample(self, resample_test_signals):
        for test_signal_pair_1, test_signal_pair_2 in resample_test_signals:
            sr1 = test_signal_pair_1[0]
            signal1 = test_signal_pair_1[1]
            sr2 = test_signal_pair_2[0]
            signal2 = test_signal_pair_2[1]
            result12 = extensions.numpy.resample(signal1, sr1, sr2)
            expected12 = signal2
            st = int(sr2/10)
            et = int(sr2 - sr2/10)
            diff12 = expected12[st:et, ] - \
                result12[st:et, ]
            print('{}, {}'.format(sr1, sr2))
            assert expected12.shape == result12.shape
            assert np.all(diff12 < 0.1)

            result21 = extensions.numpy.resample(signal2, sr2, sr1)
            expected21 = signal1
            st = int(sr1/10)
            et = int(sr1 - sr1/10)
            diff21 = expected21[st:et, ] - \
                result21[st:et, ]
            print('{}, {}'.format(sr2, sr1))
            assert expected21.shape == result21.shape
            assert np.all(diff21 < 0.1)

    def test_apply_over_subwins(self):
        func = np.mean
        # test on single row array with subwins and subwin_samples not set
        X = extensions.numpy.atleast_float_2d(np.array([1., 1., 1.]))
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=None, subwins=None, axis=0)
        assert np.array_equal(result, X)

        # test on single row array with subwin_samples not set
        X = extensions.numpy.atleast_float_2d(np.array([1., 1., 1.]))
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=None, subwins=1, axis=0)
        assert np.array_equal(result, X)

        # test on single row array with subwins not set
        X = extensions.numpy.atleast_float_2d(np.array([1., 1., 1.]))
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=1, subwins=None, axis=0)
        assert np.array_equal(result, X)

        # test on single row array with subwins to be zero
        X = extensions.numpy.atleast_float_2d(np.array([1., 1., 1.]))
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=2, subwins=None, axis=0)
        assert np.array_equal(result, X)
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=None, subwins=0, axis=0)
        assert np.array_equal(result, X)

        # test on single row array with subwin_samples to be zero
        X = extensions.numpy.atleast_float_2d(np.array([1., 1., 1.]))
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=0, subwins=None, axis=0)
        assert np.array_equal(result, X)
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=None, subwins=2, axis=0)
        assert np.array_equal(result, X)

        # test on 2d array
        X = extensions.numpy.atleast_float_2d(np.ones((10, 3)))
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=2, subwins=None, axis=0)
        assert np.array_equal(result, np.ones((5, 3)))
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=None, subwins=2, axis=0)
        assert np.array_equal(result, np.ones((2, 3)))

        # test on 2d array use subwins at first
        X = extensions.numpy.atleast_float_2d(np.ones((10, 3)))
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=2, subwins=2, axis=0)
        assert np.array_equal(result, np.ones((2, 3)))

        # test on 2d array use window parameters that are not fully dividable
        X = extensions.numpy.atleast_float_2d(
            np.concatenate((
                np.ones((1, 3)) * 2,
                np.ones((8, 3)),
                np.ones((1, 3)) * 2),
                axis=0)
        )
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=4, subwins=None, axis=0)
        assert np.array_equal(result, np.ones((2, 3)))

        # test on 2d array use window parameters that are not fully dividable
        X = extensions.numpy.atleast_float_2d(
            np.concatenate((
                np.ones((1, 3)) * 2,
                np.ones((3, 3)),
                np.ones((6, 3)) * 2),
                axis=0)
        )
        result = extensions.numpy.apply_over_subwins(
            X, func, subwin_samples=3, subwins=None, axis=0)
        assert np.array_equal(result, np.array(
            [[1, 1, 1], [2, 2, 2], [2, 2, 2]]))

    def test_regularize_sr(self, regularize_test_signals):
        for t, X, sr, expected in regularize_test_signals:
            new_t, new_X = extensions.numpy.regularize_sr(t, X, sr)
            assert len(new_t) == expected
            np.testing.assert_array_equal(new_X.shape, [expected, 3])
