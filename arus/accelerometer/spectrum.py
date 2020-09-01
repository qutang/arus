"""
Frequency domain features for numerical time series data
"""
import numpy as np
from loguru import logger
from scipy import signal

from .. import extensions as ext

SPECTRUM_FEATURE_NAME_PREFIX = [
    'DOM_FREQ',
    'DOM_FREQ_POWER',
    'TOTAL_FREQ_POWER',
    'FREQ_POWER_ABOVE_3DOT5',
    'FREQ_POWER_RATIO_ABOVE_3DOT5',
    'DOM_FREQ_POWER_RATIO',
    'FREQ_POWER_RATIO_BEWTEEN_DOT5_2DOT5',
    'DOM_FREQ_POWER_BETWEEN_DOT6_2DOT6',
    'DOM_FREQ_RATIO_PREV_BOUT',
    'SPECTRAL_ENTROPY'
]


def spectrum_features(X, sr, n=1, freq_range=None, prev_spectrum_features=None, selected=SPECTRUM_FEATURE_NAME_PREFIX):
    X = ext.numpy.atleast_float_2d(X)
    # fill nan at first, nan will be filled by spline interpolation
    X = ext.numpy.mutate_nan(X)
    freq, Sxx = _fft(X, sr, freq_range=freq_range)
    freq_peaks, Sxx_peaks = _fft_peaks(freq, Sxx)

    fv = []
    fv_names = []

    if SPECTRUM_FEATURE_NAME_PREFIX[0] in selected:
        fv.append(_dom_freq(
            freq_peaks, Sxx_peaks, n=n))
        fv_names += [
            f'{SPECTRUM_FEATURE_NAME_PREFIX[0]}_{i}' for i in range(X.shape[1])
        ]

    if SPECTRUM_FEATURE_NAME_PREFIX[1] in selected:
        fv.append(_dom_freq_power(
            freq_peaks, Sxx_peaks, n=n))
        fv_names += [
            f'{SPECTRUM_FEATURE_NAME_PREFIX[1]}_{i}' for i in range(X.shape[1])
        ]

    if SPECTRUM_FEATURE_NAME_PREFIX[2] in selected:
        fv.append(_total_freq_power(Sxx))
        fv_names += [
            f'{SPECTRUM_FEATURE_NAME_PREFIX[2]}_{i}' for i in range(X.shape[1])
        ]

    if SPECTRUM_FEATURE_NAME_PREFIX[3] in selected:
        fv.append(_freq_power_above_3_point_5(freq, Sxx))
        fv_names += [
            f'{SPECTRUM_FEATURE_NAME_PREFIX[3]}_{i}' for i in range(X.shape[1])
        ]

    if SPECTRUM_FEATURE_NAME_PREFIX[4] in selected:
        fv.append(_freq_power_ratio_above_3_point_5(freq, Sxx))
        fv_names += [
            f'{SPECTRUM_FEATURE_NAME_PREFIX[4]}_{i}' for i in range(X.shape[1])
        ]

    if SPECTRUM_FEATURE_NAME_PREFIX[5] in selected:
        fv.append(_dom_freq_power_ratio(
            freq, Sxx, freq_peaks, Sxx_peaks, n=n))
        fv_names += [
            f'{SPECTRUM_FEATURE_NAME_PREFIX[5]}_{i}' for i in range(X.shape[1])
        ]

    if SPECTRUM_FEATURE_NAME_PREFIX[6] in selected:
        fv.append(_freq_power_ratio_between_point_5_and_2_point_5(
            freq, Sxx
        ))
        fv_names += [
            f'{SPECTRUM_FEATURE_NAME_PREFIX[6]}_{i}' for i in range(X.shape[1])
        ]

    if SPECTRUM_FEATURE_NAME_PREFIX[7] in selected:
        fv.append(_dom_freq_power_between_point_6_and_2_point_6(
            freq_peaks,
            Sxx_peaks
        ))
        fv_names += [
            f'{SPECTRUM_FEATURE_NAME_PREFIX[7]}_{i}' for i in range(X.shape[1])]

    if SPECTRUM_FEATURE_NAME_PREFIX[8] in selected:
        if prev_spectrum_features is not None:
            fv.append(_dom_freq_ratio_previous_bout(
                freq_peaks, Sxx_peaks, prev_dom_freq=prev_spectrum_features[0, :], n=1))
        else:
            fv.append(_dom_freq_ratio_previous_bout(
                freq_peaks, Sxx_peaks, prev_dom_freq=None, n=1))
        fv_names += [
            f'{SPECTRUM_FEATURE_NAME_PREFIX[8]}_{i}' for i in range(X.shape[1])
        ]

    if SPECTRUM_FEATURE_NAME_PREFIX[9] in selected:
        fv.append(_spectral_entropy(freq, Sxx))
        fv_names += [
            f'{SPECTRUM_FEATURE_NAME_PREFIX[9]}_{i}' for i in range(X.shape[1])
        ]

    if len(fv) == 0:
        return None, None

    result = np.concatenate(fv, axis=1)
    return result, fv_names


def _fft(X, sr, freq_range=None):
    freq, time, Sxx = signal.spectrogram(
        X,
        fs=sr,
        window='hamming',
        nperseg=X.shape[0],
        noverlap=0,
        detrend='constant',
        return_onesided=True,
        scaling='density',
        axis=0,
        mode='magnitude')
    Sxx = np.abs(Sxx[:, :, 0])
    # interpolate to get values in the freq_range
    if freq_range is not None:
        Sxx_interpolated = np.interp(freq_range, freq, Sxx)
    else:
        Sxx_interpolated = Sxx
        freq_range = freq
    Sxx_interpolated = np.atleast_2d(Sxx_interpolated)
    return freq_range, Sxx_interpolated


def _sort_fft_peaks(freq, Sxx, i, j):
    if len(i) == 0:
        sorted_freq_peaks = np.array([0])
        sorted_Sxx_peaks = np.array([np.nanmean(Sxx, axis=0)[j]])
    else:
        freq_peaks = freq[i]
        Sxx_peaks = Sxx[i, j]
        sorted_i = np.argsort(Sxx_peaks)
        sorted_i = sorted_i[:: -1]
        sorted_freq_peaks = freq_peaks[sorted_i]
        sorted_Sxx_peaks = Sxx_peaks[sorted_i]
    # logger.debug('sxx:' + str(j) + ":" + str(sorted_Sxx_peaks.shape))
    # logger.debug('freq:' + str(j) + ":" + str(sorted_freq_peaks.shape))
    return (sorted_freq_peaks, sorted_Sxx_peaks)


def _fft_peaks(freq, Sxx):
    n_axis = Sxx.shape[1]
    m_freq = Sxx.shape[0]
    if m_freq == 1:
        freq_peaks = [freq] * n_axis
        Sxx_peaks = [Sxx[:, i] for i in range(Sxx.shape[1])]
    else:
        # at least 0.1 Hz different when looking for peak
        mpd = int(np.ceil(1.0 / (freq[1] - freq[0]) * 0.1))
        # print(self._Sxx.shape)
        # mph should not be set, because signal can be weak but there may still be some dominant frequency, 06/03/2019
        freq_peak_indices = list(map(lambda x: ext.numpy.detect_peaks(
            x, mph=None, mpd=mpd), list(Sxx.T)))
        # i = list(map(lambda x: detect_peaks(
        #     x, mph=1e-3, mpd=mpd), list(self._Sxx.T)))
        axes = range(0, n_axis)
        freq_peaks = []
        Sxx_peaks = []
        for i, j in zip(freq_peak_indices, axes):
            freq_peak, Sxx_peak = _sort_fft_peaks(freq, Sxx, i, j)
            freq_peaks.append(freq_peak)
            Sxx_peaks.append(Sxx_peak)
        # note that freq_peaks and Sxx_peaks will have structure
        # [freq_peaks_for_x_axis, freq_peaks_for_y_axis, ...]
        #  And each item would be a 1d array
    return freq_peaks, Sxx_peaks


def _spectral_entropy(freqs, Sxx):
    # normalized
    sum_of_Sxx = np.sum(Sxx, axis=0)
    psd = Sxx / sum_of_Sxx
    s_entropy = -np.sum(np.multiply(psd, np.log2(psd)), axis=0)
    result = s_entropy / np.log2(len(freqs))
    result = np.atleast_2d(result)
    return result


def _dom_freq(freq_peaks, Sxx_peaks, n=1):
    result = []
    for i in range(len(freq_peaks)):
        if len(freq_peaks[i]) >= n:
            if not np.isnan(Sxx_peaks[i][n-1]):
                result.append(freq_peaks[i][n-1])
            else:
                result.append(np.nan)
        else:
            result.append(np.nan)
    result = np.atleast_2d(np.array(result))
    return result


def _dom_freq_power(freq_peaks, Sxx_peaks, n=1):
    result = []
    for i in range(len(Sxx_peaks)):
        if len(Sxx_peaks[i]) >= n:
            result.append(Sxx_peaks[i][n-1])
        else:
            result.append(np.nan)
    result = np.atleast_2d(np.array(result))
    return result


def _total_freq_power(Sxx):
    result = np.sum(Sxx, axis=0, keepdims=True)
    return result


def _dom_freq_in_band(freq_peaks, Sxx_peaks, low=0, high=np.inf, n=1):
    result = []
    for i in range(len(Sxx_peaks)):
        freq = freq_peaks[i]
        indices = (freq >= low) & (freq <= high)
        limited_freq = freq[indices]
        if len(limited_freq) < n:
            result.append(np.nan)
        else:
            result.append(limited_freq[n-1])
    result = np.atleast_2d(np.array(result))
    return result


def _dom_freq_power_in_band(freq_peaks, Sxx_peaks, low=0, high=np.inf, n=1):
    result = []
    for i in range(len(Sxx_peaks)):
        freq = freq_peaks[i]
        Sxx = Sxx_peaks[i]
        indices = (freq >= low) & (freq <= high)
        limited_Sxx = Sxx[indices]
        if len(limited_Sxx) < n:
            result.append(np.nan)
        else:
            result.append(limited_Sxx[n-1])
    result = np.atleast_2d(np.array(result))
    return result


def _total_freq_power_in_band(freq, Sxx, low=0, high=np.inf):
    indices = (freq >= low) & (freq <= high)
    limited_Sxx = Sxx[indices, :]
    limited_total_power = np.sum(limited_Sxx, axis=0, keepdims=True)
    return limited_total_power


def _freq_power_above_3_point_5(freq, Sxx):
    result = _total_freq_power_in_band(freq, Sxx, low=3.5)
    return result


def _freq_power_ratio_above_3_point_5(freq, Sxx):
    highend_power = _freq_power_above_3_point_5(freq, Sxx)
    total_power = _total_freq_power(Sxx)
    result = np.divide(highend_power, total_power, out=np.zeros_like(
        total_power), where=total_power != 0)
    return result


def _freq_power_ratio_between_point_5_and_2_point_5(freq, Sxx):
    band_power = _total_freq_power_in_band(freq, Sxx, low=0.5, high=2.5)
    total_power = _total_freq_power(Sxx)
    result = np.divide(band_power, total_power, out=np.zeros_like(
        total_power), where=total_power != 0)
    return result


def _dom_freq_power_ratio(freq, Sxx, freq_peaks, Sxx_peaks, n=1):
    total_power = _total_freq_power(Sxx)
    dom_freq_power = _dom_freq_power(freq_peaks, Sxx_peaks, n=n)
    result = np.divide(dom_freq_power,
                       total_power, out=np.zeros_like(total_power), where=total_power != 0)
    return result


def _dom_freq_between_point_6_and_2_point_6(freq_peaks, Sxx_peaks):
    result = _dom_freq_in_band(freq_peaks, Sxx_peaks,
                               low=0.6, high=2.6, n=1)
    return result


def _dom_freq_power_between_point_6_and_2_point_6(freq_peaks, Sxx_peaks):
    result = _dom_freq_power_in_band(freq_peaks, Sxx_peaks, low=0.6, high=2.6,
                                     n=1)
    return result


def _dom_freq_ratio_previous_bout(freq_peaks, Sxx_peaks, prev_dom_freq=None, n=1):
    if prev_dom_freq is None:
        result = np.full((1, len(Sxx_peaks)), np.nan)
    else:
        current_dom_freq = _dom_freq(freq_peaks, Sxx_peaks, n=1)
        result = np.divide(current_dom_freq, np.atleast_2d(prev_dom_freq), out=np.zeros_like(
            prev_dom_freq), where=prev_dom_freq != 0)
    return result
