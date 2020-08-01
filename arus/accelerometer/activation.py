"""
Computing features related to activation samples.

Author: Qu Tang

Date: Jul 10, 2018

References:

1. Mannini A, Rosenberger M, Haskell WL, Sabatini AM, Intille SS. Activity
 Recognition in Youth Using Single Accelerometer Placed at Wrist or Ankle. Med
 Sci Sports Exerc. 2017;49(4):801–12.
 https://www.ncbi.nlm.nih.gov/pubmed/27820724

"""

import numpy as np
from .. import extensions as ext


ACTIVATION_FEATURE_NAME_PREFIX = [
    'ACTIVE_SAMPLES', 'ACTIVATIONS', 'MEAN_ACTIVATION_DURATIONS', 'STD_ACTIVATION_DURATIONS'
]


def _active_samples(X, threshold=0.2):
    active_samples = X >= threshold
    return active_samples


def _stats_active_samples_1d(active_samples):
    active_edges = np.diff(active_samples, prepend=0, append=0)
    rising_edges = np.argwhere(active_edges > 0)
    falling_edges = np.argwhere(active_edges < 0)
    durations = []
    if len(rising_edges) == 0:
        durations = [0, 0]
    else:
        if rising_edges[0] > falling_edges[0]:
            falling_edges = falling_edges[1:]
        if rising_edges[-1] > falling_edges[-1]:
            rising_edges = rising_edges[0:-1]
        if len(rising_edges) == 0:
            durations = [0, 0]
    num_activation_crossings = len(rising_edges)
    for r, f in zip(rising_edges, falling_edges):
        durations.append(f - r)
    mean_activation_duration = np.mean(durations)
    if len(durations) <= 1:
        std_activation_duration = 0
    else:
        std_activation_duration = np.std(durations, ddof=1)
    return np.array([num_activation_crossings, mean_activation_duration, std_activation_duration])


def _stats_active_samples(active_samples):
    result = np.apply_along_axis(
        _stats_active_samples_1d, axis=0, arr=active_samples)
    return result


def activation_features(X,
                        threshold=0.2,   selected=ACTIVATION_FEATURE_NAME_PREFIX):
    X = ext.numpy.atleast_float_2d(X)
    W = X.shape[0]

    fv = []
    fv_names = []

    active_samples = _active_samples(X, threshold=threshold)
    num_active_samples = np.sum(active_samples, axis=0, keepdims=True)
    stats = _stats_active_samples(active_samples)

    """
    This feature identifies the amount of activity within the window that is over the threshold, thereby providing a rough estimate of the amount of relevant activity being recorded in the window. This feature could distinguish between activities that result in relevant acceleration in most of the window and activities in which the relevant activity takes place for only a portion of the window.
    """
    if ACTIVATION_FEATURE_NAME_PREFIX[0] in selected:
        fv.append(num_active_samples / W)
        fv_names += [
            f'{ACTIVATION_FEATURE_NAME_PREFIX[0]}_{i}' for i in range(X.shape[1])]
    """
    "This feature identifies the number of threshold crossings within the window (rising edges only), normalized to the number of active samples FS, thereby capturing movement fragmentation within the window. This feature could discriminate impulsive events from longer-lasting acceleration events, since it quantifies how many times within the window the acceleration passed from the inactive to the active condition."
    """
    if ACTIVATION_FEATURE_NAME_PREFIX[1] in selected:
        num_activations = np.atleast_2d(stats[0, :])
        fv.append(np.divide(
            num_activations, num_active_samples, where=num_activations != 0.0))
        fv_names += [
            f'{ACTIVATION_FEATURE_NAME_PREFIX[1]}_{i}' for i in range(X.shape[1])]
    """
    This captures the mean duration of activation intervals within the window, normalized by the window length. An activation interval is defined as the amount of samples between two consecutive threshold crossings. This feature provides information on movement bout fragmentation within the window that could help discriminate between activities that involve stable movements, such as those in natural walking, and those with short bouts, such as sport ones.
    """
    if ACTIVATION_FEATURE_NAME_PREFIX[2] in selected:
        mean_activation_durations = np.atleast_2d(stats[1, :])
        fv.append(mean_activation_durations / W)
        fv_names += [
            f'{ACTIVATION_FEATURE_NAME_PREFIX[2]}_{i}' for i in range(X.shape[1])]
    """
    This feature captures the standard deviation of the duration of activation intervals within the window, normalized by the window length, thereby providing information on uniformity of activation intervals within the window. This feature may help discriminate between activities with cyclic movements with a very stable ratio between active and inactive phases, and more random activities. A stable cyclic movement would result in lower variability of activation intervals that would repeat themselves within the window. Fast and aperiodic movements, however, such as those in recreational activities would result in highly variable activation bouts.
    """
    if ACTIVATION_FEATURE_NAME_PREFIX[3] in selected:
        std_activation_durations = np.atleast_2d(stats[2, :])
        fv.append(std_activation_durations / W)
        fv_names += [
            f'{ACTIVATION_FEATURE_NAME_PREFIX[3]}_{i}' for i in range(X.shape[1])]

    if len(fv) == 0:
        return None, None

    result = np.concatenate(fv, axis=1)

    return result, fv_names
