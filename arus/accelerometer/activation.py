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
from .. import extensions


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


def stats_active_samples(X, threshold=0.2):
    X = extensions.numpy.atleast_float_2d(X)
    W = X.shape[0]

    active_samples = _active_samples(X, threshold=threshold)
    num_active_samples = np.sum(active_samples, axis=0, keepdims=True)
    stats = _stats_active_samples(active_samples)
    """
    This feature identifies the amount of activity within the window that is over the threshold, thereby providing a rough estimate of the amount of relevant activity being recorded in the window. This feature could distinguish between activities that result in relevant acceleration in most of the window and activities in which the relevant activity takes place for only a portion of the window.
    """
    stats_num_active_samples = num_active_samples / W

    """
    "This feature identifies the number of threshold crossings within the window (rising edges only), normalized to the number of active samples FS, thereby capturing movement fragmentation within the window. This feature could discriminate impulsive events from longer-lasting acceleration events, since it quantifies how many times within the window the acceleration passed from the inactive to the active condition."
    """
    num_activations = np.atleast_2d(stats[0, :])
    stats_num_activations = np.divide(
        num_activations, num_active_samples, where=num_activations != 0.0)
    """
    This captures the mean duration of activation intervals within the window, normalized by the window length. An activation interval is defined as the amount of samples between two consecutive threshold crossings. This feature provides information on movement bout fragmentation within the window that could help discriminate between activities that involve stable movements, such as those in natural walking, and those with short bouts, such as sport ones.
    """
    mean_activation_durations = np.atleast_2d(stats[1, :])
    stats_mean_activation_durations = mean_activation_durations / W

    """
    This feature captures the standard deviation of the duration of activation intervals within the window, normalized by the window length, thereby providing information on uniformity of activation intervals within the window. This feature may help discriminate between activities with cyclic movements with a very stable ratio between active and inactive phases, and more random activities. A stable cyclic movement would result in lower variability of activation intervals that would repeat themselves within the window. Fast and aperiodic movements, however, such as those in recreational activities would result in highly variable activation bouts.
    """
    std_activation_durations = np.atleast_2d(stats[2, :])
    stats_std_activation_durations = std_activation_durations / W

    result = np.atleast_2d(np.concatenate((stats_num_active_samples,
                                           stats_num_activations,
                                           stats_mean_activation_durations, stats_std_activation_durations), axis=1))
    names = []
    for i in range(X.shape[1]):
        names = names + ["ACTIVE_SAMPLES_" + str(i), "ACTIVATIONS_" + str(
            i), "MEAN_ACTIVATION_DURATIONS_" + str(i), "STD_ACTIVATION_DURATIONS_" + str(i)]
    return result, names
