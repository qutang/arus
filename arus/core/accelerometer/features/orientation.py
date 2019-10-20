"""

Computing features about accelerometer orientations

Author: Qu Tang

Date: Jul 10, 2018
"""
import logging
from functools import reduce

import numpy as np


from ...libs.dsp.apply_over_subwins import apply_over_subwins
from ...libs.num import format_arr
from . import stats
from ..transformation import vector_magnitude


def _gravity_angles(X, unit='rad'):
    X = format_arr(X)
    gravity = stats.mean(X)[0]
    gravity_vm = vector_magnitude(gravity)
    gravity_angles = np.arccos(
        gravity / gravity_vm) if gravity_vm != 0 else np.zeros_like(gravity)
    if unit == 'deg':
        gravity_angles = np.rad2deg(gravity_angles)
    return gravity_angles


def gravity_angles(X, subwins=None, subwin_samples=None, unit='rad'):
    result = apply_over_subwins(
        X, _gravity_angles, subwins=subwins, subwin_samples=subwin_samples, unit=unit)
    final_result = np.atleast_2d(result.flatten())
    names = []
    for i in range(result.shape[0]):
        names = names + ['G_ANGLE_X_' + str(i), 'G_ANGLE_Y_' +
                         str(i), 'G_ANGLE_Z_' + str(i)]
    return final_result, names


def gravity_angle_stats(X, subwins=None, subwin_samples=None, unit='rad'):
    result = apply_over_subwins(
        X, _gravity_angles, subwins=subwins, subwin_samples=subwin_samples, unit=unit)
    median_angles = np.nanmedian(result, axis=0, keepdims=True)
    range_angles = np.nanmax(
        result, axis=0, keepdims=True) - np.nanmin(result, axis=0, keepdims=True)
    std_angles = np.nanstd(result, axis=0, keepdims=True, ddof=1)
    final_result = np.concatenate(
        (median_angles, range_angles, std_angles), axis=1)
    return final_result, ["MEDIAN_G_ANGLE_X", "MEDIAN_G_ANGLE_Y", "MEDIAN_G_ANGLE_Z", "RANGE_G_ANGLE_X", "RANGE_G_ANGLE_Y", "RANGE_G_ANGLE_Z", "STD_G_ANGLE_X", "STD_G_ANGLE_Y", "STD_G_ANGLE_Z"]
