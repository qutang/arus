"""
Computing features about accelerometer orientations
"""

import numpy as np

from .. import extensions as ext
from . import stats

ORIENTATION_FEATURE_NAME_PREFIX = [
    'MEDIAN_G_ANGLE',
    'RANGE_G_ANGLE',
    'STD_G_ANGLE'
]


def _gravity_angles(X, unit='rad'):
    X = ext.numpy.atleast_float_2d(X)
    gravity = stats.mean(X)[0]
    gravity_vm = ext.numpy.vector_magnitude(gravity)
    gravity_angles = np.arccos(
        gravity / gravity_vm) if gravity_vm != 0 else np.zeros_like(gravity)
    if unit == 'deg':
        gravity_angles = np.rad2deg(gravity_angles)
    return gravity_angles


def gravity_angles(X, subwins=None, subwin_samples=None, unit='rad'):
    result = ext.numpy.apply_over_subwins(
        X, _gravity_angles, subwins=subwins, subwin_samples=subwin_samples, unit=unit)
    final_result = np.atleast_2d(result.flatten())
    names = []
    for i in range(result.shape[0]):
        names = names + ['G_ANGLE_X_' + str(i), 'G_ANGLE_Y_' +
                         str(i), 'G_ANGLE_Z_' + str(i)]
    return final_result, names


def orientation_features(X, subwins=None, subwin_samples=None, unit='rad', selected=ORIENTATION_FEATURE_NAME_PREFIX):

    X = ext.numpy.atleast_float_2d(X)
    result = ext.numpy.apply_over_subwins(
        X, _gravity_angles, subwins=subwins, subwin_samples=subwin_samples, unit=unit)

    fv = []
    fv_names = []

    if ORIENTATION_FEATURE_NAME_PREFIX[0] in selected:
        fv.append(np.nanmedian(result, axis=0, keepdims=True))
        fv_names += [f'{ORIENTATION_FEATURE_NAME_PREFIX[0]}_{i}' for i in [0, 1, 2]]

    if ORIENTATION_FEATURE_NAME_PREFIX[1] in selected:
        fv.append(np.nanmax(
            result, axis=0, keepdims=True) - np.nanmin(result, axis=0, keepdims=True))
        fv_names += [f'{ORIENTATION_FEATURE_NAME_PREFIX[1]}_{i}' for i in [0, 1, 2]]

    if ORIENTATION_FEATURE_NAME_PREFIX[2] in selected:
        fv.append(np.nanstd(result, axis=0, keepdims=True, ddof=1))
        fv_names += [f'{ORIENTATION_FEATURE_NAME_PREFIX[2]}_{i}' for i in [0, 1, 2]]

    if len(fv) == 0:
        return None, None

    result = np.concatenate(fv, axis=1)
    return result, fv_names
