"""
Module of functions and classes to run MUSS model.

This module includes functions to preprocess raw accelerometer data, compute features, train MUSS classifier and test with it based on the procedure described in the published manuscript.

Citation: Posture and Physical Activity Detection: Impact of Number of Sensors and Feature Type (MSSE accepted)

Author: Qu Tang
Date: 2019-12-16
License: see LICENSE file
"""
from functools import partial, reduce

import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from ..core.accelerometer.features import activation as accel_activation
from ..core.accelerometer.features import orientation as accel_ori
from ..core.accelerometer.features import spectrum as accel_spectrum
from ..core.accelerometer.features import stats as accel_stats
from ..core.accelerometer.transformation import vector_magnitude
from ..core.libs.dsp.filtering import butterworth
from ..core.libs.num import format_arr


class MUSSModel:
    def __init__(self):
        self._saved_featuresets = []
        self._saved_models = []
        self._FEATURE_NAMES = ['MEAN_0', 'STD_0', 'MAX_0', 'RANGE_0', 'DOM_FREQ_0', 'FREQ_POWER_RATIO_ABOVE_3DOT5_0', 'DOM_FREQ_POWER_RATIO_0', 'ACTIVE_SAMPLES_0', 'ACTIVATIONS_0',
                               'STD_ACTIVATION_DURATIONS_0', "MEDIAN_G_ANGLE_X", "MEDIAN_G_ANGLE_Y", "MEDIAN_G_ANGLE_Z", "RANGE_G_ANGLE_X", "RANGE_G_ANGLE_Y", "RANGE_G_ANGLE_Z"]

    def get_feature_names(self):
        return self._FEATURE_NAMES

    def combine_features(self, *input_features, placement_names=None):
        if placement_names is None:
            placement_names = range(0, len(input_features))
        if len(placement_names) != len(input_features):
            raise ValueError(
                'placement_names should have the same length as the number of input_features')
        sequence = zip(input_features, placement_names)

        def _combine(left, right):
            return pd.merge(left[0], right[0], on=['HEADER_TIME_STAMP', 'START_TIME',
                                                   'STOP_TIME'], suffixes=('_' + str(left[1]), '_' + str(right[1])))
        return reduce(_combine, sequence)

    def sync_feature_and_class(self, input_feature, input_class):
        synced_set = pd.merge(input_feature, input_class,
                              how='inner',
                              on=['HEADER_TIME_STAMP',
                                  'START_TIME', 'STOP_TIME'],
                              suffixes=('_f', '_c'))
        synced_feature = synced_set[input_feature.columns]
        synced_class = synced_set[input_class.columns]
        return synced_feature, synced_class

    def compute_features(self, input_data, sr, st, et, subwin_secs=2, ori_unit='rad', activation_threshold=0.2):
        subwin_samples = subwin_secs * sr
        X = format_arr(input_data.values[:, 1:])
        vm_feature_funcs = [
            accel_stats.mean,
            accel_stats.std,
            accel_stats.max_value,
            accel_stats.max_minus_min,
            partial(accel_spectrum.spectrum_features,
                    sr=sr, n=1, preset='muss'),
            partial(accel_activation.stats_active_samples,
                    threshold=activation_threshold)
        ]

        axis_feature_funcs = [
            partial(accel_ori.gravity_angle_stats,
                    subwin_samples=subwin_samples, unit=ori_unit)
        ]

        X_vm = vector_magnitude(X)

        X_vm_filtered = butterworth(
            X_vm, sr=sr, cut_offs=20, order=4, filter_type='low')
        X_filtered = butterworth(X, sr=sr, cut_offs=20,
                                 order=4, filter_type='low')

        result = {
            'HEADER_TIME_STAMP': [st],
            'START_TIME': [st],
            'STOP_TIME': [et]
        }
        for func in vm_feature_funcs:
            values, names = func(X_vm_filtered)
            for value, name in zip(values.transpose(), names):
                if name in self._FEATURE_NAMES:
                    result[name] = value.tolist()

        for func in axis_feature_funcs:
            values, names = func(X_filtered)
            for value, name in zip(values.transpose(), names):
                if name in self._FEATURE_NAMES:
                    result[name] = value.tolist()

        result = pd.DataFrame.from_dict(result)
        return result

    def train_classifier(self, input_feature, input_class, C=16, kernel='rbf', gamma=0.25, tol=0.0001, output_probability=True, class_weight='balanced', verbose=False, save=True):
        """Function to train MUSS classifier given input feature vectors and class labels.

        gamma with 5 better than 0.25, especially between ambulation and lying, we should use a larger gamma for higher decaying impact of neighbor samples.

        Args:
            input_features (numpy.array): A 2D array that stores the input feature vectors
            input_classes (numpy.array): A 2D array that stores the class labels
            C (int, optional): Parameter for SVM classifier. Defaults to 16.
            kernel (str, optional): Parameter for SVM classifier. Defaults to 'rbf'.
            gamma (float, optional): Parameter for SVM classifier. Defaults to 0.25.
            tol (float, optional): Parameter for SVM classifier. Defaults to 0.0001.
            output_probability (bool, optional): Parameter for SVM classifier. Defaults to True.
            class_weight (str, optional): Parameter for SVM classifier. Defaults to 'balanced'.
            verbose (bool, optional): Display training information. Defaults to True.
            save (bool, optional): Save trained model to instance variable. Defaults to True.
        """
        input_matrix = input_feature.values[:, 3:]
        input_classes = input_class.values[:, 3]

        # remove transition and unknown indices
        is_valid_label = (input_classes != 'Transition') & (
            input_classes != 'Unknown')
        input_matrix = input_matrix[is_valid_label, :]
        input_classes = input_classes[is_valid_label]

        input_matrix, input_classes = shuffle(input_matrix, input_classes)
        classifier = svm.SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            tol=tol,
            probability=output_probability,
            class_weight=class_weight,
            verbose=verbose)
        scaler = MinMaxScaler((-1, 1))
        scaled_X = scaler.fit_transform(input_matrix)
        model = classifier.fit(scaled_X, input_classes)
        train_accuracy = model.score(scaled_X, input_classes)
        result = (model, scaler, train_accuracy)
        if save:
            self._saved_models.append(result)
        return result

    def predict(self, test_features, model=-1):
        if type(model) is tuple:
            scaler = model[1]
            classifier = model[0]
        elif type(model) is int:
            scaler = self._saved_models[model][1]
            classifier = self._saved_models[model][0]
        scaled_X = scaler.transform(test_features)
        predicted_classes = classifier.predict(scaled_X)
        return predicted_classes
