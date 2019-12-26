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
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import seaborn as sns

from ..core.accelerometer.features import activation as accel_activation
from ..core.accelerometer.features import orientation as accel_ori
from ..core.accelerometer.features import spectrum as accel_spectrum
from ..core.accelerometer.features import stats as accel_stats
from ..core.accelerometer.transformation import vector_magnitude
from ..core.libs.dsp.filtering import butterworth
from ..core.libs.num import format_arr
from ..core.pipeline import Pipeline


def muss_inference_processor(chunk_list, **kwargs):
    import pandas as pd
    from .muss import MUSSModel
    model = kwargs['model']
    muss = MUSSModel()
    feature_dfs = []
    placement_names = []
    for df, st, et, prev_st, prev_et, name in chunk_list:
        feature_df = muss.compute_features(df, sr=kwargs['sr'], st=st, et=et)
        feature_dfs.append(feature_df)
        placement_names.append(name)
    sorted_feature_dfs = []
    try:
        for p in model[-2]:
            i = placement_names.index(p)
            sorted_feature_dfs.append(feature_dfs[i])
    except:
        print(
            'Make sure the placements between your data and the trained model are the same.')
    if len(sorted_feature_dfs) == 1:
        combined_df = sorted_feature_dfs[0]
    else:
        combined_df, feature_names = muss.combine_features(
            *sorted_feature_dfs, placement_names=model[-2])
    filtered_combined_df = muss.select_features(
        combined_df, feature_names=model[-1])
    predicted_probs = muss.predict(
        filtered_combined_df, model=model, output_prob=True)
    return predicted_probs


class MUSSModel:
    def __init__(self):
        self._saved_featuresets = []
        self._saved_models = []
        self._FEATURE_NAMES = ['MEAN_0', 'STD_0', 'MAX_0', 'RANGE_0', 'DOM_FREQ_0', 'FREQ_POWER_RATIO_ABOVE_3DOT5_0', 'DOM_FREQ_POWER_RATIO_0', 'ACTIVE_SAMPLES_0', 'ACTIVATIONS_0',
                               'STD_ACTIVATION_DURATIONS_0', "MEDIAN_G_ANGLE_X", "MEDIAN_G_ANGLE_Y", "MEDIAN_G_ANGLE_Z", "RANGE_G_ANGLE_X", "RANGE_G_ANGLE_Y", "RANGE_G_ANGLE_Z"]

    def get_feature_names(self):
        return self._FEATURE_NAMES

    def combine_features(self, *input_features, placement_names=None, group_col=None):
        if group_col is None:
            group_col = []
        else:
            group_col = [group_col]
        if placement_names is None:
            placement_names = range(0, len(input_features))
        if len(placement_names) != len(input_features):
            raise ValueError(
                'placement_names should have the same length as the number of input_features')
        sequence = zip(input_features, placement_names)

        def _combine(left, right):
            return pd.merge(left[0], right[0], on=['HEADER_TIME_STAMP', 'START_TIME',
                                                   'STOP_TIME'] + group_col, suffixes=('_' + str(left[1]), '_' + str(right[1])))
        combined_df = reduce(_combine, sequence)
        combined_feature_names = list(filter(lambda name: name.split('_')
                                             [-1] in placement_names, combined_df.columns))
        return combined_df, combined_feature_names

    def select_features(self, input_feature, feature_names='All', group_col=None):
        if group_col is None:
            group_col = []
        else:
            group_col = [group_col]
        if feature_names == 'All':
            return input_feature
        else:
            selected_cols = ['HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'] + \
                group_col + feature_names
            return input_feature[selected_cols]

    def sync_feature_and_class(self, input_feature, input_class, group_col=None):
        group_col = [] if group_col is None else [group_col]
        synced_set = pd.merge(input_feature, input_class,
                              how='inner',
                              on=['HEADER_TIME_STAMP',
                                  'START_TIME', 'STOP_TIME'] + group_col,
                              suffixes=('_f', '_c'))
        synced_feature = synced_set[input_feature.columns]
        synced_class = synced_set[input_class.columns]
        return synced_feature, synced_class

    def compute_features(self, input_data, sr, st, et, subwin_secs=2, ori_unit='rad', activation_threshold=0.2):
        subwin_samples = subwin_secs * sr
        X = format_arr(input_data.values[:, 1:4])
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

    def remove_classes(self, input_feature, input_class, class_col, classes_to_remove=['Unknown', 'Transition']):
        # remove transition and unknown indices
        is_valid_label = ~input_class[class_col].isin(classes_to_remove).values
        filtered_feature = input_feature.loc[is_valid_label, :]
        filtered_class = input_class[is_valid_label]
        return filtered_feature, filtered_class

    def remove_groups(self, input_feature, input_class, group_col, groups_to_remove=[]):
        is_valid_groups = ~input_class[group_col].isin(groups_to_remove)
        filtered_feature = input_feature.loc[is_valid_groups, :]
        filtered_class = input_class[is_valid_groups]
        return filtered_feature, filtered_class

    def _train_classifier(self, input_feature_arr, input_class_vec, placement_names, feature_names, C=16, kernel='rbf', gamma=0.25, tol=0.0001, output_probability=True, class_weight='balanced', verbose=False):
        input_matrix, input_classes = shuffle(
            input_feature_arr, input_class_vec)
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
        result = (model, scaler, train_accuracy,
                  placement_names, feature_names)
        return result

    def train_classifier(self, input_feature, input_class, class_col, feature_names, placement_names, save=True, **kwargs):
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
        input_matrix = input_feature[feature_names].values
        input_classes = input_class[class_col].values
        result = self._train_classifier(
            input_matrix, input_classes, placement_names=placement_names, feature_names=feature_names, **kwargs)
        if save:
            self._saved_models.append(result)
        return result

    def _predict(self, test_feature, classifier, scaler, output_prob=False):
        scaled_X = scaler.transform(test_feature)
        if output_prob:
            return classifier.predict_proba(scaled_X)
        else:
            return classifier.predict(scaled_X)

    def predict(self, test_features, model=-1, output_prob=False):
        if type(model) is tuple:
            scaler = model[1]
            classifier = model[0]
        elif type(model) is int:
            scaler = self._saved_models[model][1]
            classifier = self._saved_models[model][0]
        test_feature = test_features[model[-1]].values
        return self._predict(test_feature, classifier, scaler, output_prob=output_prob)

    def validate_classifier(self, input_feature, input_class, class_col, feature_names, placement_names, group_col, **train_kwargs):
        input_feature_arr = input_feature[feature_names].values
        input_class_vec = input_class[class_col].values
        groups = input_class[group_col].values
        logo = LeaveOneGroupOut()
        output_class_vec = input_class_vec.copy()
        for train_split, test_split in logo.split(input_feature_arr, input_class_vec, groups):
            train_feature = input_feature_arr[train_split, :]
            train_class = input_class_vec[train_split]
            test_feature = input_feature_arr[test_split, :]
            model = self._train_classifier(
                train_feature, train_class, feature_names=feature_names, placement_names=placement_names, ** train_kwargs)
            predict_class = self._predict(
                test_feature, classifier=model[0], scaler=model[1])
            output_class_vec[test_split] = predict_class
        acc = accuracy_score(input_class_vec, output_class_vec)
        return input_class_vec, output_class_vec, acc

    def _plot_confusion_matrix(self, conf_matrix, size):
        # plot confusion matrix
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 8
        rcParams['font.serif'] = ['Times New Roman']
        plt.subplots(figsize=(4, 4))
        sns.set_style({
            'font.family': 'serif',
            'font.size': 8
        })
        g = sns.heatmap(conf_matrix, annot=True, cmap="Greys",
                        cbar=False, fmt='d', robust=True, linewidths=0.2)
        g.set(xlabel="Prediction", ylabel="Ground truth")
        plt.tight_layout()
        return plt.gcf()

    def get_confusion_matrix(self, input_class, predict_class, labels, graph=False):
        conf_mat = confusion_matrix(input_class, predict_class, labels=labels)
        conf_df = pd.DataFrame(conf_mat, columns=labels, index=labels)
        conf_df.index.rename(name='Ground Truth')
        if graph:
            result = self._plot_confusion_matrix(
                conf_df, size=(len(labels), len(labels)))
        else:
            result = conf_df
        return result

    def get_classification_report(self, input_class, predict_class, labels):
        return classification_report(input_class, predict_class, labels=labels)

    @staticmethod
    def get_inference_pipeline(*streams, name='muss-pipeline', model=-1, sr=100, max_processes=2, scheduler='processes'):
        pipeline = Pipeline(
            max_processes=max_processes, scheduler=scheduler, name=name)
        for stream in streams:
            pipeline.add_stream(stream)
        pipeline.set_processor(muss_inference_processor,
                               model=model, sr=sr)
        return pipeline
