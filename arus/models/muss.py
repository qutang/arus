"""
Module of functions and classes to run MUSS model.

This module includes functions to preprocess raw accelerometer data, compute features, train MUSS classifier and test with it based on the procedure described in the published manuscript.

Citation: Posture and Physical Activity Detection: Impact of Number of Sensors and Feature Type (MSSE accepted)

Author: Qu Tang
Date: 2019-12-16
License: see LICENSE file
"""
import functools

import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn import metrics as sk_metrics
from sklearn import model_selection as sk_model_selection
from sklearn import preprocessing as sk_preprocessing
from sklearn import utils as sk_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import enum

from ..core import pipeline as arus_pipeline
from .. import mhealth_format as mh
from .. import extensions
from .. import accelerometer as accel


def muss_inference_processor(chunk_list, **kwargs):
    import pandas as pd
    from .muss import MUSSModel
    model = kwargs['model']
    muss = MUSSModel()
    feature_dfs = []
    placement_names = []
    for df, st, et, prev_st, prev_et, name in chunk_list:
        feature_df = muss.compute_features(
            df, sr=kwargs[name]['sr'], st=st, et=et)
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
    print(model[0].classes_)
    print(predicted_probs)
    return predicted_probs, filtered_combined_df, None


def muss_data_collection_processor(chunk_list, **kwargs):
    import pandas as pd
    from .muss import MUSSModel
    import pathos.pools as ppools
    output_folder = kwargs['output_folder']
    pid = kwargs['pid']
    model = kwargs['model']
    muss = MUSSModel()
    raw_data_dfs = []
    feature_dfs = []
    placement_names = []
    sorted_feature_dfs = []

    def _save_raw(chunk_data, output_folder, pid):
        print('Saving task started.')
        df = chunk_data[0]
        name = chunk_data[-1]
        writer = mh.MhealthFileWriter(
            output_folder, pid, hourly=False, date_folders=False)
        writer.set_for_sensor(
            'MetaWearR', 'AccelerationCalibrated', name, version_code='NA')
        output_paths = writer.write_csv(
            df.iloc[:, 0:4], append=True, block=True)
        print('Saved data to: ' + str(output_paths))

    io_pool = ppools.ThreadPool(nodes=len(chunk_list))
    io_pool.restart(force=True)
    save_task = io_pool.amap(_save_raw, chunk_list, len(
        chunk_list) * [output_folder], len(chunk_list) * [pid])
    for df, st, et, prev_st, prev_et, name in chunk_list:
        raw_data_dfs.append(df)
        feature_df = muss.compute_features(
            df, sr=kwargs[name]['sr'], st=st, et=et)
        feature_dfs.append(feature_df)
        placement_names.append(name)

    if model is not None:
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
    else:
        filtered_combined_df, feature_names = muss.combine_features(
            *feature_dfs, placement_names=placement_names)
        predicted_probs = None
    save_task.get(timeout=1)
    io_pool.close()
    io_pool.join()
    return predicted_probs, filtered_combined_df, raw_data_dfs


def muss_mhealth_dataset_processor(chunk_list, **kwargs):
    import pandas as pd
    from .muss import MUSSModel
    from .. import dataset
    muss = MUSSModel()
    feature_dfs = []
    for df, st, et, prev_st, prev_et, name in chunk_list:
        if name not in mh.SENSOR_PLACEMENTS:
            if not df.empty:
                class_label = dataset.parse_annotations(
                    kwargs['dataset_name'], df, kwargs['pid'], st, et)
            else:
                class_label = "Unknown"
        else:
            if df.empty:
                continue
            else:
                placement = name
                feature_df = muss.compute_features(
                    df, sr=kwargs[name]['sr'], st=st, et=et)
                feature_df['PLACEMENT'] = placement
                feature_dfs.append(feature_df)
    if len(feature_dfs) == 0:
        combined_df = pd.DataFrame()
    elif len(feature_dfs) == 1:
        combined_df = feature_dfs[0]
    else:
        combined_df = pd.concat(feature_dfs, axis=0, sort=False)
    combined_df['CLASS_LABEL_' + name] = class_label
    return combined_df


class Strategy(enum.Enum):
    REPLACE_ORIGIN = enum.auto()
    COMBINE_ORIGIN = enum.auto()
    USE_ORIGIN_ONLY = enum.auto()
    USE_NEW_ONLY = enum.auto()


class MUSSModel:
    def __init__(self):
        self._saved_featuresets = []
        self._saved_models = []
        self._FEATURE_NAMES = ['MEAN_0', 'STD_0', 'MAX_0', 'RANGE_0', 'DOM_FREQ_0', 'FREQ_POWER_RATIO_ABOVE_3DOT5_0', 'DOM_FREQ_POWER_RATIO_0', 'ACTIVE_SAMPLES_0', 'ACTIVATIONS_0',
                               'STD_ACTIVATION_DURATIONS_0', "MEDIAN_G_ANGLE_X", "MEDIAN_G_ANGLE_Y", "MEDIAN_G_ANGLE_Z", "RANGE_G_ANGLE_X", "RANGE_G_ANGLE_Y", "RANGE_G_ANGLE_Z"]

    def get_feature_names(self):
        return self._FEATURE_NAMES

    def append_placement_suffix(self, input_feature, placement_name):
        new_cols = []
        for col in input_feature.columns:
            if col in self.get_feature_names() and placement_name != '':
                col = col + '_' + placement_name
            new_cols.append(col)
        input_feature.columns = new_cols
        return(input_feature)

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
            left_df = self.append_placement_suffix(left[0], left[1])
            right_df = self.append_placement_suffix(right[0], right[1])
            merged = left_df.merge(
                right_df, on=['HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'] + group_col)
            return (merged, '')
        if len(placement_names) == 1:
            combined_df = input_features[0]
            combined_feature_names = self.get_feature_names()
        else:
            tuple_results = functools.reduce(_combine, sequence)
            combined_df = tuple_results[0]
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
            selected_cols = ['HEADER_TIME_STAMP', 'START_TIME',
                             'STOP_TIME'] + group_col + feature_names
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
        result = {
            'HEADER_TIME_STAMP': [st],
            'START_TIME': [st],
            'STOP_TIME': [et]
        }

        subwin_samples = subwin_secs * sr
        X = extensions.numpy.atleast_float_2d(input_data.values[:, 1:4])

        if input_data.shape[0] < sr:
            for name in self._FEATURE_NAMES:
                result[name] = [np.nan]
        else:
            vm_feature_funcs = [
                accel.mean,
                accel.std,
                accel.max_value,
                accel.max_minus_min,
                functools.partial(accel.spectrum_features,
                                  sr=sr, n=1, preset='muss'),
                functools.partial(accel.stats_active_samples,
                                  threshold=activation_threshold)
            ]

            axis_feature_funcs = [
                functools.partial(accel.gravity_angle_stats,
                                  subwin_samples=subwin_samples, unit=ori_unit)
            ]

            X_vm = accel.vector_magnitude(X)

            X_vm_filtered = extensions.numpy.butterworth(
                X_vm, sr=sr, cut_offs=20, order=4, filter_type='low')
            X_filtered = extensions.numpy.butterworth(X, sr=sr, cut_offs=20,
                                                      order=4, filter_type='low')

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
        input_matrix, input_classes = sk_utils.shuffle(
            input_feature_arr, input_class_vec)
        classifier = svm.SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            tol=tol,
            probability=output_probability,
            class_weight=class_weight,
            verbose=verbose)
        scaler = sk_preprocessing.MinMaxScaler((-1, 1))
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

    def replace_overlapped_classes(self, origin_feature_arr, origin_class_vec, origin_labels, new_feature_arr, new_class_vec, new_labels):
        labels_keep_in_origin = list(filter(
            lambda l: l not in new_labels, origin_labels))
        filter_condition = np.isin(origin_class_vec,
                                   labels_keep_in_origin)
        origin_feature_arr = origin_feature_arr[filter_condition, :]
        origin_class_vec = origin_class_vec[filter_condition]

        input_feature_arr = np.concatenate(
            (origin_feature_arr, new_feature_arr), axis=0)
        input_class_vec = np.concatenate(
            (origin_class_vec, new_class_vec), axis=0)
        input_labels = np.unique(input_class_vec)
        return input_feature_arr, input_class_vec, input_labels

    def combine_overlapped_classes(self, origin_feature_arr, origin_class_vec, new_feature_arr, new_class_vec):
        input_feature_arr = np.concatenate(
            (origin_feature_arr, new_feature_arr), axis=0)
        input_class_vec = np.concatenate(
            (origin_class_vec, new_class_vec), axis=0)
        input_labels = np.unique(input_class_vec)
        return input_feature_arr, input_class_vec, input_labels

    def validate_classifier(self, input_feature, input_class, class_col, feature_names, placement_names, group_col, new_input_feature=None, new_input_class=None, strategy=Strategy.USE_ORIGIN_ONLY, parallel=False, **train_kwargs):
        input_feature_arr = input_feature[feature_names].values
        input_class_vec = input_class[class_col].values
        class_labels = np.unique(input_class_vec)
        if new_input_feature is not None and new_input_class is not None:
            new_feature_arr = new_input_feature[feature_names].values
            new_class_vec = new_input_class[class_col].values
            new_labels = np.unique(new_class_vec)
        else:
            strategy = Strategy.USE_ORIGIN_ONLY

        groups = input_class[group_col].values
        logo = sk_model_selection.LeaveOneGroupOut()

        output_class_vec = np.copy(input_class_vec)

        for train_split, test_split in logo.split(input_feature_arr, input_class_vec, groups):
            train_feature_arr = input_feature_arr[train_split, :]
            train_class_vec = input_class_vec[train_split]
            train_labels = np.unique(train_class_vec)
            if strategy == Strategy.REPLACE_ORIGIN:
                train_feature_arr, train_class_vec, _ = self.replace_overlapped_classes(
                    train_feature_arr, train_class_vec, train_labels, new_feature_arr, new_class_vec, new_labels)
            elif strategy == Strategy.COMBINE_ORIGIN:
                train_feature_arr, train_class_vec, _ = self.combine_overlapped_classes(
                    train_feature_arr, train_class_vec, new_feature_arr, new_class_vec)

            test_feature_arr = input_feature_arr[test_split, :]

            model = self._train_classifier(
                train_feature_arr, train_class_vec, feature_names=feature_names, placement_names=placement_names, ** train_kwargs)

            predict_class_vec = self._predict(
                test_feature_arr, classifier=model[0], scaler=model[1])

            output_class_vec[test_split] = predict_class_vec

        acc = sk_metrics.accuracy_score(input_class_vec, output_class_vec)
        return input_class_vec, output_class_vec, class_labels, acc

    def _plot_confusion_matrix(self, conf_matrix, size, fig=None):
        sns.set_style({
            'font.family': 'serif',
            'font.size': 8,
            'font.serif': 'Times New Roman'
        })
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 8
        mpl.rcParams['font.serif'] = ['Times New Roman']
        if fig is not None:
            plt.figure(fig.number)
            plt.gcf()
            plt.cla()
        else:
            plt.subplots(figsize=(4, 4))
        # plot confusion matrix
        g = sns.heatmap(conf_matrix, annot=True, cmap="Greys",
                        cbar=False, fmt='d', robust=True, linewidths=0.2)
        g.set(xlabel="Prediction", ylabel="Ground truth")
        plt.tight_layout()
        return plt.gcf()

    def get_confusion_matrix(self, input_class, predict_class, labels=None, graph=False, fig=None):
        conf_mat = sk_metrics.confusion_matrix(
            input_class, predict_class, labels=labels)
        conf_df = pd.DataFrame(conf_mat, columns=labels, index=labels)
        conf_df.index.rename(name='Ground Truth')
        if graph:
            result = self._plot_confusion_matrix(
                conf_df, size=(len(labels), len(labels)), fig=fig)
        else:
            result = conf_df
        return result

    def get_classification_report(self, input_class, predict_class, labels):
        return sk_metrics.classification_report(input_class, predict_class, labels=labels, output_dict=True)

    @staticmethod
    def get_inference_pipeline(*streams, name='muss-inference-pipeline', **kwargs):
        pipeline = arus_pipeline.Pipeline(
            max_processes=kwargs['max_processes'], scheduler=kwargs['scheduler'], name=name)
        for stream in streams:
            pipeline.add_stream(stream)
        pipeline.set_processor(muss_inference_processor, **kwargs)
        return pipeline

    @staticmethod
    def get_data_collection_pipeline(*streams, name='muss-data-collection-pipeline', **kwargs):
        pipeline = arus_pipeline.Pipeline(
            max_processes=kwargs['max_processes'], scheduler=kwargs['scheduler'], name=name)
        for stream in streams:
            pipeline.add_stream(stream)
        pipeline.set_processor(muss_data_collection_processor, **kwargs)
        return pipeline

    @staticmethod
    def get_mhealth_dataset_pipeline(*streams, name, **kwargs):
        pipeline = arus_pipeline.Pipeline(
            max_processes=kwargs['max_processes'], scheduler=kwargs['scheduler'], name=name)
        for stream in streams:
            pipeline.add_stream(stream)
        pipeline.set_processor(muss_mhealth_dataset_processor, **kwargs)
        return pipeline
