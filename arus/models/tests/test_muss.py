from ...testing import load_test_data
from ... import mhealth_format as mh
from ..muss import MUSSModel
import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn.preprocessing import MinMaxScaler


def test_muss_compute_features():
    filepath, sr = load_test_data(file_type='mhealth', sensor_type='sensor',
                                  file_num='single', exception_type='consistent_sr')
    df = next(mh.MhealthFileReader(filepath).read_csv().get_data())
    st = df.iloc[0, 0]
    et = df.iloc[-1, 0]
    muss = MUSSModel()
    feature_vectors = muss.compute_features(df, sr=sr, st=st, et=et)
    np.testing.assert_array_equal(feature_vectors.columns[0:3], [
                                  'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'])
    np.testing.assert_array_equal(
        feature_vectors.columns[3:], muss.get_feature_names())


def test_muss_compute_features_grouped():
    filepath, sr = load_test_data(file_type='mhealth', sensor_type='sensor',
                                  file_num='single', exception_type='consistent_sr')
    df = next(mh.MhealthFileReader(filepath).read_csv().get_data())
    muss = MUSSModel()

    def _compute_features(grouped_df, sr):
        st = grouped_df.iloc[0, 0]
        et = grouped_df.iloc[-1, 0]
        feature_vectors = muss.compute_features(
            grouped_df, sr=sr, st=st, et=et)
        return feature_vectors

    result = df.groupby(pd.Grouper(key='HEADER_TIME_STAMP', freq='12s', closed='left')
                        ).apply(_compute_features, sr=sr)
    result = result.reset_index(drop=True)
    np.testing.assert_array_equal(result.columns[0:3], [
                                  'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'])
    np.testing.assert_array_equal(
        result.columns[3:], muss.get_feature_names())
    values = result['HEADER_TIME_STAMP'].diff().values[1:] / 10 ** 9
    values = values.astype('int')
    np.testing.assert_allclose(
        values, 12)


def test_muss_combine_features():
    filepath, sr = load_test_data(file_type='mhealth', sensor_type='sensor',
                                  file_num='single', exception_type='consistent_sr')
    df = next(mh.MhealthFileReader(filepath).read_csv().get_data())
    muss = MUSSModel()

    def _compute_features(grouped_df, sr):
        st = grouped_df.iloc[0, 0]
        et = grouped_df.iloc[-1, 0]
        feature_vectors = muss.compute_features(
            grouped_df, sr=sr, st=st, et=et)
        return feature_vectors

    result = df.groupby(pd.Grouper(key='HEADER_TIME_STAMP', freq='12s', closed='left')
                        ).apply(_compute_features, sr=sr)
    result = result.reset_index(drop=True)
    result2 = result.copy(deep=True)
    combined_result, combined_feature_names = muss.combine_features(
        result, result2, placement_names=['DW', 'DA'])
    np.testing.assert_array_equal(result.columns[0:3], [
                                  'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'])
    np.testing.assert_array_equal(combined_feature_names, [
                                  name + '_DW' for name in muss.get_feature_names()] + [name + '_DA' for name in muss.get_feature_names()])
    np.testing.assert_array_equal(
        combined_result.shape, (result.shape[0], 3 + len(muss.get_feature_names()) * 2))


def test_muss_train_classifier():
    muss = MUSSModel()
    feature_filepath, _ = load_test_data(file_type='mhealth', sensor_type='feature',
                                         file_num='single', exception_type='single_placement')
    class_filepath, _ = load_test_data(file_type='mhealth', sensor_type='class_labels',
                                       file_num='single', exception_type='multi_tasks')
    feature_set = pd.read_csv(
        feature_filepath, infer_datetime_format=True, parse_dates=[0, 1, 2])
    feature_set = feature_set[['HEADER_TIME_STAMP',
                               'START_TIME', 'STOP_TIME'] + muss.get_feature_names()]
    class_set = pd.read_csv(
        class_filepath, infer_datetime_format=True, parse_dates=[0, 1, 2])
    class_set = class_set[['HEADER_TIME_STAMP',
                           'START_TIME', 'STOP_TIME', 'MUSS_3_POSTURES']]
    input_feature, input_class = muss.sync_feature_and_class(
        feature_set, class_set)
    filtered_feature, filtered_class = muss.remove_classes(
        input_feature, input_class, class_col='MUSS_3_POSTURES', classes_to_remove=['Unknown', 'Transition'])
    model = muss.train_classifier(
        filtered_feature, filtered_class, class_col='MUSS_3_POSTURES', feature_names=muss.get_feature_names(), placement_names=['DW'])
    assert len(model) == 5
    assert model[2] > 0.7
    assert type(model[0]) is svm.SVC
    assert type(model[1]) is MinMaxScaler


def test_muss_validate_classifier():
    muss = MUSSModel()
    feature_filepath, _ = load_test_data(file_type='mhealth', sensor_type='feature',
                                         file_num='single', exception_type='single_placement')
    class_filepath, _ = load_test_data(file_type='mhealth', sensor_type='class_labels',
                                       file_num='single', exception_type='multi_tasks')
    feature_set = pd.read_csv(
        feature_filepath, infer_datetime_format=True, parse_dates=[0, 1, 2])
    feature_set = feature_set[['HEADER_TIME_STAMP',
                               'START_TIME', 'STOP_TIME', 'PID'] + muss.get_feature_names()]
    class_set = pd.read_csv(
        class_filepath, infer_datetime_format=True, parse_dates=[0, 1, 2])
    class_set = class_set[['HEADER_TIME_STAMP',
                           'START_TIME', 'STOP_TIME', 'PID', 'MUSS_3_POSTURES']]

    synced_feature, synced_class = muss.sync_feature_and_class(
        feature_set, class_set, group_col='PID')

    filtered_feature, filtered_class = muss.remove_classes(
        synced_feature, synced_class, class_col='MUSS_3_POSTURES', classes_to_remove=['Transition', 'Unknown'])

    groups_to_remove = filtered_class['PID'].unique()[10:]
    filtered_feature, filtered_class = muss.remove_groups(
        filtered_feature, filtered_class, group_col='PID', groups_to_remove=groups_to_remove)

    test_class, pred_class, class_labels, acc = muss.validate_classifier(
        filtered_feature, filtered_class, class_col='MUSS_3_POSTURES', feature_names=muss.get_feature_names(), placement_names=['DW'], group_col='PID')
    np.testing.assert_array_equal(test_class.shape, pred_class.shape)
    assert acc > 0.7


def test_muss_get_confusion_matrix():
    muss = MUSSModel()
    input_class = ['Sit', 'Stand', 'Stand', 'Walk']
    predict_class = ['Sit', 'Walk', 'Stand', 'Stand']
    conf_df = muss.get_confusion_matrix(input_class, predict_class, labels=[
        'Sit', 'Stand', 'Walk'], graph=False)
    np.testing.assert_array_equal(
        conf_df.values, np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]]))
