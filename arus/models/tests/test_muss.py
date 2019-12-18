from ...testing import load_test_data
from ...core.libs.mhealth_format.io import read_data_csv
from ..muss import MUSSModel
import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn.preprocessing import MinMaxScaler


def test_muss_compute_features():
    filepath, sr = load_test_data(file_type='mhealth', sensor_type='sensor',
                                  file_num='single', exception_type='consistent_sr')
    df = read_data_csv(filepath)
    data = df.values[:, 1:]
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
    df = read_data_csv(filepath)
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
    df = read_data_csv(filepath)
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
    combined_result = muss.combine_features(
        result, result2, placement_names=['DW', 'DA'])
    np.testing.assert_array_equal(result.columns[0:3], [
                                  'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'])
    np.testing.assert_array_equal(combined_result.columns[3:], [
                                  name + '_DW' for name in muss.get_feature_names()] + [name + '_DA' for name in muss.get_feature_names()])
    np.testing.assert_array_equal(
        combined_result.shape, (result.shape[0], 3 + len(muss.get_feature_names()) * 2))


def test_muss_train_classifier():
    feature_filepath, _ = load_test_data(file_type='mhealth', sensor_type='feature',
                                         file_num='single', exception_type='single_placement')
    class_filepath, _ = load_test_data(file_type='mhealth', sensor_type='class_labels',
                                       file_num='single', exception_type='single_task')
    feature_set = pd.read_csv(
        feature_filepath, infer_datetime_format=True, parse_dates=[0, 1, 2])
    class_set = pd.read_csv(
        class_filepath, infer_datetime_format=True, parse_dates=[0, 1, 2])
    muss = MUSSModel()
    input_feature, input_class = muss.sync_feature_and_class(
        feature_set, class_set)
    model = muss.train_classifier(input_feature, input_class)
    assert len(model) == 3
    assert model[2] > 0.7
    assert type(model[0]) is svm.SVC
    assert type(model[1]) is MinMaxScaler
