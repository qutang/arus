from .. import mhealth_format as mh
from .. import moment
import pytest
import pandas as pd
from concurrent import futures
import os
import numpy as np
import datetime as dt
import sys


@pytest.fixture(scope='module')
def test_data(spades_lab_data):
    sensor_files = spades_lab_data['subjects']['SPADES_1']['sensors']['DW']
    data = pd.concat(
        map(lambda f: pd.read_csv(f, parse_dates=[0]), sensor_files), sort=False)
    return data


@pytest.fixture(scope='module', params=['sensors', 'annotations'])
def test_file(request, spades_lab_data):
    if request.param == 'sensors':
        return spades_lab_data['subjects']['SPADES_2']['sensors']['DW'][0]
    else:
        return spades_lab_data['subjects']['SPADES_2']['annotations']['SPADESInLab'][0]


@pytest.fixture(scope='module')
def filepath_test_cases():
    valid_filepaths = [
        'D:\\data\\spades_lab\\SPADES_7\\MasterSynced\\2015\\11\\19\\16',
        'D:\\data\\spades_lab\\SPADES_7\\MasterSynced\\2015\\11\\19\\16\\',
        'D:/data/spades_lab/SPADES_7/MasterSynced/2015/11/19/16/',
        'D:/data/spades_lab/SPADES_7/Derived/2015/11/19/16/',
        'D:/data/spades_lab/SPADES_7/Derived/AllSensors/2015/11/19/16/',
        'D:/data/spades_lab/SPADES_7/MasterSynced/2015/11/19/16',
        '''D:/data/spades_lab/SPADES_7/MasterSynced/2015/11/19/16/
            ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.
            2015-11-19-16-00-00-000-M0500.sensor.csv''',
        '''D:/data/spades_lab/SPADES_7/MasterSynced/2015/11/19/16/
            ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.
            2015-11-19-16-00-00-000-M0500.sensor.csv.gz'''
    ]

    invalid_filepaths = [
        'C:\\',
        'C:/',
        'D:/data/spades_lab/SPADES_7/MasterSynced/2015/16',
        'D:/data/spades_lab/SPADES_7/MasterSynced/20/16',
        'D:/data/spades_lab/SPADES_7/Mastenced/2015/16/17/21/'
    ]

    valid_flat_filepaths = [
        'D:\\data\\spades_lab\\SPADES_7\\MasterSynced\\asdf.csv',
        'D:\\data\\spades_lab\\SPADES_7\\MasterSynced\\adfa.csv',
        'D:/data/spades_lab/SPADES_7/MasterSynced/sdf.csv',
        'D:/data/spades_lab/SPADES_7/Derived/dfew.csv',
        'D:/data/spades_lab/SPADES_7/Derived/AllSensors/dfsd.csv',
        '''D:/data/spades_lab/SPADES_7/MasterSynced/ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv''',
        '''D:/data/spades_lab/SPADES_7/MasterSynced/ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv.gz''',
    ]

    invalid_flat_filepaths = [
        'C:\\',
        'C:/',
        'D:/data/spades_lab/SPADES_7/MasterSyn/',
        'D:/data/spades_lab/SPADES_7/MasterSynced/20/16',
        'D:/data/spades_lab/SPADES_7/Mastenced/2015/16/17/21/',
        '''D:/data/spades_lab/SPADES_7/MasterSynced/2015/11/19/16/ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv'''
    ]

    valid_filenames = [
        'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv',
        'SPADESInLab.diego-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv',
        'SPADESInLab.DIEGO-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv',
        'SPADESInLab.DIEGO-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv.gz'
    ]

    invalid_filenames = [
        'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.annotation.csv',
        'Actig?raphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv',
        'ActigraphGT9X-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv',
        'ActigraphGT9X-AccelerationCalibrated-0,1,2.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv',
        'ActigraphGT9X-AccelerationCalibrated-NA.tas1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv',
        'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-0-000-M0500.sensor.csv',
        'SPADESInLab-sdfsdf.diego-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv'
    ]

    valid_sensor_type = [
        'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv',
        'SPADESInLab.diego-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv',
        'SPADESInLab.DIEGO-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv',
        'SPADESInLab.DIEGO-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv.gz',
    ]

    return valid_filepaths, invalid_filepaths, valid_flat_filepaths, invalid_flat_filepaths, valid_filenames, invalid_filenames, valid_sensor_type


class TestCore:
    def test_get_session_start_time(self, spades_lab_data):
        start_time = mh.get_session_start_time(
            'SPADES_12', spades_lab_data['meta']['root'])
        assert start_time.strftime(
            '%Y-%m-%d-%H-%M-%S') == '2015-12-14-11-00-00'

    def test_get_session_span(self, spades_lab_data):
        session_st, session_et = mh.get_session_span(
            'SPADES_1', spades_lab_data['meta']['root'])
        assert session_st.strftime(
            '%Y-%m-%d-%H-%M-%S') == '2015-09-24-14-00-00'
        assert session_et.strftime(
            '%Y-%m-%d-%H-%M-%S') == '2015-09-24-17-00-00'

    def test_get_date_folders(self, spades_lab_data):
        date_folders = sorted(mh.get_date_folders(
            'SPADES_1', spades_lab_data['meta']['root']))
        test_dates = [
            '2015/09/24/14',
            '2015/09/24/15',
            '2015/09/24/16',
        ]
        for date_folder, test_date in zip(date_folders, test_dates):
            assert test_date in date_folder.replace(os.sep, '/')


class TestHelper:
    def test_is_mhealth_filepath(self, filepath_test_cases):
        for test_case in filepath_test_cases[0]:
            assert mh.is_mhealth_filepath(test_case)

        for test_case in filepath_test_cases[1]:
            assert not mh.is_mhealth_filepath(test_case)

    def test_is_mhealth_flat_filepath(self, filepath_test_cases):
        for test_case in filepath_test_cases[2]:
            assert mh.is_mhealth_flat_filepath(test_case)

        for test_case in filepath_test_cases[3]:
            assert not mh.is_mhealth_flat_filepath(test_case)

    def test_is_mhealth_filename(self, filepath_test_cases):

        for test_case in filepath_test_cases[4]:
            assert mh.is_mhealth_filename(test_case)

        for test_case in filepath_test_cases[5]:
            assert not mh.is_mhealth_filename(test_case)

    @pytest.mark.skipif(sys.platform == 'linux', reason="does not run on linux")
    def test_parse_pid_from_filepath_win(self):
        correct_test_cases = [
            'D:\\data\\spades_lab\\SPADES_7\\MasterSynced\\2015\\11\\19\\16',
            'D:\\data\\spades_lab\\SPADES_7\\MasterSynced\\2015\\11\\19\\16\\',
            'D:/data/spades_lab/SPADES_7/MasterSynced/2015/11/19/16/',
            'D:/data/spades_lab/SPADES_7/Derived/2015/11/19/16/',
            'D:/data/spades_lab/SPADES_7/Derived/AllSensors/2015/11/19/16/',
            'D:/data/spades_lab/SPADES_7/MasterSynced/2015/11/19/16',
            '''D:/data/spades_lab/SPADES_7/MasterSynced/2015/11/19/16/
            ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.
            2015-11-19-16-00-00-000-M0500.sensor.csv''',
            'D:\\data\\spades_lab\\SPADES_7\\MasterSynced\\asdf.csv',
            'D:\\data\\spades_lab\\SPADES_7\\MasterSynced\\adfa.csv',
            'D:/data/spades_lab/SPADES_7/MasterSynced/sdf.csv',
            'D:/data/spades_lab/SPADES_7/Derived/dfew.csv',
            'D:/data/spades_lab/SPADES_7/Derived/AllSensors/dfsd.csv',
            '''D:/data/spades_lab/SPADES_7/MasterSynced/ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv''',
            '''D:/data/spades_lab/SPADES_7/MasterSynced/ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv.gz'''
        ]

        incorrect_test_cases = [
            'C:\\',
            'C:/',
            'D:/data/spades_lab/SPADES_7/MasterSynced/2015/16',
            'D:/data/spades_lab/SPADES_7/MasterSynced/20/16',
            'D:/data/spades_lab/SPADES_7/Mastenced/2015/16/17/21/'
        ]

        for test_case in correct_test_cases:
            print(test_case)
            assert mh.parse_pid_from_filepath(test_case) == 'SPADES_7'

        for test_case in incorrect_test_cases:
            with pytest.raises(mh.ParseError):
                mh.parse_pid_from_filepath(test_case)

    def test_parse_sensor_type_from_filepath(self, filepath_test_cases):
        for test_case in filepath_test_cases[6]:
            print(test_case)
            assert mh.parse_sensor_type_from_filepath(
                test_case) == 'ActigraphGT9X' or mh.parse_sensor_type_from_filepath(test_case) == 'SPADESInLab'

    def test_parse_data_type_from_filepath(self):
        sensor_test_cases = [
            'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv',
            'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv.gz',
        ]

        for test_case in sensor_test_cases:
            assert mh.parse_data_type_from_filepath(
                test_case) == 'AccelerationCalibrated'

    def test_parse_version_code_from_filepath(self):
        sensor_test_cases = [
            'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv',
            'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv.gz',
        ]

        for test_case in sensor_test_cases:
            assert mh.parse_version_code_from_filepath(test_case) == 'NA'

    def test_parse_sensor_id_from_filepath(self):
        sensor_test_cases = [
            'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv',
            'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv.gz',
        ]

        for test_case in sensor_test_cases:
            assert mh.parse_sensor_id_from_filepath(
                test_case) == 'TAS1E23150152'

    def test_parse_file_type_from_filepath(self):
        sensor_test_cases = [
            'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv',
            'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv.gz',
        ]

        annotation_test_cases = [
            'SPADESInLab.diego-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv',
            'SPADESInLab.DIEGO-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv',
            'SPADESInLab.DIEGO-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv.gz',
        ]

        for test_case in sensor_test_cases:
            assert mh.parse_filetype_from_filepath(test_case) == 'sensor'

        for test_case in annotation_test_cases:
            assert mh.parse_filetype_from_filepath(test_case) == 'annotation'

    def test_parse_timestamp_from_filepath(self):
        sensor_test_cases = [
            'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv',
            'ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150152-AccelerationCalibrated.2015-11-19-16-00-00-000-M0500.sensor.csv.gz',
        ]

        annotation_test_cases = [
            'SPADESInLab.diego-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv',
            'SPADESInLab.DIEGO-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv',
            'SPADESInLab.DIEGO-SPADESInLab.2015-11-19-16-00-00-000-M0500.annotation.csv.gz',
        ]

        for test_case in sensor_test_cases:
            ts = mh.parse_timestamp_from_filepath(test_case, ignore_tz=True)
            ts_unix = moment.Moment(ts).to_unix_timestamp()
            assert dt.datetime.fromtimestamp(ts_unix) == ts

        for test_case in annotation_test_cases:
            ts = mh.parse_timestamp_from_filepath(test_case, ignore_tz=True)
            ts_unix = moment.Moment(ts).to_unix_timestamp()
            assert dt.datetime.fromtimestamp(ts_unix) == ts

    def test_parse_date_from_filepath(self, spades_lab_data):
        test_sensor_files = spades_lab_data['subjects']['SPADES_1']['sensors']['DW']
        test_dates = [
            '2015-09-24-14-00-00',
            '2015-09-24-15-00-00'
            '2015-09-24-16-00-00'
        ]
        for test_case, test_date in zip(test_sensor_files, test_dates):
            date = mh.parse_date_from_filepath(test_case)
            date.strftime('%Y-%m-%d-%H-%M-%S') == test_date

        test_annotation_files = spades_lab_data['subjects']['SPADES_1']['annotations']['SPADESInLab']
        test_dates = [
            '2015-09-24-14-00-00',
            '2015-09-24-15-00-00'
            '2015-09-24-16-00-00'
        ]
        for test_case, test_date in zip(test_annotation_files, test_dates):
            date = mh.parse_date_from_filepath(test_case)
            date.strftime('%Y-%m-%d-%H-%M-%S') == test_date

    def test_parse_subject_path_from_filepath(self, spades_lab_data):
        test_sensor_file = spades_lab_data['subjects']['SPADES_1']['sensors']['DW'][0]
        subject_path = mh.parse_subject_path_from_filepath(test_sensor_file)
        assert subject_path.endswith('SPADES_1')
        test_annotation_file = spades_lab_data['subjects']['SPADES_1']['annotations']['SPADESInLab'][0]
        subject_path = mh.parse_subject_path_from_filepath(
            test_annotation_file)
        assert subject_path.endswith('SPADES_1')

    def test_transform_class_category(self, spades_lab_data):
        class_category = spades_lab_data['meta']['class_category']
        input_category = 'FINEST_ACTIVITIES'
        output_category = 'MUSS_3_POSTURES'
        input_label = 'Lying on the back'
        assert mh.transform_class_category(
            input_label, class_category, input_category, output_category) == 'Lying'


class TestIO:
    @pytest.mark.parametrize('append', [False, True])
    @pytest.mark.parametrize('date_folders', [False, True])
    @pytest.mark.parametrize('hourly', [False, True])
    @pytest.mark.parametrize('block', [False, True])
    def test_mhealthfilewriter(self, test_data, hourly, date_folders, append, block, tmpdir):
        hour = test_data.iloc[0, 0].to_pydatetime().hour
        dataset_folder = tmpdir.realpath()
        pid = 'test'
        writer = mh.MhealthFileWriter(
            dataset_folder, pid, hourly=hourly, date_folders=date_folders)
        writer.set_for_sensor('test', 'test', 'test', 'NA')
        if append:
            first_hour_filename = writer._get_output_filename(test_data)
            first_hour_folder = writer._get_output_folder(test_data)
            os.makedirs(first_hour_folder, exist_ok=True)
            existing_filepath = os.path.join(
                first_hour_folder, first_hour_filename)
            with open(existing_filepath, 'w'):
                pass
            assert os.path.exists(existing_filepath)

        results = writer.write_csv(test_data, append=append, block=block)
        if block:
            output_paths = results
            existings = [os.path.exists(f) for f in output_paths]
            np.testing.assert_array_equal(existings, True)
        else:
            done, _ = futures.wait(results)
            output_paths = [f.result() for f in done]
            existings = [os.path.exists(f) for f in output_paths]
            np.testing.assert_array_equal(existings, True)
        if not hourly:
            assert len(output_paths) == 1
        else:
            assert len(output_paths) == 3
        parent_folder_names = [os.path.basename(
            os.path.dirname(f)) for f in output_paths]
        if date_folders and hourly:
            np.testing.assert_array_equal(
                sorted(parent_folder_names), np.arange(start=hour, step=1, stop=17).astype(str))
        elif date_folders and not hourly:
            np.testing.assert_array_equal(
                sorted(parent_folder_names), str(hour))
        else:
            np.testing.assert_array_equal(
                parent_folder_names, [mh.MASTER_FOLDER]*len(output_paths))

    @pytest.mark.parametrize('chunksize', [None, 10])
    def test_mhealthfilereader(self, test_file, chunksize):
        reader = mh.MhealthFileReader(test_file)
        file_type = mh.parse_filetype_from_filepath(test_file)
        if file_type == 'sensor':
            datetime_cols = [0]
        else:
            datetime_cols = [0, 1, 2]
        reader.read_csv(chunksize=chunksize, datetime_cols=datetime_cols)
        results = []
        for data in reader.get_data():
            if chunksize is None:
                assert data.shape[0] > 10
            else:
                assert data.shape[0] == 10
            assert data.columns[0] == mh.TIMESTAMP_COL
            if file_type == 'annotation':
                np.testing.assert_array_equal(
                    data.columns[:3], mh.FEATURE_SET_TIMESTAMP_COLS)
            results.append(data)
            if len(results) == 10:
                break
        if chunksize is None:
            assert len(results) == 1
        else:
            assert len(results) > 5
