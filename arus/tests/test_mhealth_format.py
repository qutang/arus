from .. import mhealth_format as mh
import pytest
import pandas as pd
from concurrent import futures
import os
import numpy as np
import datetime as dt


@pytest.fixture(scope='module')
def test_data(spades_lab):
    sensor_files = spades_lab['subjects']['SPADES_1']['sensors']['DW']
    data = pd.concat(
        map(lambda f: pd.read_csv(f, parse_dates=[0]), sensor_files), sort=False)
    return data


@pytest.fixture(scope='module', params=['sensors', 'annotations'])
def test_file(request, spades_lab):
    if request.param == 'sensors':
        return spades_lab['subjects']['SPADES_2']['sensors']['DW'][0]
    else:
        return spades_lab['subjects']['SPADES_2']['annotations']['SPADESInLab'][0]


class TestHelper:
    def test_transform_class_category(self, spades_lab):
        class_category = spades_lab['meta']['class_category']
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
