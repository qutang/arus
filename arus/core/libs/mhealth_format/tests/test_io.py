import pytest
import numpy as np
import pandas as pd
from ..data import rename_columns
from ..data import convert_datetime_columns_to_datetime64ms
from ..io import write_data_csv
import os
from glob import glob


@pytest.fixture
def large_sensor_data():
    sr = 30.0
    st = np.datetime64('2011-06-15T08:00:00.000').astype('datetime64[ms]')
    et = st + np.timedelta64(150, 'm')
    ts = np.arange(st, et, step=int(1000 / sr))
    values = np.random.random((ts.shape[0], 3))
    data = pd.DataFrame(data=values, index=ts, columns=['X', 'Y', 'Z'])
    data = data.reset_index()
    data = rename_columns(data, file_type='sensor')
    data = convert_datetime_columns_to_datetime64ms(data, file_type='sensor')
    return data


def test_write_data_csv_split(large_sensor_data, tmpdir):
    write_data_csv(large_sensor_data, output_folder=tmpdir, pid='TEMP', file_type='sensor',
                   sensor_or_annotation_type='Simulated', data_type='Accelerometer', sensor_or_annotator_id='Unknown', split_hours=True)
    assert len(glob(os.path.join(tmpdir, 'TEMP',
                                 '**', '*.sensor.csv*'), recursive=True)) == 3


def test_write_data_csv_nosplit(large_sensor_data, tmpdir):
    write_data_csv(large_sensor_data, output_folder=tmpdir, pid='TEMP', file_type='sensor',
                   sensor_or_annotation_type='Simulated', data_type='Accelerometer', sensor_or_annotator_id='Unknown', split_hours=False)
    assert len(glob(os.path.join(tmpdir, 'TEMP',
                                 '**', '*.sensor.csv*'), recursive=True)) == 1


def test_write_data_csv_nosplit_flat(large_sensor_data, tmpdir):
    write_data_csv(large_sensor_data, output_folder=tmpdir, pid='TEMP', file_type='sensor',
                   sensor_or_annotation_type='Simulated', data_type='Accelerometer', sensor_or_annotator_id='Unknown', split_hours=False, flat=True)
    assert len(glob(os.path.join(tmpdir, 'TEMP',
                                 'MasterSynced', '*.sensor.csv*'))) == 1
